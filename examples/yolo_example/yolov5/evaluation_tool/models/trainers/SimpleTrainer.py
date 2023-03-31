import argparse
import sys
import time
from scipy import stats
import numpy as np

import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import StepLR, OneCycleLR

class SimpleTrainer:
    """
    Implements generalised computer vision classification with pruning
    """

    def __init__(self,
                 model: torch.nn.Module,
                 loss: torch.nn.Module,
                 optimizer: Optimizer,
                 device,
                 epochs: int,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 scheduler: StepLR,
                 run_name: str = 'test_delete'
                 # pruner: RigLScheduler
                 ):
        self.epochs = epochs
        self.diff = 0
        self._test_loader = test_loader
        self._train_loader = train_loader
        self._test_model = None
        self._fim_loader = None
        self.gradient_adtest = []
        self.loss_test = []
        self._stable = False
        self._overlap_queue = []
        self._loss_function = loss
        self._model = model
        self._optimizer = optimizer
        self._device = device
        self._global_steps = 0
        self.patience = 0
        self._acc_buffer = []
        self._loss_buffer = []
        self._elapsed_buffer = []
        self._scheduler = scheduler
        self.gt = []
        self.pred = []
        # self._pruner = pruner
        self.ts = None
        self.old_score = None
        self.old_grads = None
        self.gradient_flow = 0
        self.weights = None
        self._variance = 0
        self.mask2 = None
        self.newgrad = None
        self.newweight = None
        self.all_scores = None
        self.scores = []
        self.count = 0
        self._step = 0.97
        self._percentage = 0.999
        self.threshold = None

        ## Metrics for SEML ##
        self.test_acc = None
        self.train_acc = None
        self.test_loss = None
        self.train_loss = None
        self.sparse_weight = None
        self.sparse_node = None
        self.sparse_hm = None
        self.sparse_log_disk_size = None
        self.time_gpu = None
        self.flops_per_sample = None
        self.flops_log_cum = None
        self.gpu_ram = None
        self.max_gpu_ram = None
        self.batch_time = None

    def weight_reset(self):
        reset_parameters = getattr(self._model, "reset_parameters", None)
        if callable(reset_parameters):
            self._model.reset_parameters()

    def _batch_iteration(self,
                         x: torch.Tensor,
                         y: torch.Tensor,
                         train: bool = True):
        """ one iteration of forward-backward """

        # unpack
        x, y = x.to(self._device).float(), y.to(self._device)

        # forward pass
        accuracy, loss, out = self._forward_pass(x, y, train=train)
        # backward pass
        if train:
            self._backward_pass(loss)

        # free memory
        for tens in [out, y, x, loss]:
            tens.detach()

        return accuracy, loss.item(), time

    def _forward_pass(self,
                      x: torch.Tensor,
                      y: torch.Tensor,
                      train: bool = True):
        """ implementation of a forward pass """

        if train:
            self._optimizer.zero_grad()

        out = self._model(x).squeeze()
        loss = self._loss_function(
            output=out,
            target=y,
            weight_generator=self._model.parameters(),
            model=self._model
        )
        accuracy = self._get_accuracy(out, y)
        return accuracy, loss, out

    def _backward_pass(self, loss):
        """ implementation of a backward pass """

        loss.backward()
        self._optimizer.step()

    def smooth(self, scalars, weight):  # Weight between 0 and 1
        last = scalars[0]  # First value in the plot (first timestep)
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
            smoothed.append(smoothed_val)  # Save it
            last = smoothed_val  # Anchor the last smoothed value

        return smoothed

    def _epoch_iteration(self):
        """ implementation of an epoch """
        self._model.train()
        print("\n")
        div = []
        mean_abs_mag_grad = 0
        gradient_norm = []
        gradient_adtest = []
        loss_test = []
        for batch_num, batch in enumerate(self._train_loader):
            print(f"\rTraining... {batch_num}/{len(self._train_loader)}", end='')

            # Perform one batch iteration
            acc, loss, elapsed = self._batch_iteration(*batch, self._model.training)

            self._acc_buffer.append(acc)
            self._loss_buffer.append(loss)

            loss_test.append(loss)

            self._elapsed_buffer.append(elapsed)

            # self._log(batch_num)

            self._scheduler.step()

            self._optimizer.zero_grad()

        self._model.eval()

    def _log(self):
        """ logs to terminal and tensorboard if the time is right"""

        # validate on test and train set
        train_acc, train_loss = np.mean(self._acc_buffer), np.mean(self._loss_buffer)
        test_acc, test_loss, test_elapsed = self.validate()
        print("Test Accuracy: " + str(test_acc))
        self._elapsed_buffer += test_elapsed

        # reset for next log
        self._acc_buffer, self._loss_buffer, self._elapsed_buffer = [], [], []

    def validate(self):
        """ validates the model on test set """

        print("\n")

        # init test mode
        self._model.eval()
        cum_acc, cum_loss, cum_elapsed = [], [], []

        with torch.no_grad():
            for batch_num, batch in enumerate(self._test_loader):
                acc, loss, elapsed = self._batch_iteration(*batch, self._model.training)
                cum_acc.append(acc)
                cum_loss.append(loss),
                cum_elapsed.append(elapsed)
                print(f"\rEvaluating... {batch_num}/{len(self._test_loader)}", end='')
        print("\n")

        # put back into train mode
        self._model.train()

        return float(np.mean(cum_acc)), float(np.mean(cum_loss)), cum_elapsed

    def _add_metrics(self, test_acc, test_loss, train_acc, train_loss):
        """
        save metrics
        """

        sparsity = self._model.pruned_percentage
        spasity_index = 2 * ((sparsity * test_acc) / (1e-8 + sparsity + test_acc))

        if self.train_acc is not None:
            if self.train_acc < train_acc:
                self.train_acc = train_acc
        else:
            self.train_acc = train_acc

        if self.test_acc is not None:
            if self.test_acc < test_acc:
                self.test_acc = test_acc
        else:
            self.test_acc = test_acc
        self.train_loss = train_loss
        self.test_loss = test_loss
        self.sparse_weight = sparsity
        self.sparse_node = self._model.structural_sparsity
        self.sparse_hm = spasity_index
        self.sparse_log_disk_size = np.log(self._model.compressed_size)
        self.time_gpu = np.mean(self._elapsed_buffer)
        if torch.cuda.is_available():
            self.gpu_ram = torch.cuda.memory_allocated(0)
            self.max_gpu_ram = torch.cuda.max_memory_allocated(0)

    def train(self):
        """ main training function """

        self._model.train()

        # do training
        for epoch in range(0, self.epochs):
            print(f"\n\nEPOCH {epoch}  \n\n")

            # do epoch
            self._epoch_iteration()

            # do evaluation
            self._log()

    @staticmethod
    def _get_accuracy(output, y):
        # predictions = torch.round(output)
        predictions = output.argmax(dim=-1, keepdim=True).view_as(y)
        correct = y.eq(predictions).sum().item()
        return correct / output.shape[0]
