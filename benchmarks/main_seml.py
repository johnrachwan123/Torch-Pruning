import sys, os
from typing import Callable

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append('/nfs/homedirs/rachwan/Torch-Pruning')
from functools import partial
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_pruning as tp
import registry
import engine.utils as utils

import logging
from sacred import Experiment
import numpy as np
import seml
import numpy

ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))


def progressive_pruning(pruner, model, speed_up, example_inputs):
    model.eval()
    base_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    current_speed_up = 1
    while current_speed_up < speed_up:
        pruner.step(interactive=False)
        pruned_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        current_speed_up = float(base_ops) / pruned_ops
        #print(current_speed_up)
    return current_speed_up

def eval(model, test_loader, device=None):
    correct = 0
    total = 0
    loss = 0
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss += F.cross_entropy(out, target, reduction="sum")
            pred = out.max(1)[1]
            correct += (pred == target).sum()
            total += len(target)
    return (correct / total).item(), (loss / total).item()


def train_model(
        model,
        train_loader,
        test_loader,
        epochs,
        lr,
        lr_decay_milestones,
        lr_decay_gamma=0.1,
        save_as=None,

        # For pruning
        weight_decay=5e-4,
        save_state_dict_only=True,
        pruner=None,
        device=None,
        args=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
    )
    milestones = [int(ms) for ms in lr_decay_milestones.split(",")]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=lr_decay_gamma
    )
    model.to(device)
    best_acc = -1
    for epoch in range(epochs):
        model.train()
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.cross_entropy(out, target)
            loss.backward()
            if pruner is not None:
                pruner.regularize(model)  # for sparsity learning
            optimizer.step()
            if i % 10 == 0 and args.verbose:
                args.logger.info(
                    "Epoch {:d}/{:d}, iter {:d}/{:d}, loss={:.4f}, lr={:.4f}".format(
                        epoch,
                        epochs,
                        i,
                        len(train_loader),
                        loss.item(),
                        optimizer.param_groups[0]["lr"],
                    )
                )

        model.eval()
        acc, val_loss = eval(model, test_loader, device=device)
        args.logger.info(
            "Epoch {:d}/{:d}, Acc={:.4f}, Val Loss={:.4f}, lr={:.4f}".format(
                epoch, epochs, acc, val_loss, optimizer.param_groups[0]["lr"]
            )
        )
        if best_acc < acc:
            os.makedirs(args.output_dir, exist_ok=True)
            if args.mode == "prune":
                if save_as is None:
                    save_as = os.path.join(args.output_dir,
                                           "{}_{}_{}_{}.pth".format(args.dataset, args.model, args.method, args.speed_up))

                if save_state_dict_only:
                    torch.save(model.state_dict(), save_as)
                else:
                    torch.save(model, save_as)
            elif args.mode == "pretrain":
                if save_as is None:
                    save_as = os.path.join(args.output_dir, "{}_{}.pth".format(args.dataset, args.model))
                torch.save(model.state_dict(), save_as)
            best_acc = acc
        scheduler.step()
    args.logger.info("Best Acc=%.4f" % (best_acc))


def get_pruner(model, example_inputs, args, loader):
    args.sparsity_learning = False
    if args.method == "random":
        imp = tp.importance.RandomImportance()
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.method == "snip":
        imp = tp.importance.SensitivityImportance()
        pruner_entry = partial(tp.pruner.GradientPruner, global_pruning=args.global_pruning, crit='SNIP',
                               dataloader=loader, Loss=torch.nn.CrossEntropyLoss())
    elif args.method == "crop":
        imp = tp.importance.SensitivityImportance()
        pruner_entry = partial(tp.pruner.GradientPruner, global_pruning=args.global_pruning, crit='CROP',
                               dataloader=loader, Loss=torch.nn.CrossEntropyLoss())
    elif args.method == "synflow":
        imp = tp.importance.SensitivityImportance()
        pruner_entry = partial(tp.pruner.GradientPruner, global_pruning=args.global_pruning, crit='SYNFLOW',
                               dataloader=loader)
    elif args.method == "l1":
        imp = tp.importance.MagnitudeImportance(p=1)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.method == "lamp":
        imp = tp.importance.LAMPImportance(p=2)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.method == "slim":
        args.sparsity_learning = True
        imp = tp.importance.BNScaleImportance()
        pruner_entry = partial(tp.pruner.BNScalePruner, reg=args.reg, global_pruning=args.global_pruning)
    elif args.method == "group_norm":
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, global_pruning=args.global_pruning)
    elif args.method == "group_sl":
        args.sparsity_learning = True
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, reg=args.reg, global_pruning=args.global_pruning)
    else:
        raise NotImplementedError

    # args.is_accum_importance = is_accum_importance
    unwrapped_parameters = []
    ignored_layers = []
    ch_sparsity_dict = {}
    # ignore output layers
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == args.num_classes:
            ignored_layers.append(m)
        elif isinstance(m, torch.nn.modules.conv._ConvNd) and m.out_channels == args.num_classes:
            ignored_layers.append(m)

    # Here we fix iterative_steps=200 to prune the model progressively with small steps
    # until the required speed up is achieved.
    pruner = pruner_entry(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=args.iterative_steps,
        ch_sparsity=1.0,
        ch_sparsity_dict=ch_sparsity_dict,
        max_ch_sparsity=args.max_sparsity,
        ignored_layers=ignored_layers,
        unwrapped_parameters=unwrapped_parameters,
    )
    return pruner

def main(args):
    results = {}
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Logger
    if args.mode == "prune":
        prefix = 'global' if args.global_pruning else 'local'
        logger_name = "{}-{}-{}-{}".format(args.dataset, prefix, args.method, args.model)
        args.output_dir = os.path.join(args.output_dir, args.dataset, args.mode, logger_name)
        log_file = "{}/{}.txt".format(args.output_dir, logger_name)
    elif args.mode == "pretrain":
        args.output_dir = os.path.join(args.output_dir, args.dataset, args.mode)
        logger_name = "{}-{}".format(args.dataset, args.model)
        log_file = "{}/{}.txt".format(args.output_dir, logger_name)
    elif args.mode == "test":
        log_file = None
    args.logger = utils.get_logger(logger_name, output=log_file)

    # Model & Dataset
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes, train_dst, val_dst, input_size = registry.get_dataset(
        args.dataset, data_root="data"
    )
    args.num_classes = num_classes
    model = registry.get_model(args.model, num_classes=num_classes, pretrained=True, target_dataset=args.dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dst,
        batch_size=args.batch_size,
        num_workers=4,
        drop_last=True,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        val_dst, batch_size=args.batch_size, num_workers=4
    )

    for k, v in utils.utils.flatten_dict(vars(args)).items():  # print args
        args.logger.info("%s: %s" % (k, v))

    if args.restore:
        model_location = '/nfs/homedirs/rachwan/Torch-Pruning/benchmarks/run/' + args.dataset + '/pretrain/'
        loaded = torch.load(model_location + args.dataset + '_' + args.model + '.pth', map_location="cpu")
        if isinstance(loaded, nn.Module):
            model = loaded
        else:
            model.load_state_dict(loaded)
        args.logger.info("Loading model from {restore}".format(restore=args.restore))
    model = model.to(args.device)

    ######################################################
    # Training / Pruning / Testing
    example_inputs = train_dst[0][0].unsqueeze(0).to(args.device)
    if args.mode == "pretrain":
        ops, params = tp.utils.count_ops_and_params(
            model, example_inputs=example_inputs,
        )
        args.logger.info("Params: {:.2f} M".format(params / 1e6))
        args.logger.info("ops: {:.2f} M".format(ops / 1e6))
        best_acc = train_model(
            model=model,
            epochs=args.total_epochs,
            lr=args.lr,
            lr_decay_milestones=args.lr_decay_milestones,
            train_loader=train_loader,
            test_loader=test_loader,
            args=args
        )
        results['params'] = params
        results['ops'] = ops
        results['best_acc'] = best_acc

    elif args.mode == "prune":
        model.eval()
        ori_ops, ori_size = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        ori_acc, ori_val_loss = eval(model, test_loader, device=args.device)
        pruner = get_pruner(model, example_inputs=example_inputs, args=args, loader=train_loader)

        # 0. Sparsity Learning
        if args.sparsity_learning:
            reg_pth = "reg_{}_{}_{}_{}.pth".format(args.dataset, args.model, args.method, args.reg)
            reg_pth = os.path.join(os.path.join(args.output_dir, reg_pth))
            if not args.sl_restore:
                args.logger.info("Regularizing...")
                train_model(
                    model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    epochs=args.sl_total_epochs,
                    lr=args.sl_lr,
                    lr_decay_milestones=args.sl_lr_decay_milestones,
                    lr_decay_gamma=args.lr_decay_gamma,
                    regularizer=pruner.regularize,
                    save_state_dict_only=True,
                    save_as=reg_pth,
                    args=args
                )
            args.logger.info("Loading sparsity model from {}...".format(reg_pth))
            model.load_state_dict(torch.load(reg_pth, map_location=args.device))

        # 1. Pruning
        model.eval()
        args.logger.info("Pruning...")
        progressive_pruning(pruner, model, speed_up=args.speed_up, example_inputs=example_inputs)
        del pruner  # remove reference
        args.logger.info(model)
        pruned_ops, pruned_size = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        pruned_acc, pruned_val_loss = eval(model, test_loader, device=args.device)

        args.logger.info(
            "Params: {:.2f} M => {:.2f} M ({:.2f}%)".format(
                ori_size / 1e6, pruned_size / 1e6, pruned_size / ori_size * 100
            )
        )
        args.logger.info(
            "FLOPs: {:.2f} M => {:.2f} M ({:.2f}%, {:.2f}X )".format(
                ori_ops / 1e6,
                pruned_ops / 1e6,
                pruned_ops / ori_ops * 100,
                ori_ops / pruned_ops,
            )
        )
        args.logger.info("Acc: {:.4f} => {:.4f}".format(ori_acc, pruned_acc))
        args.logger.info(
            "Val Loss: {:.4f} => {:.4f}".format(ori_val_loss, pruned_val_loss)
        )

        # 2. Finetuning
        args.logger.info("Finetuning...")
        best_acc = train_model(
            model,
            epochs=args.total_epochs,
            lr=args.lr,
            lr_decay_milestones=args.lr_decay_milestones,
            train_loader=train_loader,
            test_loader=test_loader,
            device=args.device,
            save_state_dict_only=False,
            args=args
        )

        results['params'] = pruned_size
        results['ops'] = pruned_ops
        results['best_acc'] = best_acc

    elif args.mode == "test":
        model.eval()
        ops, params = tp.utils.count_ops_and_params(
            model, example_inputs=example_inputs,
        )
        args.logger.info("Params: {:.2f} M".format(params / 1e6))
        args.logger.info("ops: {:.2f} M".format(ops / 1e6))
        acc, val_loss = eval(model, test_loader)
        args.logger.info("Acc: {:.4f} Val Loss: {:.4f}\n".format(acc, val_loss))
    return results


from dataclasses import dataclass


@ex.automain
def run(arguments):
    @dataclass
    class args:
        mode = arguments['mode']  # evaluate every n batches
        model = arguments['model']
        verbose = arguments['verbose']
        dataset = arguments['dataset']
        batch_size = arguments['batch_size']
        total_epochs = arguments['total_epochs']
        lr_decay_milestones = arguments['lr_decay_milestones']
        lr_decay_gamma = arguments['lr_decay_gamma']
        lr = arguments['lr']
        restore = arguments['restore']
        output_dir = arguments['output_dir']
        method = arguments['method']
        speed_up = arguments['speed_up']
        max_sparsity = arguments['max_sparsity']
        soft_keeping_ratio = arguments['soft_keeping_ratio']
        reg = arguments['reg']
        weight_decay = arguments['weight_decay']
        seed = arguments['seed']
        global_pruning = arguments['global_pruning']
        sl_total_epochs = arguments['sl_total_epochs']
        sl_lr = arguments['sl_lr']
        sl_lr_decay_milestones = arguments['sl_lr_decay_milestones']
        sl_reg_warmup = arguments['sl_reg_warmup']
        sl_restore = arguments['sl_restore']
        iterative_steps = arguments['iterative_steps']

    results = main(args)

    # the returned result will be written into the database
    return results
