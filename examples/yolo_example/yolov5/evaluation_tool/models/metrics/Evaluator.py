from argparse import ArgumentParser
import torch
import torch_tensorrt

from evaluation_tool.models.metrics.Flops import Flops
from evaluation_tool.models.metrics.Gpu import Gpu
from evaluation_tool.models.metrics.Energy import Energy
from evaluation_tool.models.metrics.Disk import Disk
from evaluation_tool.models.metrics.Cost import Cost
from evaluation_tool.models.metrics.Duration import Duration
import math


class Evaluator:
    result = {}

    # Load evaluation modules
    flop = Flops()
    dur = Duration()
    cos = Cost()
    gp = Gpu()
    en = Energy()
    ds = Disk()

    def compile_results(self, model, inference=False, cost_multiplier=1e9):
        # if inference:
        # In Gega
        # self.result['flops'] = self.flop.get('flops') * 1e-9
        # self.result['macs'] = self.flop.get('macs') * 1e-9
        # self.result['dmas'] = self.flop.get('dmas') * 1e-9
        # Time in milliseconds
        self.result['time'] = self.dur.get_last_duration()
        # GPU RAM in GBs
        self.result['gpu'] = self.gp.get_gpu_max_ram()
        # Disk size in MBs
        self.result['disk'] = self.ds.compressed_size(model) * 1e-6
        # Cost in USD
        self.result['cost'] = self.cos.get_cost(math.ceil(self.result['time'] * cost_multiplier / (3600 * 1e3)))
        # Emissions in Grams
        self.result['emission'] = self.en.get_emissions() * 1e3
        # Energy in Wh
        # self.result['energy'] = self.en.get_energy() * 1e3

    def evaluate_inference(self, model, input_dim, device, iterations=1, batch_size=1, half=False, tensorrt=False):
        model.to(device)
        if half:
            model.half()
            temp_input = torch.rand([batch_size] + list(input_dim), dtype=torch.float16).to(device)
        else:
            temp_input = torch.rand([batch_size] + list(input_dim)).to(device)

        model.eval()
        # FLOP, MAC, DMA calculation

        # self.flop.measure(model.eval(), tuple(input_dim))
        # Time and Energy calculation
        if tensorrt and not half:
            model = torch_tensorrt.compile(model, inputs=[
                torch_tensorrt.Input(([batch_size] + list(input_dim)), dtype=torch.float32)],
                                           enabled_precisions=torch.float32,  # Run with FP32
                                           workspace_size=1 << 22
                                           )
        elif tensorrt and half:
            model = torch_tensorrt.compile(model, inputs=[
                torch_tensorrt.Input(([batch_size] + list(input_dim)), dtype=torch.half)],
                                           enabled_precisions=torch.half,  # Run with FP32
                                           workspace_size=1 << 22
                                           )
        # self.en.start()
        self.dur.start()
        for i in range(iterations):
            model(temp_input)
        self.dur.end()
        # self.en.end()
        self.dur.set_last_duration(self.dur.get_last_duration() / iterations)
        # Cost calculation
        self.cos.find_gpu(self.gp.get_gpu_max_ram(), 'GCP', 'On-demand')

        self.compile_results(model, inference=True)

    def evaluate_training(self, model, input_dim, loss, batch_size, device, iterations):
        model.to(device)
        model.train()
        temp_input = torch.rand([batch_size] + list(input_dim)).to(device, non_blocking=True)
        temp_output = torch.zeros([222, 6], dtype=int).to(device)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        for i in range(iterations):
            with torch.no_grad():

                model(temp_input)

        # Time and Energy calculation
        # self.en.start()
        self.dur.start()
        outputs = model(temp_input)
        loss, _ = loss(outputs,
                       temp_output)

        scaler.scale(loss).backward()
        self.dur.end()
        # self.en.end()
        self.dur.set_last_duration(self.dur.get_last_duration() / iterations)

        # Cost calculation
        self.cos.find_gpu(self.gp.get_gpu_max_ram(), 'GCP', 'On-demand')

        self.compile_results(model)

    def evaluate_full_training(self, trainer, model):

        # Time and Energy calculation
        self.en.start()
        self.dur.start()
        trainer.train()
        self.dur.end()
        self.en.end()

        # Cost calculation
        self.cos.find_gpu(self.gp.get_gpu_max_ram(), 'GCP', 'On-demand')

        self.compile_results(model)

    # Getters
    def get_metric(self, metric):
        return self.result[metric]

    def get_all_metrics(self):
        return self.result
