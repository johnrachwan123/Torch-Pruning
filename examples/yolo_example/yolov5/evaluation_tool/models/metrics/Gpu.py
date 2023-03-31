import torch


class Gpu:
    gpu_ram = torch.cuda.memory_allocated(0)
    max_gpu_ram = torch.cuda.max_memory_allocated(0)

    def get_gpu_ram(self):
        self.gpu_ram = torch.cuda.memory_allocated(0)
        return self.gpu_ram * 1e-9

    def get_gpu_max_ram(self):
        self.max_gpu_ram = torch.cuda.max_memory_allocated(0)
        return self.max_gpu_ram * 1e-9
