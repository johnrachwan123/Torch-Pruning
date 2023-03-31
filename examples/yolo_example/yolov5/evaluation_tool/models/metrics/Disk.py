import torch
import os
import bz2

class Disk:
    ZERO_SIGMA = -1 * 1e6
    state_size = 0
    comp_size = 0
    size = 0
    jit_size = 0
    onnx_size = 0

    def compressed_size(self, model):
        with torch.no_grad():
            for name, tensor in model.named_parameters():
                if 'weight' in name:
                    nonzero = 0
                    if 'rho' in name:
                        nonzero = torch.sum(tensor != self.ZERO_SIGMA).item()
                    else:
                        nonzero = torch.sum(tensor != 0).item()
                    temp = tensor.view(tensor.shape[0], -1).detach()
                    m, n = temp.shape[0], temp.shape[1]
                    smallest = min(m, n)
                    self.comp_size += nonzero * 34 + 2 * (smallest + 1)
        return self.comp_size

    def get_dict_size(self, model):
        torch.save(model.state_dict(), 'temp')
        self.state_size = os.path.getsize('temp')
        os.remove('temp')
        return self.state_size

    def get_size(self, model):
        torch.save(model, 'temp')
        self.size = os.path.getsize('temp')
        os.remove('temp')
        return self.size

    def get_jit_size(self, model):
        model_scripted = torch.jit.script(model)  # Export to TorchScript
        model_scripted.save('temp')  # Save
        # torch.save(model, 'temp')
        self.jit_size = os.path.getsize('temp')
        os.remove('temp')
        return self.jit_size

    def get_onnx_size(self, model, input_dim):
        ofile = bz2.BZ2File('temp', 'wb')
        torch.onnx.export(model, torch.rand([1] + list(input_dim)).cuda(), ofile)
        ofile.close()
        self.onnx_size = os.path.getsize('temp')
        os.remove('temp')
        return self.onnx_size

