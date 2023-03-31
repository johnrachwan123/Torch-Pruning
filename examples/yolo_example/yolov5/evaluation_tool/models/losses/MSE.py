import torch
from torch import nn

class MSE(nn.Module):

    def __init__(self, device, l1_reg=0, lp_reg=0, **kwargs):
        super(MSE, self).__init__(device, **kwargs)
        self.loss = nn.MSELoss()

    def forward(self, output=None, target=None, weight_generator=None, **kwargs):
        return self.loss.forward(output, target.float())