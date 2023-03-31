import torch
from torch import nn


class CrossEntropy(nn.Module):

    def __init__(self, **kwargs):
        super(CrossEntropy, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, output=None, target=None, weight_generator=None, **kwargs):
        return self.loss.forward(output, target)
