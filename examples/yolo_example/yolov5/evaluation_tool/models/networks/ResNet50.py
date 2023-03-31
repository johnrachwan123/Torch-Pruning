import torch
import torchvision


class ResNet50(torch.nn.Module):
    def __init__(self, **kwargs):
        super(ResNet50, self).__init__()
        self.model = torchvision.models.resnet50()

    def forward(self, x):
        return self.model(x)
