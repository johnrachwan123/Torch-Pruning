import torch
import torchvision


class ResNet18(torch.nn.Module):
    def __init__(self, **kwargs):
        super(ResNet18, self).__init__()
        self.model = torchvision.models.resnet18()
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        return self.model(x)
