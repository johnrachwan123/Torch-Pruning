from torchvision.models import resnet18
from torchscan import summary, crawl_module
from torchscan.utils import aggregate_info


class Flops:
    model_info = None

    def measure(self, model, input):
        module_info = crawl_module(model, input)
        self.model_info = aggregate_info(module_info, 0)

    def get(self, info):
        '''
        info: Which information to get (flops, macs, dmas)
        '''

        return self.model_info['layers'][0][info]


if __name__ == "__main__":
    # from models.networks.ResNet18 import ResNet18

    device = 'cuda:0'
    model = resnet18().to(device)
    inpts = (3, 224, 224)
    flop = flops()
    flop.measure(model.eval(), inpts)
    print(flop.model_info)
    print(flop.get('dmas'))
