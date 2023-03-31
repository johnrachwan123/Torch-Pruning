import warnings
import pprint

from utils.model_utils import *
from utils.constants import NETWORKS_DIR

from argparse import ArgumentParser

from models.metrics.Evaluator import Evaluator

warnings.filterwarnings("ignore")

parser = ArgumentParser()
parser.add_argument('--model', type=str, default='ResNet18')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--input_dim', type=int, nargs="+")
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--output_dim', type=int)
arguments = parser.parse_args()


def main():
    if arguments.checkpoint is not None:
        model = torch.load(arguments.checkpoint).to(arguments.device)
    else:
        model: nn.Module = find_right_model(
            NETWORKS_DIR, arguments.model,
            device=arguments.device,
            input_dim=tuple(arguments.input_dim),
            output_dim=arguments.output_dim,
        ).to(arguments.device)

    # Load evaluatior
    evaluator = Evaluator()

    # Evaluate model
    evaluator.evaluate_inference(model, tuple(arguments.input_dim), arguments.device)

    return evaluator.get_all_metrics()


if __name__ == '__main__':
    result = main()
    pprint.pprint(result)
