import warnings
import pprint

from utils.model_utils import *
from torch.optim.lr_scheduler import StepLR, OneCycleLR
from utils.constants import NETWORKS_DIR, LOSS_DIR, TRAINERS_DIR, OPTIMS, DATASETS

from argparse import ArgumentParser

from models.metrics.Evaluator import Evaluator

warnings.filterwarnings("ignore")

parser = ArgumentParser()
parser.add_argument('--model', type=str, default='ResNet18')
parser.add_argument('--loss', type=str, default='CrossEntropy')
parser.add_argument('--data_set', type=str, default='MNIST')
parser.add_argument('--optimizer', type=str, default='ADAM')
parser.add_argument('--train_scheme', type=str, default='SimpleTrainer')
parser.add_argument('--device', type=str, default='cuda')
#TODO: automatically get data sizes ?
parser.add_argument('--input_dim', type=int, nargs="+")
parser.add_argument('--output_dim', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=80)
parser.add_argument('--learning_rate', type=float, default=2e-3)
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--l2_reg', type=float, default=5e-5)
parser.add_argument("--tuning", action="store_true",
                    help="splits trainset into train and validationset, omits test set")
parser.add_argument('--random_shuffle_labels', action='store_true',
                        help="run with random-label experiment from zhang et al")
parser.add_argument("--preload_all_data", action="store_true", help="load all data into ram memory for speedups")

arguments = parser.parse_args()


def main():
    # Get Model
    if arguments.checkpoint is not None:
        model = torch.load(arguments.checkpoint).to(arguments.device)
    else:
        model: nn.Module = find_right_model(
            NETWORKS_DIR, arguments.model,
            device=arguments.device,
            input_dim=tuple(arguments.input_dim),
            output_dim=arguments.output_dim,
        ).to(arguments.device)

    # Get Loss
    loss = find_right_model(
        LOSS_DIR, arguments.loss,
        device=arguments.device
    )

    # get optimizer
    optimizer = find_right_model(
        OPTIMS, arguments.optimizer,
        params=model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.l2_reg,
    )

    # Get Data
    train_loader, test_loader = find_right_model(
        DATASETS, arguments.data_set,
        arguments=arguments
    )
    # Default Scheduler
    # scheduler = StepLR(optimizer, step_size=30000, gamma=0.2)
    scheduler = OneCycleLR(optimizer, max_lr=arguments.learning_rate,
                                 steps_per_epoch=len(train_loader), epochs=arguments.epochs)
    # Get Trainer
    if arguments.train_scheme == 'Default':
        # Placeholder
        trainer = None
    else:
        trainer = find_right_model(
            TRAINERS_DIR, arguments.train_scheme,
            model=model,
            loss=loss,
            optimizer=optimizer,
            device=arguments.device,
            epochs=arguments.epochs,
            train_loader=train_loader,
            test_loader=test_loader,
            scheduler=scheduler,
        )


    # Load evaluatior
    evaluator = Evaluator()

    # Evaluate model
    evaluator.evaluate_full_training(trainer, model)

    return evaluator.get_all_metrics()


if __name__ == '__main__':
    result = main()
    pprint.pprint(result)
