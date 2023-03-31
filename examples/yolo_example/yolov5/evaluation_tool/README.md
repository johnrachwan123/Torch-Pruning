# Evaluation-Tool
Evaluation tool that evaluates time, disk, gpu, energy, and cost efficiency metrics for both inference and training.
This tool has 3 possible options:

1. Evaluate inference metrics of the model
2. Evaluate the training iteration metrics of the model
3. Evaluate the full training metrics of the model

An example of each option can be found in evaluate_inference.py, evaluate_training.py, and evaluate_full_training.py respectively.

The inference example can be run as follows:

    python evaluate_inference.py --model ResNet18 --input_dim 3 32 32 --output_dim 10

The training example can be run as follows:

    python evaluate_training.py --model ResNet18 --input_dim 3 32 32 --output_dim 10 --batch_size 128
    
The full training example can be run as follows:

    python evaluate_full_training.py --model ResNet18 --data_set CIFAR10 --input_dim 3 32 32 --output_dim 10 --batch_size 128 --epochs 1
