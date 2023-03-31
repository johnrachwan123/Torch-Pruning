import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from cifar_resnet import ResNet18
import cifar_resnet as resnet

import torch_pruning as tp
import argparse
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, choices=['train', 'prune', 'test'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--total_epochs', type=int, default=100)
parser.add_argument('--step_size', type=int, default=70)
parser.add_argument('--round', type=int, default=1)
parser.add_argument('--prune_to', type=int, default=0)

args = parser.parse_args()


def get_dataloader():
    train_loader = torch.utils.data.DataLoader(
        CIFAR10('./data', train=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]), download=True), batch_size=args.batch_size, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        CIFAR10('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
        ]), download=True), batch_size=args.batch_size, num_workers=2)
    return train_loader, test_loader


def eval(model, test_loader):
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (img, target) in enumerate(test_loader):
            img = img.to(device)
            out = model(img)
            pred = out.max(1)[1].detach().cpu().numpy()
            target = target.cpu().numpy()
            correct += (pred == target).sum()
            total += len(target)
    return correct / total

def pre_train(model, train_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(train_loader):
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = F.cross_entropy(out, target)
            loss.backward()
            optimizer.step()
            if i % 10 == 0 and args.verbose:
                print("Epoch %d/%d, iter %d/%d, loss=%.4f" % (
                epoch, args.total_epochs, i, len(train_loader), loss.item()))
        scheduler.step()

def train_model(model, train_loader, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, 0.1)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.1, steps_per_epoch=len(train_loader), epochs=args.total_epochs)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_epochs)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, min_lr=0.001)

    model.to(device)

    best_acc = -1
    patience = 10
    for epoch in range(args.total_epochs):
        model.train()
        for i, (img, target) in enumerate(train_loader):
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = F.cross_entropy(out, target)
            loss.backward()
            optimizer.step()
            if i % 10 == 0 and args.verbose:
                print("Epoch %d/%d, iter %d/%d, loss=%.4f" % (
                epoch, args.total_epochs, i, len(train_loader), loss.item()))
        model.eval()
        acc = eval(model, test_loader)
        print("Epoch %d/%d, Acc=%.4f" % (epoch, args.total_epochs, acc))
        if best_acc < acc:
            torch.save(model, 'resnet18-round%d.pth' % (args.round))
            best_acc = acc
            patience = 10
        else:
            patience -= 1
        # if patience == 0:
        #     break
        scheduler.step(acc)
    print("Best Acc=%.4f" % (best_acc))


def prune_model(model, train_loader, prune_to=0):
    model.cuda()
    total_steps = 1000
    ignored_layers = []
    if prune_to > 0:
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

        model.apply(weight_reset)

        pre_train(model, train_loader, prune_to)

    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
            ignored_layers.append(m)
    example_inputs = torch.randn(1, 3, 32, 32).cuda()
    # imp = tp.importance.MagnitudeImportance(p=1)
    imp = tp.importance.MagnitudeImportance()
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 10:
            ignored_layers.append(m)

    # Compile modules
    module_list = []
    for name, layer in model.named_modules():
        if isinstance(layer, resnet.BasicBlock):
            module_list.append(layer.conv1)
            module_list.append(layer.conv2)

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        crit='SNIP',
        dataloader=train_loader,
        Loss=torch.nn.CrossEntropyLoss(),
        iterative_steps=total_steps,
        ch_sparsity=0.31,
        ignored_layers=ignored_layers,
        module_list=module_list,
        iterations=1,
        backward_needed=False,
        global_pruning=False
    )

    for i in range(total_steps):
        pruner.step()
        ori_size = tp.utils.count_params(model)

    return model


def main():
    train_loader, test_loader = get_dataloader()
    if args.mode == 'train':
        args.round = 0
        model = ResNet18(num_classes=10)
        train_model(model, train_loader, test_loader)
    elif args.mode == 'prune':
        previous_ckpt = 'resnet18-round%d.pth' % (args.round - 1)
        print("Pruning round %d, load model from %s" % (args.round, previous_ckpt))
        model = torch.load(previous_ckpt)
        prune_model(model, train_loader, prune_to=args.prune_to)
        print(model)
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.1fM" % (params))
        train_model(model, train_loader, test_loader)
    elif args.mode == 'test':
        ckpt = 'resnet18-round%d.pth' % (args.round)
        print("Load model from %s" % (ckpt))
        model = torch.load(ckpt)
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.1fM" % (params / 1e6))
        acc = eval(model, test_loader)
        print("Acc=%.4f\n" % (acc))


if __name__ == '__main__':
    main()