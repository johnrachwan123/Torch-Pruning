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
parser.add_argument('--percent', type=float, default=0.5)

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


def train_model(model, train_loader, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, 0.1)
    model.to(device)

    best_acc = -1
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
        scheduler.step()
    print("Best Acc=%.4f" % (best_acc))


def get_scores(crit, model, Loss, img, target):
    if crit == "SNIP":
        out = model(img)
        Loss = torch.nn.CrossEntropyLoss()
        loss = Loss(out, target)
        loss.backward()
    elif crit == 'SYNFLOW':
        out = model(torch.ones_like(img))
        out.sum().backward()
    elif crit == 'CROP':
        weights = []
        for layer in model.modules():
            if hasattr(layer, 'weight') and layer.weight.requires_grad == True:
                weights.append(layer.weight)
        grad_w = None
        grad_f = None
        for w in weights:
            w.requires_grad_(True)
        N = img.shape[0]
        import copy
        import torch.autograd as autograd

        outputs = model(img)
        loss = Loss(outputs, target)
        grad_w_p = autograd.grad(loss, weights, create_graph=False)
        if grad_w is None:
            grad_w = list(grad_w_p)
        else:
            for idx in range(len(grad_w)):
                grad_w[idx] += grad_w_p[idx]

        outputs = model(img)
        loss = Loss(outputs, target)
        grad_f = autograd.grad(loss, weights, create_graph=True)
        z = 0
        count = 0
        for layer in model.modules():
            if hasattr(layer, 'weight') and layer.weight.requires_grad == True:
                z += (grad_w[count] * grad_f[count]).sum()
                count += 1
        z.backward()


def prune_model(model, train_loader):
    model.cpu()
    DG = tp.DependencyGraph().build_dependency(model, torch.randn(1, 3, 32, 32))
    img, target = next(iter(train_loader))

    def prune_conv(conv, amount=0.2):
        # TODO: Do only one pass maybe by making the weights values * grads put in the weights
        # Get scores
        Loss = torch.nn.CrossEntropyLoss()
        get_scores('SNIP', model, Loss, img, target)
        strategy = tp.strategy.L1GradStrategy()
        pruning_index = strategy(conv.weight, amount=amount)
        plan = DG.get_pruning_plan(conv, tp.prune_conv_out_channel, pruning_index)
        plan.exec()

    def prune_conv_global(conv, key, amount=0.2):
        # TODO: This solution is not perfect because we are re-analyzing the gradients at every step, a better implementation could be to do everything in importance.py and basepruner.py to implement both gradient based crits and global pruning at the same time
        # Global Pruning
        Loss = torch.nn.CrossEntropyLoss()
        get_scores('CROP', model, Loss, img, target)
        # get elasticities
        grads_abs = {}
        for name, layer in model.named_modules():
            if isinstance(layer, resnet.BasicBlock):
                n = len(layer.conv1.weight)
                grads_abs[name + "conv1.weight"] = torch.norm(
                    torch.abs(layer.conv1.weight.view(n, -1) * layer.conv1.weight.grad.view(n, -1)), p=1, dim=1)
                n = len(layer.conv2.weight)
                grads_abs[name + "conv2.weight"] = torch.norm(
                    torch.abs(layer.conv2.weight.view(n, -1) * layer.conv2.weight.grad.view(n, -1)), p=1, dim=1)

        # for name, layer in model.named_modules():
        #     if hasattr(layer, 'weight') and layer.weight.requires_grad == True:
        #         n = len(layer.weight)
        #         grads_abs[name + ".weight"] = torch.norm(
        #             torch.abs(layer.weight.view(n, -1) * layer.weight.grad.view(n, -1)), p=1, dim=1)

        all_scores = torch.cat([torch.flatten(x) for _, x in grads_abs.items()])
        # threshold
        n_to_prune = int(args.percent * len(all_scores)) if args.percent < 1.0 else args.percent
        threshold = torch.kthvalue(all_scores, k=n_to_prune).values

        strategy = tp.strategy.L1GradStrategy()
        pruning_index = strategy.apply_global(grads_abs[key], threshold)
        plan = DG.get_pruning_plan(conv, tp.prune_conv_out_channel, pruning_index)
        plan.exec()

    block_prune_probs = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3]
    blk_id = 0
    for name, layer in model.named_modules():
        if isinstance(layer, resnet.BasicBlock):
            prune_conv_global(layer.conv1, name + "conv1.weight")
            prune_conv_global(layer.conv2, name + "conv2.weight")
            blk_id += 1
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
        prune_model(model, train_loader)
        print(model)
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.1fM" % (params / 1e6))
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
