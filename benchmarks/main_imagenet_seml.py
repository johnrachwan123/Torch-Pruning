import datetime
import os, sys
import time
import warnings
import registry
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append('/nfs/homedirs/rachwan/Torch-Pruning')
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

from engine.utils.imagenet_utils import presets, transforms, utils, sampler
import torch
import torch.utils.data
import torchvision
#from sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode

import torch_pruning as tp 
from functools import partial
import logging
from sacred import Experiment
import numpy as np
import seml
import numpy
# import dill
#
# torch.serialization.pickle_module = dill

ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))

def prune_to_target_flops(pruner, model, speed_up, example_inputs, args):
    model.eval()
    base_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    current_speed_up = 1
    while current_speed_up < speed_up:
        pruner.step()
        if 'vit' in args.model:
            model.hidden_dim = model.conv_proj.out_channels
        model.cpu()
        pruned_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        current_speed_up = float(base_ops) / pruned_ops
    return pruned_ops

def get_pruner(model, example_inputs, args, loader):
    unwrapped_parameters = (
        [model.encoder.pos_embedding, model.class_token] if "vit" in args.model else None
    )
    sparsity_learning = False
    data_dependency = False
    if args.method == "random":
        imp = tp.importance.RandomImportance()
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.method == "snip":
        imp = tp.importance.SensitivityImportance()
        pruner_entry = partial(tp.pruner.GradientPruner, global_pruning=args.global_pruning, crit='SNIP',
                               dataloader=loader, Loss=torch.nn.CrossEntropyLoss())
    elif args.method == "crop":
        imp = tp.importance.SensitivityImportance()
        pruner_entry = partial(tp.pruner.GradientPruner, global_pruning=args.global_pruning, crit='CROP',
                               dataloader=loader, Loss=torch.nn.CrossEntropyLoss())
    elif args.method == "synflow":
        imp = tp.importance.SensitivityImportance()
        pruner_entry = partial(tp.pruner.GradientPruner, global_pruning=args.global_pruning, crit='SYNFLOW',
                               dataloader=loader)
    elif args.method == "l1":
        imp = tp.importance.MagnitudeImportance(p=1)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.method == "lamp":
        imp = tp.importance.LAMPImportance(p=2)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.method == "slim":
        sparsity_learning = True
        imp = tp.importance.BNScaleImportance()
        pruner_entry = partial(tp.pruner.BNScalePruner, reg=args.reg, global_pruning=args.global_pruning)
    elif args.method == "group_norm":
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, global_pruning=args.global_pruning)
    elif args.method == "group_sl":
        sparsity_learning = True
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, reg=args.reg, global_pruning=args.global_pruning)
    else:
        raise NotImplementedError
    args.data_dependency = data_dependency
    args.sparsity_learning = sparsity_learning
    ignored_layers = []
    ch_sparsity_dict = {}
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
            ignored_layers.append(m)
    round_to = None
    if 'vit' in args.model:
        round_to = model.encoder.layers[0].num_heads
    pruner = pruner_entry(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=100,
        ch_sparsity=1.0,
        ch_sparsity_dict=ch_sparsity_dict,
        max_ch_sparsity=args.max_ch_sparsity,
        ignored_layers=ignored_layers,
        round_to=round_to,
        unwrapped_parameters=unwrapped_parameters,
    )
    return pruner



def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None, regularizer=None, recover=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if regularizer:
                scaler.unscale_(optimizer)
                regularizer(model)
            #if recover:
            #    recover(model.module)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if regularizer:
                regularizer(model)
            if recover:
                recover(model.module)
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))


def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(traindir, valdir, args):
    # Data loading code
    print("Loading data...")
    resize_size, crop_size = (342, 299) if args.model == 'inception_v3' else (256, 224)

    print("Loading training data...")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_train from {}".format(cache_path))
        dataset, _ = torch.load(cache_path)
    else:
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            presets.ClassificationPresetTrain(crop_size=crop_size, auto_augment_policy=auto_augment_policy,
                                              random_erase_prob=random_erase_prob))
        if args.cache_dataset:
            print("Saving dataset_train to {}...".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Data loading took", time.time() - st)

    print("Loading validation data...")
    cache_path = _get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_test from {}".format(cache_path))
        dataset_test, _ = torch.load(cache_path)
    else:
        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            presets.ClassificationPresetEval(crop_size=crop_size, resize_size=resize_size))
        if args.cache_dataset:
            print("Saving dataset_test to {}...".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders...")
    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    #     test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    # else:
    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler

def main(args):
    args.output_dir = os.path.join('/nfs/homedirs/rachwan/Torch-Pruning/benchmarks/run/imagenet/prune/', '')

    if args.output_dir:
        utils.mkdir(args.output_dir)

    # utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)

    collate_fn = None
    num_classes = len(dataset.classes)
    mixup_transforms = []
    if args.mixup_alpha > 0.0:
        mixup_transforms.append(transforms.RandomMixup(num_classes, p=1.0, alpha=args.mixup_alpha))
    if args.cutmix_alpha > 0.0:
        mixup_transforms.append(transforms.RandomCutmix(num_classes, p=1.0, alpha=args.cutmix_alpha))
    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)

        def collate_fn(batch):
            return mixupcutmix(*default_collate(batch))

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )
    print("Creating model")
    model = registry.get_model(num_classes=1000, name=args.model, pretrained=args.pretrained, target_dataset='imagenet') #torchvision.models.__dict__[args.model](pretrained=args.pretrained) #torchvision.models.get_model(args.model, weights=args.weights, num_classes=num_classes)
    model.eval()
    print("="*16)
    print(model)
    example_inputs = torch.randn(1, 3, 224, 224)
    base_ops, base_params = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    print("Params: {:.4f} M".format(base_params / 1e6))
    print("ops: {:.4f} G".format(base_ops / 1e9))
    print("="*16)
    if args.prune:
        pruner = get_pruner(model, example_inputs=example_inputs, args=args, loader=data_loader)
        if args.sparsity_learning:
            if args.sl_resume:
                print("Loading sparse model from {}...".format(args.sl_resume))
                model.load_state_dict( torch.load(args.sl_resume, map_location='cpu')['model'] )
            else:
                print("Sparsifying model...")
                if args.sl_lr is None: args.sl_lr = args.lr
                if args.sl_lr_step_size is None: args.sl_lr_step_size = args.lr_step_size
                if args.sl_lr_warmup_epochs is None: args.sl_lr_warmup_epochs = args.lr_warmup_epochs
                if args.sl_epochs is None: args.sl_epochs = args.epochs
                train(model, args.sl_epochs, 
                                        lr=args.sl_lr, lr_step_size=args.sl_lr_step_size, lr_warmup_epochs=args.sl_lr_warmup_epochs, 
                                        train_sampler=train_sampler, data_loader=data_loader, data_loader_test=data_loader_test, 
                                        device=device, args=args, regularizer=pruner.regularize, state_dict_only=True)
                #model.load_state_dict( torch.load('regularized_{:.4f}_best.pth'.format(args.reg), map_location='cpu')['model'] )
                #utils.save_on_master(
                #    model_without_ddp.state_dict(),
                #    os.path.join(args.output_dir, 'regularized-{:.4f}.pth'.format(args.reg)))

        model = model.to('cpu')
        print("Pruning model...")
        prune_to_target_flops(pruner, model, args.target_flops, example_inputs, args)
        pruned_ops, pruned_size = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        print("="*16)
        print("After pruning:")
        print(model)
        print("Params: {:.2f} M => {:.2f} M ({:.2f}%)".format(base_params / 1e6, pruned_size / 1e6, pruned_size / base_params * 100))
        print("Ops: {:.2f} G => {:.2f} G ({:.2f}%, {:.2f}X )".format(base_ops / 1e9, pruned_ops / 1e9, pruned_ops / base_ops * 100, base_ops / pruned_ops))
        print("="*16)
    
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)
    print("Finetuning..." if args.prune else "Training...")
    best_acc = train(model, args.epochs,
            lr=args.lr, lr_step_size=args.lr_step_size, lr_warmup_epochs=args.lr_warmup_epochs, 
            train_sampler=train_sampler, data_loader=data_loader, data_loader_test=data_loader_test, 
            device=device, args=args, regularizer=None, state_dict_only=(not args.prune))
    results = {}
    results['params'] = pruned_size
    results['ops'] = pruned_ops
    results['best_acc'] = best_acc
    return results

def train(
    model, 
    epochs, 
    lr, lr_step_size, lr_warmup_epochs, 
    train_sampler, data_loader, data_loader_test, 
    device, args, regularizer=None, state_dict_only=True, recover=None):

    model.to(device)
    # if args.distributed and args.sync_bn:
    #     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.label_smoothing>0:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    weight_decay = args.weight_decay if regularizer is None else 0
    bias_weight_decay = args.bias_weight_decay if regularizer is None else 0
    norm_weight_decay = args.norm_weight_decay if regularizer is None else 0

    custom_keys_weight_decay = []
    if bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    parameters = utils.set_weight_decay(
        model,
        weight_decay,
        norm_weight_decay=norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=args.momentum,
            weight_decay=weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=lr, momentum=args.momentum, weight_decay=weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = model
    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent from other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and ommit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        else:
            evaluate(model, criterion, data_loader_test, device=device)
        return
    
    start_time = time.time()
    best_acc = 0
    prefix = '' if regularizer is None else 'regularized_{:e}_'.format(args.reg)
    for epoch in range(args.start_epoch, epochs):
        # if args.distributed:
        #     train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema, scaler, regularizer, recover=recover)
        lr_scheduler.step()
        acc = evaluate(model, criterion, data_loader_test, device=device)
        if model_ema:
            acc = evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        if args.output_dir:
            checkpoint = {
                "model": model,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                # "args": args,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            if acc>best_acc:
                best_acc=acc
                utils.save_on_master(checkpoint, os.path.join(args.output_dir,
                                                              prefix + "{}_{}_{}_best.pth".format(args.model,
                                                                                                  args.method,
                                                                                                  args.target_flops)))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir,
                                                          prefix + "{}_{}_{}_latest.pth".format(args.model,
                                                                                                args.method,
                                                                                                args.target_flops)))
        print("Epoch {}/{}, Current Best Acc = {:.6f}".format(epoch, epochs, best_acc))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")
    return best_acc

from dataclasses import dataclass


@ex.automain
def run(arguments):
    @dataclass
    class args:
        data_path = arguments['data_path']
        model = arguments['model']
        pretrained = arguments['pretrained']

        device = arguments['device']
        batch_size = arguments['batch_size']
        epochs = arguments['epochs']
        workers = arguments['workers']
        opt = arguments['opt']
        lr = arguments['lr']
        momentum = arguments['momentum']
        weight_decay = arguments['weight_decay']
        norm_weight_decay = arguments['norm_weight_decay']
        bias_weight_decay = arguments['bias_weight_decay']
        transformer_embedding_decay = arguments['transformer_embedding_decay']
        label_smoothing = arguments['label_smoothing']
        mixup_alpha = arguments['mixup_alpha']
        cutmix_alpha = arguments['cutmix_alpha']
        lr_scheduler = arguments['lr_scheduler']
        lr_warmup_epochs = arguments['lr_warmup_epochs']
        lr_warmup_method = arguments['lr_warmup_method']
        lr_warmup_decay = arguments['lr_warmup_decay']
        lr_step_size = arguments['lr_step_size']
        lr_gamma = arguments['lr_gamma']
        lr_min = arguments['lr_min']
        print_freq = arguments['print_freq']
        output_dir = arguments['output_dir']
        resume = arguments['resume']
        start_epoch = arguments['start_epoch']
        cache_dataset = arguments['cache_dataset']
        sync_bn = arguments['sync_bn']
        test_only = arguments['test_only']
        auto_augment = arguments['auto_augment']
        ra_magnitude = arguments['ra_magnitude']
        augmix_severity = arguments['augmix_severity']
        random_erase = arguments['random_erase']
        amp = arguments['amp']
        world_size = arguments['world_size']
        dist_url = arguments['dist_url']
        model_ema = arguments['model_ema']
        model_ema_steps = arguments['model_ema_steps']
        model_ema_decay = arguments['model_ema_decay']
        use_deterministic_algorithms = arguments['use_deterministic_algorithms']
        interpolation = arguments['interpolation']
        val_resize_size = arguments['val_resize_size']
        val_crop_size = arguments['val_crop_size']
        train_crop_size = arguments['train_crop_size']
        clip_grad_norm = arguments['clip_grad_norm']
        ra_sampler = arguments['ra_sampler']
        ra_reps = arguments['ra_reps']
        weights = arguments['weights']
        prune = arguments['prune']
        method = arguments['method']
        global_pruning = arguments['global_pruning']
        target_flops = arguments['target_flops']
        soft_keeping_ratio = arguments['soft_keeping_ratio']
        reg = arguments['reg']
        max_ch_sparsity = arguments['max_ch_sparsity']
        sl_epochs = arguments['sl_epochs']
        sl_resume = arguments['sl_resume']
        sl_lr = arguments['sl_lr']
        sl_lr_step_size = arguments['sl_lr_step_size']
        sl_lr_warmup_epochs = arguments['sl_lr_warmup_epochs']

    results = main(args)

    # the returned result will be written into the database
    return results