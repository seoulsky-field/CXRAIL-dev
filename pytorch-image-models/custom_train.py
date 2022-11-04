#!/usr/bin/env python3
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import argparse
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from functools import partial

import torch
import torch.nn as nn
import torchvision.utils
import yaml
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, CheXpertDataset
from timm.loss import SoftTargetCrossEntropy, BinaryCrossEntropy
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, \
    convert_sync_batchnorm, model_parameters, set_fast_norm
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import ApexScaler, NativeScaler

############################################################
from libauc.losses import AUCM_MultiLabel, CrossEntropyLoss                                     # 추가
from libauc.optimizers import PESG, Adam
from libauc.metrics import auc_roc_score # for multi-task
import numpy as np
############################################################

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False


_logger = logging.getLogger('train')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset parameters
group = parser.add_argument_group('Dataset parameters')
# Keep this argument outside of the dataset group because it is positional.
parser.add_argument('data_dir', metavar='DIR',
                    help='path to dataset')
# group.add_argument('--train-split', metavar='NAME', default='train',
#                     help='dataset train split (default: train)')
#############################################################################################################
# Kyoungmin revised: default was 'validation', change 'validation' -> 'valid' to use os.path.join with root
# group.add_argument('--val-split', metavar='NAME', default='valid',
#                     help='dataset validation split (default: valid)')
#############################################################################################################
# group.add_argument('--dataset-download', action='store_true', default=False,
#                     help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
group.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')

# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                    help='Name of model to train (default: "resnet50"')
group.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
group.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
group.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
group.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
# group.add_argument('--num-classes', type=int, default=5, metavar='N',
#                     help='number of label classes (Model default if None)')
group.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
group.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                    help='Input batch size for training (default: 128)')
group.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                    help='Validation batch size override (default: None)')
group.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
scripting_group = group.add_mutually_exclusive_group()
scripting_group.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='torch.jit.script the full model')
# group.add_argument('--fast-norm', default=False, action='store_true',
#                     help='enable experimental fast-norm')
group.add_argument('--grad-checkpointing', action='store_true', default=False,
                    help='Enable gradient checkpointing through model blocks/stages')

# Optimizer parameters
group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--opt', default='pesg', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
group.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
group.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')

# Learning rate schedule parameters
group = parser.add_argument_group('Learning rate schedule parameters')
group.add_argument('--sched', type=str, default='cosine', metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
group.add_argument('--lr', type=float, default=None, metavar='LR',
                    help='learning rate, overrides lr-base if set (default: None)')
group.add_argument('--lr-base', type=float, default=0.1, metavar='LR',
                    help='base learning rate: lr = lr_base * global_batch_size / base_size')
group.add_argument('--lr-base-size', type=int, default=256, metavar='DIV',
                    help='base learning rate batch size (divisor, default: 256).')
group.add_argument('--lr-base-scale', type=str, default='', metavar='SCALE',
                    help='base learning rate vs batch_size scaling ("linear", "sqrt", based on opt if empty)')
group.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
group.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

# Augmentation & regularization parameters
group = parser.add_argument_group('Augmentation and regularization parameters')
group.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')

# Model Exponential Moving Average
### Comment: Model Soup 사용에 용이할 것으로 보임
# group = parser.add_argument_group('Model exponential moving average parameters')
# group.add_argument('--model-ema', action='store_true', default=False,
#                     help='Enable tracking moving average of model weights')
# group.add_argument('--model-ema-force-cpu', action='store_true', default=False,
#                     help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
# group.add_argument('--model-ema-decay', type=float, default=0.9998,
#                     help='decay factor for model weights moving average (default: 0.9998)')

# Misc
group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
group.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
group.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                    help='number of checkpoints to keep (default: 10)')
group.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
group.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
group.add_argument('--amp-dtype', default='float16', type=str,
                    help='lower precision AMP dtype (default: float16)')
group.add_argument('--amp-impl', default='native', type=str,
                    help='AMP impl to use, "native" or "apex" (default: native)')
group.add_argument('--no-ddp-bb', action='store_true', default=False,
                    help='Force broadcast buffers for native DDP to off.')
# group.add_argument('--pin-mem', action='store_true', default=False,
#                     help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
group.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
group.add_argument("--local_rank", default=0, type=int)
group.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
group.add_argument('--log-wandb', action='store_true', default=False,
                    help='log training and validation metrics to wandb')


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


class Config():
    root_path = '/home/dataset/chexpert/'
    ori_dir = 'CheXpert-v1.0/'
    small_dir = 'CheXpert-v1.0-small/'
    pad_dir = 'CheXpert-v1.0-pad224/'
    mode = 'train'
    use_frontal = True
    # train_cols=['Cardiomegaly']
    train_cols=['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis',  'Pleural Effusion']

    use_enhancement = False          # upsampling
    enhance_cols = ['Cardiomegaly', 'Consolidation']
    enhance_time = 1

    flip_label = False
    shuffle = True
    seed = 777
    image_size = 224
    verbose = True

    output_dir = ''
    num_classes = 5
    in_chans = 3
    args, args_text = _parse_args()
    train_workers = 4
    eval_workers = 4
    experiment = ''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        in_chans=in_chans,
        num_classes=num_classes,
        # drop_rate=args.drop,
        # drop_path_rate=args.drop_path,
        # drop_block_rate=args.drop_block,
        # global_pool=args.gp,
        # bn_momentum=args.bn_momentum,
        # bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
    )
    criterion = AUCM_MultiLabel(num_classes=num_classes)
    optimizer = PESG(model,
                    loss_fn=criterion,
                    lr=1e-4,
                    margin=1.0,
                    epoch_decay=2e-3,
                    weight_decay=1e-4)


def main():
    cfg = Config()
    utils.setup_default_logging()
    args, args_text = _parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = utils.init_distributed_device(args)
    if args.distributed:
        _logger.info(
            'Training in distributed mode with multiple processes, 1 device per process.'
            f'Process {args.rank}, total {args.world_size}, device {cfg.device}.')
    else:
        _logger.info(f'Training with a single process on 1 device ({cfg.device}).')
    assert args.rank >= 0

    if utils.is_primary(args) and args.log_wandb:
        if has_wandb:
            wandb.init(project=cfg.experiment, config=args)
        else:
            _logger.warning(
                "You've requested to log metrics to wandb but package not found. "
                "Metrics not being logged to wandb, try `pip install wandb`")

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    amp_dtype = torch.float16
    if args.amp:
        if args.amp_impl == 'apex':
            assert has_apex, 'AMP impl specified as APEX but APEX is not installed.'
            use_amp = 'apex'
            assert args.amp_dtype == 'float16'
        else:
            assert has_native_amp, 'Please update PyTorch to a version with native AMP (or use APEX).'
            use_amp = 'native'
            assert args.amp_dtype in ('float16', 'bfloat16')
        if args.amp_dtype == 'bfloat16':
            amp_dtype = torch.bfloat16

    utils.random_seed(cfg.seed, args.rank)

    # if args.fast_norm:
    #     set_fast_norm()

    if cfg.in_chans is not None:
        in_chans = cfg.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    model = cfg.model
    if cfg.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        cfg.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    if args.grad_checkpointing:
        model.set_grad_checkpointing(enable=True)

    if utils.is_primary(args):
        _logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    data_config = resolve_data_config(vars(args), model=model, verbose=utils.is_primary(args))

    # move model to GPU, enable channels last layout if set
    model.to(device=device)
    if args.channels_last:
        model.to(memory_format=torch.channels_last)

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        model = torch.jit.script(model)

    if args.lr is None:
        global_batch_size = args.batch_size * args.world_size
        batch_ratio = global_batch_size / args.lr_base_size
        if not args.lr_base_scale:
            on = args.opt.lower()
            args.lr_base_scale = 'sqrt' if any([o in on for o in ('ada', 'lamb')]) else 'linear'
        if args.lr_base_scale == 'sqrt':
            batch_ratio = batch_ratio ** 0.5
        args.lr = args.lr_base * batch_ratio
        if utils.is_primary(args):
            _logger.info(
                f'Learning rate ({args.lr}) calculated from base learning rate ({args.lr_base}) '
                f'and global batch size ({global_batch_size}) with {args.lr_base_scale} scaling.')

    ###########################################################################
    # optimizer = Adam(model, **optimizer_kwargs(cfg=args))
    optimizer = cfg.optimizer                                                # 추가
    ###########################################################################

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        assert device.type == 'cuda'
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if utils.is_primary(args):
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
        if device.type == 'cuda':
            loss_scaler = NativeScaler()
        if utils.is_primary(args):
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if utils.is_primary(args):
            _logger.info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model,
            args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=utils.is_primary(args),
        )

    # setup exponential moving average of model weights, SWA could be used here too
    # model_ema = None
    # if args.model_ema:
    #     # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
    #     model_ema = utils.ModelEmaV2(
    #         model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
    #     if args.resume:
    #         load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp == 'apex':
            # Apex DDP preferred unless native amp is activated
            if utils.is_primary(args):
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if utils.is_primary(args):
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[device], broadcast_buffers=not args.no_ddp_bb)
        # NOTE: EMA model does not need to be wrapped by DDP

    #############################################################################################################
    # Kyoungmin Custom version (with juppak T's dataset)
    dataset_train = CheXpertDataset(cfg.root_path, cfg.small_dir, cfg.mode, cfg.use_frontal, cfg.train_cols,cfg. use_enhancement, cfg.enhance_cols, cfg.enhance_time, cfg.flip_label, cfg.shuffle, cfg.seed, cfg.image_size, cfg.verbose)
        
    dataset_eval = CheXpertDataset(cfg.root_path, cfg.small_dir, 'valid', cfg.use_frontal, cfg.train_cols,cfg. use_enhancement, cfg.enhance_cols, cfg.enhance_time, cfg.flip_label, cfg.shuffle, cfg.seed, cfg.image_size, cfg.verbose)

    #############################################################################################################
    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if not train_interpolation:
        train_interpolation = data_config['interpolation']
        
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=32, num_workers=cfg.train_workers, drop_last=True, shuffle=True)
    loader_eval =  torch.utils.data.DataLoader(dataset_eval, batch_size=32, num_workers=cfg.eval_workers, drop_last=False, shuffle=False)
    #################################################################################################

    # setup loss function
    ###########################################################################
    train_loss_fn = cfg.criterion                                                            # 수정
    validate_loss_fn = cfg.criterion
    ###########################################################################
    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if utils.is_primary(args):
        if cfg.experiment:
            exp_name = cfg.experiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        output_dir = utils.get_outdir(cfg.output_dir if cfg.output_dir else './output/train', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        saver = utils.CheckpointSaver(
            model=model,
            optimizer=optimizer,
            args=args,
            # model_ema=model_ema,
            amp_scaler=loss_scaler,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=decreasing,
            max_history=args.checkpoint_hist
        )
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    # setup learning rate schedule and starting epoch
    updates_per_epoch = len(loader_train)
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        **scheduler_kwargs(args),
        updates_per_epoch=updates_per_epoch,
    )
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        if args.sched_on_updates:
            lr_scheduler.step_update(start_epoch * updates_per_epoch)
        else:
            lr_scheduler.step(start_epoch)

    if utils.is_primary(args):
        _logger.info(
            f'Scheduled epochs: {num_epochs}. LR stepped per {"epoch" if lr_scheduler.t_in_epochs else "update"}.')

    try:
        for epoch in range(start_epoch, num_epochs):
            if hasattr(dataset_train, 'set_epoch'):
                dataset_train.set_epoch(epoch)
            elif args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)

            train_metrics = train_one_epoch(
                epoch,
                model,
                loader_train,
                optimizer,
                train_loss_fn,
                args,
                lr_scheduler=lr_scheduler,
                saver=saver,
                output_dir=output_dir,
                amp_autocast=amp_autocast,
                loss_scaler=loss_scaler,
                # model_ema=model_ema,
            )

            eval_metrics = validate(
                model,
                loader_eval,
                validate_loss_fn,
                args,
                amp_autocast=amp_autocast,
            )

            # if model_ema is not None and not args.model_ema_force_cpu:

            #     ema_eval_metrics = validate(
            #         model_ema.module,
            #         loader_eval,
            #         validate_loss_fn,
            #         args,
            #         amp_autocast=amp_autocast,
            #         log_suffix=' (EMA)',
            #     )
            #     eval_metrics = ema_eval_metrics

            if output_dir is not None:
                lrs = [param_group['lr'] for param_group in optimizer.param_groups]
                utils.update_summary(
                    epoch,
                    train_metrics,
                    eval_metrics,
                    filename=os.path.join(output_dir, 'summary.csv'),
                    lr=sum(lrs) / len(lrs),
                    write_header=best_metric is None,
                    log_wandb=args.log_wandb and has_wandb,
                )

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

    except KeyboardInterrupt:
        pass

    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def train_one_epoch(
        epoch,
        model,
        loader,
        optimizer,
        loss_fn,
        args,
        device=torch.device('cuda'),
        lr_scheduler=None,
        saver=None,
        output_dir=None,
        amp_autocast=suppress,
        loss_scaler=None,
        # model_ema=None,
):

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()

    model.train()

    end = time.time()
    num_batches_per_epoch = len(loader)
    last_idx = num_batches_per_epoch - 1
    num_updates = epoch * num_batches_per_epoch
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        input, target = input.to(device), target.to(device)
            # if mixup_fn is not None:
            #     input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        with amp_autocast():
            ###########################################################################
            # output = model(input)
            # loss = loss_fn(output, target)
            ###########################################################################
            output = torch.softmax(model(input.to(device)), dim=1) # softmax for multilabel classification
            loss = loss_fn(output, target.to(device))
            ###########################################################################
            
        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad,
                clip_mode=args.clip_mode,
                parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                create_graph=second_order
            )
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                utils.dispatch_clip_grad(
                    model_parameters(model, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad,
                    mode=args.clip_mode
                )
            optimizer.step()

        # if model_ema is not None:
        #     model_ema.update(model)

        torch.cuda.synchronize()

        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))

            if utils.is_primary(args):
                _logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m)
                )

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True
                    )

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])


def validate(
        model,
        loader,
        loss_fn,
        args,
        device=torch.device('cuda'),
        amp_autocast=suppress,
        log_suffix=''
):
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        ###########################################################################
        best_val_auc = 0 
        predictions = []                                                                                    # 추가
        true_labels = []                                                                                    # 추가
        ###########################################################################
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx

            input = input.to(device)
            target = target.to(device)

            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input.to(device))                                                            # .to(device) 추가
            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = loss_fn(output, target.to(device))                                                       # .to(device) 추가
##################################################################################################
            output = torch.sigmoid(output)
            predictions.append(output.cpu().detach().numpy())
            true_labels.append(target.cpu().numpy())

        predictions = np.concatenate(predictions)
        true_labels = np.concatenate(true_labels)
        val_auc = auc_roc_score(true_labels, predictions)
        val_auc_mean = np.mean(auc_roc_score(true_labels, predictions))
        if best_val_auc < val_auc_mean:
            best_val_auc = val_auc_mean
        print(f'validation acc : {val_auc_mean} , validation best acc : {val_auc}')
        metrics = OrderedDict([('loss', loss), ('top1', best_val_auc), ('top5', val_auc_mean)])
##################################################################################################

    return metrics


if __name__ == '__main__':
    main()
