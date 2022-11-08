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
from timm.data import create_dataset, create_loader, resolve_data_config, CheXpertDataset
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import ApexScaler, NativeScaler

############################################################
from libauc.losses import AUCM_MultiLabel                                     # 추가
from libauc.optimizers import PESG
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


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Optimizer parameters
group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
group.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')

# Model Exponential Moving Average
### Comment: Model Soup 사용에 용이할 것으로 보임
group = parser.add_argument_group('Model exponential moving average parameters')
group.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
group.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
group.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
group = parser.add_argument_group('Miscellaneous parameters')
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


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()

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
    lr = 1e-4
    use_model = 'resnet50'
    checkpoint_path=''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model(
        use_model,
        pretrained=True,
        in_chans=in_chans,
        num_classes=num_classes,
        checkpoint_path=checkpoint_path,
    )
    criterion = AUCM_MultiLabel(num_classes=num_classes)
    optimizer = PESG(model,
                    loss_fn=criterion,
                    lr=lr,
                    margin=1.0,
                    epoch_decay=2e-3,
                    weight_decay=1e-4)
    log_wandb = False
    batch_size = 64
    log_interval = 500
    eval_metric = 'top1'
    sched = 'cosine'
    


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

    if utils.is_primary(args) and cfg.log_wandb:
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

    if cfg.in_chans is not None:
        in_chans = cfg.in_chans

    model = cfg.model
    if cfg.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        cfg.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    if utils.is_primary(args):
        _logger.info(
            f'Model {safe_model_name(cfg.use_model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    data_config = resolve_data_config(vars(args), model=model, verbose=utils.is_primary(args))

    # move model to GPU, enable channels last layout if set
    model.to(device=device)
    optimizer = cfg.optimizer

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

    # setup exponential moving average of model weights, SWA could be used here too
    # model_ema = None
    # if args.model_ema:
    #     # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
    #     model_ema = utils.ModelEmaV2(
    #         model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)

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


    # create datasets
    dataset_train = CheXpertDataset(cfg.root_path, cfg.small_dir, cfg.mode, cfg.use_frontal, cfg.train_cols,cfg. use_enhancement, cfg.enhance_cols, cfg.enhance_time, cfg.flip_label, cfg.shuffle, cfg.seed, cfg.image_size, cfg.verbose) 
    dataset_eval = CheXpertDataset(cfg.root_path, cfg.small_dir, 'valid', cfg.use_frontal, cfg.train_cols,cfg. use_enhancement, cfg.enhance_cols, cfg.enhance_time, cfg.flip_label, cfg.shuffle, cfg.seed, cfg.image_size, cfg.verbose)
    # create data loaders
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=cfg.batch_size, num_workers=cfg.train_workers, drop_last=True, shuffle=True)
    loader_eval =  torch.utils.data.DataLoader(dataset_eval, batch_size=cfg.batch_size, num_workers=cfg.eval_workers, drop_last=False, shuffle=False)

    # setup loss function
    ###########################################################################
    train_loss_fn = cfg.criterion                                                            # 수정
    validate_loss_fn = cfg.criterion
    ###########################################################################
    # setup checkpoint saver and eval metric tracking
    eval_metric = cfg.eval_metric
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
                safe_model_name(cfg.use_model),
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
        **scheduler_kwargs(cfg),
        updates_per_epoch=updates_per_epoch,
    )
    start_epoch = 0
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
                    log_wandb=cfg.log_wandb and has_wandb,
                )

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                # best_metric, best_epoch = saver.save_checkpoint(epoch, model=cfg.use_model, metric=save_metric)

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
    cfg = Config
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
        if last_batch or batch_idx % cfg.log_interval == 0:
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
        print(f'validation auc : {val_auc_mean} , validation best auc : {val_auc}')
        metrics = OrderedDict([('loss', loss), ('top1', best_val_auc), ('top5', val_auc_mean)])
##################################################################################################

    return metrics


if __name__ == '__main__':
    main()
