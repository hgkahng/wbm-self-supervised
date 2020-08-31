# -*- coding: utf-8 -*-

import os
import sys
import argparse

import torch

from datasets.wafer import WM811KForSimCLR
from datasets.cifar import CIFAR10ForSimCLR
from datasets.transforms import get_transform

from models.config import AttnCLRConfig, RESNET_BACKBONE_CONFIGS
from models.resnet import ResNetBackbone
from models.head import GAPProjector, AttentionProjector

from tasks.attnclr import AttnCLR

from utils.loss import AttnCLRLoss
from utils.logging import get_logger
from utils.optimization import get_optimizer, get_scheduler


AVAILABLE_MODELS = {
    'resnet': (RESNET_BACKBONE_CONFIGS, AttnCLRConfig, ResNetBackbone),
}

PROJECTOR_TYPES = {
    'linear': GAPProjector,
    'mlp': AttentionProjector,
}


def parse_args():

    parser = argparse.ArgumentParser("Attention-based Contrastive Learning Framework.", add_help=True)

    g1 = parser.add_argument_group('Data')
    g1.add_argument('--data', type=str, default='wm811k', choices=('wm811k', 'cifar10', 'stl10', 'imagenet'), required=True)
    g1.add_argument('--input_size', type=int, default=56, choices=(32, 64, 96, 224), required=True)

    g2 = parser.add_argument_group('CNN Backbone')
    g2.add_argument('--backbone_type', type=str, default='resnet', choices=('resnet'), required=True)
    g2.add_argument('--backbone_config', type=str, default='18.original', required=True)

    g3 = parser.add_argument_group('AttnCLR')
    g3.add_argument('--projector_type', type=str, default='mlp', choices=('linear', 'mlp'), required=True)
    g3.add_argument('--projector_size', type=int, default=128)
    g3.add_argument('--temperature', type=float, default=0.07)
    # g3.add_argument('--gamma', type=float, default=1.0)

    g4 = parser.add_argument_group('Model Training')
    g4.add_argument('--epochs', type=int, default=1000)
    g4.add_argument('--batch_size', type=int, default=2048)
    g4.add_argument('--num_workers', type=int, default=0)
    g4.add_argument('--device', type=str, default='cuda:0', choices=('cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cpu'))

    g5 = parser.add_argument_group('Regularization')  # pylint: disable=unused-variable

    g6 = parser.add_argument_group('Optimizer')
    g6.add_argument('--optimizer', type=str, default='sgd', choices=('sgd', 'adamw', 'lars'))
    g6.add_argument('--learning_rate', type=float, default=0.01)
    g6.add_argument('--weight_decay', type=float, default=0.001)
    g6.add_argument('--momentum', type=float, default=0.9, help='only for SGD.')
    g6.add_argument('--trust_coef', type=float, default=1.0, help='only for LARS.')

    g7 = parser.add_argument_group('Scheduler')
    g7.add_argument('--scheduler', type=str, default=None, choices=('step', 'cosine', 'restart', 'none'))
    g7.add_argument('--milestone', type=int, default=None, help='For step decay.')
    g7.add_argument('--warmup_steps', type=int, default=0, help='For linear warmups.')
    g7.add_argument('--cycles', type=int, default=1, help='For hard restarts.')

    g8 = parser.add_argument_group('Logging')
    g8.add_argument('--checkpoint_root', type=str, default='./checkpoints/')
    g8.add_argument('--write_summary', action='store_true', help='write summaries with TensorBoard.')
    g8.add_argument('--write_histogram', action='store_true', help='write attention distribution histograms.')
    g8.add_argument('--save_every', action='store_true', help='save every checkpoint w/ improvements.')

    return parser.parse_args()


def main(args):
    """Main function."""

    # 0. CONFIGURATIONS
    torch.backends.cudnn.benchmark = True
    BACKBONE_CONFIGS, Config, Backbone = AVAILABLE_MODELS[args.backbone_type]
    Projector = PROJECTOR_TYPES[args.projector_type]

    config = Config(args)
    config.save()

    logfile = os.path.join(config.checkpoint_dir, 'main.log')
    logger  = get_logger(stream=False, logfile=logfile)

    # 1. DATA
    input_transform = get_transform(config.data, size=config.input_size, mode='pretrain')
    if config.data == 'wm811k':
        in_channels = 2
        train_set = torch.utils.data.ConcatDataset(
            [
                WM811KForSimCLR('./data/wm811k/unlabeled/train/', transform=input_transform),
                WM811KForSimCLR('./data/wm811k/labeled/train/', transform=input_transform),
            ]
        )
        valid_set = torch.utils.data.ConcatDataset(
            [
                WM811KForSimCLR('./data/wm811k/unlabeled/valid/', transform=input_transform),
                WM811KForSimCLR('./data/wm811k/labeled/valid/', transform=input_transform),
            ]
        )
        test_set = torch.utils.data.ConcatDataset(
            [
                WM811KForSimCLR('./data/wm811k/unlabeled/test/', transform=input_transform),
                WM811KForSimCLR('./data/wm811k/labeled/test/', transform=input_transform),
            ]
        )
    elif config.data == 'cifar10':
        in_channels = 3
        train_set = CIFAR10ForSimCLR('./data/cifar10/', train=True, transform=input_transform)
        valid_set = CIFAR10ForSimCLR('./data/cifar10/', train=False, transform=input_transform)
        test_set  = valid_set
    elif config.data == 'stl10':
        raise NotImplementedError
    elif config.data == 'imagenet':
        raise NotImplementedError
    else:
        raise NotImplementedError
    logger.info(f"Data type: {config.data}")
    logger.info(f"Train : Valid : Test = {len(train_set):,} : {len(valid_set):,} : {len(test_set):,}")
    steps_per_epoch = len(train_set) // config.batch_size + 1
    logger.info(f"Training steps per epoch: {steps_per_epoch:,}")
    logger.info(f"Total number of training iterations: {steps_per_epoch * config.epochs:,}")

    # 2. MODEL
    backbone = Backbone(BACKBONE_CONFIGS[config.backbone_config], in_channels)
    projector = Projector(backbone.out_channels, config.projector_size)
    logger.info(f"Trainable parameters ({backbone.__class__.__name__}): {backbone.num_parameters:,}")
    logger.info(f"Trainable parameters ({projector.__class__.__name__}): {projector.num_parameters:,}")
    logger.info(f"Projector size: {config.projector_size}")

    # 3. OPTIMIZATION
    params = [{'params': backbone.parameters()}, {'params': projector.parameters()}]
    optimizer = get_optimizer(
        params=params,
        name=config.optimizer,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        momentum=config.momentum,
        trust_coef=config.trust_coef,
    )
    scheduler = get_scheduler(
        optimizer=optimizer,
        name=config.scheduler,
        epochs=config.epochs,
        milestone=config.milestone,
        warmup_steps=config.warmup_steps
    )

    # 4. EXPERIMENT (AttnCLR)
    experiment_kwargs = {
        'backbone': backbone,
        'projector': projector,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'loss_function': AttnCLRLoss(temperature=config.temperature),
        'metrics': None,
        'checkpoint_dir': config.checkpoint_dir,
        'write_summary': config.write_summary,
        'write_histogram': config.write_histogram,
    }
    experiment = AttnCLR(**experiment_kwargs)
    logger.info(f"Saving model checkpoints to: {experiment.checkpoint_dir}")

    # 5. RUN (AttnCLR)
    run_kwargs = {
        'train_set': train_set,
        'valid_set': valid_set,
        'test_set': test_set,
        'epochs': config.epochs,
        'batch_size': config.batch_size,
        'num_workers': config.num_workers,
        'device': config.device,
        'logger': logger,
        'save_every': config.save_every,
    }
    logger.info(f"Epochs: {run_kwargs['epochs']}, Batch size: {run_kwargs['batch_size']}")
    logger.info(f"Workers: {run_kwargs['num_workers']}, Device: {run_kwargs['device']}")

    experiment.run(**run_kwargs)
    logger.handlers.clear()


if __name__ == '__main__':
    try:
        main(parse_args())
    except KeyboardInterrupt:
        sys.exit(0)
