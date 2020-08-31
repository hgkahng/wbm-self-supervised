# -*- coding: utf-8 -*-

import os
import sys
import argparse

import torch
import torch.nn as nn

from datasets.wafer import WM811KForDenoising
from datasets.transforms import get_transform

from models.config import DenoisingConfig
from models.config import RESNET_ENCODER_CONFIGS
from models.config import RESNET_DECODER_CONFIGS
from models.resnet.backbone import ResNetBackbone as ResNetEncoder
from models.resnet.decoder import ResNetDecoder

from tasks.denoising import Denoising

from utils.optimization import get_optimizer, get_scheduler
from utils.logging import get_logger


AVAILABLE_MODELS = {
    'alexnet': (None, None, None),
    'vggnet': (None, None, None),
    'resnet': (RESNET_ENCODER_CONFIGS, RESNET_DECODER_CONFIGS, DenoisingConfig, ResNetEncoder, ResNetDecoder),
}

IN_CHANNELS = {'wm811k': 2}
OUT_CHANNELS = {'wm811k': 3}


def parse_args():

    parser = argparse.ArgumentParser("Denoising autoencoders.", add_help=True)

    g1 = parser.add_argument_group('General')
    g1.add_argument('--data', type=str, choices=('wm811k', ), required=True)
    g1.add_argument('--input_size', type=int, choices=(32, 64, 96, 112, 224), required=True)

    g2 = parser.add_argument_group('Network')
    g2.add_argument('--backbone_type', type=str, choices=('resnet', ), required=True)
    g2.add_argument('--backbone_config', type=str, choices=('18.original', ), required=True)

    g3 = parser.add_argument_group('Denoising')
    g3.add_argument('--augmentation', type=str, choices=('rotate', 'crop', 'shear', 'shift', 'noise', ), required=True)
    g3.add_argument('--noise', type=float, default=0.00)

    g4 = parser.add_argument_group('Model Training')
    g4.add_argument('--epochs', type=int, default=150)
    g4.add_argument('--batch_size', type=int, default=512)
    g4.add_argument('--num_workers', type=int, default=0)
    g4.add_argument('--device', type=str, default='cuda:0', choices=('cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cpu'))

    g5 = parser.add_argument_group('Regularization')  # pylint: disable=unused-variable

    g6 = parser.add_argument_group('Optimizer')
    g6.add_argument('--optimizer', type=str, default='sgd', choices=('sgd', 'adamw'))
    g6.add_argument('--learning_rate', type=float, default=0.01)
    g6.add_argument('--weight_decay', type=float, default=0.001)
    g6.add_argument('--momentum', type=float, default=0.9, help='only for SGD.')

    g7 = parser.add_argument_group('Scheduler')
    g7.add_argument('--scheduler', type=str, default='cosine', choices=('step', 'cosine', 'restart', 'none'))
    g7.add_argument('--milestone', type=int, default=None, help='For step decay.')
    g7.add_argument('--warmup_steps', type=int, default=5, help='For linear warmups.')
    g7.add_argument('--cycles', type=int, default=1, help='For hard restarts.')

    g8 = parser.add_argument_group('Logging')
    g8.add_argument('--checkpoint_root', type=str, default='./checkpoints/')
    g8.add_argument('--write_summary', action='store_true', help='write summaries with TensorBoard.')
    g8.add_argument('--save_every', type=int, default=None, help='save model checkpoint every `save_every` epochs.')

    return parser.parse_args()


def main(args):
    """Main function."""

    # 1. Configurations
    torch.backends.cudnn.benchmark = True
    ENCODER_CONFIGS, DECODER_CONFIGS, Config, Encoder, Decoder = \
        AVAILABLE_MODELS[args.backbone_type]

    config = Config(args)
    config.save()

    logfile = os.path.join(config.checkpoint_dir, 'main.log')
    logger = get_logger(stream=False, logfile=logfile)

    # 2. Data
    input_transform = get_transform(config.data, size=config.input_size, mode=config.augmentation, noise=config.noise)
    target_transform = get_transform(config.data, size=config.input_size, mode='test')
    if config.data == 'wm811k':
        train_set = torch.utils.data.ConcatDataset(
            [
                WM811KForDenoising('./data/wm811k/unlabeled/train/', input_transform, target_transform),
                WM811KForDenoising('./data/wm811k/labeled/train/', input_transform, target_transform),
            ]
        )
        valid_set = torch.utils.data.ConcatDataset(
            [
                WM811KForDenoising('./data/wm811k/unlabeled/valid/', input_transform, target_transform),
                WM811KForDenoising('./data/wm811k/labeled/valid/', input_transform, target_transform),
            ]
        )
        test_set = torch.utils.data.ConcatDataset(
            [
                WM811KForDenoising('./data/wm811k/unlabeled/test/', input_transform, target_transform),
                WM811KForDenoising('./data/wm811k/labeled/test/', input_transform, target_transform),
            ]
        )
    else:
        raise ValueError(
            f"Denoising only supports 'wm811k' data. Received '{config.data}'."
        )

    # 3. Model
    encoder = Encoder(RESNET_ENCODER_CONFIGS[config.backbone_config], in_channels=IN_CHANNELS[config.data])
    decoder = Decoder(
        RESNET_DECODER_CONFIGS[config.backbone_config],
        input_shape=encoder.output_shape,
        output_shape=(OUT_CHANNELS[config.data], config.input_size, config.input_size)
        )

    # 4. Optimization
    params = [{'params': encoder.parameters()}, {'params': decoder.parameters()}]
    optimizer = get_optimizer(
        params=params,
        name=config.optimizer,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        momentum=config.momentum
    )
    scheduler = get_scheduler(
        optimizer=optimizer,
        name=config.scheduler,
        epochs=config.epochs,
        milestone=config.milestone,
        warmup_steps=config.warmup_steps
    )

    # 5. Experiment (Denoising)
    experiment_kwargs = {
        'encoder': encoder,
        'decoder': decoder,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'loss_function': nn.CrossEntropyLoss(reduction='mean'),
        'metrics': None,
        'checkpoint_dir': config.checkpoint_dir,
        'write_summary': config.write_summary,
    }
    experiment = Denoising(**experiment_kwargs)

    # 6. Run (train, evaluate, and test model)
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

    logger.info(f"Data: {config.data}")
    logger.info(f"Augmentation: {config.augmentation}")
    logger.info(f"Train : Valid : Test = {len(train_set):,} : {len(valid_set):,} : {len(test_set):,}")
    logger.info(f"Trainable parameters ({encoder.__class__.__name__}): {encoder.num_parameters:,}")
    logger.info(f"Trainable parameters ({decoder.__class__.__name__}): {decoder.num_parameters:,}")
    logger.info(f"Saving model checkpoints to: {experiment.checkpoint_dir}")
    logger.info(f"Epochs: {run_kwargs['epochs']}, Batch size: {run_kwargs['batch_size']}")
    logger.info(f"Workers: {run_kwargs['num_workers']}, Device: {run_kwargs['device']}")

    steps_per_epoch = len(train_set) // config.batch_size + 1
    logger.info(f"Training steps per epoch: {steps_per_epoch:,}")
    logger.info(f"Total number of training iterations: {steps_per_epoch * config.epochs:,}")

    experiment.run(**run_kwargs)
    logger.handlers.clear()


if __name__ == '__main__':
    try:
        main(parse_args())
    except KeyboardInterrupt:
        sys.exit(0)
