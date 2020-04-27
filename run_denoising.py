# -*- coding: utf-8 -*-

import os
import sys
import argparse

import torch
import torch.nn as nn

from datasets.wafer import WM811kDataset
from models.config import DenoisingConfig, VGG_ENCODER_CONFIGS, VGG_DECODER_CONFIGS
from models.vgg.backbone import VGGBackbone as VGGEncoder
from models.vgg.decoder import VGGDecoder
from tasks.denoising import Denoising, MaskedBernoulliNoise
from utils.optimization import get_optimizer, get_scheduler
from utils.logging import get_logger

AVAILABLE_MODELS = {
    'alex': (),
    'vgg': (DenoisingConfig, VGGEncoder, VGGDecoder),
    'res': (),
    'dcgan': (),
}


def parse_args():

    parser = argparse.ArgumentParser("Denoising pretext task on WM811k.", add_help=True)

    g1 = parser.add_argument_group('General')
    g1.add_argument('--disable_benchmark', action='store_true', help='Benchmark mode is faster for fixed size inputs.')

    g2 = parser.add_argument_group('Network')
    g2.add_argument('--backbone_type', type=str, default='vgg', choices=('alex', 'vgg', 'res', 'dcgan'))
    g2.add_argument('--backbone_config', type=str, default='3a', choices=('3a', '6a', '9a'))
    g2.add_argument('--in_channels', type=int, default=2, choices=(1, 2))

    g3 = parser.add_argument_group('Denoising')
    g3.add_argument('--noise', type=float, default=0.01, help='Probability of noise sampled from a Bernoulli distribution.')

    g4 = parser.add_argument_group('Training')
    g4.add_argument('--epochs', type=int, default=500)
    g4.add_argument('--batch_size', type=int, default=1024)
    g4.add_argument('--num_workers', type=int, default=0)
    g4.add_argument('--device', type=str, default='cuda:1', choices=('cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cpu'))

    g5 = parser.add_argument_group('Regularization')
    g5.add_argument('--batch_norm', action='store_true', help='Use batch normalization after convolution.')

    g6 = parser.add_argument_group('Optimizer')
    g6.add_argument('--optimizer', type=str, default='adamw', choices=('sgd', 'adamw'))
    g6.add_argument('--learning_rate', type=float, default=0.001)
    g6.add_argument('--weight_decay', type=float, default=0.01)
    g6.add_argument('--optimizer_kwargs', nargs='+', default=[], help="'key1=value1' 'key2=value2'")

    g7 = parser.add_argument_group('Scheduler')
    g7.add_argument('--scheduler', type=str, default=None, choices=('step', 'exp', 'plateau'))
    g7.add_argument('--scheduler_kwargs', nargs='+', default=[], help="'key1=value1' 'key2=value2'")

    g8 = parser.add_argument_group('Logging')
    g8.add_argument('--checkpoint_root', type=str, default='./checkpoints/')
    g8.add_argument('--write_summary', action='store_true')
    g8.add_argument('--save_every', action='store_true')

    args = parser.parse_args()

    def get_kwargs(l: list):
        out = {}
        for kv in l:
            k, v = kv.split('=')
            if '.' in v:
                v = float(v)
            else:
                try:
                    v = int(v)
                except ValueError:
                    v = str(v)
            out[k] = v

        return out

    opt_kwargs = get_kwargs(args.optimizer_kwargs)
    setattr(args, 'optimizer_kwargs', opt_kwargs)

    sch_kwargs = get_kwargs(args.scheduler_kwargs)
    setattr(args, 'scheduler_kwargs', sch_kwargs)

    return args


def main(args):
    """Main function."""

    torch.backends.cudnn.benchmark = not args.disable_benchmark
    Config, Encoder, Decoder = AVAILABLE_MODELS[args.backbone_type]

    # Configurations
    config = Config(args)
    config.save()

    # Logger
    logfile = os.path.join(config.checkpoint_dir, 'main.log')
    logger = get_logger(stream=False, logfile=logfile)

    # Encoder
    encoder = Encoder(
        layer_config=VGG_ENCODER_CONFIGS[config.backbone_config],
        in_channels=config.in_channels,
        batch_norm=config.batch_norm
    )

    # Decoder
    decoder = Decoder(
        layer_config=VGG_DECODER_CONFIGS[config.backbone_config],
        input_shape=encoder.output_shape,
        output_shape=(2, 40, 40),
        batch_norm=config.batch_norm
    )
    logger.info(f"Trainable parameters ({encoder.__class__.__name__}): {encoder.num_parameters:,}")
    logger.info(f"Trainable parameters ({decoder.__class__.__name__}): {decoder.num_parameters:,}")

    # Data
    data_file = "./data/processed/WM811k.40.npz"
    train_set = WM811kDataset(data_file, mode='train')
    valid_set = WM811kDataset(data_file, mode='valid')
    test_set = WM811kDataset(data_file, mode='test')

    steps_per_epoch = len(train_set) // config.batch_size + 1
    logger.info(f"Train : Valid : Test = {len(train_set):,} : {len(valid_set):,} : {len(test_set):,}")
    logger.info(f"Training steps per epoch: {steps_per_epoch:,}")
    logger.info(f"Total number of training iterations: {steps_per_epoch * config.epochs:,}")

    # Optimizer & Scheduler
    params = [{'params': encoder.parameters()}, {'params': decoder.parameters()}]
    optimizer = get_optimizer(
        params=params,
        name=config.optimizer,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        **config.optimizer_kwargs
    )
    scheduler = get_scheduler(
        optimizer=optimizer,
        name=config.scheduler,
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        **config.scheduler_kwargs
    )

    # Experiment (Denoising)
    experiment_kws = {
        'encoder': encoder,
        'decoder': decoder,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'loss_function': nn.CrossEntropyLoss(reduction='mean'),
        'noise_function': MaskedBernoulliNoise(p=config.noise),
        'metrics': None,
        'checkpoint_dir': config.checkpoint_dir,
        'write_summary': config.write_summary,
    }
    experiment = Denoising(**experiment_kws)
    logger.info(f"Optimizer: {experiment.optimizer.__class__.__name__}")
    logger.info(f"Scheduler: {experiment.scheduler.__class__.__name__}")
    logger.info(f"Loss: {experiment.loss_function.__class__.__name__}")
    logger.info(f"Noise: {experiment.noise_function.__class__.__name__}")
    logger.info(f"Checkpoint directory: {experiment.checkpoint_dir}")

    # Run
    run_kws = {
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
    logger.info(f"Epochs: {run_kws['epochs']}, Batch size: {run_kws['batch_size']}")
    logger.info(f"Workers: {run_kws['num_workers']}, Device: {run_kws['device']}")
    experiment.run(**run_kws)
    logger.handlers.clear()


if __name__ == '__main__':
    try:
        main(parse_args())
    except KeyboardInterrupt:
        sys.exit(0)
