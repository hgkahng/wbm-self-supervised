# -*- coding: utf-8 -*-

import os
import sys
import argparse

import torch

from datasets.wafer import WM811kDataset
from models.config import BiGANConfig, VGG_BACKBONE_CONFIGS
from models.config import GENERATOR_CONFIGS, DISCRIMINATOR_CONFIGS
from models.vgg.backbone import VGGBackbone
from models.head import GAPProjector
from models.gan import Generator, Discriminator
from tasks import BiGAN
from utils.optimization import get_optimizer
from utils.logging import get_logger


AVAILABLE_MODELS = {
    'alex': (),
    'vgg': (BiGANConfig, VGGBackbone, GAPProjector),
    'res': (),
    'dcgan': (),
}


def parse_args():

    parser = argparse.ArgumentParser("BiGAN pretext task on WM811k.")

    g1 = parser.add_argument_group('General')
    g1.add_argument('--disable_benchmark', action='store_true')

    g2 = parser.add_argument_group('Network')
    g2.add_argument('--backbone_type', type=str, default='vgg', choices=('alex', 'vgg', 'res', 'dcgan'))
    g2.add_argument('--backbone_config', type=str, default='3', choices=('3', '6', '9'))
    g2.add_argument('--in_channels', type=int, default=2, choices=(1, 2))

    g3 = parser.add_argument_group('BiGAN')
    g3.add_argument('--gan_type', type=str, default='dcgan', choices=('dcgan', 'vgg'))
    g3.add_argument('--gan_config', type=str, default='3', choices=('3', '6', '9'))
    g3.add_argument('--latent_size', type=int, default=50)

    g4 = parser.add_argument_group('Training')
    g4.add_argument('--epochs', type=int, default=500)
    g4.add_argument('--batch_size', type=int, default=1024)
    g4.add_argument('--num_workers', type=int, default=1)
    g4.add_argument('--device', type=str, default='cuda:1')

    g5 = parser.add_argument_group('Regularization')
    g5.add_argument('--batch_norm', action='store_true', help='Use batch normalization after convolution.')

    g6 = parser.add_argument_group('Stochastic Gradient Descent')
    g6.add_argument('--optimizer_type', type=str, default='adamw', choices=('adamw', 'sgd'))
    g6.add_argument('--learning_rate', type=float, default=.001)
    g6.add_argument('--weight_decay', type=float, default=0.01)
    g6.add_argument('--beta1', type=float, default=0.9, help='Hyperparameter of Adam & AdamW.')

    g7 = parser.add_argument_group('Logging')
    g7.add_argument('--checkpoint_root', type=str, default='./checkpoints/')
    g7.add_argument('--write_summary', action='store_true')
    g7.add_argument('--save_every', action='store_true')

    return parser.parse_args()


def main(args):
    """Main function."""

    torch.backends.cudnn.benchmark = not args.disable_benchmark
    Config, Encoder, Projector = AVAILABLE_MODELS[args.backbone_type]

    # Configurations
    config = Config(args)
    config.save()

    # Logger
    logfile = os.path.join(config.checkpoint_dir, 'main.log')
    logger = get_logger(stream=False, logfile=logfile)

    # Encoder
    encoder = Encoder(
        layer_config=VGG_BACKBONE_CONFIGS[config.backbone_config],
        in_channels=config.in_channels,
        batch_norm=config.batch_norm,
    )

    # Projector
    projector = Projector(
        input_shape=encoder.output_shape,
        num_features=config.latent_size,
    )

    # Generator
    generator = Generator(
        model_type=config.gan_type,
        layer_config=GENERATOR_CONFIGS[config.gan_config],
        latent_size=config.latent_size,
        output_shape=(config.in_channels, 40, 40),
    )

    # Discriminator
    discriminator = Discriminator(
        model_type=config.gan_type,
        layer_config=DISCRIMINATOR_CONFIGS[config.gan_config],
        in_channels=config.in_channels,
        latent_size=config.latent_size,
    )

    logger.info(f"Trainable parameters: ({encoder.__class__.__name__}): {encoder.num_parameters:,}")
    logger.info(f"Trainable parameters: ({projector.__class__.__name__}): {projector.num_parameters:,}")
    logger.info(f"Trainable parameters: ({generator.__class__.__name__}): {generator.num_parameters:,}")
    logger.info(f"Trainable parameters: ({discriminator.__class__.__name__}): {discriminator.num_parameters:,}")
    logger.info(f"Latent size: {config.latent_size}")

    # Optimizers (E, G, D)
    opt_E = get_optimizer(
        params=[{'params':encoder.parameters()}, {'params': projector.parameters()}],
        name=config.optimizer_type,
        lr=config.learning_rate, weight_decay=config.weight_decay, beta1=config.beta1)
    opt_G = get_optimizer(
        params=generator.parameters(), name=config.optimizer_type,
        lr=config.learning_rate, weight_decay=config.weight_decay, beta1=config.beta1)
    opt_D = get_optimizer(
        params=discriminator.parameters(), name=config.optimizer_type,
        lr=config.learning_rate, weight_decay=config.weight_decay, beta1=config.beta1)

    # Data
    train_set = WM811kDataset(mode='train')
    valid_set = WM811kDataset(mode='valid')
    test_set = WM811kDataset(mode='test')
    logger.info(f"Train : Valid : Test = {len(train_set):,} : {len(valid_set):,} : {len(test_set):,}")
    logger.info(f"Training steps per epoch: {len(train_set) // config.batch_size + 1:,}")
    logger.info(f"Total number of training iterations: {(len(train_set) // config.batch_size + 1) * config.epochs:,}")

    # Experiment
    experiment_kws = {
        'encoder': encoder,
        'projector': projector,
        'generator': generator,
        'discriminator': discriminator,
        'optimizer_E': opt_E,
        'optimizer_G': opt_G,
        'optimizer_D': opt_D,
        'metrics': None,
        'checkpoint_dir': config.checkpoint_dir,
        'write_summary': config.write_summary,
    }
    experiment = BiGAN(**experiment_kws)
    logger.info(f"Optimizer (E): {experiment.optimizer_E.__class__.__name__}")
    logger.info(f"Optimizer (G): {experiment.optimizer_G.__class__.__name__}")
    logger.info(f"Optimizer (D): {experiment.optimizer_D.__class__.__name__}")
    logger.info(f"Loss: {experiment.loss_function.__class__.__name__}")
    logger.info(f"Checkpoint directory: {experiment.checkpoint_dir}")

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
