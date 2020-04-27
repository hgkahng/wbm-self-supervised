# -*- coding: utf-8 -*-

import os
import sys
import argparse

import torch

from datasets.wafer import WM811kDataset
from models.config import JigsawConfig, VGG_BACKBONE_CONFIGS
from models.vgg.backbone import VGGBackbone
from models.head import PatchGAPClassifier
from tasks import Jigsaw
from utils.loss import LabelSmoothingLoss
from utils.metrics import MultiAccuracy
from utils.optimization import get_optimizer, get_scheduler
from utils.logging import get_logger

AVAILABLE_MODELS = {
    'alex': (),
    'vgg': (JigsawConfig, VGGBackbone, PatchGAPClassifier),
    'res': (),
    'dcgan': (),
}


def parse_args():

    parser = argparse.ArgumentParser("Jigsaw pretext task on WM811k.")

    g1 = parser.add_argument_group('General')
    g1.add_argument('--disable_benchmark', action='store_true')

    g2 = parser.add_argument_group('Backbone')
    g2.add_argument('--backbone_type', type=str, default='vgg', choices=('alex', 'vgg', 'res', 'dcgan'))
    g2.add_argument('--backbone_config', type=str, default='3', choices=('3', '6', '9'))
    g2.add_argument('--in_channels', type=int, default=2, choices=(1, 2))

    g3 = parser.add_argument_group('Jigsaw')
    g3.add_argument('--num_patches', type=int, default=9, choices=(4, 9, 16))
    g3.add_argument('--num_permutations', type=int, default=100, choices=(50, 100))

    g4 = parser.add_argument_group('Training')
    g4.add_argument('--epochs', type=int, default=500)
    g4.add_argument('--batch_size', type=int, default=1024)
    g4.add_argument('--num_workers', type=int, default=1)
    g4.add_argument('--device', type=str, default='cuda:1')

    g5 = parser.add_argument_group('Regularization')
    g5.add_argument('--batch_norm', action='store_true', help='Use batch normalization after convolution')
    g5.add_argument('--smoothing', type=float, default=0.0)

    g6 = parser.add_argument_group('Stochastic Gradient Descent')
    g6.add_argument('--optimizer_type', type=str, default='adamw', choices=('adamw', 'sgd'))
    g6.add_argument('--learning_rate', type=float, default=.001)
    g6.add_argument('--weight_decay', type=float, default=0.01)
    g6.add_argument('--beta1', type=float, default=0.9, help='Hyperparameter of Adam & AdamW.')
    g6.add_argument('--scheduler', type=str, default=None, choices=('warm_restart', 'cyclic'))

    g7 = parser.add_argument_group('Logging')
    g7.add_argument('--checkpoint_root', type=str, default='./checkpoints/')
    g7.add_argument('--write_summary', action='store_true')
    g7.add_argument('--save_every', action='store_true')

    return parser.parse_args()


def main(args):
    """Main function."""

    torch.backends.cudnn.benchmark = not args.disable_benchmark
    Config, Backbone, Classifier = AVAILABLE_MODELS[args.backbone_type]

    # Configurations
    config = Config(args)
    config.save()

    # Logger
    logfile = os.path.join(config.checkpoint_dir, 'main.log')
    logger = get_logger(stream=False, logfile=logfile)

    # Backbone
    backbone = Backbone(
        layer_config=VGG_BACKBONE_CONFIGS[config.backbone_config],
        in_channels=config.in_channels,
        batch_norm=config.batch_norm
    )

    # Classifier
    classifier = Classifier(
        patch_input_shape=(config.num_patches, ) + backbone.output_shape,
        num_classes=config.num_permutations,
    )
    logger.info(f"Trainable parameters ({backbone.__class__.__name__}): {backbone.num_parameters:,}")
    logger.info(f"Trainable parameters ({classifier.__class__.__name__}): {classifier.num_parameters:,}")

    # Optimizer (mandatory) & scheduler (optional)
    params = [{'params': backbone.parameters()}, {'params': classifier.parameters()}]
    optimizer = get_optimizer(
        params=params, name=config.optimizer_type,
        lr=config.learning_rate, weight_decay=config.weight_decay, beta1=config.beta1)
    scheduler = get_scheduler(optimizer, name=config.scheduler, epochs=config.epochs)

    # Experiment
    experiment_kws = {
        'backbone': backbone,
        'classifier': classifier,
        'num_patches': config.num_patches,
        'num_permutations': config.num_permutations,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'loss_function': LabelSmoothingLoss(
            num_classes=config.num_permutations,
            smoothing=config.smoothing,
            reduction='mean'),
        'checkpoint_dir': config.checkpoint_dir,
        'write_summary': config.write_summary,
        'metrics': {
            'accuracy': MultiAccuracy(config.num_permutations),
        },
    }
    experiment = Jigsaw(**experiment_kws)
    logger.info(f"Optimizer: {experiment.optimizer.__class__.__name__}")
    logger.info(f"Scheduler: {experiment.scheduler.__class__.__name__}")
    logger.info(f"Loss: {experiment.loss_function.__class__.__name__}")
    logger.info(f"Number of patches: {experiment.num_patches:,}")
    logger.info(f"Number of permutations: {experiment.num_permutations:,}")
    logger.info(f"Checkpoint directory: {experiment.checkpoint_dir}")

    # Data
    train_set = WM811kDataset(mode='train')
    valid_set = WM811kDataset(mode='valid')
    test_set = WM811kDataset(mode='test')
    logger.info(f"Train : Valid : Test = {len(train_set):,} : {len(valid_set):,} : {len(test_set):,}")
    logger.info(f"Training steps per epoch: {len(train_set) // config.batch_size + 1:,}")
    logger.info(f"Total number of training iterations: {(len(train_set) // config.batch_size + 1) * config.epochs:,}")

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
