 # -*- coding: utf-8 -*-

import os
import sys
import argparse

import torch
import torch.nn as nn
import numpy as np

from datasets.wafer import WM811K
from datasets.transforms import get_transform

from configs.task_configs import MixupConfig
from configs.network_configs import ALEXNET_BACKBONE_CONFIGS
from configs.network_configs import VGGNET_BACKBONE_CONFIGS
from configs.network_configs import RESNET_BACKBONE_CONFIGS
from models.alexnet import AlexNetBackbone
from models.vggnet import VggNetBackbone
from models.resnet import ResNetBackbone
from models.head import GAPClassifier

from tasks.mixup import Mixup

from utils.logging import get_logger
from utils.metrics import MultiAccuracy, MultiF1Score
from utils.optimization import get_optimizer, get_scheduler


AVAILABLE_MODELS = {
    'alexnet': (ALEXNET_BACKBONE_CONFIGS, MixupConfig, AlexNetBackbone),
    'vggnet': (VGGNET_BACKBONE_CONFIGS, MixupConfig, VggNetBackbone),
    'resnet': (RESNET_BACKBONE_CONFIGS, MixupConfig, ResNetBackbone),
}

CLASSIFIER_TYPES = {'linear': GAPClassifier,}
IN_CHANNELS = {"wm811k": 2}
NUM_CLASSES = {"wm811k": 9}


def parse_args():

    def convert_arg_line_to_args(arg_line):
        for arg in arg_line.split():
            if not arg.strip():
                continue
            yield arg

    parser = argparse.ArgumentParser("WM-811K classification with Mixup.", fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    
    g0 = parser.add_argument_group('Randomness')
    g0.add_argument('--seed', type=int, default=0, help='Random seed for repeated trials of experiments.')

    g1 = parser.add_argument_group('Data')
    g1.add_argument('--data', type=str, choices=('wm811k', ), required=True)
    g1.add_argument('--input_size', type=int, choices=(32, 64, 96, 112, 224), required=True)

    g2 = parser.add_argument_group('CNN Backbone')
    g2.add_argument('--backbone_type', type=str, choices=('alexnet', 'vggnet', 'resnet'), required=True)
    g2.add_argument(
        '--backbone_config',
        type=str,
        choices=(
            'batch_norm',                                # alexnet
            '16.batch_norm',                             # vggnet
            '18.original', '34.original', '50.original'  # resnet
        ),
        required=True)

    g3 = parser.add_argument_group('Classification')
    g3.add_argument('--label_proportion', type=float, default=1.0, help='proportion of labeled data. (0, 1].')

    g4 = parser.add_argument_group('Model Training')
    g4.add_argument('--epochs', type=int, default=100, help='number of training epochs.')
    g4.add_argument('--batch_size', type=int, default=256, help='batch size used during training.')
    g4.add_argument('--num_workers', type=int, default=0, help='number of cpu threads for data loading.')
    g4.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cpu'])
    g4.add_argument('--augmentation', type=str, default='test', choices=['rotate', 'test'])
    g4.add_argument('--disable_mixup', action='store_true', help='Use only for comparison purposes.')

    g5 = parser.add_argument_group('Regularization')
    g5.add_argument('--dropout', type=float, default=0.0, help='dropout rate in fully-connected layers.')
    g5.add_argument('--balance', action='store_true', help='Balance class distribution within batches.')

    g6 = parser.add_argument_group('Optimizer')
    g6.add_argument('--optimizer', type=str, default='sgd', choices=('sgd', 'adamw', 'lars'))
    g6.add_argument('--learning_rate', type=float, default=0.01)
    g6.add_argument('--weight_decay', type=float, default=0.001)
    g6.add_argument('--momentum', type=float, default=0.9, help='only for SGD.')

    g7 = parser.add_argument_group('Scheduler')
    g7.add_argument('--scheduler', type=str, default=None, choices=('step', 'cosine', 'restart', 'none'))
    g7.add_argument('--milestone', type=int, default=None, help='For step decay.')
    g7.add_argument('--warmup_steps', type=int, default=0, help='For linear warmups.')
    g7.add_argument('--cycles', type=int, default=1, help='For hard restarts.')

    g8 = parser.add_argument_group('Logging')
    g8.add_argument('--checkpoint_root', type=str, default='./checkpoints/')
    g8.add_argument('--write_summary', action='store_true', help='Write summaries with TensorBoard.')
    g8.add_argument('--eval_metric', type=str, default='loss', choices=('loss', 'f1', 'accuracy'))

    return parser.parse_args()


def main(args):
    """Main function."""

    # 0. Main configurations
    BACKBONE_CONFIGS, Config, Backbone = AVAILABLE_MODELS[args.backbone_type]
    Classifier = CLASSIFIER_TYPES['linear']
    config = Config(args)
    config.save()

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)  # For reproducibility
    torch.backends.cudnn.benchmark = True

    logfile = os.path.join(config.checkpoint_dir, 'main.log')
    logger = get_logger(stream=False, logfile=logfile)

    in_channels = IN_CHANNELS[config.data]
    num_classes = NUM_CLASSES[config.data]

    # 1. Dataset
    if config.data == 'wm811k':
        train_transform = get_transform(config.data, size=config.input_size, mode=config.augmentation)
        test_transform  = get_transform(config.data, size=config.input_size, mode='test')
        train_set = WM811K(
            './data/wm811k/labeled/train/',
            transform=train_transform,
            proportion=config.label_proportion,
            seed=config.seed
        )
        valid_set = WM811K('./data/wm811k/labeled/valid/', transform=test_transform)
        test_set  = WM811K('./data/wm811k/labeled/test/', transform=test_transform)
    else:
        raise NotImplementedError
    
    steps_per_epoch = len(train_set) // config.batch_size + 1
    logger.info(f"Data type: {config.data}")
    logger.info(f"Train : Valid : Test = {len(train_set):,} : {len(valid_set):,} : {len(test_set):,}")
    logger.info(f"Training steps per epoch: {steps_per_epoch:,}")
    logger.info(f"Total number of training iterations: {steps_per_epoch * config.epochs:,}")

    # 2. Model
    backbone = Backbone(BACKBONE_CONFIGS[config.backbone_config], in_channels)
    classifier = Classifier(
        in_channels=backbone.out_channels,
        num_classes=num_classes,
        dropout=config.dropout,
    )

    # 3. Optimization (TODO: add LARS)
    params = [
        {'params': backbone.parameters(), 'lr': config.learning_rate},
        {'params': classifier.parameters(), 'lr': config.learning_rate},
    ]
    optimizer = get_optimizer(
        params=params,
        name=config.optimizer,
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = get_scheduler(
        optimizer=optimizer,
        name=config.scheduler,
        epochs=config.epochs,
        milestone=config.milestone,
        warmup_steps=config.warmup_steps
    )

    # 4. Experiment (Mixup)
    experiment_kwargs = {
        'backbone': backbone,
        'classifier': classifier,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'loss_function': nn.CrossEntropyLoss(),
        'checkpoint_dir': config.checkpoint_dir,
        'write_summary': config.write_summary,
        'metrics': {
            'accuracy': MultiAccuracy(num_classes=num_classes),
            'f1': MultiF1Score(num_classes=num_classes, average='macro'),
        },
    }
    experiment = Mixup(**experiment_kwargs)
    logger.info(f"Saving model checkpoints to: {experiment.checkpoint_dir}")

    # 9. RUN (Mixup)
    run_kwargs = {
        'train_set': train_set,
        'valid_set': valid_set,
        'test_set': test_set,
        'epochs': config.epochs,
        'batch_size': config.batch_size,
        'num_workers': config.num_workers,
        'device': config.device,
        'logger': logger,
        'eval_metric': config.eval_metric,
        'balance': config.balance,
        'disable_mixup': config.disable_mixup,
    }
    logger.info(f"Epochs: {run_kwargs['epochs']}, Batch size: {run_kwargs['batch_size']}")
    logger.info(f"Workers: {run_kwargs['num_workers']}, Device: {run_kwargs['device']}")
    logger.info(f"Mixup enabled: {not config.disable_mixup}")

    experiment.run(**run_kwargs)
    logger.handlers.clear()


if __name__ == '__main__':
    try:
        main(parse_args())
    except KeyboardInterrupt:
        sys.exit(0)
