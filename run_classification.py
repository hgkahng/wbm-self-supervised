# -*- coding: utf-8 -*-

import os
import sys
import argparse

import torch
import numpy as np

from datasets.wafer import WM811K
from datasets.cifar import CustomCIFAR10
from datasets.transforms import get_transform

from models.config import ClassificationConfig
from models.config import ALEXNET_BACKBONE_CONFIGS
from models.config import VGGNET_BACKBONE_CONFIGS
from models.config import RESNET_BACKBONE_CONFIGS
from models.alexnet import AlexNetBackbone
from models.vggnet import VggNetBackbone
from models.resnet import ResNetBackbone
from models.head import GAPClassifier

from tasks.classification import Classification

from utils.loss import LabelSmoothingLoss
from utils.logging import get_logger
from utils.metrics import MultiAccuracy, MultiF1Score, MultiAUPRC
from utils.optimization import get_optimizer, get_scheduler


AVAILABLE_MODELS = {
    'alexnet': (ALEXNET_BACKBONE_CONFIGS, ClassificationConfig, AlexNetBackbone),
    'vggnet': (VGGNET_BACKBONE_CONFIGS, ClassificationConfig, VggNetBackbone),
    'resnet': (RESNET_BACKBONE_CONFIGS, ClassificationConfig, ResNetBackbone),
}

CLASSIFIER_TYPES = {
    'linear': GAPClassifier,
    'mlp': None,
}

IN_CHANNELS = {
    "wm811k": 2,
    "cifar10": 3,
    "stl10": 3,
    "imagenet": 3,
}

NUM_CLASSES = {
    "wm811k": 9,
    "cifar10": 10,
    "stl10": None,
    "imagenet": 1000,
}


def parse_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser("Downstream classification, supports both linear evaluation & fine-tuning.", add_help=True)

    g0 = parser.add_argument_group('Randomness')
    g0.add_argument('--seed', type=int, required=True)

    g1 = parser.add_argument_group('Data')
    g1.add_argument('--data', type=str, choices=('wm811k', 'cifar10', 'stl10', 'imagenet'), required=True)
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
    g3.add_argument('--freeze', action='store_true', help='for linear evaluation, freeze backbone weights.')
    g3.add_argument('--label_proportion', type=float, default=1.0, help='proportion of labeled data. (0, 1].')

    g4 = parser.add_argument_group('Model Training')
    g4.add_argument('--epochs', type=int, default=100, help='number of training epochs.')
    g4.add_argument('--batch_size', type=int, default=256, help='batch size used during training.')
    g4.add_argument('--num_workers', type=int, default=0, help='number of cpu threads for data loading.')
    g4.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cpu'])
    g4.add_argument('--augmentation', type=str, default='rotate', 
                    choices=['crop', 'rotate', 'cutout', 'shift', 'noise', 'test', 'crop+rotate', 'shift+crop+rotate', 'shift+crop+rotate+cutout+noise']
    )

    g5 = parser.add_argument_group('Regularization')
    g5.add_argument('--dropout', type=float, default=0.0, help='dropout rate in fully-connected layers.')
    g5.add_argument('--smoothing', type=float, default=0.0, help='label smoothing ratio.')
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

    # Subparsers are defined for configuring pretrained models.
    subparsers = parser.add_subparsers(title='Pretext Tasks')

    scratch = subparsers.add_parser('scratch', add_help=True)
    scratch.set_defaults(pretext=None)

    denoising = subparsers.add_parser('denoising', add_help=True)
    denoising.add_argument('--root', type=str, required=True)
    denoising.add_argument('--noise', type=float, default=0., choices=[0.0, 0.01, 0.05, 0.1])
    denoising.set_defaults(pretext='denoising')

    inpainting = subparsers.add_parser('inpainting', add_help=True)
    inpainting.add_argument('--root', type=str, required=True)
    inpainting.add_argument('--size', type=tuple, default=(10, 10))
    inpainting.set_defaults(pretext='inpainting')

    jigsaw = subparsers.add_parser('jigsaw', add_help=True)
    jigsaw.add_argument('--root', type=str, required=True)
    jigsaw.add_argument('--num_patches', type=int, default=9, choices=(4, 9, 16, 25))
    jigsaw.add_argument('--num_permutations', type=int, default=100, choices=(50, 100))
    jigsaw.set_defaults(pretext='jigsaw')

    rotation = subparsers.add_parser('rotation', add_help=True)
    rotation.add_argument('--root', type=str, required=True)
    rotation.set_defaults(pretext='rotation')

    bigan = subparsers.add_parser('bigan', add_help=True)
    bigan.add_argument('--root', type=str, required=True)
    bigan.add_argument('--gan_type', type=str, default='dcgan', choices=('dcgan', 'vgg'))
    bigan.add_argument('--gan_config', type=str, default=None, choices=('3', '6', '9'))
    bigan.set_defaults(pretext='bigan')

    pirl = subparsers.add_parser('pirl', add_help=True)
    pirl.add_argument('--root', type=str, required=True)
    pirl.add_argument('--projector_type', type=str, default='linear', choices=('linear', 'mlp'))
    pirl.add_argument('--projector_size', type=int, default=128)
    pirl.add_argument('--num_negatives', type=int, default=5000)
    pirl.set_defaults(pretext='pirl')

    moco = subparsers.add_parser('moco', add_help=True)
    moco.add_argument('--root', type=str, required=True)
    moco.set_defaults(pretext='moco')

    simclr = subparsers.add_parser('simclr', add_help=True)
    simclr.add_argument('--root', type=str, required=True)
    simclr.add_argument('--projector_type', type=str, default='mlp', choices=('linear', 'mlp'))
    simclr.add_argument('--projector_size', type=int, default=128)
    simclr.set_defaults(pretext='simclr')

    semiclr = subparsers.add_parser('semiclr', add_help=True)
    semiclr.add_argument('--root', type=str, required=True)
    semiclr.add_argument('--projector_type', type=str, default='mlp', choices=('linear', 'mlp'))
    semiclr.add_argument('--projector_size', type=int, default=128)
    semiclr.set_defaults(pretext='semiclr')

    attnclr = subparsers.add_parser('attnclr', add_help=True)
    attnclr.add_argument('--root', type=str, required=True)
    attnclr.add_argument('--projector_type', type=str, default='mlp', choices=('linear', 'mlp'))
    attnclr.add_argument('--projector_size', type=int, default=128)
    attnclr.set_defaults(pretext='attnclr')

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

    elif config.data == 'cifar10':
        input_transform = get_transform(config.data, size=config.input_size, mode='test')
        train_set = CustomCIFAR10('./data/cifar10/', train=True, transform=input_transform, proportion=config.label_proportion)
        valid_set = CustomCIFAR10('./data/cifar10/', train=False, transform=input_transform)
        test_set  = valid_set

    elif config.data == 'stl10':
        raise NotImplementedError

    elif config.data == 'imagenet':
        raise NotImplementedError

    else:
        raise KeyError

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

    # Load pretrained weights if pretext task is specified
    if config.pretext is not None:
        pretrained_model_file, pretrained_config = \
            config.find_pretrained_model(root=config.root)
        if config.pretext == 'denoising':
            backbone.load_weights_from_checkpoint(path=pretrained_model_file, key='encoder')
        elif config.pretext == 'inpainting':
            backbone.load_weights_from_checkpoint(path=pretrained_model_file, key='encoder')
        elif config.pretext == 'jigsaw':
            backbone.load_weights_from_checkpoint(path=pretrained_model_file, key='backbone')
        elif config.pretext == 'rotation':
            backbone.load_weights_from_checkpoint(path=pretrained_model_file, key='backbone')
        elif config.pretext == 'bigan':
            backbone.load_weights_from_checkpoint(path=pretrained_model_file, key='encoder')
        elif config.pretext == 'moco':
            backbone.load_weights_from_checkpoint(path=pretrained_model_file, key='backbone')
        elif config.pretext == 'pirl':
            backbone.load_weights_from_checkpoint(path=pretrained_model_file, key='backbone')
        elif config.pretext == 'simclr':
            backbone.load_weights_from_checkpoint(path=pretrained_model_file, key='backbone')
        elif config.pretext == 'semiclr':
            backbone.load_weights_from_checkpoint(path=pretrained_model_file, key='backbone')
        elif config.pretext == 'attnclr':
            backbone.load_weights_from_checkpoint(path=pretrained_model_file, key='backbone')
        else:
            raise NotImplementedError
    logger.info(f"Pretrained model: {config.pretext}")

    # Optionally freeze backbone layers
    if config.freeze:
        _ = backbone.freeze_weights(to_freeze=['all'])
        logger.info("Freezed weights for CNN backbone.")
    logger.info(f"Trainable parameters ({backbone.__class__.__name__}): {backbone.num_parameters:,}")
    logger.info(f"Trainable parameters ({classifier.__class__.__name__}): {classifier.num_parameters:,}")

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

    # 4. Experiment (classification)
    experiment_kwargs = {
        'backbone': backbone,
        'classifier': classifier,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'loss_function': LabelSmoothingLoss(num_classes, smoothing=config.smoothing),
        'checkpoint_dir': config.checkpoint_dir,
        'write_summary': config.write_summary,
        'metrics': {
            'accuracy': MultiAccuracy(num_classes=num_classes),
            'f1': MultiF1Score(num_classes=num_classes, average='macro'),
            'auprc': MultiAUPRC(num_classes=num_classes),
        },
    }
    experiment = Classification(**experiment_kwargs)
    logger.info(f"Saving model checkpoints to: {experiment.checkpoint_dir}")

    # 9. RUN (classification)
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
