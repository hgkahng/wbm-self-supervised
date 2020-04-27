# -*- coding: utf-8 -*-

import os
import sys
import argparse

import torch

from tasks import Classification
from models.config import ClassificationConfig, VGG_BACKBONE_CONFIGS, RESNET_BACKBONE_CONFIGS
from models.vgg import VGGBackbone
from models.resnet import ResNetBackbone
from models.head import GAPClassifier
from datasets.wafer import LabeledWM811kFolder
from datasets.transforms import BasicTransform
from utils.loss import LabelSmoothingLoss
from utils.metrics import MultiAccuracy, MultiF1Score
from utils.optimization import get_optimizer, get_scheduler
from utils.logging import get_logger


AVAILABLE_MODELS = {
    'vgg': (VGG_BACKBONE_CONFIGS, ClassificationConfig, VGGBackbone, GAPClassifier),
    'resnet': (RESNET_BACKBONE_CONFIGS, ClassificationConfig, ResNetBackbone, GAPClassifier),
}

FINETUNE_TYPES = {
    'end_to_end': 0,
    'freeze': 1,
    'blockwise_learning_rates': 2,
}

NUM_CLASSES = LabeledWM811kFolder.NUM_CLASSES


def parse_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser("Classification on Wafer40.", add_help=True)

    g1 = parser.add_argument_group('General')
    g1.add_argument('--data_index', type=int, required=True)
    g1.add_argument('--input_size', type=int, default=112, choices=(32, 64, 112, 224))
    g1.add_argument('--disable_benchmark', action='store_true', help='Benchmark mode is faster for fixed size inputs.')

    g2 = parser.add_argument_group('Backbone')
    g2.add_argument('--backbone_type', type=str, default=None, choices=('vgg', 'resnet'), required=True)
    g2.add_argument('--backbone_config', type=str, default=None, required=True)
    g2.add_argument('--in_channels', type=int, default=2, choices=(1, 2))

    g4 = parser.add_argument_group('Training')
    g4.add_argument('--labeled', type=float, default=1.0, help='Proportion of labeled data.')
    g4.add_argument('--balance', action='store_true', help='Balance class label distribution.')
    g4.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    g4.add_argument('--batch_size', type=int, default=256, help='Batch size used during training.')
    g4.add_argument('--num_workers', type=int, default=0, help='Number of cpu threads for data loading.')
    g4.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cpu'])

    g5 = parser.add_argument_group('Regularization')
    g5.add_argument('--dropout', type=float, default=0.0, help='Dropout rate in fully-connected layers.')
    g5.add_argument('--smoothing', type=float, default=0.0, help='Zero equals general cross entropy loss.')
    g5.add_argument('--class_weights', action='store_true', help='Calculate class weights.')

    g6 = parser.add_argument_group('Optimizer')
    g6.add_argument('--optimizer', type=str, default='adamw', choices=('adamw', 'sgd'))
    g6.add_argument('--learning_rate', type=float, default=0.001)
    g6.add_argument('--weight_decay', type=float, default=0.01)
    g6.add_argument('--optimizer_kwargs', nargs='+', default=[], help="Additional arguments for optimizers.")

    g7 = parser.add_argument_group('Scheduler')
    g7.add_argument('--scheduler', type=str, default=None, choices=('step', 'multi_step', 'plateau', 'cosine', 'restart'))
    g7.add_argument('--scheduler_kwargs', nargs='+', default=[], help="Arguments for learning rate scheduling.")

    g8 = parser.add_argument_group('Logging')
    g8.add_argument('--checkpoint_root', type=str, default='./checkpoints/')
    g8.add_argument('--write_summary', action='store_true', help='Write summaries with TensorBoard.')
    g8.add_argument('--eval_metric', type=str, default='loss', choices=('loss', 'f1', 'accuracy'))

    g9 = parser.add_argument_group('Fine-tuning')
    g9_finetune = g9.add_mutually_exclusive_group()
    g9_finetune.add_argument('--freeze', nargs='+', default=[], help='Freeze the weights of backbone blocks.')
    g9_finetune.add_argument('--blockwise_learning_rates', nargs='+', default=[], help='Set blockwise learning rates of backbone.')

    subparsers = parser.add_subparsers(title='Pretext Tasks')

    scratch = subparsers.add_parser('scratch', add_help=True)
    scratch.set_defaults(pretext=None)

    denoising = subparsers.add_parser('denoising', add_help=True)
    denoising.add_argument('--root', type=str, default=None)
    denoising.add_argument('--noise', type=float, default=0.1)
    denoising.set_defaults(pretext='denoising')

    inpainting = subparsers.add_parser('inpainting', add_help=True)
    inpainting.add_argument('--root', type=str, default=None)
    inpainting.add_argument('--size', type=tuple, default=(10, 10))
    inpainting.set_defaults(pretext='inpainting')

    jigsaw = subparsers.add_parser('jigsaw', add_help=True)
    jigsaw.add_argument('--root', type=str, default=None)
    jigsaw.add_argument('--num_patches', type=int, default=9, choices=(4, 9, 16, 25))
    jigsaw.add_argument('--num_permutations', type=int, default=100, choices=(50, 100))
    jigsaw.set_defaults(pretext='jigsaw')

    rotation = subparsers.add_parser('rotation', add_help=True)
    rotation.add_argument('--root', type=str, default=None)
    rotation.set_defaults(pretext='rotation')

    bigan = subparsers.add_parser('bigan', add_help=True)
    bigan.add_argument('--root', type=str, default=None)
    bigan.add_argument('--gan_type', type=str, default='dcgan', choices=('dcgan', 'vgg'))
    bigan.add_argument('--gan_config', type=str, default=None, choices=('3', '6', '9'))
    bigan.set_defaults(pretext='bigan')

    pirl = subparsers.add_parser('pirl', add_help=True)
    pirl.add_argument('--root', type=str, default=None)
    pirl.add_argument('--model_type', type=str, default='best', choices=('best', 'last'))
    pirl.add_argument('--projector_type', type=str, default='linear', choices=('linear', 'mlp'))
    pirl.add_argument('--projector_size', type=int, default=128)
    pirl.add_argument('--num_negatives', type=int, default=5000)
    pirl.add_argument('--noise', type=float, default=0.05)
    pirl.add_argument('--rotate', action='store_true')
    pirl.set_defaults(pretext='pirl')

    npid = subparsers.add_parser('npid', add_help=True)
    npid.add_argument('--root', type=str, default=None)
    npid.set_defaults(pretext='npid')

    moco = subparsers.add_parser('moco', add_help=True)
    moco.add_argument('--root', type=str, default=None)
    moco.set_defaults(pretext='moco')

    simclr = subparsers.add_parser('simclr', add_help=True)
    simclr.add_argument('--root', type=str, default=None)
    simclr.set_defaults(pretext='simclr')

    args = parser.parse_args()

    if args.pretext is not None:
        assert args.root is not None, "Provide a root directory under which a pretrained model might exist."

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

    blr_kwargs = get_kwargs(args.blockwise_learning_rates)
    setattr(args, 'blockwise_learning_rates', blr_kwargs)

    if len(args.freeze) > 0:
        setattr(args, 'finetune_type', FINETUNE_TYPES['freeze'])
    elif len(args.blockwise_learning_rates) > 0:
        setattr(args, 'finetune_type', FINETUNE_TYPES['blockwise_learning_rates'])
    else:
        setattr(args, 'finetune_type', FINETUNE_TYPES['end_to_end'])

    return args


def main(args):
    """Main function."""

    torch.backends.cudnn.benchmark = not args.disable_benchmark
    BACKBONE_CONFIGS, Config, Backbone, Classifier = AVAILABLE_MODELS[args.backbone_type]

    # 1. Configurations
    config = Config(args)
    config.save()

    # 2. Logging
    logfile = os.path.join(config.checkpoint_dir, 'main.log')
    logger = get_logger(stream=False, logfile=logfile)

    # 3-1. Backbone
    backbone = Backbone(BACKBONE_CONFIGS[config.backbone_config], config.in_channels)

    # 3-2. Load pretrained weights if pretext task is specified
    if config.pretext is not None:
        pretrained_model_file, pretrained_config = \
            config.find_pretrained_model(root=config.root, model_type=config.model_type)  # 'best_model.pt' or 'last_model.pt'
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
        else:
            raise NotImplementedError
    logger.info(f"Pretrained model: {config.pretext}")

    # 3-3. Optionally freeze backbone layers
    if config.finetune_type == 1:
        freezed_layer_names = backbone.freeze_weights(to_freeze=config.freeze)
        for l in freezed_layer_names:
            logger.info(f"'{l}' freezed.")

    # 4. Classifier
    classifier = Classifier(
        in_channels=backbone.output_shape[0],
        num_classes=NUM_CLASSES,
        dropout=config.dropout)

    # 5. Set learning rates
    params = [{'params': classifier.parameters(), 'lr': config.learning_rate}]
    if config.finetune_type == 2:
        for name, block in backbone.layers.named_children():
            params += [
                {
                    'params': block.parameters(),
                    'lr': config.blockwise_learning_rates.get(name, config.learning_rate)
                }
            ]
    else:
        params += [{'params': backbone.parameters(), 'lr': config.learning_rate}]
    logger.info(f"Finetune type: {config.finetune_type}")
    logger.info(f"Trainable parameters ({backbone.__class__.__name__}): {backbone.num_parameters:,}")
    logger.info(f"Trainable parameters ({classifier.__class__.__name__}): {classifier.num_parameters:,}")

    # 6. Set optimizer and learning rate scheduler
    optimizer = get_optimizer(
        params=params, name=config.optimizer,
        lr=config.learning_rate, weight_decay=config.weight_decay, **config.optimizer_kwargs
        )
    scheduler = get_scheduler(
        optimizer=optimizer, name=config.scheduler,
        epochs=config.epochs, **config.scheduler_kwargs
        )

    # 7. Configure input image transforms and load datasets
    transforms = BasicTransform.get(size=(config.input_size, config.input_size))
    train_set = LabeledWM811kFolder('./data/images/labeled/train/', transform=transforms, proportion=config.labeled)
    valid_set = LabeledWM811kFolder('./data/images/labeled/valid/', transform=transforms)
    test_set = LabeledWM811kFolder('./data/images/labeled/test/', transform=transforms)

    steps_per_epoch = len(train_set) // config.batch_size + 1
    logger.info(f"Train : Valid : Test = {len(train_set):,} : {len(valid_set):,} : {len(test_set):,}")
    logger.info(f"Training steps per epoch: {steps_per_epoch:,}")
    logger.info(f"Total number of training iterations: {steps_per_epoch * config.epochs:,}")

    # 8. Configure experiment (classification)
    experiment_kws = {
        'backbone': backbone,
        'classifier': classifier,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'loss_function': LabelSmoothingLoss(
            num_classes=NUM_CLASSES,
            smoothing=config.smoothing,
            reduction='mean',
            class_weights=train_set.class_weights if config.class_weights else None,
        ),
        'checkpoint_dir': config.checkpoint_dir,
        'write_summary': config.write_summary,
        'metrics': {
            'accuracy': MultiAccuracy(num_classes=NUM_CLASSES),
            'f1': MultiF1Score(num_classes=NUM_CLASSES, average='macro'),
        },
    }
    experiment = Classification(**experiment_kws)
    logger.info(f"Optimizer: {experiment.optimizer.__class__.__name__}")
    logger.info(f"Scheduler: {experiment.scheduler.__class__.__name__}")
    logger.info(f"Loss: {experiment.loss_function.__class__.__name__}")
    logger.info(f"Checkpoint directory: {experiment.checkpoint_dir}")

    # 9. Run experiment (classification)
    run_kws = {
        'train_set': train_set,
        'valid_set': valid_set,
        'test_set': test_set,
        'epochs': config.epochs,
        'batch_size': config.batch_size,
        'num_workers': config.num_workers,
        'device': config.device,
        'logger': logger,
        'eval_metric': config.eval_metric,
    }

    if config.class_weights:
        from datasets.wafer import WM811K_LABELS
        idx2label = {i: l for l, i in WM811K_LABELS.items()}
        for l, i in WM811K_LABELS.items():
            logger.info(f"Class weights - {l:>10},({i}): {train_set.class_weights[i]}")
    logger.info(f"Epochs: {run_kws['epochs']}, Batch size: {run_kws['batch_size']}")
    logger.info(f"Workers: {run_kws['num_workers']}, Device: {run_kws['device']}")

    experiment.run(**run_kws)
    logger.handlers.clear()


if __name__ == '__main__':
    try:
        main(parse_args())
    except KeyboardInterrupt:
        sys.exit(0)
