# -*- coding: utf-8 -*-

import os
import sys
import copy
import argparse

import torch

from datasets.wafer import UnlabeledWM811kFolderForSemiCLR
from datasets.wafer import LabeledWM811kFolderForSemiCLR
from datasets.transforms import RotationTransform
from models.config import SemiCLRConfig, VGG_BACKBONE_CONFIGS, RESNET_BACKBONE_CONFIGS
from models.vgg import VGGBackbone
from models.resnet import ResNetBackbone
from models.head import GAPProjector, NonlinearProjector
from tasks.semiclr import SemiCLR
from utils.loss import SimCLRLoss
from utils.optimization import get_optimizer, get_scheduler
from utils.logging import get_logger


AVAILABLE_MODELS = {
    'vgg': (VGG_BACKBONE_CONFIGS, SemiCLRConfig, VGGBackbone),
    'resnet': (RESNET_BACKBONE_CONFIGS, SemiCLRConfig, ResNetBackbone),
}

PROJECTOR_TYPES = {
    'linear': GAPProjector,
    'mlp': NonlinearProjector
}


def parse_args():

    parser = argparse.ArgumentParser("SemiCLR pretext task on WM811k.", add_help=True)

    g1 = parser.add_argument_group('General')
    g1.add_argument('--input_size', type=int, default=56, choices=(28, 56, 112, 224))

    g2 = parser.add_argument_group('CNN Backbone')
    g2.add_argument('--backbone_type', type=str, default='resnet', choices=('vgg', 'resnet'))
    g2.add_argument('--backbone_config', type=str, default='18.original')
    g2.add_argument('--in_channels', type=int, default=2, choices=(1, 2))

    g3 = parser.add_argument_group('SemiCLR')
    g3.add_argument('--projector_type', type=str, default='linear', choices=('linear', 'mlp'))
    g3.add_argument('--projector_size', type=int, default=128)
    g3.add_argument('--temperature', type=float, default=0.07)

    g4 = parser.add_argument_group('Training')
    g3.add_argument('--label_proportion', type=float, default=1.0, help='Proportion of labeled data.')
    g4.add_argument('--epochs', type=int, default=100)
    g4.add_argument('--unlabeled_batch_size', type=int, default=1024)
    g4.add_argument('--labeled_batch_size', type=int, default=1024)
    g4.add_argument('--num_workers', type=int, default=0)
    g4.add_argument('--device', type=str, default='cuda:0', choices=('cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cpu'))

    g5 = parser.add_argument_group('Regularization')  # pylint: disable=unused-variable

    g6 = parser.add_argument_group('Optimizer')
    g6.add_argument('--optimizer', type=str, default='sgd', choices=('sgd', 'adamw'))
    g6.add_argument('--learning_rate', type=float, default=0.01)
    g6.add_argument('--weight_decay', type=float, default=0.001)
    g6.add_argument('--momentum', type=float, default=0.9, help='only for SGD.')

    g7 = parser.add_argument_group('Scheduler')
    g7.add_argument('--scheduler', type=str, default=None, choices=('step', 'cosine'))
    g7.add_argument('--milestone', type=int, default=None, help='only for step decay.')
    g7.add_argument('--warmup_steps', type=int, default=0, help='only for linear warmups.')

    g8 = parser.add_argument_group('Logging')
    g8.add_argument('--checkpoint_root', type=str, default='./checkpoints/')
    g8.add_argument('--write_summary', action='store_true', help='write summaries with TensorBoard.')
    g8.add_argument('--save_every', action='store_true', help='save every checkpoint w/ improvements.')

    g9 = parser.add_argument_group('Resuming training from a checkpoint')
    g9.add_argument('--resume_from_checkpoint', type=str, default=None)

    return parser.parse_args()


def main(args):
    """Main function."""

    torch.backends.cudnn.benchmark = True
    BACKBONE_CONFIGS, Config, Backbone = AVAILABLE_MODELS[args.backbone_type]
    Projector = PROJECTOR_TYPES[args.projector_type]

    # 1. Configurations
    config = Config(args)
    config.save()

    # 2. Logger
    logfile = os.path.join(config.checkpoint_dir, 'main.log')
    logger  = get_logger(stream=False, logfile=logfile)

    # 3. Backbone
    backbone = Backbone(BACKBONE_CONFIGS[config.backbone_config], config.in_channels)

    # 4. Projector
    projector = Projector(
        in_channels=backbone.out_channels,
        num_features=config.projector_size,
    )
    logger.info(f"Trainable parameters ({backbone.__class__.__name__}): {backbone.num_parameters:,}")
    logger.info(f"Trainable parameters ({projector.__class__.__name__}): {projector.num_parameters:,}")
    logger.info(f"Projector size: {config.projector_size}")

    # 5. Set optimizer and learning rate scheduler
    params = [{'params': backbone.parameters()}, {'params': projector.parameters()}]
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

    # 6. Configure input datasets and transforms
    unlabeled_data_kwargs = {
        'transform': RotationTransform.get(size=(config.input_size, config.input_size))
    }
    unlabeled_train_set = UnlabeledWM811kFolderForSemiCLR('./data/images/unlabeled/train/', **unlabeled_data_kwargs)
    unlabeled_valid_set = UnlabeledWM811kFolderForSemiCLR('./data/images/unlabeled/valid/', **unlabeled_data_kwargs)
    unlabeled_test_set  = UnlabeledWM811kFolderForSemiCLR('./data/images/unlabeled/test/', **unlabeled_data_kwargs)

    labeled_data_kwargs = copy.deepcopy(unlabeled_data_kwargs)
    labeled_data_kwargs.update({'proportion': config.label_proportion})
    labeled_train_set = LabeledWM811kFolderForSemiCLR('./data/images/labeled/train/', **labeled_data_kwargs)
    labeled_valid_set = LabeledWM811kFolderForSemiCLR('./data/images/labeled/valid/', **labeled_data_kwargs)
    labeled_test_set  = LabeledWM811kFolderForSemiCLR('./data/images/labeled/test/', **labeled_data_kwargs)

    steps_per_epoch = len(unlabeled_train_set) // config.unlabeled_batch_size + 1
    logger.info(f"(Unlabeled) Train : Valid : Test = {len(unlabeled_train_set):,} : {len(unlabeled_valid_set):,} : {len(unlabeled_test_set):,}")
    logger.info(f"(  Labeled) Train : Valid : Test = {len(labeled_train_set):,} : {len(labeled_valid_set):,} : {len(labeled_test_set):,}")
    logger.info(f"Training steps per epoch: {steps_per_epoch:,}")
    logger.info(f"Total number of training iterations: {steps_per_epoch * config.epochs:,}")

    # 7. Configure experiment (SemiCLR)
    experiment_kwargs = {
        'backbone': backbone,
        'projector': projector,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'loss_function': SimCLRLoss(temperature=config.temperature),
        'augment_function': None,
        'metrics': None,
        'checkpoint_dir': config.checkpoint_dir,
        'write_summary': config.write_summary,
    }
    experiment = SemiCLR(**experiment_kwargs)
    logger.info(f"Saving model checkpoints to: {experiment.checkpoint_dir}")

    # 8. Run experiment (SimCLR)
    run_kwargs = {
        'unlabeled_train_set': unlabeled_train_set,
        'unlabeled_valid_set': unlabeled_valid_set,
        'unlabeled_test_set': unlabeled_test_set,
        'labeled_train_set': labeled_train_set,
        'labeled_valid_set': labeled_valid_set,
        'labeled_test_set': labeled_test_set,
        'epochs': config.epochs,
        'unlabeled_batch_size': config.unlabeled_batch_size,
        'labeled_batch_size': config.labeled_batch_size,
        'num_workers': config.num_workers,
        'device': config.device,
        'logger': logger,
        'save_every': config.save_every,
    }
    logger.info(f"Epochs: {run_kwargs['epochs']}")
    logger.info(f"Unlabeled batch size: {run_kwargs['unlabeled_batch_size']}")
    logger.info(f"Labeled batch size: {run_kwargs['labeled_batch_size']}")
    logger.info(f"Workers: {run_kwargs['num_workers']}, Device: {run_kwargs['device']}")

    if config.resume_from_checkpoint is not None:
        logger.info(f"Resuming from checkpoint: {config.resume_from_checkpoint}")
        model_ckpt = os.path.join(config.resume_from_checkpoint, 'best_model.pt')
        experiment.load_model_from_checkpoint(model_ckpt)

        # Assign optimizer variables to appropriate device
        for state in experiment.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(config.device)

    experiment.run(**run_kwargs)
    logger.handlers.clear()


if __name__ == '__main__':
    try:
        main(parse_args())
    except KeyboardInterrupt:
        sys.exit(0)
