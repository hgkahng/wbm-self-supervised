# -*- coding: utf-8 -*-

import os
import sys
import glob
import argparse

import torch

from datasets.wafer import UnlabeledWM811kFolderForPIRL
from datasets.transforms import BasicTransform, RotationTransform
from models.config import PIRLConfig, VGG_BACKBONE_CONFIGS, RESNET_BACKBONE_CONFIGS
from models.vgg.backbone import VGGBackbone
from models.resnet.backbone import ResNetBackbone
from models.head import GAPProjector, NonlinearProjector
from tasks.pirl import PIRL, MemoryBank
from tasks.denoising import MaskedBernoulliNoise
from utils.loss import NCELoss
from utils.optimization import get_optimizer, get_scheduler
from utils.logging import get_logger


AVAILABLE_MODELS = {
    'vgg': (VGG_BACKBONE_CONFIGS, PIRLConfig, VGGBackbone),
    'resnet': (RESNET_BACKBONE_CONFIGS, PIRLConfig, ResNetBackbone),
}

PROJECTOR_TYPES = {
    'linear': GAPProjector,
    'mlp': NonlinearProjector,
}


def parse_args():

    parser = argparse.ArgumentParser("PIRL pretext task on WM811k.", add_help=True)

    g1 = parser.add_argument_group('General')
    g1.add_argument('--input_size', type=int, default=112, choices=(56, 112, 224))
    g1.add_argument('--disable_benchmark', action='store_true')

    g2 = parser.add_argument_group('Backbone')
    g2.add_argument('--backbone_type', type=str, default='resnet', choices=('vgg', 'resnet'))
    g2.add_argument('--backbone_config', type=str, default='18.original')
    g2.add_argument('--in_channels', type=int, default=2, choices=(1, 2))

    g3 = parser.add_argument_group('PIRL')
    g3.add_argument('--projector_type', type=str, default='linear', choices=('linear', 'mlp'))
    g3.add_argument('--projector_size', type=int, default=128)
    g3.add_argument('--num_negatives', type=int, default=1000)
    g3.add_argument('--loss_weight', type=float, default=0.9)
    g3.add_argument('--temperature', type=float, default=0.07)
    g3.add_argument('--noise', type=float, default=None)
    g3.add_argument('--rotate', action='store_true')

    g4 = parser.add_argument_group('Training')
    g4.add_argument('--epochs', type=int, default=100)
    g4.add_argument('--batch_size', type=int, default=1024)
    g4.add_argument('--num_workers', type=int, default=0)
    g4.add_argument('--device', type=str, default='cuda:1', choices=('cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cpu'))

    g5 = parser.add_argument_group('Regularization')

    g6 = parser.add_argument_group('Optimizer')
    g6.add_argument('--optimizer', type=str, default='adamw', choices=('sgd', 'adamw'))
    g6.add_argument('--learning_rate', type=float, default=0.0001)
    g6.add_argument('--weight_decay', type=float, default=0.0005)
    g6.add_argument('--optimizer_kwargs', nargs='+', default=[])

    g7 = parser.add_argument_group('Scheduler')
    g7.add_argument('--scheduler', type=str, default=None, choices=('step', 'plateau', 'cosine', 'restart'))
    g7.add_argument('--scheduler_kwargs', nargs='+', default=[])

    g8 = parser.add_argument_group('Logging')
    g8.add_argument('--checkpoint_root', type=str, default='./checkpoints/')
    g8.add_argument('--write_summary', action='store_true')
    g8.add_argument('--save_every', action='store_true')

    g9 = parser.add_argument_group('Resuming training from a checkpoint')
    g9.add_argument('--resume_from_checkpoint', type=str, default=None)

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
    BACKBONE_CONFIGS, Config, Backbone = AVAILABLE_MODELS[args.backbone_type]
    Projector = PROJECTOR_TYPES[args.projector_type]

    # 1. Configurations
    config = Config(args)
    config.save()

    # 2. Logger
    logfile = os.path.join(config.checkpoint_dir, 'main.log')
    logger = get_logger(stream=False, logfile=logfile)

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

    # 6. Set optimizer and learning rate scheduler
    params = [{'params': backbone.parameters()}, {'params': projector.parameters()}]
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
        **config.scheduler_kwargs
    )

    # 7. Configure input image transforms and load datasets
    data_kws = {'transform': BasicTransform.get(size=(config.input_size, config.input_size))}
    if config.rotate:
        data_kws['positive_transform'] = RotationTransform.get(size=(config.input_size, config.input_size))
    else:
        data_kws['positive_transform'] = BasicTransform.get(size=(config.input_size, config.input_size))

    train_set = UnlabeledWM811kFolderForPIRL('./data/images/unlabeled/train/', **data_kws)
    valid_set = UnlabeledWM811kFolderForPIRL('./data/images/unlabeled/valid/', **data_kws)
    test_set  = UnlabeledWM811kFolderForPIRL('./data/images/unlabeled/test/', **data_kws)

    steps_per_epoch = len(train_set) // config.batch_size + 1
    logger.info(f"Train : Valid : Test = {len(train_set):,} : {len(valid_set):,} : {len(test_set):,}")
    logger.info(f"Training steps per epoch: {steps_per_epoch:,}")
    logger.info(f"Total number of training iterations: {steps_per_epoch * config.epochs:,}")

    # 8. Configure experiment (PIRL)
    experiment_kws = {
        'backbone': backbone,
        'projector': projector,
        'memory': MemoryBank(
            size=(len(train_set), config.projector_size),
            device=config.device
            ),
        'noise_function': MaskedBernoulliNoise(config.noise) if config.noise > 0 else None,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'loss_function': NCELoss(temperature=config.temperature),
        'loss_weight': config.loss_weight,
        'num_negatives': config.num_negatives,
        'metrics': None,
        'checkpoint_dir': config.checkpoint_dir,
        'write_summary': config.write_summary,
    }
    experiment = PIRL(**experiment_kws)
    logger.info(f"Optimizer: {experiment.optimizer.__class__.__name__}")
    logger.info(f"Scheduler: {experiment.scheduler.__class__.__name__}")
    logger.info(f"Loss: {experiment.loss_function.__class__.__name__}")
    logger.info(f"Negative samples: {experiment.num_negatives:,}")
    logger.info(f"Loss weight: {experiment.loss_weight:.2f}")
    logger.info(f"Temperature: {experiment.loss_function.temperature:.2f}")
    logger.info(f"Noise: {config.noise:.2f}")
    logger.info(f"Rotate: {config.rotate}")
    logger.info(f"Checkpoint directory: {experiment.checkpoint_dir}")

    # 9-1. Experiment (PIRL)
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

    # 9-2. Optionally load from checkpoint
    if config.resume_from_checkpoint is not None:
        logger.info(f"Resuming from a checkpoint: {config.resume_from_checkpoint}")
        model_ckpt = os.path.join(config.resume_from_checkpoint, 'best_model.pt')
        memory_ckpt = os.path.join(config.resume_from_checkpoint, 'best_memory.pt')
        experiment.load_model_from_checkpoint(model_ckpt)  # load model & optimizer
        experiment.memory.load(memory_ckpt)                # load memory bank

        # Assign optimizer variables to appropriate device
        for state in experiment.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(config.device)

    experiment.run(**run_kws)
    logger.handlers.clear()


if __name__ == '__main__':
    try:
        main(parse_args())
    except KeyboardInterrupt:
        sys.exit(0)
