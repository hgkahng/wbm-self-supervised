# -*- coding: utf-8 -*-

import os
import sys

import torch

from datasets.wafer import WM811KForPIRL
from datasets.transforms import get_transform

from configs.task_configs import PIRLConfig
from configs.network_configs import ALEXNET_BACKBONE_CONFIGS
from configs.network_configs import VGGNET_BACKBONE_CONFIGS
from configs.network_configs import RESNET_BACKBONE_CONFIGS
from models.alexnet import AlexNetBackbone
from models.vggnet import VggNetBackbone
from models.resnet import ResNetBackbone
from models.head import LinearHead, MLPHead
from tasks.pirl import PIRL, MemoryBank
from utils.loss import PIRLLoss
from utils.metrics import TopKAccuracy
from utils.logging import get_logger
from utils.optimization import get_optimizer, get_scheduler


AVAILABLE_MODELS = {
    'alexnet': (ALEXNET_BACKBONE_CONFIGS, AlexNetBackbone),
    'vggnet': (VGGNET_BACKBONE_CONFIGS, VggNetBackbone),
    'resnet': (RESNET_BACKBONE_CONFIGS, ResNetBackbone),
}

PROJECTOR_TYPES = {
    'linear': LinearHead,
    'mlp': MLPHead,
}

IN_CHANNELS = {'wm811k': 2}


def main():
    """Main function."""

    # 1. Configurations
    config = PIRLConfig.from_command_line()
    config.save()
    logfile = os.path.join(config.checkpoint_dir, 'main.log')
    logger = get_logger(stream=False, logfile=logfile)

    BACKBONE_CONFIGS, Backbone = AVAILABLE_MODELS[config.backbone_type]
    Projector = PROJECTOR_TYPES[config.projector_type]

    # 2. Data
    if config.data == 'wm811k':
        data_transforms = {
            'transform': get_transform(
                data=config.data,
                size=config.input_size,
                mode='test'
            ),
            'positive_transform': get_transform(
                data=config.data,
                size=config.input_size,
                mode=config.augmentation,
            ),
        }
        train_set = torch.utils.data.ConcatDataset(
            [
                WM811KForPIRL('./data/wm811k/unlabeled/train/', **data_transforms),
                WM811KForPIRL('./data/wm811k/labeled/train/', **data_transforms),
            ]
        )
        valid_set = torch.utils.data.ConcatDataset(
            [
                WM811KForPIRL('./data/wm811k/unlabeled/valid/', **data_transforms),
                WM811KForPIRL('./data/wm811k/labeled/valid/', **data_transforms),
            ]
        )
        test_set = torch.utils.data.ConcatDataset(
            [
                WM811KForPIRL('./data/wm811k/unlabeled/test/', **data_transforms),
                WM811KForPIRL('./data/wm811k/labeled/test/', **data_transforms),
            ]
        )
    else:
        raise ValueError(
            f"PIRL only supports 'wm811k' data. Received '{config.data}'."
        )

    # 3. Model
    backbone = Backbone(BACKBONE_CONFIGS[config.backbone_config], in_channels=IN_CHANNELS[config.data])
    projector = Projector(backbone.out_channels, config.projector_size)

    # 4. Optimization
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

    # 5. Experiment (PIRL)
    experiment_kwargs = {
        'backbone': backbone,
        'projector': projector,
        'memory': MemoryBank(
            size=(len(train_set), config.projector_size),
            device=config.device
            ),
        'optimizer': optimizer,
        'scheduler': scheduler,
        'loss_function': PIRLLoss(temperature=config.temperature),
        'loss_weight': config.loss_weight,
        'num_negatives': config.num_negatives,
        'metrics': {
            'top@1': TopKAccuracy(num_classes=1 + config.num_negatives, k=1),
            'top@5': TopKAccuracy(num_classes=1 + config.num_negatives, k=5)
            },
        'checkpoint_dir': config.checkpoint_dir,
        'write_summary': config.write_summary,
    }
    experiment = PIRL(**experiment_kwargs)

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
    logger.info(f"Trainable parameters ({backbone.__class__.__name__}): {backbone.num_parameters:,}")
    logger.info(f"Trainable parameters ({projector.__class__.__name__}): {projector.num_parameters:,}")
    logger.info(f"Projector type: {config.projector_type}")
    logger.info(f"Projector dimension: {config.projector_size}")
    logger.info(f"Saving model checkpoints to: {experiment.checkpoint_dir}")
    logger.info(f"Epochs: {run_kwargs['epochs']}, Batch size: {run_kwargs['batch_size']}")
    logger.info(f"Workers: {run_kwargs['num_workers']}, Device: {run_kwargs['device']}")

    steps_per_epoch = len(train_set) // config.batch_size + 1
    logger.info(f"Training steps per epoch: {steps_per_epoch:,}")
    logger.info(f"Total number of training iterations: {steps_per_epoch * config.epochs:,}")

    experiment.run(**run_kwargs)
    logger.handlers.clear()


if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
