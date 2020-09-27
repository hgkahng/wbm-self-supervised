# -*- coding: utf-8 -*-

import os
import sys

import torch
import torch.nn as nn
import numpy as np

from datasets.wafer import WM811K
from datasets.cifar import CustomCIFAR10
from datasets.transforms import get_transform

from configs import ClassificationConfig
from configs import ALEXNET_BACKBONE_CONFIGS
from configs import VGGNET_BACKBONE_CONFIGS
from configs import RESNET_BACKBONE_CONFIGS
from models.alexnet import AlexNetBackbone
from models.vggnet import VggNetBackbone
from models.resnet import ResNetBackbone
from models.head import LinearClassifier

from tasks.classification import Classification

from utils.loss import LabelSmoothingLoss
from utils.logging import get_logger
from utils.metrics import MultiAccuracy, MultiF1Score
from utils.optimization import get_optimizer, get_scheduler


AVAILABLE_MODELS = {
    'alexnet': (ALEXNET_BACKBONE_CONFIGS, AlexNetBackbone),
    'vggnet': (VGGNET_BACKBONE_CONFIGS, VggNetBackbone),
    'resnet': (RESNET_BACKBONE_CONFIGS, ResNetBackbone),
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


def main():

    # 1. Configurations
    config = ClassificationConfig.from_command_line()
    config.save()

    logfile = os.path.join(config.checkpoint_dir, 'main.log')
    logger = get_logger(stream=False, logfile=logfile)

    in_channels = IN_CHANNELS[config.data]
    num_classes = NUM_CLASSES[config.data]

    # 2. Dataset
    if config.data == 'wm811k':
        train_transform = get_transform(config.data, size=config.input_size, mode=config.augmentation)
        test_transform  = get_transform(config.data, size=config.input_size, mode='test')
        train_set = WM811K('./data/wm811k/labeled/train/', transform=train_transform, proportion=config.label_proportion)
        valid_set = WM811K('./data/wm811k/labeled/valid/', transform=test_transform)
        test_set  = WM811K('./data/wm811k/labeled/test/', transform=test_transform)
    elif config.data == 'cifar10':
        train_transform = get_transform(config.data, size=config.input_size, mode=config.augmentation)
        test_transform = get_transform(config.data, size=config.input_size, mode='test')
        train_set = CustomCIFAR10('./data/cifar10/', train=True, transform=train_transform, proportion=config.label_proportion)
        valid_set = CustomCIFAR10('./data/cifar10/', train=False, transform=test_transform)
        test_set  = valid_set
    elif config.data == 'stl10':
        raise NotImplementedError
    elif config.data == 'imagenet':
        raise NotImplementedError
    else:
        raise KeyError

    # 3. Model
    BACKBONE_CONFIGS, Backbone = AVAILABLE_MODELS[config.backbone_type]
    backbone = Backbone(BACKBONE_CONFIGS[config.backbone_config], in_channels=in_channels)
    classifier = LinearClassifier(in_channels=backbone.out_channels, num_classes=num_classes)

    # 3-1. Load pre-trained weights (if provided)
    if config.pretrained_model_file is not None:
        try:
            backbone.load_weights_from_checkpoint(path=config.pretrained_model_file, key='backbone')
        except KeyError:
            backbone.load_weights_from_checkpoint(path=config.pretrained_model_file, key='encoder')
        finally:
            logger.info(f"Loaded pre-trained model from: {config.pretrained_model_file}")
    else:
        logger.info("No pre-trained model provided.")

    # 3-2. Finetune or freeze weights of backbone
    if config.freeze:
        backbone.freeze_weights()
        logger.info("Freezing backbone weights.")
        

    # 4. Optimization
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

    # 5. Experiment (classification)
    experiment_kwargs = {
        'backbone': backbone,
        'classifier': classifier,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'loss_function': LabelSmoothingLoss(num_classes, smoothing=config.label_smoothing),
        'checkpoint_dir': config.checkpoint_dir,
        'write_summary': config.write_summary,
        'metrics': {
            'accuracy': MultiAccuracy(num_classes=num_classes),
            'f1': MultiF1Score(num_classes=num_classes, average='macro'),
        },
    }
    experiment = Classification(**experiment_kwargs)

    # 6. Run (classification)
    run_kwargs = {
        'train_set': train_set,
        'valid_set': valid_set,
        'test_set': test_set,
        'epochs': config.epochs,
        'batch_size': config.batch_size,
        'num_workers': config.num_workers,
        'device': config.device,
        'logger': logger,
    }
    experiment.run(**run_kwargs)
    logger.handlers.clear()


if __name__ == '__main__':

    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
