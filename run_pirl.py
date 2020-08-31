# -*- coding: utf-8 -*-

import os
import sys
import argparse

import torch

from datasets.wafer import WM811KForPIRL
from datasets.transforms import get_transform

from models.config import PIRLConfig
from models.config import ALEXNET_BACKBONE_CONFIGS
from models.config import VGGNET_BACKBONE_CONFIGS
from models.config import RESNET_BACKBONE_CONFIGS
from models.alexnet import AlexNetBackbone
from models.vggnet import VggNetBackbone
from models.resnet import ResNetBackbone
from models.head import GAPProjector, NonlinearProjector

from tasks.pirl import PIRL, MemoryBank

from utils.loss import PIRLLoss
from utils.metrics import TopKAccuracy
from utils.logging import get_logger
from utils.optimization import get_optimizer, get_scheduler



AVAILABLE_MODELS = {
    'alexnet': (ALEXNET_BACKBONE_CONFIGS, PIRLConfig, AlexNetBackbone),
    'vggnet': (VGGNET_BACKBONE_CONFIGS, PIRLConfig, VggNetBackbone),
    'resnet': (RESNET_BACKBONE_CONFIGS, PIRLConfig, ResNetBackbone),
}

PROJECTOR_TYPES = {
    'linear': GAPProjector,
    'mlp': NonlinearProjector,
}


IN_CHANNELS = {'wm811k': 2}


def parse_args():

    parser = argparse.ArgumentParser("Pretext Invariant Representation Learning.", add_help=True)

    g1 = parser.add_argument_group('General')
    g1.add_argument('--data', type=str, choices=('wm811k', ), required=True)
    g1.add_argument('--input_size', type=int, choices=(32, 64, 96, 112, 224), required=True)

    g2 = parser.add_argument_group('CNN Backbone')
    g2.add_argument('--backbone_type', type=str, default='resnet', choices=('alexnet', 'vggnet', 'resnet'), required=True)
    g2.add_argument('--backbone_config', type=str, default='18.original', required=True)

    g3 = parser.add_argument_group('PIRL')
    g3.add_argument('--projector_type', type=str, default='linear', choices=('linear', 'mlp'), required=True)
    g3.add_argument('--projector_size', type=int, default=128, help='Dimension of projection head.')
    g3.add_argument('--temperature', type=float, default=0.07, help='Logit scaling factor for contrastive learning.')
    g3.add_argument('--num_negatives', type=int, default=5000, help='Number of negative examples for contrastive learning.')
    g3.add_argument('--loss_weight', type=float, default=0.5, help='Weighting factor of loss function, [0, 1].')
    g3.add_argument('--augmentation', type=str, choices=('rotate+crop', 'rotate', 'crop', 'cutout', 'shift', 'noise'), required=True)

    g4 = parser.add_argument_group('Model Training')
    g4.add_argument('--epochs', type=int, default=150)
    g4.add_argument('--batch_size', type=int, default=512)
    g4.add_argument('--num_workers', type=int, default=0)
    g4.add_argument('--device', type=str, default='cuda:0', choices=('cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cpu'))

    g5 = parser.add_argument_group('Regularization')  # pylint: disable=unused-variable

    g6 = parser.add_argument_group('Optimizer')
    g6.add_argument('--optimizer', type=str, default='sgd', choices=('sgd', 'adamw', 'lars'))
    g6.add_argument('--learning_rate', type=float, default=0.01)
    g6.add_argument('--weight_decay', type=float, default=0.001)
    g6.add_argument('--momentum', type=float, default=0.9, help='only for SGD.')

    g7 = parser.add_argument_group('Scheduler')
    g7.add_argument('--scheduler', type=str, default='cosine', choices=('step', 'cosine', 'restart', 'none'))
    g7.add_argument('--milestone', type=int, default=None, help='For step decay.')
    g7.add_argument('--warmup_steps', type=int, default=5, help='For linear warmups.')
    g7.add_argument('--cycles', type=int, default=1, help='For hard restarts.')

    g8 = parser.add_argument_group('Logging')
    g8.add_argument('--checkpoint_root', type=str, default='./checkpoints/')
    g8.add_argument('--write_summary', action='store_true', help='write summaries with TensorBoard.')
    g8.add_argument('--save_every', type=int, default=None, help='save model checkpoint every `save_every` epochs.')

    g9 = parser.add_argument_group('Resuming training from a checkpoint')
    g9.add_argument('--resume_from_checkpoint', type=str, default=None)

    return parser.parse_args()


def main(args):
    """Main function."""

    # 1. Configurations
    torch.backends.cudnn.benchmark = True
    BACKBONE_CONFIGS, Config, Backbone = AVAILABLE_MODELS[args.backbone_type]
    Projector = PROJECTOR_TYPES[args.projector_type]

    config = Config(args)
    config.save()

    logfile = os.path.join(config.checkpoint_dir, 'main.log')
    logger = get_logger(stream=False, logfile=logfile)

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

    experiment.run(**run_kwargs)
    logger.handlers.clear()


if __name__ == '__main__':
    try:
        main(parse_args())
    except KeyboardInterrupt:
        sys.exit(0)
