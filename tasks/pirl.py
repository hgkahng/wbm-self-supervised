# -*- coding: utf-8 -*-

import os
import json
import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical

from models.base import BackboneBase, HeadBase
from tasks.base import Task
from datasets.wafer import get_dataloader
from utils.loss import NCELoss
from utils.logging import get_tqdm_config


class PIRL(Task):
    def __init__(self,
                 backbone: BackboneBase,
                 projector: HeadBase,
                 memory: object,
                 noise_function: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 loss_function: nn.Module,
                 loss_weight: float, num_negatives: int,
                 metrics: dict, checkpoint_dir: str, write_summary: bool):
        super(PIRL, self).__init__()

        assert isinstance(memory, MemoryBank)
        assert isinstance(loss_function, NCELoss)

        self.backbone = backbone
        self.projector = projector
        self.memory = memory
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.loss_weight = loss_weight
        self.num_negatives = num_negatives
        self.metrics = metrics if isinstance(metrics, dict) else None
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.checkpoint_dir) if write_summary else None

        self.noise_function = noise_function

    def run(self, train_set, valid_set, epochs, batch_size, num_workers=1, device='cuda', **kwargs):

        assert isinstance(train_set, torch.utils.data.Dataset)
        assert isinstance(valid_set, torch.utils.data.Dataset)
        assert isinstance(epochs, int)
        assert isinstance(batch_size, int)
        assert isinstance(num_workers, int)
        assert device.startswith('cuda') or device == 'cpu'

        logger = kwargs.get('logger', None)

        self.backbone = self.backbone.to(device)
        self.projector = self.projector.to(device)

        train_loader = get_dataloader(train_set, batch_size, shuffle=True, num_workers=num_workers)
        valid_loader = get_dataloader(valid_set, batch_size, shuffle=False, num_workers=num_workers // 2)

        # Initialize training memory
        if not self.memory.initialized:
            self.memory.initialize(self.backbone, self.projector, train_loader)

        with tqdm.tqdm(**get_tqdm_config(total=epochs, leave=True, color='blue')) as pbar:

            best_valid_loss = float('inf')
            best_epoch = 0

            for epoch in range(1, epochs + 1):

                # 0. Train & evaluate
                train_history = self.train(train_loader, device)
                valid_history = self.evaluate(valid_loader, device)

                # 1. Epoch history (loss)
                epoch_history = {
                    'loss': {
                        'train': train_history.get('loss'),
                        'valid': valid_history.get('loss')
                    },
                    'original': {
                        'train': train_history.get('original'),
                        'valid': valid_history.get('original'),
                    },
                    'transformed': {
                        'train': train_history.get('transformed'),
                        'valid': valid_history.get('transformed')
                    }
                }

                # 2. Epoch history (other metrics if provided)
                if self.metrics is not None:
                    raise NotImplementedError

                # 3. Tensorboard
                if self.writer is not None:
                    for metric_name, metric_dict in epoch_history.items():
                        self.writer.add_scalars(
                            main_tag=metric_name,
                            tag_scalar_dict=metric_dict,
                            global_step=epoch
                        )

                # 4-1. Save model if it is the current best
                valid_loss = epoch_history['loss']['valid']
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_epoch = epoch
                    self.save_checkpoint(self.best_ckpt, epoch=epoch, **epoch_history)
                    self.memory.save(os.path.join(os.path.dirname(self.best_ckpt), 'best_memory.pt'), epoch=epoch)
                    if kwargs.get('save_every', False):
                        new_ckpt = os.path.join(
                            self.checkpoint_dir,
                            f'epoch_{epoch:04d}.loss_{valid_loss:.4f}.pt'
                        )
                        self.save_checkpoint(new_ckpt, epoch=epoch, **epoch_history)

                # 4-2. Update scheduler
                if isinstance(self.scheduler, torch.optim.lr_scheduler.StepLR):
                    self.scheduler.step()
                elif isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(valid_loss)
                elif isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                    self.scheduler.step()
                elif isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    self.scheduler.step()

                # 5. Logging
                desc = f" Epoch [{epoch:>04}/{epochs:>04}] ({best_epoch:04}) |"
                for metric_name, metric_dict in epoch_history.items():
                    for k, v in metric_dict.items():
                        desc += f" {k}_{metric_name}: {v:.4f} |"
                pbar.set_description_str(desc)
                pbar.update(1)
                if logger is not None:
                    logger.info(desc)

        # 6. Save last model
        self.save_checkpoint(self.last_ckpt, epoch=epoch, **epoch_history)
        self.memory.save(os.path.join(os.path.dirname(self.last_ckpt), 'last_memory.pt'), epoch=epoch)

        # 7. Evaluate best model on test set (optional if `test_set` is given)
        if 'test_set' in kwargs.keys():
            test_loader = get_dataloader(kwargs.get('test_set'), batch_size, num_workers=num_workers // 2)
            self.test(test_loader, device=device, logger=logger)

    def train(self, data_loader: torch.utils.data.DataLoader, device: str, **kwargs):
        """Train function defined for a single epoch."""

        train_loss = 0.
        loss_T = 0.
        loss_O = 0.
        num_samples = len(data_loader.dataset)
        steps_per_epoch = len(data_loader)
        self._set_learning_phase(train=True)

        with tqdm.tqdm(**get_tqdm_config(steps_per_epoch, leave=False, color='green')) as pbar:
            for i, batch in enumerate(data_loader):

                x, m = batch['x'].to(device), batch['m'].to(device)
                x_t, m_t = batch['x_t'].to(device), batch['m_t'].to(device)
                j = batch['idx']

                if self.noise_function is not None:
                    x_t = self.noise_function(x_t, m_t)

                self.optimizer.zero_grad()
                original_features = self.predict(x, m)
                transformed_features = self.predict(x_t, m_t)

                representations = self.memory.get_representations(j).to(device)
                negatives = self.memory.get_negatives(self.num_negatives, exclude=j)

                transformed_loss = self.loss_function(
                    queries=representations,
                    positives=transformed_features,
                    negatives=negatives
                    )
                original_loss = self.loss_function(
                    queries=representations,
                    positives=original_features,
                    negatives=negatives
                    )
                loss = self.loss_weight * transformed_loss + \
                    (1 - self.loss_weight) * original_loss
                loss.backward()
                self.optimizer.step()

                # Update memory bank
                self.memory.update(j, original_features.detach())

                loss_T += transformed_loss.item() * x.size(0)
                loss_O += original_loss.item() * x.size(0)
                train_loss += loss.item() * x.size(0)

                desc = f" Batch [{i+1:>04}/{steps_per_epoch:>04}]"
                desc += f" Loss: {train_loss / (data_loader.batch_size * (i + 1)):.4f} "
                pbar.set_description_str(desc)
                pbar.update(1)

            # Update weighted count at the end of each epoch
            self.memory.update_weighted_count()

        out = {
            'loss': train_loss / num_samples,
            'transformed': loss_T / num_samples,
            'original': loss_O / num_samples,
            }
        if self.metrics is not None:
            raise NotImplementedError

        return out

    def evaluate(self, data_loader: torch.utils.data.DataLoader, device: str, **kwargs):
        """Evaluate current model. A single pass through the given dataset."""

        valid_loss = 0.
        loss_T = 0.
        loss_O = 0.
        num_samples = len(data_loader.dataset)
        self._set_learning_phase(train=False)

        with torch.no_grad():
            for _, batch in enumerate(data_loader):

                x, m = batch['x'].to(device), batch['m'].to(device)
                x_t, m_t = batch['x_t'].to(device), batch['m_t'].to(device)
                j = batch['idx']

                if self.noise_function is not None:
                    x_t = self.noise_function(x_t, m_t)

                original_features = self.predict(x, m)
                transformed_features = self.predict(x_t, m_t)

                negatives = self.memory.get_negatives(self.num_negatives, exclude=j)

                transformed_loss = self.loss_function(
                    queries=original_features,
                    positives=transformed_features,
                    negatives=negatives
                    )
                original_loss = self.loss_function(
                    queries=original_features,
                    positives=original_features,
                    negatives=negatives
                    )

                loss = self.loss_weight * transformed_loss \
                    + (1 - self.loss_weight) * original_loss
                valid_loss += loss.item() * x.size(0)
                loss_T += transformed_loss.item() * x.size(0)
                loss_O += original_loss.item() * x.size(0)

            out = {
                'loss': valid_loss / num_samples,
                'transformed': loss_T / num_samples,
                'original': loss_O / num_samples,
                }
            if self.metrics is not None:
                raise NotImplementedError

            return out

    def predict(self, x: torch.Tensor, m: torch.Tensor = None, train: bool = False):
        self._set_learning_phase(train)
        if self.backbone.in_channels == 2:
            if m is None:
                raise ValueError("Backbone requires a mask tensor `m`.")
            x = torch.cat([x, m], dim=1)
        return self.projector(self.backbone(x))

    def test(self, data_loader: torch.utils.data.DataLoader, device: str, logger=None):
        """Evaluate best model on test data."""

        def test_on_ckpt(ckpt: str):
            """Load checkpoint history and add test metric values."""
            
            self.load_model_from_checkpoint(ckpt)
            self.memory.load(ckpt.replace('_model.pt', '_memory.pt'))
            ckpt_history = self.load_history_from_checkpoint(ckpt)

            test_history = self.evaluate(data_loader, device)
            for metric_name, metric_val in test_history.items():
                ckpt_history[metric_name]['test'] = metric_val

            return ckpt_history

        # 1. Best model (based on validation loss)
        best_history = test_on_ckpt(self.best_ckpt)
        desc = f" Best model ({best_history.get('epoch', -1):04d}): "
        for metric_name, metric_dict in best_history.items():
            if metric_name == 'epoch':
                continue
            for k, v in metric_dict.items():
                desc += f" {k}_{metric_name}: {v:.4f} |"

        print(desc)
        if logger is not None:
            logger.info(desc)

        with open(os.path.join(self.checkpoint_dir, "best_history.json"), 'w') as fp:
            json.dump(best_history, fp, indent=2)

        # 2. Last model
        last_history = test_on_ckpt(self.last_ckpt)
        desc = f" Last model ({last_history.get('epoch', -1):04d}): "
        for metric_name, metric_dict in last_history.items():
            if metric_name == 'epoch':
                continue
            for k, v in metric_dict.items():
                desc += f" {k}_{metric_name}: {v:.4f} |"

        print(desc)
        if logger is not None:
            logger.info(desc)

        with open(os.path.join(self.checkpoint_dir, "last_history.json"), 'w') as fp:
            json.dump(last_history, fp, indent=2)

    def _set_learning_phase(self, train=False):
        if train:
            self.backbone.train()
            self.projector.train()
        else:
            self.backbone.eval()
            self.projector.eval()

    def save_checkpoint(self, path: str, **kwargs):
        ckpt = {
            'backbone': self.backbone.state_dict(),
            'projector': self.projector.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if \
                self.scheduler is not None else None
        }
        if kwargs:
            ckpt.update(kwargs)
        torch.save(ckpt, path)

    def load_model_from_checkpoint(self, path: str, **kwargs):
        ckpt = torch.load(path)
        self.backbone.load_state_dict(ckpt['backbone'])
        self.projector.load_state_dict(ckpt['projector'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(ckpt['scheduler'])

    def load_history_from_checkpoint(self, path: str):
        ckpt = torch.load(path)
        del ckpt['backbone']
        del ckpt['projector']
        del ckpt['optimizer']
        del ckpt['scheduler']
        return ckpt


class MemoryBank(object):
    def __init__(self, size: tuple, device: str):
        self.device = device
        self.weight = .5
        self.weighted_count = 0
        self.buffer = torch.zeros(*size).to(self.device)
        self.weighted_sum = torch.zeros_like(self.buffer)
        self.initialized = False

    def initialize(self,
                   backbone: nn.Module,
                   projector: nn.Module,
                   train_loader: torch.utils.data.DataLoader):
        self.update_weighted_count()

        with tqdm.tqdm(desc="Initializing", total=len(train_loader), dynamic_ncols=True) as pbar:

            with torch.no_grad():
                for _, batch in enumerate(train_loader):
                    x, m = batch['x'].to(self.device), batch['m'].to(self.device)
                    j = batch['idx']
                    if backbone.in_channels == 2:
                        x = torch.cat([x, m], dim=1)
                    self.weighted_sum[j, :] = projector(backbone(x)).detach()
                    self.buffer[j, :] = self.weighted_sum[j, :]
                    pbar.update(1)

        self.intialized = True

    def update(self, index: list, values: torch.Tensor):
        """Update memory with weighted moving average."""
        self.weighted_sum[index, :] = \
            values.to(self.device) + (1 - self.weight) * self.weighted_sum[index, :]
        self.buffer[index, :] = \
            self.weighted_sum[index, :] / self.weighted_count

    def update_weighted_count(self):
        self.weighted_count = 1 + (1 - self.weight) * self.weighted_count

    def get_representations(self, index: int or tuple or list):
        return self.buffer[index]

    def get_negatives(self, size: int, exclude: int or tuple or list):
        logits = torch.ones(self.buffer.size(0), device=self.device)
        logits[exclude] = 0
        sample_size = torch.Size([size])
        return self.buffer[Categorical(logits=logits).sample(sample_size), :]

    def save(self, path: str, **kwargs):
        ckpt = {
            'weight': self.weight,
            'weighted_count': self.weighted_count,
            'buffer': self.buffer.cpu(),
            'weighted_sum': self.weighted_sum.cpu(),
        }
        if kwargs:
            ckpt.update(kwargs)
        torch.save(ckpt, path)

    def load(self, path: str):
        ckpt = torch.load(path)
        self.weight = ckpt['weight']
        self.weighted_count = ckpt['weighted_count']
        self.buffer = ckpt['buffer'].to(self.device)
        self.weighted_sum = ckpt['weighted_sum'].to(self.device)
        self.initialized = True