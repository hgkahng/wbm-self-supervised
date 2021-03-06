# -*- coding: utf-8 -*-

import os
import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from tasks.base import Task
from models.base import BackboneBase, HeadBase
from datasets.loaders import get_dataloader
from utils.loss import SimCLRLoss
from utils.logging import get_tqdm_config
from utils.logging import make_epoch_description


class AttnCLR(Task):
    def __init__(self,
                 backbone: BackboneBase,
                 projector: HeadBase,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 loss_function: nn.Module,
                 metrics: dict,
                 checkpoint_dir: str,
                 write_summary: bool,
                 **kwargs):
        super(AttnCLR, self).__init__()

        self.backbone = backbone
        self.projector = projector

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.metrics = metrics if isinstance(metrics, dict) else None

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.checkpoint_dir) if write_summary else None

    def run(self,
            train_set,
            valid_set,
            epochs: int,
            batch_size: int,
            num_workers: int = 0,
            device: str = 'cuda',
            **kwargs):

        assert isinstance(train_set, torch.utils.data.Dataset)
        assert isinstance(valid_set, torch.utils.data.Dataset)
        assert isinstance(epochs, int)
        assert isinstance(batch_size, int)
        assert isinstance(num_workers, int)
        assert device.startswith('cuda') or device == 'cpu'

        logger = kwargs.get('logger', None)

        self.backbone.to(device)
        self.projector.to(device)

        train_loader = get_dataloader(train_set, batch_size, num_workers=num_workers)
        valid_loader = get_dataloader(valid_set, batch_size, num_workers=num_workers)

        with tqdm.tqdm(**get_tqdm_config(total=epochs, leave=True, color='green')) as pbar:

            best_valid_loss = float('inf')
            best_epoch = 0

            for epoch in range(1, epochs + 1):

                # 0. Train & evaluate
                train_history = self.train(train_loader, device=device)
                valid_history = self.evaluate(valid_loader, device=device)

                # 1. Epoch history (loss)
                epoch_history = {
                    'loss': {
                        'train': train_history.get('loss'),
                        'valid': valid_history.get('loss'),
                    }
                }

                # 2. Epoch history (other metrics if provided)
                if self.metrics is not None:
                    assert isinstance(self.metrics, dict)
                    for metric_name, _ in self.metrics.items():
                        epoch_history[metric_name] = {
                            'train': train_history.get(metric_name),
                            'valid': valid_history.get(metric_name)
                        }

                # 3. TensorBoard
                if self.writer is not None:
                    for metric_name, metric_dict in epoch_history.items():
                        self.writer.add_scalars(
                            main_tag=metric_name,
                            tag_scalar_dict=metric_dict,
                            global_step=epoch
                        )
                        if self.scheduler is not None:
                            self.writer.add_scalar(
                                tag='lr',
                                scalar_value=self.scheduler.get_last_lr()[0],
                                global_step=epoch
                            )

                # 4-1. Save model if it is the current best
                valid_loss = epoch_history['loss']['valid']
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_epoch = epoch
                    self.save_checkpoint(self.best_ckpt, epoch=epoch, **epoch_history)

                # 4-2. Save intermediate models
                if isinstance(kwargs.get('save_every'), int):
                    if epoch % kwargs.get('save_every') == 0:
                        new_ckpt = os.path.join(self.checkpoint_dir, f'epoch_{epoch:04d}.loss_{valid_loss:.4f}.pt')
                        self.save_checkpoint(new_ckpt, epoch=epoch, **epoch_history)

                # 5. Update learning rate scheduler
                if self.scheduler is not None:
                    self.scheduler.step()

               # 6. Logging
                desc = make_epoch_description(
                    history=epoch_history,
                    current=epoch,
                    total=epochs,
                    best=best_epoch
                )
                pbar.set_description_str(desc)
                pbar.update(1)
                if logger is not None:
                    logger.info(desc)

        # 7. Save last model
        self.save_checkpoint(self.last_ckpt, epoch=epoch, **epoch_history)

        # 8. Evaluate best model on test set (optional if `test_set` is given)
        if 'test_set' in kwargs.keys():
            test_loader = get_dataloader(kwargs.get('test_set'), batch_size=batch_size, num_workers=num_workers)
            self.test(test_loader, device=device, logger=logger)

    def train(self, data_loader: torch.utils.data.DataLoader, device: str, **kwargs):  # pylint: disable=unused-argument
        """Train function defined for a single epoch."""

        out = {'loss': 0.}
        steps_per_epoch = len(data_loader)
        self._set_learning_phase(train=True)

        with tqdm.tqdm(**get_tqdm_config(steps_per_epoch, leave=False, color='red')) as pbar:
            for i, batch in enumerate(data_loader):

                x1, x2 = batch['x1'].to(device), batch['x2'].to(device)
                z1, z1_depth, z1_spatial = self.predict(x1)
                z2, z2_depth, z2_spatial = self.predict(x2)
                loss_z, logits, mask = self.loss_function(features=torch.stack([z1, z2], dim=1))
                loss_depth, _, _ = self.loss_function(features=torch.stack([z1_depth, z2_depth], dim=1))
                loss_spatial, _, _ = self.loss_function(features=torch.stack([z1_spatial, z2_spatial], dim=1))
                loss = loss_z + loss_depth + loss_spatial
                loss = 1/3 * loss

                # Backpropagation & update
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Accumulate loss & metrics
                out['loss'] += loss.item()
                if self.metrics is not None:
                    assert isinstance(self.metrics, dict)
                    for metric_name, metric_function in self.metrics.items():
                        if metric_name not in out.keys():
                            out[metric_name] = 0.
                        with torch.no_grad():
                            logits = logits.detach()
                            targets = mask.detach().eq(1).nonzero(as_tuple=True)[1]
                            out[metric_name] += metric_function(logits, targets)

                desc = f" Batch: [{i+1:>4}/{steps_per_epoch:>4}]: "
                desc += " | ".join ([f"{k}: {v/(i+1):.4f}" for k, v in out.items()])
                pbar.set_description_str(desc)
                pbar.update(1)

            return {k: v / steps_per_epoch for k, v in out.items()}

    def evaluate(self, data_loader: torch.utils.data.DataLoader, device: str, **kwargs):  # pylint: disable=unused-argument
        """Evaluate current model. Running a single pass through the given dataset."""

        out = {'loss': 0.}
        steps_per_epoch = len(data_loader)
        self._set_learning_phase(train=False)

        with torch.no_grad():
            for _, batch in enumerate(data_loader):

                x1, x2 = batch['x1'].to(device), batch['x2'].to(device)
                z1, z1_depth, z1_spatial = self.predict(x1)
                z2, z2_depth, z2_spatial = self.predict(x2)
                loss_z, logits, mask = self.loss_function(features=torch.stack([z1, z2], dim=1))
                loss_depth, _, _ = self.loss_function(features=torch.stack([z1_depth, z2_depth], dim=1))
                loss_spatial, _, _ = self.loss_function(features=torch.stack([z1_spatial, z2_spatial], dim=1))
                loss = loss_z + loss_depth + loss_spatial
                loss = 1/3 * loss

                # Accumulate loss & metrics
                out['loss'] += loss.item()
                if self.metrics is not None:
                    assert isinstance(self.metrics, dict)
                    for metric_name, metric_function in self.metrics.items():
                        if metric_name not in out.keys():
                            out[metric_name] = 0.
                        logits = logits.detach()
                        targets = mask.detach().eq(1).nonzero(as_tuple=True)[1]
                        out[metric_name] += metric_function(logits, targets)

            return {k: v / steps_per_epoch for k, v in out.items()}

    def predict(self, x: torch.Tensor):
        z, z_depth, z_spatial = self.projector(self.backbone(x))
        return z, z_depth, z_spatial

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

    def load_model_from_checkpoint(self, path: str):
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
