# -*- coding: utf-8 -*-

import os
import json
import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

from models.base import BackboneBase, HeadBase
from tasks.base import Task
from datasets.loaders import get_dataloader
from utils.loss import SimCLRLoss
from utils.logging import get_tqdm_config
from utils.logging import make_epoch_description


class SimCLR(Task):
    def __init__(self,
                 backbone: BackboneBase,
                 projector: HeadBase,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 loss_function: SimCLRLoss,
                 metrics: dict,
                 checkpoint_dir: str,
                 write_summary: bool
                 ):
        super(SimCLR, self).__init__()

        self.backbone = backbone
        self.projector = projector
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.metrics = metrics if isinstance(metrics, dict) else None

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.checkpoint_dir) if write_summary else None

    def run(self, train_set, valid_set, epochs: int, batch_size: int, num_workers: int = 0, device: str = 'cuda', **kwargs):  # pylint: disable=unused-argument

        assert isinstance(train_set, torch.utils.data.Dataset)
        assert isinstance(valid_set, torch.utils.data.Dataset)
        assert isinstance(epochs, int)
        assert isinstance(batch_size, int)
        assert isinstance(num_workers, int)
        assert device.startswith('cuda') or device == 'cpu'

        logger = kwargs.get('logger', None)

        self.backbone = self.backbone.to(device)
        self.projector = self.projector.to(device)

        train_loader = get_dataloader(train_set, batch_size, num_workers=num_workers)
        valid_loader = get_dataloader(valid_set, batch_size, num_workers=num_workers)

        with tqdm.tqdm(**get_tqdm_config(total=epochs, leave=True, color='blue')) as pbar:

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
                    raise NotImplementedError

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

                # 4. Save model if it is the current best
                valid_loss = epoch_history['loss']['valid']
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_epoch = epoch
                    self.save_checkpoint(self.best_ckpt, epoch=epoch, **epoch_history)
                    if kwargs.get('save_every', False):
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

        # 8. Test model (optional)
        if 'test_set' in kwargs.keys():
            test_loader = get_dataloader(kwargs.get('test_set'), batch_size=batch_size, num_workers=num_workers)
            self.test(test_loader, device=device, logger=logger)

    def train(self, data_loader: torch.utils.data.DataLoader, device: str, **kwargs):  # pylint: disable=unused-argument
        """Train function defined for a single epoch."""

        train_loss = 0.
        steps_per_epoch = len(data_loader)
        self._set_learning_phase(train=True)

        with tqdm.tqdm(**get_tqdm_config(steps_per_epoch, leave=False, color='red')) as pbar:
            for i, batch in enumerate(data_loader):

                self.optimizer.zero_grad()
                x1, x2 = batch['x1'].to(device), batch['x2'].to(device)
                z1, z2 = self.predict(x1), self.predict(x2)
                loss = self.loss_function(features=torch.stack([z1, z2], dim=1))
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                desc = f" Batch [{i+1:>4}/{steps_per_epoch:>4}]"
                desc += f" Loss: {train_loss/(i+1):.4f} "
                pbar.set_description_str(desc)
                pbar.update(1)

            out = {
                'loss': train_loss / steps_per_epoch,
            }
            if self.metrics is not None:
                raise NotImplementedError

            return out

    def evaluate(self, data_loader: torch.utils.data.DataLoader, device: str, **kwargs):  # pylint: disable=unused-argument
        """Evaluate current model. Running a single pass through the given dataset."""

        valid_loss = 0.
        steps_per_epoch = len(data_loader)
        self._set_learning_phase(train=False)

        with torch.no_grad():
            for _, batch in enumerate(data_loader):

                x1, x2 = batch['x1'].to(device), batch['x2'].to(device)
                z1, z2 = self.predict(x1), self.predict(x2)
                loss = self.loss_function(features=torch.stack([z1, z2], dim=1))
                valid_loss += loss.item()

            out = {'loss': valid_loss / steps_per_epoch}
            if self.metrics is not None:
                raise NotImplementedError

            return out

    def predict(self, x: torch.Tensor):
        return self.projector(self.backbone(x))

    def test(self, data_loader: torch.utils.data.DataLoader, device: str, logger = None):
        """Evaluate best model on test data."""

        def test_on_ckpt(ckpt: str):
            """Load checkpoint history and add test metric values."""
            self.load_model_from_checkpoint(ckpt)
            ckpt_history = self.load_history_from_checkpoint(ckpt)
            test_history = self.evaluate(data_loader, device)
            for metric_name, metric_val in test_history.items():
                ckpt_history[metric_name]['test'] = metric_val
            return ckpt_history

        def make_description(history: dict, prefix: str = ''):
            desc = f" {prefix} ({history['epoch']:>4d}): "
            for metric_name, metric_dict in history.items():
                if metric_name == 'epoch':
                    continue
                for k, v in metric_dict.items():
                    desc += f" {k}_{metric_name}: {v:.4f} |"
            return desc

        # 1. Best model
        best_history = test_on_ckpt(self.best_ckpt)
        desc = make_description(best_history, prefix='Best model')
        print(desc)
        if logger is not None:
            logger.info(desc)

        with open(os.path.join(self.checkpoint_dir, 'best_history.json'), 'w') as fp:
            json.dump(best_history, fp, indent=2)

        # 2. Last model
        last_history = test_on_ckpt(self.last_ckpt)
        desc = make_description(last_history, prefix='Last model')
        print(desc)
        if logger is not None:
            logger.info(desc)

        with open(os.path.join(self.checkpoint_dir, 'last_history.json'), 'w') as fp:
            json.dump(best_history, fp, indent=2)

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
