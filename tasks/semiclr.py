# -*- coding: utf-8 -*-

import os
import json
import tqdm

import torch
import torch.nn as nn

from models.base import BackboneBase, HeadBase
from tasks.simclr import SimCLR
from datasets.loaders import get_dataloader
from utils.loss import SimCLRLoss
from utils.logging import get_tqdm_config


class SemiCLR(SimCLR):
    def __init__(self,
                 backbone: BackboneBase,
                 projector: HeadBase,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 loss_function: SimCLRLoss,
                 augment_function: nn.Module,
                 metrics: dict,
                 checkpoint_dir: str,
                 write_summary: bool
                 ):
        super(SemiCLR, self).__init__(
            backbone=backbone,
            projector=projector,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_function=loss_function,
            augment_function=augment_function,
            metrics=metrics,
            checkpoint_dir=checkpoint_dir,
            write_summary=write_summary
        )

    def run(self,
            unlabeled_train_set, unlabeled_valid_set,
            labeled_train_set, labeled_valid_set,
            epochs: int,
            unlabeled_batch_size: int, labeled_batch_size: int,
            num_workers: int = 1, device: str = 'cuda',
            **kwargs):

        assert isinstance(unlabeled_train_set, torch.utils.data.Dataset)
        assert isinstance(unlabeled_valid_set, torch.utils.data.Dataset)
        assert isinstance(labeled_train_set, torch.utils.data.Dataset)
        assert isinstance(labeled_valid_set, torch.utils.data.Dataset)
        assert isinstance(epochs, int)
        assert isinstance(unlabeled_batch_size, int)
        assert isinstance(labeled_batch_size, int)
        assert isinstance(num_workers, int)
        assert device.startswith('cuda') or device == 'cpu'

        logger = kwargs.get('logger', None)

        self.backbone = self.backbone.to(device)
        self.projector = self.projector.to(device)

        unlabeled_train_loader = get_dataloader(
            unlabeled_train_set, batch_size=unlabeled_batch_size,
            shuffle=True, num_workers=num_workers
        )
        unlabeled_valid_loader = get_dataloader(
            unlabeled_valid_set, batch_size=unlabeled_batch_size,
            shuffle=True, num_workers=num_workers
        )
        labeled_train_loader   = get_dataloader(
            labeled_train_set, batch_size=labeled_batch_size,
            shuffle=True, num_workers=num_workers
        )
        labeled_valid_loader   = get_dataloader(
            labeled_valid_set, batch_size=labeled_batch_size,
            shuffle=True, num_workers=num_workers
        )

        with tqdm.tqdm(**get_tqdm_config(total=epochs, leave=True, color='blue')) as pbar:

            best_valid_loss = float('inf')
            best_epoch = 0

            for epoch in range(1, epochs + 1):

                # 0. Train & evaluate
                train_history = self.train(
                    unlabeled_train_loader, labeled_train_loader,
                    device=device
                )
                valid_history = self.evaluate(
                    unlabeled_valid_loader, labeled_valid_loader,
                    device=device
                )

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
                            global_step=epoch,
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
                desc = f" Epoch [{epoch:>04}/{epochs:>04}] ({best_epoch:04}) |"
                for metric_name, metric_dict in epoch_history.items():
                    for k, v in metric_dict.items():
                        desc += f" {k}_{metric_name}: {v:.4f} |"
                pbar.set_description_str(desc)
                pbar.update(1)
                if logger is not None:
                    logger.info(desc)

        # 7. Save last model
        self.save_checkpoint(self.last_ckpt, epoch=epoch, **epoch_history)

        # 8. Evaluate best model on test set (optional if `test_set` is given)
        if all([dset in kwargs for dset in ['unlabeled_test_set', 'labeled_test_set']]):
            unlabeled_test_loader = get_dataloader(
                kwargs.get('unlabeled_test_set'), batch_size=unlabeled_batch_size,
                shuffle=True, num_workers=num_workers
            )
            labeled_test_loader = get_dataloader(
                kwargs.get('labeled_test_set'), batch_size=labeled_batch_size,
                shuffle=True, num_workers=num_workers
            )
            self.test(unlabeled_test_loader, labeled_test_loader, device=device, logger=logger)

    def train(self,
              unlabeled_data_loader: torch.utils.data.DataLoader,
              labeled_data_loader: torch.utils.data.DataLoader,
              device: str, **kwargs):

        train_loss = 0.
        steps_per_epoch = len(unlabeled_data_loader)
        self._set_learning_phase(train=True)

        labeled_iter = iter(labeled_data_loader)

        with tqdm.tqdm(**get_tqdm_config(steps_per_epoch, leave=False, color='cyan')) as pbar:
            for i, batch_u in enumerate(unlabeled_data_loader):

                x1_u, m1_u = batch_u['x1'].to(device), batch_u['m1'].to(device)
                x2_u, m2_u = batch_u['x2'].to(device), batch_u['m2'].to(device)

                try:
                    batch_l = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(labeled_data_loader)
                    batch_l = next(labeled_iter)

                x1_l, m1_l = batch_l['x1'].to(device), batch_l['m1'].to(device)
                x2_l, m2_l = batch_l['x2'].to(device), batch_l['m2'].to(device)

                x1, m1 = torch.cat([x1_u, x1_l], dim=0), torch.cat([m1_u, m1_l], dim=0)
                x2, m2 = torch.cat([x2_u, x2_l], dim=0), torch.cat([m2_u, m2_l], dim=0)

                if self.augment_function is not None:
                    x1 = self.augment_function(x1, m1)
                    x2 = self.augment_function(x2, m2)

                self.optimizer.zero_grad()
                z1 = self.predict(x1, m1, train=True)
                z2 = self.predict(x2, m2, train=True)

                mask = SimCLRLoss.semisupervised_mask(
                    unlabeled_size=x1_u.size(0),  # unlabeled batch size
                    labels=batch_l.get('y')       # labels
                )

                loss = self.loss_function(features=torch.stack([z1, z2], dim=1), mask=mask)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                desc = f" Batch [{i+1:>04}/{steps_per_epoch:>04}]"
                desc += f" Loss {train_loss/(i+1):.4f} "
                pbar.set_description_str(desc)
                pbar.update(1)

            out = {
                'loss': train_loss / steps_per_epoch
            }
            if self.metrics is not None:
                raise NotImplementedError

            return out

    def evaluate(self,
                 unlabeled_data_loader: torch.utils.data.DataLoader,
                 labeled_data_loader: torch.utils.data.DataLoader,
                 device: str, **kwargs):

        valid_loss = 0.
        steps_per_epoch = len(unlabeled_data_loader)
        self._set_learning_phase(train=False)

        with torch.no_grad():

            labeled_iter = iter(labeled_data_loader)
            for _, batch_u in enumerate(unlabeled_data_loader):

                x1_u, m1_u = batch_u['x1'].to(device), batch_u['m1'].to(device)
                x2_u, m2_u = batch_u['x2'].to(device), batch_u['m2'].to(device)

                try:
                    batch_l = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(labeled_data_loader)
                    batch_l = next(labeled_iter)

                x1_l, m1_l = batch_l['x1'].to(device), batch_l['m1'].to(device)
                x2_l, m2_l = batch_l['x2'].to(device), batch_l['m2'].to(device)

                x1, m1 = torch.cat([x1_u, x1_l], dim=0), torch.cat([m1_u, m1_l], dim=0)
                x2, m2 = torch.cat([x2_u, x2_l], dim=0), torch.cat([m2_u, m2_l], dim=0)

                if self.augment_function is not None:
                    x1 = self.augment_function(x1, m1)
                    x2 = self.augment_function(x2, m2)

                z1 = self.predict(x1, m1, train=False)
                z2 = self.predict(x2, m2, train=False)

                mask = SimCLRLoss.semisupervised_mask(
                    unlabeled_size=x1_u.size(0),
                    labels=batch_l.get('y')
                )

                loss = self.loss_function(features=torch.stack([z1, z2], dim=1), mask=mask)
                valid_loss += loss.item()

            out = {'loss': valid_loss / steps_per_epoch}
            if self.metrics is not None:
                raise NotImplementedError

            return out

    def test(self,
             unlabeled_data_loader: torch.utils.data.DataLoader,
             labeled_data_loader: torch.utils.data.DataLoader,
             device: str, logger=None):
        """Evaluate best model on test data."""

        def test_on_ckpt(ckpt: str):
            """Load checkpoint history and add test metric values."""
            self.load_model_from_checkpoint(ckpt)
            ckpt_history = self.load_history_from_checkpoint(ckpt)
            test_history = self.evaluate(unlabeled_data_loader, labeled_data_loader, device)
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

        with open(os.path.join(self.checkpoint_dir, 'best_history.json'), 'w') as fp:
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

        with open(os.path.join(self.checkpoint_dir, 'last_history.json'), 'w') as fp:
            json.dump(best_history, fp, indent=2)
