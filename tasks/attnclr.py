# -*- coding: utf-8 -*-

import os
import tqdm

import torch
import torch.nn as nn

from models.base import BackboneBase, HeadBase
from tasks.simclr import SimCLR
from datasets.wafer import get_dataloader
from utils.loss import SimCLRLoss, AttnCLRLoss
from utils.logging import get_tqdm_config
from utils.logging import make_epoch_description


class AttnCLR(SimCLR):
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
        super(AttnCLR, self).__init__(
            backbone=backbone,
            projector=projector,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_function=loss_function,
            metrics=metrics,
            checkpoint_dir=checkpoint_dir,
            write_summary=write_summary
        )
        self.write_histogram = kwargs.get('write_histogram', False)

    def run(self, train_set, valid_set, epochs: int, batch_size: int, num_workers: int = 0, device: str = 'cuda', **kwargs):

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

        with tqdm.tqdm(**get_tqdm_config(total=epochs, leave=True, color='green')) as pbar:

            best_valid_loss = float('inf')
            best_epoch = 0

            for epoch in range(1, epochs + 1):

                # 0. Train & evaluate
                train_history = self.train(train_loader, device=device)
                valid_history = self.evaluate(valid_loader, device=device, current_epoch=epoch - 1)

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

        # 8. Evaluate best model on test set (optional if `test_set` is given)
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
                z, attn = self.predict(torch.cat([x1, x2], dim=0))
                z = z.view(x1.size(0), 2, -1)

                # Calculate attention-based contrastive loss
                loss, _ = self.loss_function(features=z, attention_scores=attn)
                loss.backward()

                # Clip the gradients (XXX: why is this necessary?)
                # nn.utils.clip_grad_norm_(self.backbone.parameters(), 1.)
                # nn.utils.clip_grad_norm_(self.projector.parameters(), 1.)

                # Update weights
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

    def evaluate(self, data_loader: torch.utils.data.DataLoader, device: str, current_epoch: int = None, **kwargs):  # pylint: disable=unused-argument
        """Evaluate current model. Running a single pass through the given dataset."""

        valid_loss = 0.
        steps_per_epoch = len(data_loader)
        self._set_learning_phase(train=False)

        with torch.no_grad():
            for i, batch in enumerate(data_loader):

                x1, x2 = batch['x1'].to(device), batch['x2'].to(device)
                z, attn = self.predict(torch.cat([x1, x2], dim=0))
                z = z.view(x1.size(0), 2, -1)

                loss, masked_attn_scores = self.loss_function(features=z, attention_scores=attn)
                valid_loss += loss.item()

                if self.write_histogram and (self.writer is not None):  # FIXME
                    assert current_epoch is not None, ""
                    self.writer.add_histogram(
                        tag='valid/masked_attention',
                        values=masked_attn_scores,
                        global_step=(current_epoch * steps_per_epoch) + i,
                    )

            out = {'loss': valid_loss / steps_per_epoch}
            if self.metrics is not None:
                raise NotImplementedError

            return out

    def predict(self, x: torch.Tensor):
        z, attention = self.projector(self.backbone(x))
        return z, attention

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
