# -*- coding: utf-8 -*-

import os
import json
import tqdm

import torch
import torch.nn as nn

from torch.distributions.beta import Beta
from tasks.classification import Classification
from datasets.loaders import get_dataloader
from utils.logging import get_tqdm_config
from utils.logging import make_epoch_description


class Mixup(Classification):
    def __init__(self,  **kwargs):
        super(Mixup, self).__init__(**kwargs)

    def run(self, train_set, valid_set, epochs: int, batch_size: int, num_workers: int = 0, device: str = 'cuda', **kwargs):
        """Train, evaluate and optionally test."""

        assert isinstance(train_set, torch.utils.data.Dataset)
        assert isinstance(valid_set, torch.utils.data.Dataset)
        assert isinstance(epochs, int)
        assert isinstance(batch_size, int)
        assert isinstance(num_workers, int)
        assert device.startswith('cuda') or device == 'cpu'

        logger = kwargs.get('logger', None)
        disable_mixup = kwargs.get('disable_mixup', False)

        self.backbone = self.backbone.to(device)
        self.classifier = self.classifier.to(device)
        
        balance = kwargs.get('balance', False)
        if logger is not None:
            logger.info(f"Class balance: {balance}")
        shuffle = not balance

        train_loader = get_dataloader(train_set, batch_size, num_workers=num_workers, shuffle=shuffle, balance=balance)
        valid_loader = get_dataloader(valid_set, batch_size, num_workers=num_workers, balance=False)

        with tqdm.tqdm(**get_tqdm_config(total=epochs, leave=True, color='blue')) as pbar:

            # Determine model selection metric. Defaults to 'loss'.
            eval_metric = kwargs.get('eval_metric', 'loss')
            if eval_metric == 'loss':
                best_metric_val = float('inf')
            elif eval_metric in ['accuracy', 'precision', 'recall', 'f1', 'auroc', 'auprc']:
                best_metric_val = 0
            else:
                raise NotImplementedError

            best_epoch = 0
            for epoch in range(1, epochs + 1):

                # 0. Train & evaluate
                if disable_mixup:
                    train_history = self.train(train_loader, device)
                else:
                    train_history = self.train_with_mixup(train_loader, device)
                valid_history = self.evaluate(valid_loader, device)

                # 1. Epoch history (loss)
                epoch_history = {
                    'loss': {
                        'train': train_history.get('loss'),
                        'valid': valid_history.get('loss')
                    }
                }

                # 2. Epoch history (other metrics if provided)
                if isinstance(self.metrics, dict):
                    for metric_name, _ in self.metrics.items():
                        epoch_history[metric_name] = {
                            'train': train_history[metric_name],
                            'valid': valid_history[metric_name],
                        }

                 # 3. Tensorboard
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
                metric_val = epoch_history[eval_metric]['valid']
                if eval_metric == 'loss':
                    if metric_val <= best_metric_val:
                        best_metric_val = metric_val
                        best_epoch = epoch
                        self.save_checkpoint(self.best_ckpt, epoch=epoch, **epoch_history)
                elif eval_metric in ['accuracy', 'f1', 'auroc', 'auprc']:
                    if metric_val >= best_metric_val:
                        best_metric_val = metric_val
                        best_epoch = epoch
                        self.save_checkpoint(self.best_ckpt, epoch=epoch, **epoch_history)
                else:
                    raise NotImplementedError

                # 5. Update learning rate scheduler (optional)
                if self.scheduler is not None:
                    self.scheduler.step()

                # 6. Logging
                desc = make_epoch_description(
                    history=epoch_history,
                    current=epoch,
                    total=epochs,
                    best=best_epoch,
                )
                pbar.set_description_str(desc)
                pbar.update(1)
                if logger is not None:
                    logger.info(desc)

        # 7. Save last model
        self.save_checkpoint(self.last_ckpt, epoch=epoch, **epoch_history)

        # 8. Test model (optional)
        if 'test_set' in kwargs.keys():
            test_loader = get_dataloader(kwargs.get('test_set'), batch_size, num_workers=num_workers)
            self.test(test_loader, device=device, logger=logger)

    def train_with_mixup(self, data_loader: torch.utils.data.DataLoader, device: str):
            
        train_loss = 0.
        trues_a, trues_b, preds = [], [], []
        self._set_learning_phase(train=True)

        for _, batch in enumerate(data_loader):

            x = batch['x'].to(device)
            y = batch['y'].to(device)

            x, y_a, y_b, lam = self.mixup_data(x=x, y=y, alpha=1.0)
            y_pred = self.predict(x)
            loss = self.mixup_loss(
                loss_function=self.loss_function,
                pred=y_pred,
                y_a=y_a,
                y_b=y_b,
                lam=lam
            )
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            train_loss += loss.item()
            trues_a += [y_a.cpu().detach()]
            trues_b += [y_b.cpu().detach()]
            preds += [y_pred.cpu().detach()]

        out = {'loss': train_loss / len(data_loader)}
        if self.metrics is not None:
            assert isinstance(self.metrics, dict)
            trues_a = torch.cat(trues_a, dim=0)
            trues_b = torch.cat(trues_b, dim=0)
            preds = torch.cat(preds, dim=0)
            for metric_name, metric_function in self.metrics.items():
                out[metric_name] = 0.5 * (
                    metric_function(preds, trues_a).item() + \
                    metric_function(preds, trues_b).item()
                )
        
        return out

    @staticmethod
    def mixup_data(x: torch.FloatTensor,
                   y: torch.LongTensor,
                   alpha: float = 1.0):
        
        if not len(x) == len(y):
            raise ValueError("The size of `x` and `y` must match in the first dim.")
        
        if alpha > 0.:
            alpha = float(alpha)
            beta_dist = Beta(torch.tensor([alpha]), torch.tensor([alpha]))
            lam = beta_dist.sample().item()
        else:
            lam = 1.

        batch_size, num_channels, _, _ = x.size()
        index = torch.randperm(batch_size).to(x.device)
        
        # For WM811K, the input tensors `x` have two channels, where
        # the first channel has values of either one (for fail) or zero (for pass),
        # while the second channel has values of either one (for valid bins) or zeros (null bins).
        if num_channels == 2:
            mixed_x0 = \
                lam * x[:, 0, :, :] + (1 - lam) * x[index, 0, :, :]  # (B, H, W)
            mixed_x1 = (x[:, 1, :, :] + x[index, 1, :, :])           # (B, H, W)
            mixed_x1 = torch.clamp(mixed_x1, min=0, max=1)           # (B, H, W)
            mixed_x = torch.stack([mixed_x0, mixed_x1], dim=1)       # (B, 2, H, W)
        else:
            raise NotImplementedError
        
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    @staticmethod
    def mixup_loss(loss_function: nn.Module,
                   pred: torch.FloatTensor,
                   y_a: torch.LongTensor,
                   y_b: torch.LongTensor,
                   lam: float = 1.0):
        return lam * loss_function(pred, y_a) + (1 - lam) * loss_function(pred, y_b)
