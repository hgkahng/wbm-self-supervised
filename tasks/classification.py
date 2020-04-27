# -*- coding: utf-8 -*-

import os
import json
import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from tasks.base import Task
from datasets.wafer import get_dataloader
from utils.logging import get_tqdm_config


class Classification(Task):
    def __init__(self, backbone, classifier, optimizer, scheduler, loss_function,
                 metrics: dict, checkpoint_dir: str, write_summary: bool):
        super(Classification, self).__init__()

        self.backbone = backbone
        self.classifier = classifier
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.metrics = metrics if isinstance(metrics, dict) else None
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.checkpoint_dir) if write_summary else None

    def run(self, train_set, valid_set, epochs, batch_size, num_workers=0, device='cuda', **kwargs):
        """Train, evaluate and optionally test."""

        assert isinstance(train_set, torch.utils.data.Dataset)
        assert isinstance(valid_set, torch.utils.data.Dataset)
        assert isinstance(epochs, int)
        assert isinstance(batch_size, int)
        assert isinstance(num_workers, int)
        assert device.startswith('cuda') or device == 'cpu'

        logger = kwargs.get('logger', None)

        self.backbone = self.backbone.to(device)
        self.classifier = self.classifier.to(device)

        train_loader = get_dataloader(train_set, batch_size, shuffle=True, num_workers=num_workers)
        valid_loader = get_dataloader(valid_set, batch_size, shuffle=False, num_workers=num_workers)

        with tqdm.tqdm(**get_tqdm_config(total=epochs, leave=True, color='yellow')) as pbar:

            eval_metric = kwargs.get('eval_metric', 'loss')
            if eval_metric == 'loss':
                best_metric_val = float('inf')
            elif eval_metric in ['accuracy', 'f1']:
                best_metric_val = 0
            
            best_epoch = 0

            for epoch in range(1, epochs + 1):

                # 0. Train & evaluate
                train_history = self.train(train_loader, device, current_epoch=epoch)
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
                if isinstance(self.writer, SummaryWriter):
                    for metric_name, metric_dict in epoch_history.items():
                        self.writer.add_scalars(
                            main_tag=metric_name,
                            tag_scalar_dict=metric_dict,
                            global_step=epoch
                        )

                # 4-1. Save model if it is the current best
                metric_val = epoch_history[eval_metric]['valid']
                if eval_metric == 'loss':
                    if metric_val <= best_metric_val:
                        best_metric_val = metric_val
                        best_epoch = epoch
                        self.save_checkpoint(self.best_ckpt, epoch=epoch, **epoch_history)
                elif eval_metric in ['accuracy', 'f1']:
                    if metric_val >= best_metric_val:
                        best_metric_val = metric_val
                        best_epoch = epoch
                        self.save_checkpoint(self.best_ckpt, epoch=epoch, **epoch_history)
                else:
                    raise NotImplementedError

                # 4-2. Update learning rate scheduler (optional)
                if isinstance(self.scheduler, torch.optim.lr_scheduler.StepLR):
                    self.scheduler.step()
                if isinstance(self.scheduler, torch.optim.lr_scheduler.MultiStepLR):
                    self.scheduler.step()

                # 5. Logging
                desc = f" Epoch [{epoch:>04}/{epochs:>04}] ({best_epoch:>04}) |"
                for metric_name, metric_dict in epoch_history.items():
                    for k, v in metric_dict.items():
                        desc += f" {k}_{metric_name}: {v:.4f} |"
                pbar.set_description_str(desc)
                pbar.update(1)
                if logger is not None:
                    logger.info(desc)

        # 6. Save last model
        self.save_checkpoint(self.last_ckpt, epoch=epoch, **epoch_history)

        # 7. Evaluate best model on test set (optional if `test_set` exists)
        if 'test_set' in kwargs.keys():
            test_loader = get_dataloader(kwargs.get('test_set'), batch_size, num_workers=num_workers)
            self.test(test_loader, device=device, logger=logger)

    def train(self, data_loader: torch.utils.data.DataLoader, device: str, current_epoch: int):
        """Train function defined for a single epoch."""

        train_loss = 0.
        trues, preds = [], []
        self._set_learning_phase(train=True)

        for _, batch in enumerate(data_loader):

            x, y, m = batch['x'].to(device), batch['y'].to(device), batch['m'].to(device)

            self.optimizer.zero_grad()
            y_pred = self.predict(x, m, train=True)
            loss = self.loss_function(y_pred, y)
            loss.backward()
            self.optimizer.step()

            trues += [y]
            preds += [y_pred.detach()]

            train_loss += loss.item() * y.size(0)  # reverse operation of 'average'

        out = {'loss': train_loss / len(data_loader.dataset)}
        if isinstance(self.metrics, dict):
            trues = torch.cat(trues, dim=0)  # [(1,  ), (1,  ), (1,  ), ...] --> (N,  )
            preds = torch.cat(preds, dim=0)  # [(1, C), (1, C), (1, C), ...] --> (N, C)
            for metric_name, metric_function in self.metrics.items():
                out[metric_name] = metric_function(preds, trues).item()

        return out

    def evaluate(self, data_loader: torch.utils.data.DataLoader, device: str):
        """Evaluate current model. A single pass through the given dataset."""

        valid_loss = 0.
        trues, preds = [], []
        self._set_learning_phase(train=False)

        with torch.no_grad():
            for _, batch in enumerate(data_loader):

                x, y, m = batch['x'].to(device), batch['y'].to(device), batch['m'].to(device)
                y_pred = self.predict(x, m, train=False)
                loss = self.loss_function(y_pred, y)
                valid_loss += loss.item() * y.size(0)

                trues += [y]
                preds += [y_pred]

            out = {'loss': valid_loss / len(data_loader.dataset)}
            if isinstance(self.metrics, dict):
                trues = torch.cat(trues, dim=0)
                preds = torch.cat(preds, dim=0)
                for metric_name, metric_function in self.metrics.items():
                    out[metric_name] = metric_function(preds, trues).item()

            return out

    def predict(self, x: torch.Tensor, m: torch.Tensor = None, train: bool = False):
        """Make a prediction provided a batch of samples."""

        self._set_learning_phase(train)
        if self.backbone.in_channels == 2:
            if m is None:
                raise ValueError("Backbone requires a mask tensor `m`.")
            x = torch.cat([x, m], dim=1)
        return self.classifier(self.backbone(x))

    def test(self, data_loader: torch.utils.data.DataLoader, device: str, logger=None):
        """Evaluate model on test data."""

        def test_on_ckpt(ckpt: str):
            """Load checkpoint history and add test metric values."""
            self.load_model_from_checkpoint(ckpt)
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
            self.classifier.train()
        else:
            self.backbone.eval()
            self.classifier.eval()

    def save_checkpoint(self, path: str, **kwargs):
        ckpt = {
            'backbone': self.backbone.state_dict(),
            'classifier': self.classifier.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if \
                self.scheduler is not None else None
        }
        if kwargs:
            for k, v in kwargs.items():
                ckpt[k] = v
        torch.save(ckpt, path)

    def load_model_from_checkpoint(self, path: str):
        ckpt = torch.load(path)
        self.backbone.load_state_dict(ckpt['backbone'])
        self.classifier.load_state_dict(ckpt['classifier'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(ckpt['scheduler'])

    def load_history_from_checkpoint(self, path: str):
        ckpt = torch.load(path)
        del ckpt['backbone']
        del ckpt['classifier']
        del ckpt['optimizer']
        del ckpt['scheduler']

        return ckpt
