# -*- coding: utf-8 -*-

import os
import json
import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from tasks.base import Task
from datasets.loaders import get_dataloader
from utils.logging import get_tqdm_config
from utils.logging import make_epoch_description


class Classification(Task):
    def __init__(self,
                 backbone,
                 classifier,
                 optimizer,
                 scheduler,
                 loss_function,
                 metrics: dict,
                 checkpoint_dir: str,
                 write_summary: bool,
                 **kwargs):
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

    def run(self, train_set, valid_set, epochs: int, batch_size: int, num_workers: int = 0, device: str = 'cuda', **kwargs):
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
                train_history = self.train(train_loader, device)
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

    def train(self, data_loader: torch.utils.data.DataLoader, device: str):
        """Train function defined for a single epoch."""

        train_loss = 0.
        trues, preds = [], []
        self._set_learning_phase(train=True)

        for _, batch in enumerate(data_loader):

            x = batch['x'].to(device)
            y = batch['y'].to(device)
            y_pred = self.predict(x)
            loss = self.loss_function(y_pred, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            train_loss += loss.item()
            trues += [y.cpu().detach()]
            preds += [y_pred.cpu().detach()]

        out = {'loss': train_loss / len(data_loader)}
        if self.metrics is not None:
            assert isinstance(self.metrics, dict)
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

                x = batch['x'].to(device)
                y = batch['y'].to(device)
                y_pred = self.predict(x)
                loss = self.loss_function(y_pred, y)
                valid_loss += loss.item()

                trues += [y.cpu()]
                preds += [y_pred.cpu()]

            out = {'loss': valid_loss / len(data_loader)}
            if isinstance(self.metrics, dict):
                trues = torch.cat(trues, dim=0)
                preds = torch.cat(preds, dim=0)
                for metric_name, metric_function in self.metrics.items():
                    out[metric_name] = metric_function(preds, trues).item()

            return out

    def predict(self, x: torch.Tensor):
        """Make a prediction provided a batch of samples."""
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

        def make_description(history: dict, prefix: str = ''):
            desc = f" {prefix} ({history['epoch']:>4d}): "
            for metric_name, metric_dict in history.items():
                if metric_name == 'epoch':
                    continue
                for k, v in metric_dict.items():
                    desc += f" {k}_{metric_name}: {v:.4f} |"
            return desc

        # 1. Best model (based on validation loss)
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
            ckpt.update(kwargs)
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
