# -*- coding: utf-8 -*-

import os
import time
import json
import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Bernoulli
from kornia.filters import GaussianBlur2d

from models.base import *
from tasks.base import Task
from datasets.wafer import get_dataloader
from utils.logging import get_tqdm_config
from utils.plotting import save_image_dpi


class Denoising(Task):
    def __init__(self,
                 encoder: BackboneBase,
                 decoder: DecoderBase,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 loss_function: nn.Module,
                 noise_function: nn.Module,
                 metrics: dict, checkpoint_dir: str, write_summary: bool):
        super(Denoising, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.noise_function = noise_function
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

        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)

        train_loader = get_dataloader(train_set, batch_size, shuffle=True, num_workers=num_workers)
        valid_loader = get_dataloader(valid_set, batch_size, shuffle=False, num_workers=0)

        with tqdm.tqdm(**get_tqdm_config(total=epochs, leave=True, color='blue')) as pbar:

            best_valid_loss = float('inf')
            best_epoch = 0

            for epoch in range(1, epochs + 1):

                # 0. Train & evaluate
                train_history = self.train(train_loader, device=device, current_epoch=epoch)
                valid_history = self.evaluate(valid_loader, device=device)

                # 1. Epoch history (loss)
                epoch_history = {
                    'loss': {
                        'train': train_history.get('loss'),
                        'valid': valid_history.get('loss')
                    }
                }
                # 2. Epoch history (Other metrics if provided)
                if isinstance(self.metrics, dict):
                    raise NotImplementedError

                # 3. Tensorboard
                if isinstance(self.writer, SummaryWriter):
                    for metric_name, metric_dict in epoch_history.items():
                        self.writer.add_scalars(
                            main_tag=metric_name,
                            tag_scalar_dict=metric_dict,
                            global_step=epoch
                        )

                # 4. Save model if it is the current best
                valid_loss = epoch_history['loss']['valid']
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_epoch = epoch
                    self.save_checkpoint(self.best_ckpt, epoch=epoch, **epoch_history)
                    if kwargs.get('save_every', False):
                        new_ckpt = os.path.join(
                            self.checkpoint_dir,
                            f'epoch_{epoch:04d}.loss_{valid_loss:.4f}.pt'
                        )
                        self.save_checkpoint(new_ckpt, epoch=epoch, **epoch_history)

                # 5. Update scheduler
                if isinstance(self.scheduler, torch.optim.lr_scheduler.StepLR):
                    self.scheduler.step()
                elif isinstance(self.scheduler, torch.optim.lr_scheduler.ExponentialLR):
                    self.scheduler.step()
                elif isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(valid_loss)

                # 6. Logging
                desc = f" Epoch [{epoch:>04}/{epochs:>04}] ({best_epoch:04}) |"
                for metric_name, metric_dict in epoch_history.items():
                    for k, v in metric_dict.items():
                        desc += f" {k}_{metric_name}: {v:.4f} |"
                pbar.set_description_str(desc)
                pbar.update(1)
                if logger is not None:
                    logger.info(desc)

        # 7. Evaluate best model on test set (optional if `test_set` exists)
        if 'test_set' in kwargs.keys():
            test_loader = get_dataloader(kwargs.get('test_set'), batch_size, num_workers=0)
            self.test(test_loader, device=device, logger=logger)

    def train(self, data_loader: torch.utils.data.DataLoader, device: str, current_epoch: int):
        """Train function defined for a single epoch."""

        train_loss = 0.
        steps_per_epoch = len(data_loader)
        self._set_learning_phase(train=True)

        with tqdm.tqdm(**get_tqdm_config(total=steps_per_epoch, leave=False, color='green')) as pbar:
            for i, (x, _, m) in enumerate(data_loader):

                x, m = x.to(device), m.to(device)

                self.optimizer.zero_grad()
                _, _, decoded = self.predict(x, m, train=True)  # (B, 2, H, W)
                loss = self.loss_function(decoded, x.detach().long().squeeze(1))
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * x.size(0)  # reverse 'average'

                desc = f" Batch [{i+1:>04}/{steps_per_epoch:>04}] "
                desc += f" Loss: {train_loss / (data_loader.batch_size * (i + 1)):.4f} "
                pbar.set_description_str(desc)
                pbar.update(1)

        out = {'loss': train_loss / len(data_loader.dataset)}
        if isinstance(self.metrics, dict):
            raise NotImplementedError

        return out

    def evaluate(self, data_loader: torch.utils.data.DataLoader, device: str):
        """Evaluate current model. A single pass through the given dataset."""

        valid_loss = 0.
        self._set_learning_phase(train=False)

        with torch.no_grad():
            for _, (x, _, m) in enumerate(data_loader):

                x, m = x.to(device), m.to(device)
                _, _, decoded = self.predict(x, m, train=False)  # (B, 2, H, W)
                loss = self.loss_function(decoded, x.long().squeeze(1))
                valid_loss += loss.item() * x.size(0)

        out = {'loss': valid_loss / len(data_loader.dataset)}
        if isinstance(self.metrics, dict):
            raise NotImplementedError

        return out

    def predict(self, x: torch.Tensor, m: torch.Tensor = None, train: bool = False):
        """Note that the output is a tuple of length 3."""

        self._set_learning_phase(train)
        x_with_noise = self.noise_function(x, m)
        if self.encoder.in_channels == 2:
            if m is None:
                raise ValueError("Encoder requires a mask tensor `m`.")
            model_input = torch.cat([x_with_noise, m], dim=1)
        else:
            model_input = x_with_noise
        encoded = self.encoder(model_input)
        decoded = self.decoder(encoded)

        return x, x_with_noise, decoded

    def test(self, data_loader: torch.utils.data.DataLoader, device: str, logger=None):
        """Evaluate best model on test data."""

        self.load_model_from_checkpoint(self.best_ckpt)
        best_history = self.load_history_from_checkpoint(self.best_ckpt)

        test_history = self.evaluate(data_loader, device)
        for metric_name, metric_val in test_history.items():
            best_history[metric_name]['test'] = metric_val

        desc = f" Best model ({best_history.get('epoch', -1):04d}): "
        for metric_name, metric_dict in best_history.items():
            if metric_name == 'epoch':
                continue
            for k, v in metric_dict.items():
                desc += f" {k}_{metric_name}: {v:.4f} |"

        print(desc)
        if logger is not None:
            logger.info(desc)

        # Save final loss & metrics to a single json file
        with open(os.path.join(self.checkpoint_dir, 'best_history.json'), 'w') as fp:
            json.dump(best_history, fp, indent=2)

        # Visualize wafer bin maps
        plot_dir = os.path.join(self.checkpoint_dir, 'visualizations')
        os.makedirs(plot_dir, exist_ok=True)
        with torch.no_grad():
            self._set_learning_phase(train=False)
            for i, (x, _, m) in enumerate(data_loader):
                x, m = x.to(device), m.to(device)
                x, x_with_noise, decoded = self.predict(x, m, train=False)
                decoded = m * decoded.argmax(dim=1, keepdims=True).float()  # (B, 1, 40, 40)
                self.visualize(
                    *[x, x_with_noise, decoded], mask=m,
                    root=plot_dir, start=data_loader.batch_size * i,
                    )
                break

    def _set_learning_phase(self, train: bool = True):
        if train:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()

    def save_checkpoint(self, path: str, **kwargs):
        ckpt = {
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if \
                self.scheduler is not None else None
        }
        if kwargs:
            ckpt.update(kwargs)
        torch.save(ckpt, path)

    def load_model_from_checkpoint(self, path: str):
        ckpt = torch.load(path)
        self.encoder.load_state_dict(ckpt['encoder'])
        self.decoder.load_state_dict(ckpt['decoder'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(ckpt['scheduler'])

    def load_history_from_checkpoint(self, path: str):
        ckpt = torch.load(path)
        del ckpt['encoder']
        del ckpt['decoder']
        del ckpt['optimizer']
        del ckpt['scheduler']

        return ckpt

    @staticmethod
    def visualize(*tensors,
                  root: str,
                  prefix: str = '',
                  mask: torch.Tensor = None,
                  feature_range: tuple = (0, 1),
                  normalize: bool = True,
                  dpi: tuple = (500, 500),
                  start: int = 0):
        """
        Visualize input `torch.Tensor`s provided as positional arguments.\n
        Arguments:
            tensor: a 4D `torch.Tensor` of shape (B, C, H, W).
            root: str, root directory under which the images will be saved.
            prefix: str, filename prefix.
            mask: a single `torch.Tensor` of shape (B, 1, H, W), optional.
            feature_range: tuple, (min, max) values for normalization, optional.
            start: int, the index from which to start the output file names.
        """
        if not all([tensor.size() == tensors[0].size() for tensor in tensors]):
            raise ValueError("Input tensors provided as positional arguments must have the same shape.")
        nrow = len(tensors)  # Number of columns in a row
        tensors = torch.stack(tensors, axis=1)    # N * (B, 1, 40, 40) -> (B, N, 1, 40, 40)
        if mask is not None:
            # Unsqueezing in the 1st dimension of the mask gives shape (B, 1, 1, 40, 40),
            # making broadcasting available. Adding it to the input tensors and dividing
            # by a value of 2 normalizes the tensors to [0, 1]. Specifically, failures will
            # have a value of 1, normal dies a value of 1/2, and out-of-borders a value of 0.
            tensors = (tensors + mask.unsqueeze(1)) / 2
        plot_configs = dict(nrow=nrow, range=feature_range, normalize=normalize, dpi=dpi)
        for i, t in enumerate(tensors):
            filepath = os.path.join(root, f"{prefix}_{start+i:05d}.png")
            save_image_dpi(t, filepath, **plot_configs)


class MaskedBernoulliNoise(nn.Module):
    """
    Adds noise to input by sampling from a Bernoulli distribution.
    The range of noise is specified by [min_val, max_val].
    """
    def __init__(self, p: float, min_val: int = 0, max_val: int = 1):
        super(MaskedBernoulliNoise, self).__init__()
        self.bernoulli = Bernoulli(p)
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x: torch.Tensor, m: torch.Tensor = None):
        """
        Arguments:
            x: 4d (B, 1, H, W) torch tensor. 1 for defects, 0 for normal.
            m: 4d (B, 1, H, W) torch tensor. 1 for valid wafer regions, 0 for out-of-border regions.
        """
        with torch.no_grad():
            noise_mask = self.bernoulli.sample(x.size()).to(x.device)
            noise_value = torch.randint_like(x, self.min_val, self.max_val + 1).to(x.device)
            if m is not None:
                noise_mask = noise_mask * m

            return x * (1 - noise_mask) + noise_value * noise_mask


class MaskedGaussianSmoothing(nn.Module):
    """Applies Gaussian smoothing to the input."""
    def __init__(self, kernel_size: int, sigma: float, border_type: str = 'reflect'):
        super(MaskedGaussianSmoothing, self).__init__()
        self.gblur2d = GaussianBlur2d(
            kernel_size=(kernel_size, kernel_size),
            sigma=(sigma, sigma),
            border_type=border_type
        )

    def forward(self, x: torch.Tensor, m: torch.Tensor = None):
        with torch.no_grad():
            y = self.gblur2d(x)
            if m is not None:
                y = y * m + x * (1 - m)

            return y
