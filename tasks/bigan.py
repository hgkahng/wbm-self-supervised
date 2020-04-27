# -*- coding: utf-8 -*-

import os
import json
import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from tasks.base import Task
from tasks.denoising import Denoising

from datasets.wafer import get_dataloader
from utils.logging import get_tqdm_config


class BiGAN(Task):
    def __init__(self,
                 encoder, projector, generator, discriminator,
                 optimizer_E, optimizer_G, optimizer_D,
                 metrics: dict, checkpoint_dir: str, write_summary: bool):
        super(BiGAN, self).__init__()

        self.encoder = encoder
        self.projector = projector
        self.generator = generator
        self.discriminator = discriminator

        self.optimizer_E = optimizer_E
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D

        self.metrics = metrics if isinstance(metrics, dict) else None
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.checkpoint_dir) if write_summary else None

        self.latent_size = self.generator.latent_size
        self.loss_function = nn.BCEWithLogitsLoss(reduction='mean')

    def run(self, train_set, valid_set, epochs, batch_size, num_workers=1, device='cuda', **kwargs):
        """Train, evaluate and optionally test."""

        assert isinstance(train_set, torch.utils.data.Dataset)
        assert isinstance(valid_set, torch.utils.data.Dataset)
        assert isinstance(epochs, int)
        assert isinstance(batch_size, int)
        assert isinstance(num_workers, int)
        assert device.startswith('cuda') or device == 'cpu'

        logger = kwargs.get('logger', None)

        self.encoder = self.encoder.to(device)
        self.projector = self.projector.to(device)
        self.generator = self.generator.to(device)
        self.discriminator = self.discriminator.to(device)

        train_loader = get_dataloader(train_set, batch_size, shuffle=True, num_workers=num_workers)
        valid_loader = get_dataloader(valid_set, batch_size, shuffle=False)

        with tqdm.tqdm(**get_tqdm_config(total=epochs, leave=True, color='blue')) as pbar:

            best_G_loss = float('inf')
            best_epoch = 0

            for epoch in range(1, epochs + 1):

                # Train & Evaluate
                train_history = self.train(train_loader, device=device, current_epoch=epoch)
                valid_history = self.evaluate(valid_loader, device)

                # 1. Epoch history (loss)
                epoch_history = {
                    'D_loss': {
                        'train': train_history.get('D_loss'),
                        'valid': valid_history.get('D_loss'),
                    },
                    'G_loss': {
                        'train': train_history.get('G_loss'),
                        'valid': valid_history.get('G_loss'),
                    }
                }

                # 2. Epoch history (other metrics if provided)
                if isinstance(self.metrics, dict):
                    raise NotImplementedError

                # 3. Tensorboard
                if self.writer is not None:
                    for metric_name, metric_dict in epoch_history.items():
                        self.writer.add_scalars(
                            main_tag=metric_name,
                            tag_scalar_dict=metric_dict,
                            global_step=epoch
                        )

                # 4. Save model if it is the current best
                valid_G_loss = epoch_history['G_loss']['valid']
                if valid_G_loss < best_G_loss:
                    best_G_loss = valid_G_loss
                    best_epoch = epoch
                    self.save_checkpoint(self.best_ckpt, epoch=epoch, **epoch_history)
                    if kwargs.get('save_every', False):
                        new_ckpt = f'epoch_{epoch:04d}.loss_{valid_G_loss:.4f}.pt'
                        new_ckpt = os.path.join(self.checkpoint_dir, new_ckpt)
                        self.save_checkpoint(new_ckpt, epoch=epoch, **epoch_history)

                # 5. Logging
                desc = f" Epoch [{epoch:>04}/{epochs:>04}] |"
                for metric_name, metric_dict in epoch_history.items():
                    for k, v in metric_dict.items():
                        desc += f" {k}_{metric_name}: {v:.4f} |"
                pbar.set_description_str(desc)
                pbar.update(1)
                if logger is not None:
                    logger.info(desc)

        # 6. Evaluate best model on test set (optional if `test_set` is given)
        if 'test_set' in kwargs.keys():
            test_loader = get_dataloader(kwargs.get('test_set'), batch_size, num_workers=0)
            self.test(test_loader, device=device, logger=logger)

    def train(self, data_loader: torch.utils.data.DataLoader, device: str, current_epoch: int):
        """Training BiGAN for a single epoch."""

        G_loss = 0.
        D_loss = 0.
        steps_per_epoch = len(data_loader)
        self._set_learning_phase(train=True)

        with tqdm.tqdm(**get_tqdm_config(total=steps_per_epoch, leave=False, color='magenta')) as pbar:
            for i, (x, _, m) in enumerate(data_loader):

                x, m = x.to(device), m.to(device)

                E_x = self.encoder_predict(x, m)
                z = torch.randn(x.size(0), self.latent_size, 1, 1).to(device)
                G_z = self.generator(z)
                logits_fake = self.discriminator(G_z, z)
                if self.discriminator.in_channels == 2:
                    logits_real = self.discriminator(torch.cat([x, m], dim=1), E_x)
                else:
                    logits_real = self.discriminator(x, E_x)

                fake = torch.zeros_like(logits_fake)
                real = torch.ones_like(logits_real)

                d_loss = self.loss_function(logits_real, real) + \
                    self.loss_function(logits_fake, fake)
                d_loss.div_(2)

                g_loss = self.loss_function(logits_real, fake) + \
                    self.loss_function(logits_fake, real)
                g_loss.div_(2)

                # Update discriminator (D)
                self.optimizer_D.zero_grad()
                d_loss.backward(retain_graph=True)
                self.optimizer_D.step()

                # Update generator (G) & encoder (E)
                self.optimizer_E.zero_grad()
                self.optimizer_G.zero_grad()
                g_loss.backward()
                self.optimizer_E.step()
                self.optimizer_G.step()

                D_loss += d_loss.item() * x.size(0)
                G_loss += g_loss.item() * x.size(0)

                desc = f" Batch [{i+1:>04}/{steps_per_epoch:>04}] "
                pbar.set_description_str(desc)
                pbar.update(1)

        out = {
            'D_loss': D_loss / len(data_loader.dataset),
            'G_loss': G_loss / len(data_loader.dataset),
            }
        if isinstance(self.metrics, dict):
            raise NotImplementedError

        return out

    def evaluate(self, data_loader: torch.utils.data.DataLoader, device: str):
        """Evaluate current model. A single pass through the given dataset."""

        D_loss = 0.
        G_loss = 0.
        self._set_learning_phase(train=False)

        with torch.no_grad():
            for _, (x, _, m) in enumerate(data_loader):

                x, m = x.to(device), m.to(device)

                E_x = self.encoder_predict(x, m)
                z = torch.randn(x.size(0), self.latent_size, 1, 1).to(device)
                G_z = self.generator(z)
                logits_fake = self.discriminator(G_z, z)
                if self.discriminator.in_channels == 2:
                    logits_real = self.discriminator(torch.cat([x, m], dim=1), E_x)
                else:
                    logits_real = self.discriminator(x, E_x)

                fake = torch.zeros_like(logits_fake)
                real = torch.ones_like(logits_real)

                d_loss = self.loss_function(logits_real, real) + \
                    self.loss_function(logits_fake, fake)
                d_loss.div_(2)

                g_loss = self.loss_function(logits_real, fake) + \
                    self.loss_function(logits_fake, real)
                g_loss.div_(2)

                D_loss += d_loss.item() * x.size(0)
                G_loss += g_loss.item() * x.size(0)

        out = {
            'D_loss': D_loss / len(data_loader.dataset),
            'G_loss': G_loss / len(data_loader.dataset)
        }
        if isinstance(self.metrics, dict):
            raise NotImplementedError

        return out

    def encoder_predict(self, x: torch.Tensor, m: torch.Tensor = None):
        if self.encoder.in_channels == 2:
            if m is None:
                raise ValueError("Encoder requires a mask tensor `m`.")
            x = torch.cat([x, m], dim=1)
        return self.projector(self.encoder(x))

    def predict(self, x: torch.Tensor, m: torch.Tensor = None):
        return self.encoder_predict(x, m)

    def test(self, data_loader: torch.utils.data.DataLoader, device: str, logger=None):
        """Evaluate best model on test data."""

        self.load_model_from_checkpoint(self.best_ckpt)
        best_history = self.load_history_from_checkpoint(self.best_ckpt)

        test_history = self.evaluate(data_loader, device)
        for metric_name, metric_val in test_history.items():
            best_history[metric_name]['test'] = metric_val

        best_epoch = best_history.get('epoch', -1)
        desc = f" Best model obtained on epoch {best_epoch} |"
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

        # Visualize generated wafer bin map images
        plot_dir = os.path.join(self.checkpoint_dir, 'visualizations')
        os.makedirs(plot_dir, exist_ok=True)
        with torch.no_grad():
            self._set_learning_phase(train=False)
            for i, (x, _, m) in enumerate(data_loader):
                x, m = x.to(device), m.to(device)
                z = torch.randn(x.size(0), self.latent_size, 1, 1).to(device)
                G_z = self.generator(z)  # (B, ?, 40, 40)
                Denoising.visualize(
                    G_z.index_select(dim=1, index=torch.zeros(1).long().to(device)),
                    mask=G_z.index_select(dim=1, index=torch.ones(1).long().to(device)).round(),
                    root=plot_dir, start=data_loader.batch_size * i,
                    )
                break

    def _set_learning_phase(self, train: bool = True):
        if train:
            self.encoder.train()
            self.projector.train()
            self.generator.train()
            self.discriminator.train()
        else:
            self.encoder.eval()
            self.projector.eval()
            self.generator.eval()
            self.discriminator.eval()

    def save_checkpoint(self, path: str, **kwargs):
        ckpt = {
            'encoder': self.encoder.state_dict(),
            'projector': self.projector.state_dict(),
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_E': self.optimizer_E.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
        }
        if kwargs:
            ckpt.update(kwargs)
        torch.save(ckpt, path)

    def load_model_from_checkpoint(self, path: str):
        ckpt = torch.load(path)
        self.encoder.load_state_dict(ckpt['encoder'])
        self.projector.load_state_dict(ckpt['projector'])
        self.generator.load_state_dict(ckpt['generator'])
        self.discriminator.load_state_dict(ckpt['discriminator'])
        self.optimizer_E.load_state_dict(ckpt['optimizer_E'])
        self.optimizer_G.load_state_dict(ckpt['optimizer_G'])
        self.optimizer_D.load_state_dict(ckpt['optimizer_D'])

    def load_history_from_checkpoint(self, path: str):
        ckpt = torch.load(path)
        del ckpt['encoder']
        del ckpt['projector']
        del ckpt['generator']
        del ckpt['discriminator']
        del ckpt['optimizer_E']
        del ckpt['optimizer_G']
        del ckpt['optimizer_D']
        return ckpt
