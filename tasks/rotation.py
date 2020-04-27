# -*- coding: utf-8 -*-

import os
import json
import random
import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from tasks.base import Task
from datasets.wafer import get_dataloader
from utils.logging import get_tqdm_config
from utils.gradcam import GradCAM
from utils.plotting import save_image_dpi


class Rotation(Task):
    num_rotations = 4
    def __init__(self, backbone, classifier,
                 optimizer, scheduler, loss_function,
                 metrics: dict, checkpoint_dir: str, write_summary: bool):
        super(Rotation, self).__init__()

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
        valid_loader = get_dataloader(valid_set, batch_size, shuffle=False, num_workers=0)

        with tqdm.tqdm(**get_tqdm_config(total=epochs, leave=True, color='blue')) as pbar:

            best_valid_loss = float('inf')
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
                            'train': train_history.get(metric_name),
                            'valid': valid_history.get(metric_name)
                        }

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
                        new_ckpt = f'epoch_{epoch:04d}.loss_{valid_loss:.4f}.pt'
                        new_ckpt = os.path.join(self.checkpoint_dir, new_ckpt)
                        self.save_checkpoint(new_ckpt, epoch=epoch, **epoch_history)

                # 5. Update Scheduler
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

        # 7. Evaluate best model on test set (optional if `test_set` is given)
        if 'test_set' in kwargs.keys():
            test_loader = get_dataloader(kwargs.get('test_set'), batch_size, num_workers=0)
            self.test(test_loader, device=device, logger=logger)

    def train(self, data_loader: torch.utils.data.DataLoader, device: str, current_epoch: int):
        """Train function defined for a single epoch."""

        train_loss = 0.
        steps_per_epoch = len(data_loader)
        self._set_learning_phase(train=True)

        with tqdm.tqdm(**get_tqdm_config(total=steps_per_epoch, leave=False, color='red')) as pbar:
            for i, (x, _, m) in enumerate(data_loader):

                x, m = x.to(device), m.to(device)
                xs, ms = self.rotate_4x(x), self.rotate_4x(m)  # B --> 4B
                y = torch.repeat_interleave(torch.arange(4), x.size(0)).long().to(device)

                self.optimizer.zero_grad()
                y_pred = self.predict(xs, ms, train=True)
                loss = self.loss_function(y_pred, y)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * y.size(0)

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
                xs, ms = self.rotate_4x(x), self.rotate_4x(m)
                y = torch.repeat_interleave(torch.arange(4), x.size(0)).long().to(device)
                y_pred = self.predict(xs, ms, train=False)
                loss = self.loss_function(y_pred, y)
                valid_loss += loss.item() * y.size(0)  # reverse operation of 'average'

            out = {'loss': valid_loss / len(data_loader.dataset)}
            if isinstance(self.metrics, dict):
                raise NotImplementedError

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
        """Evaluate best model on test data."""

        self.load_model_from_checkpoint(self.best_ckpt)
        best_history = self.load_history_from_checkpoint(self.best_ckpt)

        test_history = self.evaluate(data_loader, device)
        for metric_name, metric_val in test_history.items():
            best_history[metric_name]['test'] = metric_val

        desc = f" Best model {best_history.get('epoch', -1):04d}: "
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

        # Visualize grad-CAM
        plot_dir = os.path.join(self.checkpoint_dir, 'gradcam_visualizations')
        os.makedirs(plot_dir, exist_ok=True)
        gradcam = GradCAM.from_config(
            backbone=self.backbone, classifier=self.classifier,
            model_type='vgg', layer_name='layers.block3.conv1')

        for i, (x, _, m) in enumerate(data_loader):

            x, m = x.to(device), m.to(device)
            xs = self.rotate_4x(x)  # (4B, 1, 40, 40)
            ms = self.rotate_4x(m)  # (4B, 1, 40, 40)

            if self.backbone.in_channels == 2:
                model_input = torch.cat([xs, ms], dim=1)
            else:
                model_input = xs

            ss, _ = gradcam(model_input)  # (4B, 1, 40, 40)

            xs = torch.stack(xs.split(x.size(0)), dim=1)  # (B, 4, 1, 40, 40)
            ms = torch.stack(ms.split(x.size(0)), dim=1)  # (B, 4, 1, 40, 40)
            ss = torch.stack(ss.split(x.size(0)), dim=1)  # (B, 4, 1, 40, 40)
            for j, (x4, m4, s4) in enumerate(zip(xs, ms, ss)):

                imgs = []
                heatmaps = []
                results = []
                for x_, m_, s_ in zip(x4, m4, s4):
                    img, heatmap, result = GradCAM.visualize_cam(smap=s_, img=x_, optional_mask=m_)
                    imgs += [img]
                    heatmaps += [heatmap]
                    results += [result]

                save_image_dpi(
                    tensor=[*imgs, *heatmaps, *results],
                    dpi=(500, 500),
                    fp=os.path.join(plot_dir, f"gradcam_{x.size(0)*i+j:06d}.png"),
                    nrow=len(imgs),  # 4
                    normalize=False)
            break

    def _set_learning_phase(self, train: bool = False):
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

    @classmethod
    def rotate_4x(cls, x: torch.Tensor):
        """Returns a 4 x B tensor by rotating (0, 90, 180, 270) degrees."""
        return torch.cat([x, cls.rot90(x), cls.rot180(x), cls.rot270(x)], dim=0)

    @staticmethod
    def rot90(x: torch.Tensor):
        """Rotates 90 degrees counter-clockwise."""
        assert x.ndim == 4, "(B, C, H, W)"
        return torch.rot90(x, 1, dims=(2, 3))

    @staticmethod
    def rot180(x: torch.Tensor):
        """Rotates tensor 180 degrees counter-clockwise."""
        assert x.ndim == 4, "(B, C, H, W)"
        return torch.rot90(x, 2, dims=(2, 3))

    @staticmethod
    def rot270(x: torch.Tensor):
        """Rotates tensor 270 degrees counter-clockwise."""
        assert x.ndim == 4, "(B, C, H, W)"
        return torch.rot90(x, 3, dims=(2, 3))


class RotateTransform(nn.Module):
    """Rotates input randomly in 4 different degrees."""
    def __init__(self):
        super(RotateTransform, self).__init__()

    def forward(self, x: torch.Tensor, m: torch.Tensor = None):
        """
        Arguments:
            x: 4d (B, 1, H, W) torch tensor. 1 for defects, 0 for normal.
            m: 4d (B, 1, H, W) torch tensor. 1 for valid wafer regions, 0 for out-of-border regions.
        """
        with torch.no_grad():
            if m is None:
                out = self.random_rotate(x)[0]
            else:
                out = self.random_rotate(x, m)
            return out

    @staticmethod
    def random_rotate(*tensors):
        assert tensors[0].ndim == 4, "(B, C, H, W)"
        assert all([tensor.size() == tensors[0].size() for tensor in tensors])

        angles = torch.randint(
            low=0, high=4, 
            size=(tensors[0].size(0), ),
            device=tensors[0].device
        )
        out = []
        for tensor in tensors:
            rotated = [
                torch.rot90(t.unsqueeze(0), a, dims=(2, 3)) \
                    for t, a in zip(tensor, angles)
                ]
            rotated = torch.cat(rotated, dim=0)
            out += [rotated]
        return out
