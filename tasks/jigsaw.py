# -*- coding: utf-8 -*-

import os
import json
import math
import itertools
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import kornia
import numpy as np
from scipy.spatial.distance import cdist

from tasks.base import Task
from datasets.wafer import get_dataloader
from utils.logging import get_tqdm_config


class Jigsaw(Task):
    def __init__(self, backbone, classifier,
                 num_patches, num_permutations,
                 optimizer, scheduler, loss_function,
                 metrics: dict, checkpoint_dir: str, write_summary: bool):
        super(Jigsaw, self).__init__()

        self.backbone = backbone
        self.classifier = classifier
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.num_patches = num_patches
        self.num_classes = self.num_permutations = num_permutations
        self.permutations = JigsawTransform.make_permutations(num_patches, num_permutations)

        self.loss_function = loss_function
        self.metrics = metrics if isinstance(metrics, dict) else None
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.checkpoint_dir) if write_summary else None

    def run(self, train_set, valid_set, epochs, batch_size, num_workers=1, device='cuda', **kwargs):
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

        with tqdm.tqdm(**get_tqdm_config(epochs, leave=True, color='blue')) as pbar:

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
                        'valid': valid_history.get('loss'),
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
                if self.writer is not None:
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

                # 5. Logging
                desc = f" Epoch [{epoch:>04}/{epochs:>04}] | "
                for metric_name, metric_dict in epoch_history.items():
                    for k, v in metric_dict.items():
                        desc += f" {k}_{metric_name}: {v:.4f} |"
                pbar.set_description_str(desc)
                pbar.update(1)
                if logger is not None:
                    logger.info(desc)

        # 6. Evaluate best model on test set (optional if `test_set` is given)
        if 'test_set' in kwargs.keys():
            test_loader = get_dataloader(kwargs.get('test_set'), batch_size)
            self.test(test_loader, device=device, logger=logger)

    def train(self, data_loader: torch.utils.data.DataLoader, device: str, current_epoch: int):
        """Train function defined for a single epoch."""

        train_loss = 0.
        trues, preds = [], []
        steps_per_epoch = len(data_loader)
        self._set_learning_phase(train=True)

        self.permutations = self.permutations.to(device)

        with tqdm.tqdm(**get_tqdm_config(steps_per_epoch, leave=False, color='red')) as pbar:
            for i, (x, _, m) in enumerate(data_loader):

                y = torch.randint(0, self.num_permutations, size=(x.size(0), ))
                x, y, m = x.to(device), y.to(device), m.to(device)

                x, m = JigsawTransform.make_patches(x, m, num_patches=self.num_patches)     # (B, P, C, H, W)
                x, m = JigsawTransform.permute_patches(x, m, indices=self.permutations[y])  # (B, P, C, H, W)

                self.optimizer.zero_grad()
                y_pred = self.predict(x, m, train=True)                     # (B, K)
                loss = self.loss_function(y_pred, y)
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step(current_epoch-1 + i/steps_per_epoch)

                trues += [y]
                preds += [y_pred.detach()]

                train_loss += loss.item() * y.size(0)

                desc = f" Batch [{i+1:>04}/{steps_per_epoch:>04}] "
                pbar.set_description_str(desc)
                pbar.update(1)

        out = {'loss': train_loss / len(data_loader.dataset)}
        if isinstance(self.metrics, dict):
            trues = torch.cat(trues, dim=0)
            preds = torch.cat(preds, dim=0)
            for metric_name, metric_function in self.metrics.items():
                out[metric_name] = metric_function(preds, trues).item()

        return out

    def evaluate(self, data_loader: torch.utils.data.DataLoader, device: str):
        """Evaluate current model. A single pass through the given dataset."""

        valid_loss = 0.
        trues, preds = [], []
        self._set_learning_phase(train=False)

        self.permutations = self.permutations.to(device)

        with torch.no_grad():
            for _, (x, _, m) in enumerate(data_loader):

                y = torch.randint(0, self.num_permutations, size=(x.size(0), ))
                x, y, m = x.to(device), y.to(device), m.to(device)

                x, m = JigsawTransform.make_patches(x, m, num_patches=self.num_patches)
                x, m = JigsawTransform.permute_patches(x, m, indices=self.permutations[y])

                y_pred = self.predict(x, m, train=False)
                loss = self.loss_function(y_pred, y)
                valid_loss += loss.item() * y.size(0)

                trues += [y]
                preds += [y_pred.detach()]

            out = {'loss': valid_loss / len(data_loader.dataset)}
            if isinstance(self.metrics, dict):
                trues = torch.cat(trues, dim=0)
                preds = torch.cat(preds, dim=0)
                for metric_name, metric_function in self.metrics.items():
                    out[metric_name] = metric_function(preds, trues).item()

            return out

    def predict(self, x: torch.Tensor, m: torch.Tensor = None, train: bool = False):
        """Make a prediction provided a batch of samples."""

        assert x.ndim == 5, "(B, P, C, H, W)"
        x = x.transpose(0, 1)
        if m is not None:
            assert m.ndim == 5, "(B, P, C, H, W)"
            m = m.transpose(0, 1)

        bout = [self._predict_patch(x_, m_, train) for x_, m_ in zip(x, m)]
        bout = torch.stack(bout, dim=1)  # (B, P, C', H', W')

        return self.classifier(bout)     # (B, K)

    def _predict_patch(self, x: torch.Tensor, m: torch.Tensor = None, train: bool = False):
        """Make a prediction for a single patch of batched samples."""

        assert x.ndim == 4, "(B, 1, H, W)"
        self._set_learning_phase(train)
        if self.backbone.in_channels == 2:
            if m is None:
                raise ValueError("Backbone requires a mask tensor `m`.")
            assert m.ndim == 4, "(B, 1, H, W)"
            x = torch.cat([x, m], dim=1)
        return self.backbone(x)

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
        with open(os.path.join(self.checkpoint_dir, "best_history.json"), "w") as fp:
            json.dump(best_history, fp, indent=2)

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
                self.scheduler is not None else None,
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

    @staticmethod
    def make_permutations(num_patches: int = 9,
                          num_permutations: int = 100,
                          selection: str = 'max'):

        P_hat = list(itertools.permutations(list(range(num_patches)), num_patches))
        P_hat = np.array(P_hat)

        for i in range(num_permutations):
            if i == 0:
                j = np.random.randint(len(P_hat))
                P = P_hat[j].reshape(1, -1)
            else:
                P = np.concatenate([P, P_hat[j].reshape(1, -1)], axis=0)

            P_hat = np.delete(P_hat, j, axis=0)
            D = cdist(P, P_hat, metric='hamming').mean(axis=0).flatten()

            if selection == 'max':
                j = D.argmax()
            elif selection == 'mean':
                m = int(D.shape[0] / 2)
                S = D.argsort()
                j = S[np.random.randint(m - 10, m + 10)]
            else:
                raise NotImplementedError

        return torch.from_numpy(P).long()

    @staticmethod
    def make_patches(x: torch.Tensor, num_patches: int = 9, resize=True):

        assert x.ndim == 4, "(B, C, H, W)"
        assert num_patches in [n ** 2 for n in [2, 3, 4, 5]]

        _, _, h, w = x.size()
        n = int(math.sqrt(num_patches))           # sqrt(9) =  3
        patch_size = [h // n, w // n]             # 40 // 3 = 13
        patch_size = [p - 1 for p in patch_size]  # 13 -  1 = 12
        crop_size  = [s * n for s in patch_size]  # 12 *  3 = 36

        x = kornia.center_crop(x, size=crop_size)  # (B, C, 36, 36)

        tl = [[i                , j                ] for i in range(n) for j in range(n)]
        tr = [[i                , j + patch_size[1]] for i in range(n) for j in range(n)]
        br = [[i + patch_size[0], j + patch_size[1]] for i in range(n) for j in range(n)]
        bl = [[i + patch_size[0], j                ] for i in range(n) for j in range(n)]

        def crop(x: torch.Tensor, boxes: list or tuple):
            assert x.ndim == 4 & len(boxes) == 4
            tl, _, br, bl = boxes
            return x[:, :, tl[0]:bl[0], bl[1]:br[1]]

        out = []
        for boxes in zip(tl, tr, br, bl):
            o = crop(x, boxes)
            if resize:
                o = F.interpolate(o, size=(h, w), mode='nearest')
            out.append(o)

        return torch.stack(out, dim=1)  # (B, P, C, H, W)

    @staticmethod
    def permute_patches(x: torch.Tensor, indices: torch.Tensor):
        """Permute a 5D tensor."""
        assert x.ndim == 5,       "(B, P, C, H, W)"
        assert indices.ndim == 2, "(B, P)"
        permuted = [x_.index_select(0, index) for x_, index in zip(x, indices)]

        return torch.stack(permuted, dim=0)


class JigsawTransform(nn.Module):
    """Creates a jigsaw puzzle input."""
    def __init__(self, num_patches: int, num_permutations: int):
        super(JigsawTransform, self).__init__()
        self.num_patches = num_patches
        self.num_permutations = num_permutations
        self.permutations = self.make_permutations(self.num_patches, self.num_permutations)

    def forward(self, x: torch.Tensor, m: torch.Tensor = None, indices: torch.Tensor = None):
        """
        Arguments:
            x: 4d (B, 1, H, W) torch tensor. 1 for defects, 0 for normal.
            m: 4d (B, 1, H, W) torch tensor. 1 for valid wafer regions, 0 for out-of-border regions.
        """

        if indices is None:
            raise ValueError("`indices` is required.")

        with torch.no_grad():
            if m is None:
                x_patches = self.make_patches(x, num_patches=self.num_patches)
                x_permuted = self.permute_patches(x_patches, indices=self.permutations[indices])
                return x_permuted  # (B, P, C, H, W)
            else:
                x_patches, m_patches = self.make_patches(x, m, num_patches=self.num_patches)
                x_permuted, m_permuted = self.permute_patches(x_patches, m_patches, indices=self.permutations[indices])
                return x_permuted, m_permuted

    @staticmethod
    def make_permutations(num_patches: int = 9,
                          num_permutations: int = 100,
                          selection: str = 'max'):

        P_hat = list(itertools.permutations(list(range(num_patches)), num_patches))
        P_hat = np.array(P_hat)

        for i in range(num_permutations):
            if i == 0:
                j = np.random.randint(len(P_hat))
                P = P_hat[j].reshape(1, -1)
            else:
                P = np.concatenate([P, P_hat[j].reshape(1, -1)], axis=0)

            P_hat = np.delete(P_hat, j, axis=0)
            D = cdist(P, P_hat, metric='hamming').mean(axis=0).flatten()

            if selection == 'max':
                j = D.argmax()
            elif selection == 'mean':
                m = int(D.shape[0] / 2)
                S = D.argsort()
                j = S[np.random.randint(m - 10, m + 10)]
            else:
                raise NotImplementedError

        return torch.from_numpy(P).long()

    @staticmethod
    def make_patches(*tensors, num_patches: int = 9, resize=True):

        assert tensors[0].ndim == 4, "(B, C, H, W)"
        assert all([tensor.size() == tensors[0].size() for tensor in tensors])
        assert num_patches in [n ** 2 for n in [2, 3, 4, 5]]

        _, _, h, w = tensors[0].size()
        n = int(math.sqrt(num_patches))           # sqrt(9) =  3
        patch_size = [h // n, w // n]             # 40 // 3 = 13
        patch_size = [p - 1 for p in patch_size]  # 13 -  1 = 12
        crop_size  = [s * n for s in patch_size]  # 12 *  3 = 36

        tl = [[i                , j                ] for i in range(n) for j in range(n)]
        tr = [[i                , j + patch_size[1]] for i in range(n) for j in range(n)]
        br = [[i + patch_size[0], j + patch_size[1]] for i in range(n) for j in range(n)]
        bl = [[i + patch_size[0], j                ] for i in range(n) for j in range(n)]

        def crop(x: torch.Tensor, boxes: list or tuple):
            assert x.ndim == 4 & len(boxes) == 4
            tl, _, br, bl = boxes
            return x[:, :, tl[0]:bl[0], bl[1]:br[1]]

        out = []
        for tensor in tensors:
            tensor = kornia.center_crop(tensor, size=crop_size)  # (B, C, 36, 36)

            patches = []
            for boxes in zip(tl, tr, br, bl):
                patch = crop(tensor, boxes)
                if resize:
                    patch = F.interpolate(patch, size=(h, w), mode='nearest')
                patches.append(patch)

            out.append(torch.stack(patches, dim=1))  # (B, P, C, H, W)

        if len(out) == 1:
            return out[0]
        else:
            return out

    @staticmethod
    def permute_patches(*tensors, indices: torch.Tensor):
        """Permute a 5D tensor."""
        assert tensors[0].ndim == 5, "(B, P, C, H, W)"
        assert all([tensor.size() == tensors[0].size() for tensor in tensors])
        assert indices.ndim == 2, "(B, P)"

        out = []
        for tensor in tensors:
            permuted = [x4d.index_select(0, index) for x4d, index in zip(tensor, indices)]
            permuted = torch.stack(permuted, dim=0)  # (B, P, C, H, W)
            out.append(permuted)

        if len(out) == 1:
            return out[0]
        else:
            return out


if __name__ == '__main__':

    # Configurations
    NUM_PATCHES = 4
    NUM_PERMUTATIONS = 10
    BATCH_SIZE = 3

    # Make permutations
    permutations_ = JigsawTransform.make_permutations(NUM_PATCHES, NUM_PERMUTATIONS)

    # Make patches
    X = torch.randint(0, 2, size=(BATCH_SIZE, 1, 10, 10)).float()
    M = torch.randint_like(X, 0, 2)
    patches_ = JigsawTransform.make_patches(X, M, num_patches=NUM_PATCHES)

    # Permute patches
    permuted_ = JigsawTransform.permute_patches(*patches_, indices=permutations_[:BATCH_SIZE])
