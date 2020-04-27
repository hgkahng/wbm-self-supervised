# -*- coding: utf-8 -*-

import os
import glob
import pathlib

import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


WM811K_LABELS = {
    'center'    : 0,
    'donut'     : 1,
    'edge-loc'  : 2,
    'edge-ring' : 3,
    'loc'       : 4,
    'random'    : 5,
    'scratch'   : 6,
    'near-full' : 7,
    'none'      : 8,
}


def load_image(filepath: str):
    """Load image with PIL."""
    return Image.open(filepath)


def decouple_mask(x: torch.Tensor):
    """
    Decouple input with existence mask.
    Defect bins = 2, Normal bins = 1, Null bins = 0
    """
    m = x.gt(0).float()
    x = torch.clamp(x - 1, min=0., max=1.)
    return x, m


class LabeledWM811kFolder(Dataset):
    NUM_CLASSES = len(WM811K_LABELS)  # 9
    def __init__(self, root: str, transform=None, proportion: float = 1.0, **kwargs):

        self.root = root
        self.transform = transform
        self.proportion = proportion

        images  = sorted(glob.glob(os.path.join(root, '**/*.png'), recursive=True))  # Get paths to images
        labels  = [pathlib.PurePath(image).parent.name for image in images]          # Parent directory names are class labels
        targets = [WM811K_LABELS[l] for l in labels]                                 # Convert class labels to integer target values
        samples = [(image, target) for image, target in zip(images, targets)]        # Make (path, target) pairs

        if self.proportion < 1.0:
            self.samples, _ = train_test_split(
                samples,
                train_size=self.proportion,
                stratify=[s[1] for s in samples],
                shuffle=True,
                random_state=2015010720 + kwargs.get('random_seed', 0),
            )
        else:
            self.samples = samples

        self.class_weights = self._compute_effective_samples_class_weights()
        assert isinstance(self.class_weights, torch.Tensor)

    def __getitem__(self, idx):

        path, y = self.samples[idx]
        x = load_image(path)

        if self.transform is not None:
            x = self.transform(x)

        x, m = decouple_mask(x)

        return dict(x=x, y=y, m=m, idx=idx)

    def __len__(self):
        return len(self.samples)

    def _compute_class_weights(self):
        """Compute class weights."""

        y = np.array([s[1] for s in self.samples], dtype=int)
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes, y)

        return torch.from_numpy(class_weights.astype(np.float32))

    def _compute_effective_samples_class_weights(self, beta: float = 0.999):
        """Compute class weights, based on the following paper:
            Cui, Y., Jia, M., Lin, T. Y., Song, Y., & Belongie, S. (2019).
            Class-balanced loss based on effective number of samples.
            In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 9268-9277).
        """

        samples_per_class = np.zeros(self.NUM_CLASSES)
        y = np.array([s[1] for s in self.samples], dtype=int)
        for c in np.unique(y):
            samples_per_class[c] = (y == c).sum()

        effective_num = 1.0 - np.power(beta, np.sqrt(samples_per_class)) + 1e-8
        class_weights = (1.0 - beta) / effective_num
        class_weights = class_weights / np.sum(class_weights) * self.NUM_CLASSES

        return torch.from_numpy(class_weights.astype(np.float32))


class UnlabeledWM811kFolder(Dataset):
    label2target = {'-': 0}
    def __init__(self, root, transform=None):

        self.root = root
        self.transform = transform

        images = sorted(glob.glob(os.path.join(root, '**/*.png'), recursive=True))
        labels = [pathlib.PurePath(image).parent.name for image in images]
        targets = [self.label2target[l] for l in labels]
        self.samples = [(image, target) for image, target in zip(images, targets)]

    def __getitem__(self, idx):
        return self.make_batch(idx)

    def __len__(self):
        return len(self.samples)

    def make_batch(self, idx):
        """Create a batch of samples. """
        path, y = self.samples[idx]
        img = load_image(path)

        if self.transform is not None:
            x = self.transform(img)

        x, m = decouple_mask(x)

        return dict(x=x, y=y, m=m, idx=idx)


class UnlabeledWM811kFolderForPIRL(UnlabeledWM811kFolder):
    def __init__(self, root, transform=None, positive_transform=None):
        super(UnlabeledWM811kFolderForPIRL, self).__init__(root, transform)
        self.positive_transform = positive_transform

    def make_batch(self, idx):
        path, y = self.samples[idx]
        img = load_image(path)

        if self.transform is not None:
            x = self.transform(img)

        if self.positive_transform is not None:
            x_t = self.positive_transform(img)

        x, m = decouple_mask(x)
        x_t, m_t = decouple_mask(x_t)

        return dict(x=x, y=y, m=m, idx=idx, x_t=x_t, m_t=m_t)


class WM811kDataset(Dataset):
    def __init__(self, filepath: str, mode: str = 'train', return_mask: bool = True):

        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)
        self.filepath = filepath

        if mode not in ['train', 'valid', 'test']:
            raise ValueError(mode)
        self.mode = mode
        self.return_mask = return_mask

        npzfile = np.load(self.filepath)

        self.y = torch.from_numpy(npzfile['y_' + mode]).long()      # only used for labeled data
        self.x = torch.from_numpy(npzfile['x_' + mode]).float()     # {0, 1, 2}
        self.m = self.x.ge(1).float()                               # {0, 1}

        self.x = torch.clamp(self.x - 1, min=0., max=1.)            # {0, 1}

        assert self.x.shape == self.m.shape
        assert len(self.x.shape) == 4 and len(self.y.shape) == 1
        assert len(self.x) == len(self.y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.return_mask:
            return self.x[idx], self.y[idx], self.m[idx]
        else:
            return self.x[idx], self.y[idx]

    @property
    def input_size(self):
        return self.x.size()[2:]


class LabeledWM811kDataset(WM811kDataset):
    NUM_CLASSES = 8
    def __init__(self, filepath: str, mode: str = 'train', return_mask: bool = True, proportion: float = 1.0):
        super(LabeledWM811kDataset, self).__init__(filepath, mode, return_mask)

        is_labeled = self.y != 8  # 'none'
        self.y = self.y[is_labeled]
        self.x = self.x[is_labeled]
        self.m = self.m[is_labeled]

        assert 0. < proportion <= 1.
        if proportion < 1.:
            assert mode == 'train', "Only supported for training sets."

        if 0. < proportion < 1.:
            split_configs = {
                'random_state': 2015010720,
                'stratify': self.y.numpy(),
                'train_size': proportion
            }
            indices, _ = train_test_split(np.arange(len(self.x)), **split_configs)
            self.x, self.y, self.m = self.x[indices], self.y[indices], self.m[indices]
        else:
            pass


class Wafer40Dataset(Dataset):
    NUM_CLASSES = 5
    INPUT_SIZE = (40, 40)
    def __init__(self, filepath: str, mode: str = 'train', return_mask: bool = True):

        if not os.path.exists(filepath):
            raise FileNotFoundError
        self.filepath = filepath

        if mode not in ['train', 'valid', 'test']:
            raise ValueError
        self.mode = mode

        self.return_mask = return_mask

        npzfile = np.load(self.filepath)
        self.y = torch.from_numpy(npzfile['y_' + mode]).long()
        self.x = torch.from_numpy(npzfile['x_' + mode]).float()  # {0, 1, 2}
        self.m = self.x.ge(1).float()                            # {0, 1}

        self.x = torch.clamp(self.x - 1, min=0., max=1.)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.return_mask:
            return self.x[idx], self.y[idx], self.m[idx]
        else:
            return self.x[idx], self.y[idx]


class PartialWafer40Dataset(Wafer40Dataset):
    def __init__(self, filepath: str, mode: str = 'train', proportion: float = 1.0, return_mask: bool = True):
        super(PartialWafer40Dataset, self).__init__(filepath, mode, return_mask)

        assert mode == 'train', "Only supported for training sets."
        assert 0. < proportion <= 1.

        if 0. < proportion < 1.:
            split_configs = {
                'random_state': 2015010720,
                'stratify': self.y.numpy(),
                'train_size': proportion
            }
            indices, _ = train_test_split(np.arange(len(self.x)), **split_configs)
            self.x, self.y, self.m = self.x[indices], self.y[indices], self.m[indices]
        else:
            pass


def get_dataloader(dataset: torch.utils.data.Dataset,
                   batch_size: int,
                   shuffle: bool = False,
                   num_workers: int = 0):
    """Return a `DataLoader` instance."""
    assert isinstance(dataset, torch.utils.data.Dataset)
    loader_configs = {
        'dataset': dataset,
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'drop_last': False,
        'pin_memory': True,
    }
    return DataLoader(**loader_configs)
