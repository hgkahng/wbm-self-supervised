# -*- coding: utf-8 -*-

import os
import glob
import pathlib

import numpy as np
import torch
import cv2

from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


class WM811K(Dataset):
    label2idx = {
        'center'    : 0,
        'donut'     : 1,
        'edge-loc'  : 2,
        'edge-ring' : 3,
        'loc'       : 4,
        'random'    : 5,
        'scratch'   : 6,
        'near-full' : 7,
        'none'      : 8,
        '-'         : 9,
    }
    idx2label = [k for k in label2idx.keys()]
    num_classes = len(idx2label) - 1  # exclude unlabeled (-)

    def __init__(self, root, transform=None, proportion=1.0, **kwargs):
        super(WM811K, self).__init__()

        self.root = root
        self.transform = transform
        self.proportion = proportion

        images  = sorted(glob.glob(os.path.join(root, '**/*.png'), recursive=True))  # Get paths to images
        labels  = [pathlib.PurePath(image).parent.name for image in images]          # Parent directory names are class label strings
        targets = [self.label2idx[l] for l in labels]                                # Convert class label strings to integer target values
        samples = list(zip(images, targets))                                         # Make (path, target) pairs

        if self.proportion < 1.0:
            # Randomly sample a proportion of the data
            self.samples, _ = train_test_split(
                samples,
                train_size=self.proportion,
                stratify=[s[1] for s in samples],
                shuffle=True,
                random_state=1993 + kwargs.get('seed', 0),
            )
        else:
            self.samples = samples

    def __getitem__(self, idx):

        path, y = self.samples[idx]
        x = self.load_image_cv2(path)

        if self.transform is not None:
            x = self.transform(x)
        x = self.decouple_mask(x)

        return dict(x=x, y=y, idx=idx)

    def __len__(self):
        return len(self.samples)

    def _compute_class_weights(self):
        """Compute class weights."""

        y = np.array([s[1] for s in self.samples], dtype=int)
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes, y)

        return torch.from_numpy(class_weights.astype(np.float32))

    def _compute_effective_samples_class_weights(self, beta: float = 0.999):
        """
        Compute class weights, based on the following paper:
        Cui, Y., Jia, M., Lin, T. Y., Song, Y., & Belongie, S. (2019).
        Class-balanced loss based on effective number of samples.
        In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 9268-9277).
        """

        samples_per_class = np.zeros(self.num_classes)
        y = np.array([s[1] for s in self.samples], dtype=int)
        for c in np.unique(y):
            samples_per_class[c] = (y == c).sum()

        effective_num = (1.0 - np.power(beta, samples_per_class)) + 1e-8
        class_weights = (1.0 - beta) / effective_num
        class_weights = class_weights / np.sum(class_weights) * self.num_classes

        return torch.from_numpy(class_weights.astype(np.float32))

    @staticmethod
    def load_image_pil(filepath: str):
        """Load image with PIL."""
        return Image.open(filepath)

    @staticmethod
    def load_image_cv2(filepath: str):
        """Load image with cv2."""
        out = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # 2D; (H, W)
        return np.expand_dims(out, axis=2)                # 3D; (H, W, 1)

    @staticmethod
    def decouple_mask(x: torch.Tensor):
        """
        Decouple input with existence mask.
        Defect bins = 2, Normal bins = 1, Null bins = 0
        """
        m = x.gt(0).float()
        x = torch.clamp(x - 1, min=0., max=1.)

        return torch.cat([x, m], dim=0)


class WM811KForDenoising(WM811K):
    def __init__(self, root, transform, target_transform):
        super(WM811KForDenoising, self).__init__(root, transform)
        self.target_transform = target_transform

    def __getitem__(self, idx):

        path, _ = self.samples[idx]
        img = self.load_image(path)

        x = self.transform(img)
        x = self.decouple_mask(x)
        y = self.target_transform(img).long().squeeze(0)  # 3d -> 2d

        return dict(x=x, y=y, idx=idx)


class WM811KForPIRL(WM811K):
    def __init__(self, root, transform=None, positive_transform=None):
        super(WM811KForPIRL, self).__init__(root, transform)
        self.positive_transform = positive_transform

    def __getitem__(self, idx):

        path, y = self.samples[idx]
        img = self.load_image_cv2(path)

        if self.transform is not None:
            x = self.transform(img)

        if self.positive_transform is not None:
            x_t = self.positive_transform(img)

        x = self.decouple_mask(x)
        x_t = self.decouple_mask(x_t)

        return dict(x=x, x_t=x_t, y=y, idx=idx)


class WM811KForSimCLR(WM811K):
    def __init__(self, root, transform=None, proportion: float = 1.0):
        super(WM811KForSimCLR, self).__init__(root, transform, proportion)

    def __getitem__(self, idx):

        path, y = self.samples[idx]
        img = self.load_image_cv2(path)

        if self.transform is not None:
            x1 = self.transform(img)
            x2 = self.transform(img)

        x1 = self.decouple_mask(x1)
        x2 = self.decouple_mask(x2)

        return dict(x1=x1, x2=x2, y=y, idx=idx)
