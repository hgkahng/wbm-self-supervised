# -*- coding: utf-8 -*-

import numpy as np

from sklearn.model_selection import train_test_split
from torchvision.datasets.cifar import CIFAR10
from PIL import Image


class CustomCIFAR10(CIFAR10):
    num_classes = 10
    def __init__(self, root, train=True, transform=None, proportion=1.0, **kwargs):
        super(CustomCIFAR10, self).__init__(root=root, 
                                            train=train,
                                            transform=transform,
                                            target_transform=None,
                                            download=False)
        
        assert isinstance(self.data, np.ndarray)
        assert isinstance(self.targets, list)

        self.proportion = proportion
        if self.proportion < 1.0:
            indices, _ = train_test_split(
                np.arange(len(self.data)),
                train_size=self.proportion,
                stratify=self.targets,
                shuffle=True,
                random_state=2020 + kwargs.get('random_seed', 0)
            )
            self.data = self.data[indices]
            self.targets = self.targets[indices]
        else:
            pass

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        
        return dict(x=img, y=target, idx=index)


class CIFAR10ForSimCLR(CIFAR10):
    def __init__(self, root, train=True, transform=None):
        super(CIFAR10ForSimCLR, self).__init__(root=root,
                                               train=train,
                                               transform=transform,
                                               target_transform=None,
                                               download=True)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            x1 = self.transform(img)
            x2 = self.transform(img)
        
        return dict(x1=x1, x2=x2, y=target, idx=index)
        