# -*- coding: utf-8 -*-

import cv2
import torch
import albumentations as A
import numpy as np

from torchvision.transforms import Compose
from torchvision.transforms import ToTensor, Normalize
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip
from torchvision.transforms import RandomApply, RandomChoice
from torchvision.transforms import ColorJitter, RandomGrayscale
from albumentations.core.transforms_interface import BasicTransform
from albumentations.core.transforms_interface import ImageOnlyTransform

from torch.distributions import Bernoulli
from kornia.filters import GaussianBlur2d, MedianBlur


class RandomFlip(object):
    def __init__(self, p: float):
        self.p = p
        self.transform = RandomChoice(
            [
                RandomHorizontalFlip(self.p),
                RandomVerticalFlip(self.p),
            ]
        )

    def __call__(self, x: torch.Tensor):
        return self.transform(x)


class ToWBM(BasicTransform):
    def __init__(self, always_apply: bool = True, p: float = 1.0):
        super(ToWBM, self).__init__(always_apply, p)

    @property
    def targets(self):
        return {"image": self.apply}

    def apply(self, img: np.ndarray, **kwargs):  # pylint: disable=unused-argument
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = img[:, :, None]
            img = torch.from_numpy(img.transpose(2, 0, 1))
            if isinstance(img, torch.ByteTensor):
                img = img.float().div(255)
        return torch.ceil(img * 2)

    def get_transform_init_args_names(self):
        return []

    def get_params_dependent_on_targets(self, params):
        return {}


class MaskedBernoulliNoise(ImageOnlyTransform):
    def __init__(self, noise: float, always_apply: bool = False, p: float = 1.0):
        super(MaskedBernoulliNoise, self).__init__(always_apply, p)
        self.noise = noise
        self.min_ = 0
        self.max_ = 1
        self.bernoulli = Bernoulli(probs=noise)

    def apply(self, x: torch.Tensor, **kwargs):  # pylint: disable=unused-argument
        assert x.ndim == 3
        m = self.bernoulli.sample(x.size()).to(x.device)
        m = m * x.gt(0).float()
        noise_value = 1 + torch.randint_like(x, self.min_, self.max_ + 1).to(x.device)  # 1 or 2
        return x * (1 - m) + noise_value * m

    def get_params(self):
        return {'noise': self.noise}


class MaskedGaussianBlur(object):
    """Applies gaussian filter to the input."""
    def __init__(self,
                 kernel_size: int or tuple = 3,
                 sigma: float or tuple = 1.,
                 border_type: str = 'reflect'):
        super(MaskedGaussianBlur, self).__init__()

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            assert isinstance(kernel_size, tuple)
            self.kernel_size = kernel_size
        if isinstance(sigma, float):
            self.sigma = (sigma, sigma)
        else:
            assert isinstance(sigma, tuple)
            self.sigma = sigma
        self.gblur = GaussianBlur2d(
            kernel_size=self.kernel_size,
            sigma=(sigma, sigma),
            border_type=border_type
        )

    def __call__(self, x: torch.Tensor):
        """
        Arguments:
            x: 3d (1, H, W) torch tensor. 2 for defects, 1 for normal, 0 for out-of-border regions.
        """
        with torch.no_grad():
            x = x.unsqueeze(0)      # (1, H, W) -> (1, 1, H, W)
            m = x.gt(0).float()     # 1 for valid wafer regions, 0 for out-of-border regions
            out = x * (1 - m) + self.gblur(x) * m
            return out.squeeze(0)   # (1, 1, H, W) -> (1, H, W)


class MaskedMedianBlur(object):
    """Applies median filter to the input."""
    def __init__(self, kernel_size: int or tuple = 3):
        super(MaskedMedianBlur, self).__init__()
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            assert isinstance(kernel_size, tuple)
            self.kernel_size = kernel_size
        self.mblur = MedianBlur(kernel_size=self.kernel_size)

    def __call__(self, x: torch.Tensor):
        """
        Arguments:
            x: 3d (1, H, W) torch tensor. 1 for defects, 0 for normal, 0 for out-of-border regions.
        """
        with torch.no_grad():
            x = x.unsqueeze(0)      # (1, H, W) -> (1, 1, H, W)
            m = x.gt(0).float()     # 1 for valid regions, 0 for out-of-border regions
            out = x * (1 - m) + self.mblur(x) * m
            return out.squeeze(0)   # (1, 1, H, W) -> (1, H, W)


class ImageNetTransform(object):
    def __init__(self, size: tuple = (224, 224), mode: str = 'pretrain'):
        raise NotImplementedError


class STL10Transform(object):
    def __init__(self, size: tuple = (96, 96), mode: str = 'pretrain'):
        raise NotImplementedError


class CIFAR10Transform(object):
    def __init__(self, size: tuple = (32, 32), mode: str = 'pretrain', **kwargs):
        if mode == 'pretrain':
            self.transforms = Compose(
                [
                    RandomResizedCrop(size=size),
                    RandomHorizontalFlip(p=0.5),
                    RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    RandomGrayscale(p=0.2),
                    ToTensor(),
                    Normalize(mean=self.mean, std=self.std)
                ]
            )
        elif mode == 'test':
            self.transforms = Compose(
                [
                    ToTensor(),
                    Normalize(mean=self.mean, std=self.std)
                ]
            )
        else:
            raise NotImplementedError

        self.size = size
        self.mode = mode

    def __call__(self, img):
        return self.transforms(img)

    def __repr__(self):
        return self.__class__.__name__ + f'({self.size}, {self.mode})'

    @property
    def mean(self):
        return [0.4914, 0.4822, 0.4465]

    @property
    def std(self):
        return [0.2023, 0.1994, 0.2010]


class WM811KTransform(object):
    """Transformations for wafer bin maps from WM-811K."""
    def __init__(self,
                 size: tuple = (96, 96),
                 mode: str = 'test',
                 **kwargs):

        defaults = dict(
            size=size,
            mode=mode,
        )
        defaults.update(kwargs)
        self.defaults = defaults

        if mode == 'crop+rotate':
            transform = self.crop_rotate_transform(**defaults)
        elif mode == 'shift+crop+rotate':
            transform = self.shift_crop_rotate_transform(**defaults)
        elif mode == 'shift+crop+rotate+cutout+noise':
            transform = self.shift_crop_rotate_cutout_noise_transform(**defaults)
        elif mode == 'rotate':
            transform = self.rotate_transform(**defaults)
        elif mode == 'crop':
            transform = self.crop_transform(**defaults)
        elif mode == 'shear':
            transform = self.shear_transform(**defaults)
        elif mode == 'shift':
            transform = self.shift_transform(**defaults)
        elif mode == 'noise':
            transform = self.noise_transform(**defaults)
        elif mode == 'cutout':
            transform = self.cutout_transform(**defaults)
        elif mode == 'test':
            transform = [
                A.Resize(*size, interpolation=cv2.INTER_NEAREST),
                ToWBM(),
            ]
        else:
            raise NotImplementedError

        self.transform = A.Compose(transform)

    def __call__(self, img):
        return self.transform(image=img)['image']

    def __repr__(self):
        repr_str = self.__class__.__name__
        for k, v in self.defaults.items():
            repr_str += f"\n{k}: {v}"
        return repr_str

    @staticmethod
    def crop_transform(size: tuple,
                       scale: tuple = (0.5, 1.0),
                       ratio: tuple = (0.9, 1.1),
                       **kwargs):  # pylint: disable=unused-argument
        """
        Crop-based augmentation, with `albumentations`.
        Expects a 3D numpy array of shape [H, W, C] as input.
        """
        transform = [
            A.Flip(p=0.5),
            A.RandomResizedCrop(*size, scale=scale, ratio=ratio, interpolation=cv2.INTER_NEAREST, p=1.0),
            ToWBM(),
        ]

        return transform

    @staticmethod
    def rotate_transform(size: tuple, **kwargs):  # pylint: disable=unused-argument
        """
        Rotation-based augmentation, with `albumentations`.
        Expects a 3D numpy array of shape [H, W, C] as input.
        """
        transform = [
            A.Flip(p=0.5),
            A.Rotate(limit=180, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, p=1.0),
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            ToWBM(),
        ]

        return transform

    @staticmethod
    def shift_transform(size: tuple,
                        shift: float = 0.25,
                        **kwargs):  # pylint: disable=unused-argument
        transform = [
            A.Flip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=shift,
                scale_limit=0,
                rotate_limit=0,
                interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1.0
            ),
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            ToWBM(),
        ]

        return transform

    @staticmethod
    def cutout_transform(size: tuple, num_holes: int = 4, **kwargs):  # pylint: disable=unused-argument
        cut_h = size[0] // 5
        cut_w = size[1] // 5
        transform = [
            A.Flip(p=0.5),
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            A.Cutout(num_holes=num_holes, max_h_size=cut_h, max_w_size=cut_w, fill_value=0, p=0.5),
            ToWBM()
        ]

        return transform

    @staticmethod
    def noise_transform(size: tuple, noise: float = 0.05, **kwargs):  # pylint: disable=unused-argument
        if noise <= 0.:
            raise ValueError("'noise' probability must be larger than zero.")
        transform = [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            ToWBM(),
            MaskedBernoulliNoise(noise=noise),
        ]

        return transform

    @staticmethod
    def crop_rotate_transform(size: tuple,
                              scale: tuple = (0.5, 1.0),
                              ratio: tuple = (0.9, 1.1),
                              **kwargs):  # pylint: disable=unused-argument
        """
        Rotation & crop-based augmentation, with `albumentations`.
        Expects a 3D numpy array of shape [H, W, C] as input.
        """
        transform = [
            A.Flip(p=0.5),
            A.Rotate(limit=180, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, p=1.0),
            A.RandomResizedCrop(*size, scale=scale, ratio=ratio, interpolation=cv2.INTER_NEAREST, p=1.0),
            ToWBM(),
        ]

        return transform

    @staticmethod
    def shift_crop_rotate_transform(size: tuple,
                                    shift: float = 0.25,
                                    scale: tuple = (0.5, 1.0),
                                    ratio: tuple = (0.9, 1.1),
                                    **kwargs):  # pylint: disable=unused-argument
        transform = [
            A.Flip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=shift,
                scale_limit=0,
                rotate_limit=0,
                interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.9,
            ),
            A.Rotate(limit=180, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, p=1.0),
            A.RandomResizedCrop(*size, scale=scale, ratio=ratio, interpolation=cv2.INTER_NEAREST, p=1.0),
            ToWBM(),
        ]

        return transform

    @staticmethod
    def shift_crop_rotate_cutout_noise_transform(size: tuple,
                                                 shift: float = 0.25,
                                                 scale: tuple = (0.5, 1.0),
                                                 ratio: tuple = (0.9, 1.1),
                                                 num_holes: int = 4,
                                                 noise: float = 0.05,
                                                 **kwargs):  # pylint: disable=unused-argument
        cut_h = size[0] // 5
        cut_w = size[1] // 5
        transform = [
            A.Flip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=shift,
                scale_limit=0,
                rotate_limit=0,
                interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.75,
            ),
            A.Rotate(limit=180, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, p=1.0),
            A.RandomResizedCrop(*size, scale=scale, ratio=ratio, interpolation=cv2.INTER_NEAREST, p=1.0),
            A.Cutout(num_holes=num_holes, max_h_size=cut_h, max_w_size=cut_w, fill_value=0, p=0.5),
            ToWBM(),
            MaskedBernoulliNoise(noise=noise),
        ]

        return transform


def get_transform(data: str, size: int or tuple, mode: str, **kwargs):
    """Get data transformation module by name string."""
    assert data in ['wm811k', 'cifar10', 'stl10', 'imagenet']
    if isinstance(size, int):
        size = (size, size)
    if data == 'wm811k':
        return WM811KTransform(size=size, mode=mode, **kwargs)
    elif data == 'cifar10':
        return CIFAR10Transform(size=size, mode=mode, **kwargs)
    elif data == 'stl10':
        raise NotImplementedError
    elif data == 'imagenet':
        raise NotImplementedError
    else:
        raise NotImplementedError
