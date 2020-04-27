# -*- coding: utf-8 -*-

import torch.optim as optim
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def get_optimizer(params, name: str, lr: float, weight_decay: float, **kwargs):
    """Configure optimizer."""

    if name == 'adamw':
        return AdamW(params=params, lr=lr, weight_decay=weight_decay, betas=(kwargs.get('beta1', 0.9), 0.999))
    elif name == 'sgd':
        return SGD(params=params, lr=lr, weight_decay=weight_decay, momentum=kwargs.get('momentum', 0.9), nesterov=False)
    else:
        raise NotImplementedError


def get_scheduler(optimizer: optim.Optimizer, name: str, epochs: int, **kwargs):
    """Configure learning rate scheduler."""

    if name == 'step':
        step_kws = dict(
            step_size=kwargs.get('step_size', int(epochs * 0.8)),
            gamma=kwargs.get('gamma', 0.1)
        )
        return StepLR(optimizer, **step_kws)

    elif name == 'multi_step':
        multi_kws = dict(
            milestones=[epochs // 5 * 3, epochs // 5 * 4],
            gamma=kwargs.get('gamma', 0.1),
        )
        return MultiStepLR(optimizer, **multi_kws)

    elif name == 'plateau':
        plateau_kws = dict(
            factor=kwargs.get('factor', 0.1),
            patience=kwargs.get('patience', 10),
            threshold=kwargs.get('threshold', 1e-4),
            cooldown=kwargs.get('cooldown', 0),
            min_lr=kwargs.get('min_lr', 0),
        )
        return ReduceLROnPlateau(optimizer, **plateau_kws)

    elif name == 'cosine':
        cosine_kws = dict(
            T_max=kwargs.get('T_max', epochs),
            eta_min=kwargs.get('eta_min', 0.)
        )
        return CosineAnnealingLR(optimizer, **cosine_kws)

    elif name == 'restart':
        T_mult = kwargs.get('T_mult')
        if kwargs.get('T_0') is not None:
            T_0 = kwargs.get('T_0')
        else:
            num_restarts = kwargs.get('num_restarts', 4)
            if T_mult == 1:
                T_0 = epochs // num_restarts + 1
            else:
                T_0 = epochs // (T_mult ** num_restarts - 1) + 1
        return CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult)

    else:
        return None
