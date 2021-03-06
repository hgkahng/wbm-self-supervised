# -*- coding: utf-8 -*-

import math
import torch
import torch.optim as optim
from torch.optim import Optimizer, SGD, AdamW
from torch.optim.lr_scheduler import StepLR, LambdaLR


class LARS(Optimizer):
    """
    Implementation of `Large Batch Training of Convolutional Networks, You et al`.
    Code is borrowed from the following:
        https://github.com/cmpark0126/pytorch-LARS
    """
    def __init__(self, params, lr, momentum=0., weight_decay=0., trust_coef=1.):
        if lr < 0.:
            raise ValueError(f"Invalid `lr` value: {lr}")
        if momentum < 0.:
            raise ValueError(f"Invalid `momentum` value: {momentum}")
        if weight_decay < 0.:
            raise ValueError(f"Invalid `weight_decay` value: {weight_decay}")
        if trust_coef < 0.:
            raise ValueError(f"Invalid `trust_coef` value: {trust_coef}")

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, trust_coef=trust_coef)

        super(LARS, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LARS, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step."""

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum     = group['momentum']
            trust_coef   = group['trust_coef']
            global_lr    = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                p_norm = torch.norm(p.data, p=2)
                d_p_norm = torch.norm(d_p, p=2)

                local_lr = torch.div(p_norm, d_p_norm + weight_decay * p_norm)
                local_lr.mul_(trust_coef)

                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf

                p.data.add_(d_p, alpha=-(global_lr * local_lr))

        return loss


class WarmupCosineSchedule(LambdaLR):
    """Linear warmup & cosine decay.
       Implementation from `pytorch_transformers.optimization.WarmupCosineSchedule`.
       Assuming that the initial learning rate of `optimizer` is set to 1., this scheduler
       linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
       Decreases learning rate for 1. to 0. over remaining `t_total - warmup_steps` following a cosine curve.
       If `cycles` (default=0.5) is different from default, learning r ate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, min_lr=1e-4, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        self.min_lr = min_lr
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return self.min_lr + float(step) / float(max(1.0, self.warmup_steps))
        # Progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return self.min_lr + max(0., 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


class WarmupCosineWithHardRestartsSchedule(LambdaLR):
    """ Linear warmup and then cosine cycles with hard restarts.
        Implementation from `pytorch_transformers.optimization.WarmupCosineWithHardRestartsSchedule`.
        Assuming that the initial learning rate of `optimizer` is set to 1., this scheduler
        linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        If `cycles` (default=1.) is different from default, learning rate follows `cycles` times a cosine decaying
        learning rate (with hard restarts).
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=1., last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineWithHardRestartsSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        # Progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1. + math.cos(math.pi * ((float(self.cycles) * progress) % 1.0))))


def get_optimizer(params, name: str, lr: float, weight_decay: float, **kwargs):
    """Configure optimizer."""
    
    if name == 'adamw':
        return AdamW(params=params, lr=lr, weight_decay=weight_decay)
    elif name == 'sgd':
        return SGD(params=params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif name == 'lars':
        trust_coef = kwargs.get('trust_coef', 1.)
        return LARS(params=params, lr=lr, momentum=0.9, weight_decay=weight_decay, trust_coef=trust_coef)
    else:
        raise NotImplementedError


def get_scheduler(optimizer: optim.Optimizer, name: str, epochs: int, **kwargs):
    """Configure learning rate scheduler."""
    if name == 'step':
        step_size = kwargs.get('milestone', epochs // 10 * 9)
        gamma = kwargs.get('gamma', 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif name == 'cosine':
        warmup_steps = kwargs.get('warmup_steps', epochs // 10)
        return WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, t_total=epochs)
    elif name == 'restart':
        warmup_steps = kwargs.get('warmup_steps', epochs // 10)
        cycles = kwargs.get('cycles', 4)
        return WarmupCosineWithHardRestartsSchedule(optimizer, warmup_steps=warmup_steps, t_total=epochs, cycles=cycles)
    else:
        return None
