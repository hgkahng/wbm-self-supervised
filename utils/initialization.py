
# -*- coding: utf-8 -*-

import torch.nn as nn

def initialize_weights(model: nn.Module, activation='relu'):
    """Initialize trainable weights."""

    for _, m in model.named_modules():

        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 1)

        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            nn.init.constant_(m.bias, 0)
