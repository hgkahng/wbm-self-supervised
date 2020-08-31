# -*- coding: utf-8 -*-

import cv2
import albumentations as A

from datasets.transforms import ToWBM


class CNNWDITransform(object):
    def __init__(self,
                 size: tuple = (96, 96),
                 mode: str = 'test',
                 **kwargs):
        defaults = dict(size=size, mode=mode)
        defaults.update(kwargs)
        self.defaults = defaults

        if mode == 'train':
            transform = [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.2,
                    scale_limit=0.1,
                    rotate_limit=10,
                    interpolation=cv2.INTER_NEAREST,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5
                ),
                A.Resize(*size, interpolation=cv2.INTER_NEAREST),
                ToWBM()
            ]
        elif mode == 'test':
            transform = [
                A.Resize(*size, interpolation=cv2.INTER_NEAREST),
                ToWBM(),
            ]
        self.transform = A.Compose(transform)

    def __call__(self, img):
        return self.transform(image=img)['image']
