# -*- coding: utf-8 -*-

from torchvision.datasets import STL10


if __name__ == '__main__':
    dataset = STL10('./data/stl10/', download=True)
