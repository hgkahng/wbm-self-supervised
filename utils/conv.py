# -*- coding: utf-8 -*-

from collections.abc import Iterable


def conv_output_shape(hw, kernel_size=1, stride=1, pad=0, dilation=1):
    """Computes output shape of 2D conv operation."""
    if not isinstance(hw, Iterable):
        assert isinstance(hw, int)
        hw = (hw, hw)
    else:
        assert len(hw) == 2

    if not isinstance(kernel_size, Iterable):
        assert isinstance(kernel_size, int)
        kernel_size = (kernel_size, kernel_size)
    else:
        assert len(kernel_size) == 2

    if not isinstance(stride, Iterable):
        assert isinstance(stride, int)
        stride = (stride, stride)
    else:
        assert len(stride) == 2

    if not isinstance(pad, Iterable):
        assert isinstance(pad, int)
        pad = (pad, pad)
    else:
        assert len(pad) == 2

    h = (
        hw[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1
    ) // stride[0] + 1
    w = (
        hw[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1
    ) // stride[1] + 1

    return h, w


def transpose_conv_output_shape(hw, kernel_size=1, stride=1, pad=0, dilation=1):
    """Computes output shape of 2D conv transpose operation."""
    if not isinstance(hw, Iterable):
        assert isinstance(hw, int)
        hw = (hw, hw)
    else:
        assert len(hw) == 2

    if not isinstance(kernel_size, Iterable):
        assert isinstance(kernel_size, int)
        kernel_size = (kernel_size, kernel_size)
    else:
        assert len(kernel_size) == 2

    if not isinstance(stride, Iterable):
        assert isinstance(stride, int)
        stride = (stride, stride)
    else:
        assert len(stride) == 2

    if not isinstance(pad, Iterable):
        assert isinstance(pad, int)
        pad = (pad, pad)
    else:
        assert len(pad) == 2

    h = (hw[0] - 1) * stride[0] - (2 * pad[0]) + (dilation * kernel_size[0])
    w = (hw[1] - 1) * stride[1] - (2 * pad[1]) + (dilation * kernel_size[1])

    return h, w


if __name__ == '__main__':

    HEIGHT, WIDTH = (96, 96)
    KERNEL_SIZE = 7
    STRIDE = 2
    PAD = 3
    DILATION = 1

    print("Kernel size:", KERNEL_SIZE)
    print("Stride:", STRIDE)
    print("Padding:", PAD)
    print("Dilation:", DILATION)
    print("2D convolution operation shape check:")
    output_shape = conv_output_shape(HEIGHT, KERNEL_SIZE, STRIDE, PAD, DILATION)
    print(f"INPUT SHAPE: ({HEIGHT}, {WIDTH}) -> OUTPUT SHAPE: ({output_shape[0]}, {output_shape[1]})")

    print("2D transpose convolution operation shape check:")
    output_shape = transpose_conv_output_shape(HEIGHT, KERNEL_SIZE, STRIDE, PAD, DILATION)
    print(f"INPUT SHAPE: ({HEIGHT}, {WIDTH}) -> OUTPUT SHAPE: ({output_shape[0]}, {output_shape[1]})")
