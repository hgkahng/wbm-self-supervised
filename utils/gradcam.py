# -*- coding: utf-8 -*-

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

layer_finders = {}

def register_layer_finder(model_type):
    def register(func):
        layer_finders[model_type] = func
        return func
    return register


@register_layer_finder('vgg')
def find_vgg_layer(model, target_layer_name):
    """
    Examples of supported names are as follows:
        'block1.conv1'
        'layers.block1.conv1'
        ...
    """
    target_layer_name = target_layer_name.replace('layers.', '')
    for name, module in model.layers.named_modules():
        if name == target_layer_name:
            return module
    raise KeyError(f"Target layer '{target_layer_name}' does not exist.")

@register_layer_finder('alex')
def find_alex_layer(model, target_layer_name):
    raise NotImplementedError

@register_layer_finder('res')
def find_res_layer(model, target_layer_name):
    raise NotImplementedError


class GradCAM(object):
    def __init__(self, backbone: nn.Module, classifier: nn.Module, target_layer: nn.Module):
        self.backbone = backbone
        self.classifier = classifier
        self.gradients = {}
        self.activations = {}

        def forward_hook(m, input, output):
            # A forward hook is executed during the forward pass.
            self.activations['value'] = output

        def backward_hook(m, grad_input, grad_output):
            # A backward hook is executed during the backward pass.
            self.gradients['value'] = grad_output[0]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def __call__(self, x, class_idx=None, retain_graph=False):
        return self.forward(x, class_idx, retain_graph)

    def forward(self, x, class_idx=None, retain_graph=False):
        assert x.ndim == 4, "(B, C, H, W)"
        b, _, h, w = x.size()

        logit = self.classifier(self.backbone(x))
        if class_idx is None:
            index = torch.argmax(logit, dim=1, keepdim=True)
            score = torch.gather(logit, 1, index)
        else:
            index = torch.as_tensor(class_idx).view(-1, 1)
            score = torch.gather(logit, 1, class_idx)

        self.backbone.zero_grad()
        self.classifier.zero_grad()
        score.mean().backward(retain_graph=retain_graph)
        gradients = self.gradients['value']      # (B, k, u, v)
        activations = self.activations['value']  # (B, k, u, v)
        b, k, _, _ = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights * activations).sum(1, keepdim=True)  # (B, 1, u, v)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)

        saliency_min = saliency_map.view(b, -1).min(dim=1)[0].view(b, 1, 1, 1)
        saliency_max = saliency_map.view(b, -1).max(dim=1)[0].view(b, 1, 1, 1)
        saliency_map = (saliency_map - saliency_min).div(saliency_max - saliency_min).data

        return saliency_map, logit

    @classmethod
    def from_config(cls, backbone: nn.Module, classifier: nn.Module, model_type: str, layer_name: str):
        target_layer = layer_finders[model_type](backbone, layer_name)
        return cls(backbone, classifier, target_layer)

    @staticmethod
    def visualize_cam(smap, img, optional_mask=None, alpha=1.0):
        """Make heatmap from saliency map and synthesize GradCAM result image using heatmap and input.
        Arguments:
            smap (torch.tensor): saliency map of shape (1, H, W), range in [0, 1].
            img (torch.tensor): img shape of (1, H, W), range in [0, 1].
            optional_mask (torch.tensor): mask of shape (1, H, W), range in [0, 1].
        Returns:
            img (torch.tensor): img of shape (3, H, W)
            heatmap (torch.tensor): heatmap of shape (3, H, W).
            result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
        """
        assert smap.ndim == img.ndim == 3, "(1, H, W)"
        heatmap = (255 * smap.squeeze()).type(torch.uint8).cpu().numpy()       # (H, W)    ~ [0, 255]
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)                 # (H, W, 3) ~ [0, 225]
        heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)  # (3, H, W) ~ [0,   1]
        b, g, r = heatmap.split(1)                                             # 3 x (1, H, W)
        heatmap = torch.cat([r, g, b], dim=0) * alpha                          # (3, H, W)

        # The wafer image must be normalized as well
        if optional_mask is not None:
            assert optional_mask.ndim == 3, "(1, H, W)"
            img = (img + optional_mask) / 2                # [0, 1] + [0, 1] -> [0, 2] -> 1/2 -> [0, 1]
        result = heatmap + img.cpu()
        result = result.div(result.max()).squeeze()        # (1, 3, H, W) -> (3, H, W) ~ [0, 1]

        return img.cpu().repeat(3, 1, 1), heatmap, result  # equal shape of (3, H, W)


class GradCAMpp(GradCAM):
    # https://github.com/vickyliin/gradcam_plus_plus-pytorch/blob/4aa264b3ecb7c145ab1618572ffff3ab837e663b/gradcam/gradcam.py#L93
    def forward(self, x, class_idx=None, retain_graph=False):
        raise NotImplementedError
