# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """
    Implementation of cross entropy loss with label smoothing.
    Follows the implementation of the two followings:
        https://github.com/pytorch/pytorch/issues/7455#issuecomment-513062631
        https://github.com/pytorch/pytorch/issues/7455#issuecomment-513735962
    """
    def __init__(self, num_classes, smoothing=.0, dim=1, reduction='mean', class_weights=None):
        """
        Arguments:
            num_classes: int, specifying the number of target classes.
            smoothing: float, default value of 0 is equal to general cross entropy loss.
            dim: int, aggregation dimension.
            reduction: str, default 'mean'.
            class_weights: 1D tensor of shape (C, ) or (C, 1).
        """
        super(LabelSmoothingLoss, self).__init__()
        assert 0 <= smoothing < 1
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.dim = dim

        assert reduction in ['sum', 'mean']
        self.reduction = reduction

        self.class_weights = class_weights

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Arguments:
            pred: 2D torch tensor of shape (B, C)
            target: 1D torch tensor of shape (B, )
        """
        pred = F.log_softmax(pred, dim=self.dim)
        true_dist = self.smooth_one_hot(target, self.num_classes, self.smoothing)
        multiplied = -true_dist * pred

        if self.class_weights is not None:
            weights = self.class_weights.to(multiplied.device)
            summed = torch.matmul(multiplied, weights.view(self.num_classes, 1))  # (B, C) @ (C, 1) -> (B, 1)
            summed = summed.squeeze()                                             # (B, 1) -> (B, )
        else:
            summed = torch.sum(multiplied, dim=self.dim)                          # (B, C) -> sum -> (B, )

        if self.reduction == 'sum':
            return summed
        elif self.reduction == 'mean':
            return torch.mean(summed)
        else:
            raise NotImplementedError

    @staticmethod
    def smooth_one_hot(target: torch.Tensor, num_classes: int, smoothing=0.):
        assert 0 <= smoothing < 1
        confidence = 1. - smoothing
        label_shape = torch.Size((target.size(0), num_classes))
        with torch.no_grad():
            true_dist = torch.zeros(label_shape, device=target.device)
            true_dist.fill_(smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), confidence)
        return true_dist  # (B, C)


class SoftF1Loss(nn.Module):
    def __init__(self, num_classes: int, smoothing: float = 0.0, reduction: str = 'mean', class_weights: torch.Tensor = None, epsilon=1e-7):
        super(SoftF1Loss, self).__init__()

        self.num_classes = num_classes
        self.smoothing = smoothing
        self.epsilon = epsilon

        assert reduction in ['sum', 'mean']
        self.reduction = reduction

        if class_weights is None:
            self.class_weights = torch.ones(1, self.num_classes)
        else:
            self.class_weights = class_weights.view(1, self.num_classes)

    def forward(self, logits, target):

        assert logits.ndim == 2
        assert target.ndim == 1

        pred = F.softmax(logits, dim=1)
        true = LabelSmoothingLoss.smooth_one_hot(target, self.num_classes, smoothing=self.smoothing)
        assert pred.size() == true.size()

        class_weights = self.class_weights.to(pred.device)

        tp = ((  true) * (  pred)).sum(dim=0) * class_weights
        _  = ((1-true) * (1-pred)).sum(dim=0) * class_weights
        fp = ((1-true) * (  pred)).sum(dim=0) * class_weights
        fn = ((  true) * (1-pred)).sum(dim=0) * class_weights

        precision = tp / (tp + fp + self.epsilon)
        recall    = tp / (tp + fn + self.epsilon)

        f1 = 2 * precision * recall / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)

        if self.reduction == 'sum':
            return 1 - f1.mean(dim=1)
        elif self.reduction == 'mean':
            return 1 - f1.mean()
        else:
            raise NotImplementedError


class BinaryF1Loss(nn.Module):
    """
    Reference:
        https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354#gistcomment-3055663
    """
    def __init__(self, epsilon=1e-7):
        super(BinaryF1Loss, self).__init__()
        self.epsilon = epsilon

    def forward(self, logits, target):

        assert logits.ndim == 2
        assert target.ndim == 1

        pred = F.softmax(logits, dim=1)
        true = F.one_hot(target, 2).to(torch.float32)
        assert pred.size() == true.size()

        tp = (true * pred).sum(dim=0)
        _  = ((1-true) * (1-pred)).sum(dim=0)
        fp = ((1-true) * pred).sum(dim=0)
        fn = (true * (1-pred)).sum(dim=0)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * precision * recall / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)  # (1, C)

        return 1 - f1.mean()


class NCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(NCELoss, self).__init__()
        self.temperature = temperature
        self.similarity_1d = nn.CosineSimilarity(dim=1)
        self.similarity_2d = nn.CosineSimilarity(dim=2)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    def forward(self,
                queries: torch.Tensor,
                positives: torch.Tensor,
                negatives: torch.Tensor):

        assert queries.size() == positives.size()
        batch_size, _ = queries.size()
        num_negatives, _ = negatives.size()

        negatives = negatives.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, numN, F)
        negatives = negatives.detach()

        # Similarity between queries & positive samples
        sim_q2p = self.similarity_1d(queries, positives)             # (B, )
        sim_q2p = sim_q2p.div(self.temperature).unsqueeze(1)         # (B, 1)

        # Similarity between positive & negative samples
        sim_p2n = self.similarity_2d(
            positives.unsqueeze(1).repeat(1, num_negatives, 1),      # (B, numN, F)
            negatives                                                # (B, numN, F)
        )
        sim_p2n = sim_p2n.div(self.temperature)                      # (B, numN)

        # Get class logits
        logits = torch.cat([sim_q2p, sim_p2n], dim=1)                # (B, 1 + numN)

        # Get cross entropy loss
        loss = self.cross_entropy(
            logits,                                                  # (B, 1 + numN)
            torch.zeros(logits.size(0)).long().to(logits.device)     # (B, )
        )

        return loss
