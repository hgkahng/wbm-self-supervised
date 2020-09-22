# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from entmax import sparsemax, entmax15


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


class PIRLLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super(PIRLLoss, self).__init__()
        self.temperature = temperature
        self.similarity_1d = nn.CosineSimilarity(dim=1)
        self.similarity_2d = nn.CosineSimilarity(dim=2)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    def forward(self,
                anchors: torch.Tensor,
                positives: torch.Tensor,
                negatives: torch.Tensor):

        assert anchors.size() == positives.size()
        batch_size, _ = anchors.size()
        num_negatives, _ = negatives.size()

        negatives = negatives.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, numN, F)
        negatives = negatives.detach()

        # Similarity between queries & positive samples
        sim_a2p = self.similarity_1d(anchors, positives)             # (B, )
        sim_a2p = sim_a2p.div(self.temperature).unsqueeze(1)         # (B, 1)

        # Similarity between positive & negative samples
        sim_a2n = self.similarity_2d(
            positives.unsqueeze(1).repeat(1, num_negatives, 1),      # (B, numN, F)
            negatives                                                # (B, numN, F)
        )
        sim_a2n = sim_a2n.div(self.temperature)                      # (B, numN)

        # Get class logits
        logits = torch.cat([sim_a2p, sim_a2n], dim=1)                # (B, 1 + numN)

        # Get cross entropy loss
        loss = self.cross_entropy(
            logits,                                                  # (B, 1 + numN)
            torch.zeros(logits.size(0)).long().to(logits.device)     # (B, )
        )

        return loss, logits.detach()


class SimCLRLoss(nn.Module):
    """
    Modified implementation of the following:
        https://github.com/HobbitLong/SupContrast
    """
    def __init__(self, temperature: float = 0.07, contrast_mode: str = 'all', reduction: str = 'mean'):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.reduction = reduction
        assert self.reduction in ['mean', 'sum', 'none']

    def forward(self, features: torch.Tensor, labels: torch.Tensor = None, mask: torch.Tensor = None):

        if features.ndim < 3:
            raise ValueError("Expecting `features` to be a shape of (B, N, ...)")
        if features.ndim > 3:
            features = features.view(features.size(0), features.size(1), -1)

        batch_size, num_views, _ = features.size()                                # (B, N, F)
        device = features.device

        # Normalize features such that |f| = |f'| = 1.
        # Note that normalization must be performed befoce reshaping.
        features = nn.functional.normalize(features, dim=-1)

        # The `mask` is an indicator for the positive examples. 1 if positive, 0 if negative.
        # In most cases, the `mask` will have values of 1 along the diagonal, implying that
        # two or more views of the same sample must map to a nearby manifold.
        if labels is not None and mask is not None:
            raise ValueError("Only one of `labels` and `mask` should be specified.")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)      # (B, B)
        elif labels is not None:
            # Use only with fully supervised data (100% labels)
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Number of labels does not match the number of samples.")
            mask = torch.eq(labels, labels.T).float()
        else:
            mask = mask.float().to(device)

        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)       # (B, N, F) -> (N x B, F)

        # `anchor_features` has shape of (K x B, F)
        if self.contrast_mode == 'one':
            anchor_features = features[:, 0]                                      # (K x B, F),
            anchor_count = 1                                                      # where K = 1.
        elif self.contrast_mode == 'all':
            anchor_features = contrast_features                                   # (K x B, F),
            anchor_count = num_views                                              # where K = N.
        else:
            raise NotImplementedError

        # Compute logits (a.k.a similarity scores)
        anchor_dot_contrast = torch.matmul(anchor_features, contrast_features.T)  # (K x B, F) @ (F, N x B)
        anchor_dot_contrast.div_(self.temperature)                                # (K x B, N x B)

        # For numerical stability, subtract the largest logit value
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)       # (K x B, N x B) -> (K x B, 1)
        logits = anchor_dot_contrast - logits_max.detach()                        # (K x B, N x B) - (K x B, 1)

        # Tile mask (B, B) -> (K x B, N x B)
        mask = mask.repeat(anchor_count, num_views)

        # Mask out self-contrasts. The `logits_mask` is also used to exclude
        # self-contrasts from being calculated in the denominator.
        # Now, the mask will have zero entries along the diagonal.
        logits_mask = torch.ones_like(mask)  # (K x B, N x B)
        logits_mask = torch.scatter(
            logits_mask,
            1,
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1),   # (K x B, 1)
            0
        )
        mask = mask * logits_mask                                                 # (K x B, N x B)

        # Compute log probabilities
        exp_logits = torch.exp(logits) * logits_mask                              # (K x B, N x B)
        log_prob   = logits - torch.log(exp_logits.sum(1, keepdim=True))          # (K x B, N x B) - (K x B, 1)

        # Compute mean of log likelihood over positives
        mean_log_prob_pos = (log_prob * mask).sum(1) / mask.sum(1)                # (K x B, )

        # Loss (negative log likelihood)
        loss = torch.neg(mean_log_prob_pos)
        loss = loss.view(anchor_count, batch_size)                                # (K, B)

        if self.reduction == 'mean':
            return loss.mean(), logits, mask
        elif self.reduction == 'sum':
            return loss.sum(), logits, mask
        elif self.reduction is 'none':
            return loss, logits, mask
        else:
            raise NotImplementedError

    @staticmethod
    def semisupervised_mask(unlabeled_size: int, labels: torch.Tensor):
        """Create mask for semi-supervised contrastive learning."""

        labels = labels.view(-1, 1)
        labeled_size = labels.size(0)
        mask_size = unlabeled_size + labeled_size
        mask = torch.zeros(mask_size, mask_size, dtype=torch.float32).to(labels.device)

        L = torch.eq(labels, labels.T).float()
        mask[unlabeled_size:, unlabeled_size:] = L
        U = torch.eye(unlabeled_size, dtype=torch.float32).to(labels.device)
        mask[:unlabeled_size, :unlabeled_size] = U
        mask.clamp_(0, 1)  # Just in case. This might not be necessary.

        return mask


class AttnCLRLoss(nn.Module):
    """
    Loss function for attention-based contrastive learning.
    Arguments:
        temperature: float, default = 0.07.
        reduction: str, default = mean.
    Returns:

    """
    def __init__(self,
                 temperature: float = 0.07,
                 gamma: float = 1.0,
                 reduction: str = 'mean'):
        super(AttnCLRLoss, self).__init__()

        self.temperature = temperature
        self.gamma = gamma
        self.reduction = reduction
        assert self.reduction in ['mean', 'sum', 'none']

    def forward(self,
                features: torch.Tensor,
                attention_scores: torch.Tensor,
                labels: torch.Tensor = None,
                mask: torch.Tensor = None):

        if features.ndim < 3:
            raise ValueError("Expecting `features` to be a shape of (B, N, ...)")
        if features.ndim > 3:
            features = features.view(features.size(0), features.size(1), -1)

        batch_size, num_views, _ = features.size()
        device = features.device

        if labels is not None and mask is not None:
            raise ValueError("Only one of `labels` and `mask` should be specified.")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Shape of `labels` does not match the batch size.")
            mask = torch.eq(labels, labels.T).float()
        else:
            mask = mask.float().to(device)

        # Normalize features such that |f| = |f'| = 1
        features = nn.functional.normalize(features, dim=-1)

        # Reshape features
        features = features.view(num_views * batch_size, -1)

        # Compute similarity scores
        similarities = torch.matmul(features, features.T)  # (N x B, F) @ (F, N x B)
        similarities.div_(self.temperature)                # (N x B, N x B)

        # Subtract the largest value, for numerical stability
        similarities = similarities - torch.max(similarities, dim=-1, keepdim=True)[0].detach()

        mask = mask.repeat(num_views, num_views)  # Tile mask; (B, B) -> (N x B, N x B)
        positive_mask = mask - torch.eye(mask.size(0), device=device)  # mask - diagonals
        negative_mask = 1. - mask                                      # reciprocal of the mask
        
        # Refine attention scores. Before re-normalization, remove self-contrast & positive-pairs' scores.
        masked_scores = attention_scores * negative_mask
        masked_scores.div_(self.temperature)
        masked_scores = sparsemax(masked_scores, dim=-1)
        # masked_scores.div_(masked_scores.sum(dim=-1, keepdim=True) + 1e-12)

        # Compute exponentials of similarities, to be used as the denominator
        exp_similarities = torch.exp(similarities)
        exp_similarities = exp_similarities * (positive_mask + negative_mask - masked_scores)

        # Compute log probabilities
        log_prob = similarities - torch.log(exp_similarities.sum(dim=-1, keepdim=True))

        # Compute mean of log-likelihood over positives
        mean_log_prob_pos = (log_prob * positive_mask).sum(dim=1) / positive_mask.sum(dim=1)  # usually 1

        # Entropy penalty on the masked attention scores
        #entropy = masked_scores * torch.log(masked_scores + 1e-12) * negative_mask
        #entropy = torch.neg(entropy.sum(dim=1))

        # We will be minimizing the negative mean log-likelihood over positives
        #loss = torch.neg(mean_log_prob_pos) + entropy * self.gamma
        loss = torch.neg(mean_log_prob_pos)

        if self.reduction == 'mean':
            return loss.mean(), masked_scores
        elif self.reduction == 'sum':
            return loss.sum(), masked_scores
        elif self.reduction is 'none':
            return loss.view(batch_size, num_views), masked_scores
        else:
            raise NotImplementedError
