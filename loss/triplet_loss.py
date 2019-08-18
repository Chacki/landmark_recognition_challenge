"""
File: triplet_loss.py
Description: Implementation of margin ranking loss
"""
import torch
from torch import nn


class DistanceWeightedSampling:
    def __init__(self, batch_k, cutoff=0.5, nonzero_loss_cutoff=1.4):
        self.batch_k = batch_k
        self.cutoff = cutoff
        self.nonzero_loss_cutoff = nonzero_loss_cutoff

    def get_triplets(self, feats):
        with torch.no_grad():
            n = feats.size(0)
            d = feats.size(1)
            k = self.batch_k
            dist = torch.cdist(feats, feats).clamp(min=self.cutoff)
            log_weights = (
                (2.0 - d) * torch.log(dist) - (d - 3) / 2
            ) * torch.log(1 - 0.25 * dist ** 2)
            weights = torch.exp(log_weights - torch.max(log_weights))

            mask = torch.ones_like(weights)
            for i in range(0, n, k):
                mask[i : i + k, i : i + k] = 0

            weights = (
                (weights + 1e-7)
                * mask
                * ((dist < self.nonzero_loss_cutoff).float() + 1e-7)
            )

            n_indices = torch.multinomial(weights, k - 1).view(-1)
            a_indices = torch.arange(n).unsqueeze(-1).repeat(1, k - 1).view(-1)
            p_indices = (
                torch.arange(n)
                .view(n // k, -1)
                .unsqueeze(1)
                .repeat(1, k, 1)
                .masked_select(~torch.diag_embed(torch.ones(n // k, k).bool()))
            )
        return a_indices, p_indices, n_indices


class TripletLoss:
    def __init__(self, margin):
        self.loss = nn.TripletMarginLoss(margin=margin)
        self.selector = DistanceWeightedSampling(batch_k=4)

    def __call__(self, feats, labels):
        a_idx, p_idx, n_idx = self.selector.get_triplets(feats)
        loss = self.loss(feats[a_idx], feats[p_idx], feats[n_idx])
        return loss


class OnlineHardMining:
    """ Triplet loss which samples both, hard negative and hard postive samples.
    """

    def __init__(self, margin):
        self.loss = nn.TripletMarginLoss(margin=margin, swap=True)

    def __call__(self, feats, labels):
        with torch.no_grad():
            N = labels.size(0)
            dist = torch.cdist(feats, feats)
            is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
            is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
            # `dist_ap` means distance(anchor, positive)
            # both `dist_ap` and `relative_p_inds` with shape [N, 1]
            relative_p_inds = torch.argmax(
                dist[is_pos].contiguous().view(N, -1), 1, keepdim=True
            )
            # `dist_an` means distance(anchor, negative)
            # both `dist_an` and `relative_n_inds` with shape [N, 1]
            relative_n_inds = torch.argmin(
                dist[is_neg].contiguous().view(N, -1), 1, keepdim=True
            )
        # shape [N]
        loss = self.loss(feats, feats[relative_p_inds], feats[relative_n_inds])
        return loss


class OnlineHardNegativeMining:
    """ Triplet loss with online hard negative mining.
        Positive samples are given through preprocessing.
    """

    def __init__(self, margin):
        self.loss = nn.TripletMarginLoss(margin=margin)

    def __call__(self, feats, labels):
        """
        feats and labels are reshaped matching pairs
        [id0, id0, id1, id1, ...]
        """
        assert (labels[::2] == labels[1::2]).all()
        with torch.no_grad():
            N = labels.size(0)
            dist = torch.cdist(feats, feats)
            is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
            # `dist_an` means distance(anchor, negative)
            # both `dist_an` and `relative_n_inds` with shape [N, 1]
            relative_n_inds = torch.argmin(
                dist[is_neg].contiguous().view(N, -1), 1, keepdim=True
            )
        # shape [N]
        loss = self.loss(
            feats[::2], feats[1::2], feats[relative_n_inds].squeeze()[::2]
        )
        return loss
