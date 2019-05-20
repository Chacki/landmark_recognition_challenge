"""
File: triplet_loss.py
Description: Implementation of margin ranking loss
"""
import torch
from absl import flags
from torch import nn

FLAGS = flags.FLAGS


class TripletLoss:
    def __init__(self, margin):
        self.loss = nn.MarginRankingLoss(margin=margin)
        self.coords_i, self.coords_j = torch.triu_indices(FLAGS.batch_size, 1)

    def __call__(self, feats, labels):
        N = labels.size(0)
        dist = torch.cdist(feats, feats)
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap, relative_p_inds = torch.max(
            dist[is_pos].contiguous().view(N, -1), 1, keepdim=True
        )
        # `dist_an` means distance(anchor, negative)
        # both `dist_an` and `relative_n_inds` with shape [N, 1]
        dist_an, relative_n_inds = torch.min(
            dist[is_neg].contiguous().view(N, -1), 1, keepdim=True
        )
        # shape [N]
        dist_ap = dist_ap.squeeze(1)
        dist_an = dist_an.squeeze(1)
        ones = torch.ones_like(labels)
        return self.loss(dist_an, dist_ap, ones)
