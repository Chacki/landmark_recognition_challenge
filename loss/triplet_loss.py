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
        self.loss = nn.TripletMarginLoss(margin=margin)

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
