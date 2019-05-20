"""
File: triplet_loss.py
Description: Implementation of margin ranking loss
"""
import numpy as np
import torch
from torch import nn

from absl import flags
FLAGS = flags.FLAGS

class TripletLoss:
    def __init__(self, margin):
        self.loss = nn.MarginRankingLoss(margin=margin)
        self.coords_i, self.coords_j = np.triu_indices(FLAGS.batch_size, 1)

    def __call__(self, feats, labels):
        dist = nn.functional.pdist(feats)
        dist_p = torch.zeros_like(labels)
        dist_n = torch.zeros_like(labels)
        for dis, idx_i, idx_j in zip(dist, self.coords_i, self.coords_j):
            if labels[idx_i] == labels[idx_j]:
                dist_p[idx_i] = max(dist_p[idx_i], dis)
            else:
                dist_n[idx_i] = max(dist_n[idx_i], dis)
        ones = torch.ones_like(labels)
        return self.loss(dist_n, dist_p, ones)
