"""
File: center_loss.py
Description: Center loss
"""

import torch
from torch import nn


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feature_dim, sparse=False):
        super().__init__()

        self.centers = nn.Embedding(num_classes, feature_dim, sparse=sparse)
        self.loss = nn.MSELoss()

    def forward(self, feats, labels):
        centers = self.centers(labels.squeeze())
        # centers = nn.functional.normalize(centers)
        # feats = nn.functional.normalize(feats)
        # loss = (1 - (feats * centers).sum(-1)).mean()
        loss = self.loss(feats, centers)
        return loss
