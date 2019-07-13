"""
File: center_loss.py
Description: Center loss
"""

import torch
from torch import nn


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feature_dim):
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))
        self.loss = nn.MSELoss()

    def forward(self, feats, labels):
        centers = self.centers.gather(dim=0, index=labels)
        loss = self.loss(feats, centers)
        return loss
