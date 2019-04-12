import torch
import numpy as np
from torch import nn
from ..utils import class2one_hot, one_hot
from torch import einsum


class IndividualBoundGenerator(nn.Module):

    def __init__(self, eps=0.1, num_classes=2):
        super().__init__()
        self.eps = eps
        self.num_classes = num_classes

    def forward(self, gt: torch.Tensor):
        assert gt.dtype == torch.int64, f"gt type {gt.dtype} not supported."
        gt_onehot = class2one_hot(gt, self.num_classes)
        assert one_hot(gt_onehot)
        _size = einsum('bchd->c', gt_onehot)
        lowbound = (_size.float() * (1 - self.eps)).long()
        highbound = (_size.float() * (1 + self.eps)).long()
        return torch.stack((lowbound, highbound))


class GlobalBoundGenerator(nn.Module):
    def __init__(self, lowbound=0, highbound=1000):
        super().__init__()
        self.lowbound = torch.Tensor([0, lowbound]).long()
        self.highbound = torch.Tensor([0, highbound]).long()

    def forward(self, gt: torch.Tensor):
        return torch.stack((self.lowbound, self.highbound))


