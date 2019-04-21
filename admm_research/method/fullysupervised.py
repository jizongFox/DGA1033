from typing import *

import torch
from torch import nn as nn
from ..dataset.metainfoGenerator import IndividualBoundGenerator
from admm_research.models import Segmentator
from .ADMM_refactor import AdmmBase


class FullySupervisedWrapper(AdmmBase):

    def __init__(self, model: Segmentator, device='cpu', *args,
                 **kwargs) -> None:
        super().__init__(model, device=device, *args, **kwargs)
        self.criterion = nn.CrossEntropyLoss()
        print(f'loss used here is {self.criterion}')

    def set_input(self, img, gt, weak_gt, bounds, *args, **kwargs):
        self.img: torch.Tensor = img.to(self.device)
        _, _, _, _ = self.img.shape
        self.gt: torch.Tensor = gt.to(self.device)
        _, _, _ = self.gt.shape
        self.weak_gt: torch.Tensor = weak_gt.to(self.device)
        self.lowbound: torch.Tensor = bounds[:, 0].to(self.device)
        self.highbound: torch.Tensor = bounds[:, 1].to(self.device)
        self.score: torch.Tensor = self.model.predict(img, logit=True)
        _, _, _, _ = self.score.shape
        # self.s: torch.Tensor = pred2segmentation(self.score)  # b, w, h
        # _, _, _ = self.s.shape
        # self.v = torch.zeros_like(self.s, dtype=torch.float).to(self.device)  # b w h
        # _, _, _ = self.v.shape

    def update(self, *args, **kwargs):
        self.model.optimizer.zero_grad()
        pred = self.model.predict(self.img, logit=True)
        loss = self.criterion(pred, self.gt.squeeze(1))
        loss.backward()
        self.model.optimizer.step()


class Soft3DConstrainedWrapper(FullySupervisedWrapper):
    def __init__(self, model: Segmentator, device='cpu', new_eps=0.1,*args, **kwargs) -> None:
        super().__init__(model, device, *args, **kwargs)
        self.new_eps = new_eps
        self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor([0, 1]), ignore_index=-1).to(self.device)
        self.threeD_size_generator = IndividualBoundGenerator(eps=self.new_eps)

    def set_input(self, img: torch.Tensor, gt: torch.Tensor, weak_gt: torch.Tensor, bounds: torch.Tensor,
                  paths: Tuple[str] = None, *args, **kwargs):
        self.img: torch.Tensor = img.to(self.device)
        _, _, _, _ = self.img.shape
        self.gt: torch.Tensor = gt.to(self.device)
        _, _, _ = self.gt.shape
        self.score: torch.Tensor = self.model.predict(img, logit=True)
        self.prior: torch.Tensor = weak_gt.to(self.device)
        self.ce_prior = self.prior.squeeze(1).clone()
        self.ce_prior[(0 < self.ce_prior) & (self.ce_prior < 1)] = -1
        self.ce_prior = self.ce_prior.long()
        assert set(self.ce_prior.unique().cpu().numpy()).issubset(set([0, 1, -1]))

        bounds = self.threeD_size_generator(self.gt)
        self.lowbound: torch.Tensor = bounds[0, 1].to(self.device).float()
        self.highbound: torch.Tensor = bounds[1, 1].to(self.device).float()

    def update(self, *args, **kwargs):
        self.model.optimizer.zero_grad()
        pred = self.model.predict(self.img, logit=True)
        partialCELoss = self.criterion(pred, self.ce_prior)
        softFGsize = pred[:, 1].sum()
        if self.lowbound <= softFGsize <= self.highbound:
            sizeLoss = 0
        elif softFGsize > self.highbound:
            sizeLoss = (softFGsize - self.highbound) ** 2 / (pred.view(0,-1).size(0)/2)
        elif softFGsize < self.lowbound:
            sizeLoss = (softFGsize - self.lowbound) ** 2 / (pred.view(-1).size(0)/2)
        else:
            raise ValueError

        loss = partialCELoss + sizeLoss
        loss.backward()
        self.model.optimizer.step()
