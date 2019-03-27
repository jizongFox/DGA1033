from torch import nn as nn
from admm_research.loss import CrossEntropyLoss2d
from admm_research.models import Segmentator
from admm_research.utils import pred2segmentation
from .ADMM_refactor import AdmmBase
import torch


class FullySupervisedWrapper(AdmmBase):

    def __init__(self, model: Segmentator, device='cpu', *args,
                 **kwargs) -> None:
        super().__init__(model, device=device, *args, **kwargs)
        self.criterion = nn.CrossEntropyLoss()
        print(f'loss used here is {self.criterion}')

    def set_input(self, img, gt, weak_gt, bounds):
        self.img: torch.Tensor = img.to(self.device)
        _, _, _, _ = self.img.shape
        self.gt: torch.Tensor = gt.to(self.device)
        _, _, _ = self.gt.shape
        self.weak_gt: torch.Tensor = weak_gt.to(self.device)
        self.lowbound: torch.Tensor = bounds[:, 0].to(self.device)
        self.highbound: torch.Tensor = bounds[:, 1].to(self.device)
        self.score: torch.Tensor = self.model.predict(img, logit=True)
        _, _, _, _ = self.score.shape
        self.s: torch.Tensor = pred2segmentation(self.score)  # b, w, h
        _, _, _ = self.s.shape
        self.v = torch.zeros_like(self.s, dtype=torch.float).to(self.device)  # b w h
        _, _, _ = self.v.shape

    def update(self, *args, **kwargs):
        self.model.optimizer.zero_grad()
        pred = self.model.predict(self.img, logit=True)
        loss = self.criterion(pred, self.gt.squeeze(1))
        loss.backward()
        self.model.optimizer.step()
