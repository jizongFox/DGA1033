from typing import *

import torch
import torch.nn.functional as F
from torch import nn as nn
import matplotlib.pyplot as plt
from admm_research.models import Segmentator
from .ADMM_refactor import AdmmBase
from ..dataset.metainfoGenerator import IndividualBoundGenerator
from ..metrics2 import AverageValueMeter
from ..utils import pred2segmentation


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
    def __init__(self, model: Segmentator, device='cpu', new_eps=0.1, *args, **kwargs) -> None:
        super().__init__(model, device, *args, **kwargs)
        self.new_eps = new_eps
        self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor([0, 1]), ignore_index=-1).to(self.device)
        self.threeD_size_generator = IndividualBoundGenerator(eps=self.new_eps)
        self.ce_loss_Meter = AverageValueMeter()
        self.size_Meter = AverageValueMeter()

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
        partialCELoss = self.criterion(self.score, self.ce_prior)
        self.softpred = F.softmax(self.score, 1)[:, 1]
        softFGsize = self.softpred.sum()
        if self.lowbound <= softFGsize <= self.highbound:
            sizeLoss = torch.tensor(0)
        elif softFGsize > self.highbound:
            sizeLoss = (softFGsize - self.highbound) ** 2 / float((self.score.view(-1).size(0)))
        elif softFGsize < self.lowbound:
            sizeLoss = (softFGsize - self.lowbound) ** 2 /  float((self.score.view(-1).size(0)))
        else:
            raise ValueError

        loss = partialCELoss + 0.01* sizeLoss
        loss.backward()
        self.model.optimizer.step()
        self.size_Meter.add(sizeLoss.item())
        self.ce_loss_Meter.add(partialCELoss.item())

        if self.visualization:
            self.show('ce_prior', fig_num=1)

        # plt.figure(3)
        # plt.clf()
        # plt.imshow(self.img[int(self.img.shape[0] / 2)].cpu().data.numpy().squeeze(), cmap='gray')
        # plt.imshow(self.softpred[int(self.img.shape[0] / 2)].cpu().data.numpy().squeeze(), alpha=0.5)
        # plt.colorbar()
        # plt.show(block=False)
        # plt.pause(0.01)
        # print(f'size loss:{sizeLoss.item()}, CE_loss:{partialCELoss.item()}, mean_FG:{self.softpred.mean()}')

    def show(self, name=None, fig_num=1):
        try:
            getattr(self, name)
        except Exception as e:
            return
        plt.figure(fig_num, figsize=(5, 5))
        plt.clf()
        plt.subplot(1, 1, 1)
        plt.imshow(self.img[int(self.img.shape[0] / 2)].cpu().data.numpy().squeeze(), cmap='gray')

        plt.contour(self.gt[int(self.img.shape[0] / 2)].squeeze().cpu().data.numpy(), level=[0], colors="yellow",
                    alpha=0.2, linewidth=0.001,
                    label='GT')
        if name is not None and name != 'img':
            try:
                plt.contour(getattr(self, name)[int(self.img.shape[0] / 2)].detach().cpu(), level=[0], colors="red",
                            alpha=0.2, linewidth=0.001,
                            label=name)
            except AttributeError:
                plt.contour(getattr(self, name)[int(self.img.shape[0] / 2)], level=[0], colors="red", alpha=0.2,
                            linewidth=0.001,
                            label=name)
        plt.contour(pred2segmentation(self.score)[int(self.img.shape[0] / 2)].squeeze().cpu().data.numpy(), level=[0],
                    colors="green", alpha=0.2, linewidth=0.001, label='CNN')
        plt.title(name)
        plt.show(block=False)
        plt.pause(0.01)
