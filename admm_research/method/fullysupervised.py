from torch import nn as nn
from admm_research.loss import CrossEntropyLoss2d
from .ADMM import AdmmBase


class FullySupervisedWrapper(AdmmBase):
    @classmethod
    def setup_arch_flags(cls):
        super().setup_arch_flags()

    def __init__(self, torchnet: nn.Module, hparams: dict) -> None:
        super().__init__(torchnet, hparams)
        self.criterion = CrossEntropyLoss2d()

    def reset(self, image):
        pass

    def update(self, **kwargs):
        pass

    def update_1(self, img_gt_wgt):
        (img, gt, wgt) = img_gt_wgt
        self.optim.zero_grad()
        pred = self.torchnet(img)
        loss = self.criterion(pred, gt.squeeze(1))
        loss.backward()
        self.optim.step()

    def update_2(self, a):
        pass

    def _update_theta(self):
        pass

    def forward_img(self, **kwargs):
        pass
