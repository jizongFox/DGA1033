import time
from abc import ABC, abstractmethod
from multiprocessing.dummy import Pool
# from torch.multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from functools import partial
from itertools import repeat

from admm_research import ModelMode
from admm_research.models import Segmentator
from admm_research.utils import pred2segmentation
from .gc import _update_gamma


class AdmmBase(ABC):

    def __init__(self, model: Segmentator,
                 OptimInnerLoopNum: int = 1,
                 ADMMLoopNum: int = 2,
                 device='cpu',
                 *args,
                 **kwargs
                 ) -> None:
        super().__init__()
        self.p_v = 10
        self.model = model
        self.OptimInnerLoopNum = OptimInnerLoopNum
        self.ADMMLoopNum = ADMMLoopNum
        self.device = torch.device(device)

    def set_input(self, img, gt, weak_gt, *args, **kwargs):
        pass

    @property
    def save_dict(self):
        return self.torchnet.state_dict()

    @abstractmethod
    def update(self, **kwargs):
        pass

    @abstractmethod
    def _update_theta(self, **kwargs):
        pass

    def set_mode(self, mode):
        assert mode in (ModelMode.TRAIN, ModelMode.EVAL)
        if mode == ModelMode.TRAIN:
            self.model.torchnet.train()
        else:
            self.model.torchnet.eval()

    def show(self, name=None, fig_num=1):
        try:
            getattr(self, name)
        except Exception as e:
            return
        plt.figure(fig_num, figsize=(5, 5))
        plt.clf()
        plt.subplot(1, 1, 1)
        plt.imshow(self.img[0].cpu().data.numpy().squeeze(), cmap='gray')

        plt.contour(self.weak_gt[0].squeeze().cpu().data.numpy(), level=[0], colors="yellow", alpha=0.2,
                    linewidth=0.001,
                    label='GT')
        plt.contour(self.gt[0].squeeze().cpu().data.numpy(), level=[0], colors="yellow", alpha=0.2, linewidth=0.001,
                    label='GT')
        if name is not None and name != 'img':
            try:
                plt.contour(getattr(self, name)[0].detach().cpu(), level=[0], colors="red", alpha=0.2, linewidth=0.001,
                            label=name)
            except AttributeError:
                plt.contour(getattr(self, name)[0], level=[0], colors="red", alpha=0.2, linewidth=0.001,
                            label=name)
        plt.contour(pred2segmentation(self.score)[0].squeeze().cpu().data.numpy(), level=[0],
                    colors="green", alpha=0.2, linewidth=0.001, label='CNN')
        plt.title(name)
        plt.show(block=False)
        plt.pause(0.01)

    def to(self, device):
        self.model.to(device)


class AdmmSize(AdmmBase):

    def __init__(self, model: Segmentator,
                 OptimInnerLoopNum: int = 1,
                 ADMMLoopNum: int = 2,
                 device: str = 'cpu'
                 ) -> None:
        super().__init__(model, OptimInnerLoopNum, ADMMLoopNum, device)

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

    def update(self, criterion):
        for iteration in range(self.ADMMLoopNum):
            self._update_s_torch()
            self._update_theta(criterion)
            self._update_v()
            # self.show('s', fig_num=1)
            # self.show('v', fig_num=2)
            # time1 = time.time()
            # results = _multiprocess_Call(self.img.cpu().numpy().squeeze(1), F.softmax(self.score.detach(), 1).cpu().numpy(),
            #                          self.v.cpu().numpy(), self.gt.cpu().numpy(), self.weak_gt.cpu().numpy())
            # print(f'time used for gc of batch {self.img.shape[0]}: {time.time()-time1}')

    def _update_s_torch(self):
        s_score = 0.5 - (F.softmax(self.score, 1)[:, 1].squeeze() + self.v)
        original_shape = s_score[0].shape
        for i, a in enumerate(s_score):
            if self.highbound[i] == 0:
                self.s[i] = torch.zeros(original_shape).to(self.device)
                continue

            sorted_value, sorted_index = torch.sort(a.view(-1))
            useful_pixel_number = (sorted_value < 0).sum()
            if self.lowbound[i] < useful_pixel_number and self.highbound[i] > useful_pixel_number:
                self.s[i] = ((a < 0) * 1.0).reshape(original_shape)
            elif useful_pixel_number <= self.lowbound[i]:
                self.s[i] = ((a <= sorted_value[self.lowbound[i] + 1]) * 1.0).reshape(original_shape)
            elif useful_pixel_number >= self.highbound[i]:
                self.s[i] = ((a <= sorted_value[self.highbound[i] - 1] * 1.0) * 1).reshape(original_shape)
            else:
                raise ('something wrong here.')
        self.s = self.s.detach()
        assert self.s.shape.__len__() == 3

    def _update_theta(self, criterion):
        self.score = self.model.predict(self.img, logit=True)

        for i in range(self.OptimInnerLoopNum):
            CE_loss = criterion(self.score, self.weak_gt.squeeze(1).long())
            unlabled_loss = self.p_v / 2 * (
                    F.softmax(self.score, dim=1)[:, 1] + (-self.s.float() + self.v.float())).norm(
                dim=[1, 2]).mean() ** 2

            unlabled_loss /= list(self.s.reshape(-1).size())[0]

            loss = CE_loss + unlabled_loss
            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.optimizer.step()
            self.score = self.model.predict(self.img, logit=True)
            # print(f'CE:{CE_loss.item()}, unlabeled:{unlabled_loss.item()}')

    def _update_v(self):
        self.v = self.v + (F.softmax(self.score, dim=1)[:, 1].squeeze().detach() - self.s.float()) * 0.1
        assert self.v.shape.__len__() == 3


class AdmmGCSize(AdmmSize):

    def __init__(self, model: Segmentator, OptimInnerLoopNum: int = 1, ADMMLoopNum: int = 2,
                 device: str = 'cpu', lamda=10, sigma=0.02) -> None:
        super().__init__(model, OptimInnerLoopNum, ADMMLoopNum, device)
        self.lamda = lamda
        self.sigma = sigma
        self.p_u = 10

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
        self.gamma: np.ndarray = pred2segmentation(self.score).cpu().numpy()
        _, _, _ = self.gamma.shape
        self.s: torch.Tensor = pred2segmentation(self.score)  # b, w, h
        _, _, _ = self.s.shape
        self.u = np.zeros_like(self.s.cpu())
        _, _, _ = self.u.shape
        self.v = torch.zero
        s_like(self.s, dtype=torch.float).to(self.device)  # b w h
        _, _, _ = self.v.shape

    def update(self, criterion):
        for iteration in range(self.ADMMLoopNum):
            self._update_s_torch()
            self._update_gamma()
            self._update_theta(criterion)
            self._update_u()
            self._update_v()
            self.show('gamma',fig_num=1)
            self.show('s',fig_num=2)

    def _update_gamma(self):
        new_gamma = _multiprocess_Call(self.img.cpu().numpy().squeeze(1),
                                       F.softmax(self.score.detach(), 1).cpu().numpy(),
                                       self.u, self.gt.cpu().numpy(), self.weak_gt.cpu().numpy(), self.lamda,
                                       self.sigma)
        new_gamma = np.stack(new_gamma, axis=0)
        _, _, _ = new_gamma.shape
        self.gamma = new_gamma

    def update_2(self, criterion):
        for iteration in range(self.ADMMLoopNum):
            self._update_theta(criterion)
            self._update_u()
            self._update_v()

    def _update_theta(self, criterion):
        for i in range(self.OptimInnerLoopNum):
            current_n_gamma_p_u = torch.from_numpy(-self.gamma + self.u).float().to(self.device)

            CE_loss = criterion(self.score, self.weak_gt.squeeze(1).long())
            size_loss = self.p_v / 2 * \
                        (F.softmax(self.score, dim=1)[:, 1] + (-self.s.float() + self.v.float())).norm(
                            dim=[1, 2]).mean() ** 2
            gamma_loss = self.p_u / 2 * \
                         (F.softmax(self.score, dim=1)[:, 1] + current_n_gamma_p_u).norm(dim=[1,2]).mean() ** 2

            unlabled_loss = (size_loss+gamma_loss)/ list(self.s.reshape(-1).size())[0]

            loss = CE_loss + unlabled_loss
            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.optimizer.step()
            self.score = self.model.predict(self.img,logit=True)

    def _update_u(self):
        new_u:np.ndarray = self.u + (F.softmax(self.score, dim=1)[:, 1].cpu().data.numpy().squeeze() - self.gamma) * 0.01
        assert new_u.shape.__len__() == 3
        self.u = new_u


# helper function to call graphcut
def _multiprocess_Call(imgs, scores, us, gts, weak_gts, lamda, sigma):
    P = Pool()
    results = P.starmap(Update_gamma, zip(imgs, scores, us, gts, weak_gts, repeat(lamda), repeat(sigma)))
    P.close()
    return results

Update_gamma = partial(_update_gamma, kernelsize=3)
