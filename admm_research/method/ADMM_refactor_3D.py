import re
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import maxflow
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from admm_research.loss import Entropy
from admm_research.models import Segmentator
from admm_research.utils import pred2segmentation
from .ADMM_refactor import AdmmGCSize
from ..dataset.metainfoGenerator import IndividualBoundGenerator
from ..scheduler import customized_scheduler
from admm_research.postprocessing._viewer import multi_slice_viewer


def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape) / 255.0, cdf


class AdmmGCSize3D(AdmmGCSize):

    def __init__(
            self,
            model: Segmentator,
            OptimInnerLoopNum: int = 1,
            ADMMLoopNum: int = 2,
            device: str = 'cpu',
            lamda=0.5,
            sigma=0.005,
            kernel_size=5,
            visualization=False,
            gc_method='method3',
            p_u: float = 10.0,
            p_v: float = 10.0,
            new_eps: float = 0.1,
            weight_scheduler_dict: dict = {},
            balance_scheduler_dict: dict = {},
            gc_scheduler_dict: dict = {},
            *args,
            **kwargs
    ) -> None:
        super().__init__(model, OptimInnerLoopNum, ADMMLoopNum, device, lamda, sigma, kernel_size, visualization,
                         gc_method, p_v, p_u, *args, **kwargs)
        self.new_eps = float(new_eps)
        ## overide the 3D size generator
        self.threeD_size_generator = IndividualBoundGenerator(eps=self.new_eps)
        self.weight_scheduler: customized_scheduler.Scheduler = getattr(customized_scheduler,
                                                                        weight_scheduler_dict['name'])(
            **{k: v for k, v in weight_scheduler_dict.items() if k != "name"})
        self.balance_scheduler: customized_scheduler.Scheduler = getattr(customized_scheduler,
                                                                         balance_scheduler_dict['name'])(
            **{k: v for k, v in balance_scheduler_dict.items() if k != "name"})
        self.gc_scheduler: customized_scheduler.Scheduler = getattr(customized_scheduler, gc_scheduler_dict['name'])(
            **{k: v for k, v in gc_scheduler_dict.items() if k != "name"})

    def step(self):
        self.weight_scheduler.step()
        self.balance_scheduler.step()
        self.weight_scheduler.step()

    def update(self, criterion):
        for iteration in range(self.ADMMLoopNum):
            if self.p_v > 0:
                self._update_s_torch()
            if self.p_u > 0:
                self._update_gamma(self.gc_scheduler.value)
            self._update_theta(criterion)
            if self.p_u > 0:
                self._update_u()
            if self.p_v > 0:
                self._update_v()
            if self.visualization:
                self.show('gamma', fig_num=1)
                self.show('s', fig_num=2)

    def set_input(self, img: torch.Tensor, gt: torch.Tensor, weak_gt: torch.Tensor, bounds: torch.Tensor,
                  paths: Tuple[str] = None, *args, **kwargs):
        self.img: torch.Tensor = img.to(self.device)
        _, _, _, _ = self.img.shape
        self.gt: torch.Tensor = gt.to(self.device)
        _, _, _ = self.gt.shape
        self.prior: torch.Tensor = weak_gt.to(self.device)
        self.ce_prior = self.prior.squeeze(1).clone()
        self.ce_prior[(0 < self.ce_prior) & (self.ce_prior < 1)] = -1
        self.ce_prior = self.ce_prior.long()
        assert set(self.ce_prior.unique().cpu().numpy()).issubset(set([0, 1, -1]))

        self.cropMin = np.array(np.nonzero(self.prior.squeeze(1).cpu()).min(0)[0]) - 1
        self.cropMax = np.array(np.nonzero(self.prior.squeeze(1).cpu()).max(0)[0]) + 1
        self.cropMin = np.array([max(x, 0) for x in self.cropMin])
        self.cropMax = np.array([min(x, self.prior.squeeze(1).shape[i]) for i, x in enumerate(self.cropMax)])

        self.path = [Path(p).stem for p in paths]
        self.parent_path = Path(paths[0]).parent
        self.num_patient = int(re.compile(r'\d+').findall(self.path[0])[0])
        bounds = self.threeD_size_generator(self.gt)
        self.lowbound: torch.Tensor = bounds[0, 1].to(self.device)
        self.highbound: torch.Tensor = bounds[1, 1].to(self.device)
        self.score: torch.Tensor = self.model.predict(img, logit=True)
        _, _, _, _ = self.score.shape
        self.gamma: np.ndarray = np.zeros_like(self.gt.squeeze().cpu())
        # pred2segmentation(self.score).cpu().numpy()
        _, _, _ = self.gamma.shape
        self.s: torch.Tensor = torch.zeros_like(self.gt.squeeze()).to(self.device)
        # pred2segmentation(self.score)  # b, w, h
        _, _, _ = self.s.shape
        self.u = np.zeros_like(self.s.cpu())
        _, _, _ = self.u.shape
        self.v = torch.zeros_like(self.s, dtype=torch.float).to(self.device)  # b w h
        _, _, _ = self.v.shape

    def _update_gamma(self, ratio: float = 0.0):
        cropMin = self.cropMin.astype(int)
        cropMax = self.cropMax.astype(int)

        crop_img = self.img.squeeze().cpu().numpy()[
                   int(cropMin[0]):int(cropMax[0]) + 1,
                   int(cropMin[1]):int(cropMax[1] + 1),
                   int(cropMin[2]):int(cropMax[2] + 1)
                   ]
        # crop_img, _ = image_histogram_equalization(crop_img)
        mask_crop = self.gt.squeeze().cpu().numpy()[
                    int(cropMin[0]):int(cropMax[0] + 1),
                    int(cropMin[1]):int(cropMax[1] + 1),
                    int(cropMin[2]):int(cropMax[2] + 1)
                    ]
        assert crop_img.shape == mask_crop.shape
        priorCrop = self.prior.squeeze(1)[
                    int(cropMin[0]):int(cropMax[0] + 1),
                    int(cropMin[1]):int(cropMax[1] + 1),
                    int(cropMin[2]):int(cropMax[2] + 1)
                    ].cpu().numpy().copy()
        priorCrop[priorCrop >= 1] = 1e6
        priorCrop[priorCrop <= 0] = -1e6

        # priorCrop = np.moveaxis(priorCrop, 2, 0)
        assert crop_img.shape == priorCrop.shape
        g = maxflow.Graph[float](0, 0)
        nodeids = g.add_grid_nodes(list(priorCrop.shape))
        g = self._set_boundary_term(g, nodeids, crop_img, lumda=100, sigma=0.0001, kernelsize=5)

        crop_probability = F.softmax(self.score, 1)[:, 1].detach().cpu().numpy().squeeze()[
                           int(cropMin[0]):int(cropMax[0] + 1),
                           int(cropMin[1]):int(cropMax[1] + 1),
                           int(cropMin[2]):int(cropMax[2] + 1)
                           ]

        g.add_grid_tedges(nodeids, (-0.5 + (1 - ratio) * priorCrop + ratio * crop_probability + self.u[
                                                                                                int(cropMin[0]):int(
                                                                                                    cropMax[0] + 1),
                                                                                                int(cropMin[1]):int(
                                                                                                    cropMax[1] + 1),
                                                                                                int(cropMin[2]):int(
                                                                                                    cropMax[2] + 1)
                                                                                                ]),
                          np.zeros_like(priorCrop))
        g.maxflow()
        sgm = g.get_grid_segments(nodeids) * 1
        crop_gamma = np.int_(np.logical_not(sgm))
        new_gamma = np.zeros_like(self.gamma)
        new_gamma[
        int(cropMin[0]):int(cropMax[0]) + 1,
        int(cropMin[1]):int(cropMax[1] + 1),
        int(cropMin[2]):int(cropMax[2] + 1)
        ] = crop_gamma
        assert self.gamma.shape == new_gamma.shape
        self.gamma = new_gamma

    def _set_boundary_term(self, g, nodeids, img, lumda, sigma, kernelsize):
        lumda = float(lumda)
        sigma = float(sigma)
        kernelsize = int(kernelsize)
        kernel = np.ones((kernelsize, kernelsize, kernelsize))
        kernel[int(kernel.shape[0] / 2), int(kernel.shape[1] / 2), int(kernel.shape[2] / 2)] = 0
        transfer_function = lambda pixel_difference: lumda * np.exp((-1 / sigma) * pixel_difference ** 2)
        # =====new =========================================
        padding_size = int(max(kernel.shape) / 2)
        position = np.array(list(zip(*np.where(kernel != 0))))
        for p in position[:int(len(position) / 2)]:
            structure = np.zeros(kernel.shape)
            structure[p[0], p[1], p[2]] = kernel[p[0], p[1], p[2]]
            pad_im = np.pad(img,
                            ((padding_size, padding_size), (padding_size, padding_size), (padding_size, padding_size)),
                            'constant',
                            constant_values=0)
            shifted_im = self.shift_matrix(pad_im, structure)
            weights_ = transfer_function(
                np.abs(pad_im - shifted_im)
                [
                padding_size:-padding_size,
                padding_size:-padding_size,
                padding_size:-padding_size
                ]
            )

            g.add_grid_edges(nodeids, structure=structure, weights=weights_, symmetric=True)
        return g

    @staticmethod
    def shift_matrix(matrix, kernel):
        center_x, center_y, center_z = int(kernel.shape[0] / 2), int(kernel.shape[1] / 2), int(kernel.shape[2] / 2)
        [kernel_x, kernel_y, kernel_z] = np.array(list(zip(*np.where(kernel == 1))))[0]
        dx, dy, dz = kernel_x - center_x, kernel_y - center_y, kernel_z - center_z
        shifted_matrix = np.roll(matrix, -dx, axis=0)
        shifted_matrix = np.roll(shifted_matrix, -dy, axis=1)
        shifted_matrix = np.roll(shifted_matrix, -dz, axis=2)
        return shifted_matrix

    def _update_s_torch(self):
        s_score = 0.5 - (F.softmax(self.score, 1)[:, 1].squeeze() + self.v)
        _, _, _ = s_score.shape
        original_shape = s_score.shape
        if self.highbound == 0:
            self.s = torch.zeros(original_shape).to(self.device)
            return

        sorted_value, sorted_index = torch.sort(s_score.view(-1))
        useful_pixel_number = (sorted_value < 0).sum()
        if self.lowbound < useful_pixel_number < self.highbound:
            self.s = ((s_score < 0) * 1.0).reshape(original_shape)
        elif useful_pixel_number <= self.lowbound:
            self.s = ((s_score <= sorted_value[self.lowbound + 1]) * 1.0).reshape(original_shape)
        elif useful_pixel_number >= self.highbound:
            self.s = ((s_score <= sorted_value[self.highbound - 1] * 1.0) * 1).reshape(original_shape)
        else:
            raise ('something wrong here.')
        self.s = self.s.detach()
        assert self.s.shape.__len__() == 3

    def _update_theta(self, criterion):
        # for cross entropy

        for i in range(self.OptimInnerLoopNum):

            CE_loss = nn.CrossEntropyLoss(ignore_index=-1)(self.score, self.ce_prior)

            current_n_gamma_p_u = torch.from_numpy(-self.gamma + self.u).float().to(self.device)
            size_loss = self.p_v / 2 * \
                        (F.softmax(self.score, dim=1)[:, 1] + (-self.s.float() + self.v.float())).norm(
                            dim=[1, 2]).mean() ** 2
            size_loss /= list(self.s.reshape(-1).size())[0]
            gamma_loss = self.p_u / 2 * \
                         (F.softmax(self.score, dim=1)[:, 1] + current_n_gamma_p_u).norm(dim=[1, 2]).mean() ** 2
            gamma_loss /= self.gamma.reshape(-1).size

            total_loss = CE_loss + \
                         self.weight_scheduler.value / (self.weight_scheduler.value + 1) * (
                                 self.balance_scheduler.value * size_loss + (
                                 1 - self.balance_scheduler.value) * gamma_loss)

            if self.p_v > 0:
                total_loss -= Entropy()(F.softmax(self.score, 1)).mean() * 0.01

            self.model.optimizer.zero_grad()
            total_loss.backward()
            self.model.optimizer.step()
            self.score = self.model.predict(self.img, logit=True)

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
