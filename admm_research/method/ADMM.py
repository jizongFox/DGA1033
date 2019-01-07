from abc import ABC, abstractmethod
from enum import Enum

import cv2
import matplotlib.pyplot as plt
import maxflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from admm_research import flags
from admm_research.utils import AverageMeter, dice_loss, pred2segmentation, extract_from_big_dict, dice_batch, \
    probs2one_hot, class2one_hot
from admm_research import LOGGER

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ModelMode(Enum):
    """ Different mode of model """
    TRAIN = 'TRAIN'  # during training
    EVAL = 'EVAL'  # eval mode. On validation data
    PRED = 'PRED'

    @staticmethod
    def from_str(mode_str):
        """ Init from string
            :param mode_str: ['train', 'eval', 'predict']
        """
        if mode_str == 'train':
            return ModelMode.TRAIN
        elif mode_str == 'eval':
            return ModelMode.EVAL
        elif mode_str == 'predict':
            return ModelMode.PRED
        else:
            raise ValueError('Invalid argument mode_str {}'.format(mode_str))


class AdmmBase(ABC):
    optim_hparam_keys = ['lr', 'weight_decay', 'amsgrad', 'optim_inner_loop_num']
    arch_hparam_keys = ['arch', 'num_classes']

    @classmethod
    def setup_arch_flags(cls):
        """ Setup the arch_hparams """
        flags.DEFINE_float('weight_decay', default=0, help='decay of learning rate schedule')
        flags.DEFINE_float('lr', default=0.001, help='learning rate')
        flags.DEFINE_boolean('amsgrad', default=False, help='amsgrad')
        flags.DEFINE_integer('optim_inner_loop_num', default=5, help='optim_inner_loop_num')
        flags.DEFINE_string('arch', default='enet', help='arch_name')
        flags.DEFINE_integer('num_classes', default=2, help='num of classes')
        flags.DEFINE_string('method', default='admm_gc_size', help='arch_name')
        flags.DEFINE_boolean('ignore_negative', default=False, help='ignore negative examples in the training')

    def __init__(self, torchnet: nn.Module, hparams: dict) -> None:
        super().__init__()
        self.hparams = hparams
        self.p_u = 10
        self.p_v = 10
        optim_hparams = extract_from_big_dict(hparams, AdmmBase.optim_hparam_keys)
        self.torchnet = torchnet
        self.optim_inner_loop_num = optim_hparams['optim_inner_loop_num']
        optim_hparams.pop('optim_inner_loop_num')
        self.optim = torch.optim.Adam(self.torchnet.parameters(), **optim_hparams)

    @abstractmethod
    def forward_img(self, img, gt, weak_gt):
        assert img.size(0) == 1, 'batchsize of 1 is permitted, given %d' % img.size(0)
        self.img = img
        self.gt = gt
        self.weak_gt = weak_gt
        self.img_size = torch.sum(gt)
        self.score = self.torchnet(img)

    @property
    def save_dict(self):
        return self.torchnet.state_dict()

    @abstractmethod
    def update(self, **kwargs):
        pass

    @abstractmethod
    def _update_theta(self, **kwargs):
        pass

    @abstractmethod
    def reset(self, **kwargs):
        pass

    def set_mode(self, mode):
        assert mode in (ModelMode.TRAIN, ModelMode.EVAL)
        if mode == ModelMode.TRAIN:
            self.torchnet.train()
        else:
            self.torchnet.eval()

    def show(self, name=None, fig_num=1):
        try:
            getattr(self, name)
        except Exception as e:
            return
        plt.figure(fig_num, figsize=(5, 5))
        plt.clf()
        plt.subplot(1, 1, 1)
        plt.imshow(self.img[0].cpu().data.numpy().squeeze(), cmap='gray')

        plt.contour(self.weak_gt.squeeze().cpu().data.numpy(), level=[0], colors="yellow", alpha=0.2, linewidth=0.001,
                    label='GT')
        plt.contour(self.gt.squeeze().cpu().data.numpy(), level=[0], colors="yellow", alpha=0.2, linewidth=0.001,
                    label='GT')
        if name is not None:
            plt.contour(getattr(self, name), level=[0], colors="red", alpha=0.2, linewidth=0.001, label=name)
        plt.contour(pred2segmentation(self.score).squeeze().cpu().data.numpy(), level=[0],
                    colors="green", alpha=0.2, linewidth=0.001, label='CNN')
        plt.title(name)
        plt.show(block=False)
        plt.pause(0.01)

    def evaluate(self, dataloader, mode='3Ddice'):
        b_dice_meter = AverageMeter()
        f_dice_meter = AverageMeter()
        threeD_dice = AverageMeter()
        # self.torchnet.eval()
        datalaoder_original_state = dataloader.dataset.training
        dataloader.dataset.set_mode('eval')
        assert dataloader.dataset.training == ModelMode.EVAL
        assert self.torchnet.training == True

        with torch.no_grad():

            for i, (image, mask, weak_mask, pathname) in enumerate(dataloader):
                if self.hparams['ignore_negative']:
                    if weak_mask.sum() == 0 or mask.sum() == 0:
                        continue
                image, mask, weak_mask = image.to(device), mask.to(device), weak_mask.to(device)
                proba = F.softmax(self.torchnet(image), dim=1)
                predicted_mask = proba.max(1)[1]
                [b_iou, f_iou] = dice_loss(predicted_mask, mask)

                if mode == '3Ddice':
                    predicted_mask = probs2one_hot(proba)
                    mask_oh = class2one_hot(mask.squeeze(1), 2)
                    batch_dice = dice_batch(predicted_mask, mask_oh)
                    threeD_dice.update(batch_dice[1], 1)

                b_dice_meter.update(b_iou, image.size(0))
                f_dice_meter.update(f_iou, image.size(0))

        self.torchnet.train()
        dataloader.dataset.set_mode(datalaoder_original_state)
        assert dataloader.dataset.training == datalaoder_original_state
        assert self.torchnet.training == True
        return b_dice_meter.avg, f_dice_meter.avg, threeD_dice.avg

    def to(self, device):
        self.torchnet.to(device)


class AdmmSize(AdmmBase):
    size_hparam_keys = ['individual_size_constraint', 'eps', 'global_upbound', 'global_lowbound']
    optim_hparam_keys = [] + AdmmBase.optim_hparam_keys

    @classmethod
    def setup_arch_flags(cls):
        super().setup_arch_flags()
        flags.DEFINE_boolean('individual_size_constraint', default=True,
                             help='Individual size constraint for each input image')
        flags.DEFINE_float('eps', default=0.2, help='eps for individual size constraint')
        flags.DEFINE_integer('global_upbound', default=2000,
                             help='global upper bound if individual_size_constraint is False')
        flags.DEFINE_integer('global_lowbound', default=20,
                             help='global lower bound if individual_size_constraint is False')

    def __init__(self, torchnet: nn.Module, hparams: dict) -> None:
        super().__init__(torchnet, hparams)
        size_hparams = extract_from_big_dict(hparams, AdmmSize.size_hparam_keys)
        self.individual_size_constraint = size_hparams['individual_size_constraint']
        if self.individual_size_constraint:
            self.eps = size_hparams['eps']
        else:
            self.upbound = size_hparams['global_upbound']
            self.lowbound = size_hparams['global_lowbound']

    def reset(self, img, gt, wg):
        self.s = np.zeros(img.squeeze().shape)
        self.v = np.zeros(img.squeeze().shape)
        self.initilize = False

    def initialize_dummy_variables(self, score):
        self.s = pred2segmentation(score).cpu().data.numpy().squeeze()  # b, w, h
        self.v = np.zeros(list(self.s.shape))  # b w h
        self.initilize = True

    def forward_img(self, img, gt, weak_gt):
        super().forward_img(img, gt, weak_gt)
        if self.individual_size_constraint:
            self.upbound = int((1.0 + self.eps) * self.img_size.item())
            self.lowbound = int((1.0 - self.eps) * self.img_size.item())

    def update(self, img_gt_weakgt, criterion):
        self.forward_img(*img_gt_weakgt)
        if self.initilize == False:
            self.initialize_dummy_variables(self.score)
        self._update_s()
        self._update_theta(criterion)
        self._update_v()

    def update_1(self, img_gt_weakgt):
        self.forward_img(*img_gt_weakgt)
        if self.initilize == False:
            self.initialize_dummy_variables(self.score)
        self._update_s()

    def update_2(self, criterion):
        self._update_theta(criterion)
        self._update_v()

    def _update_s(self):
        if self.weak_gt.sum() == 0 or self.gt.sum() == 0:
            self.s = np.zeros(self.img.squeeze().shape)
            return

        a = 0.5 - (F.softmax(self.score, 1)[:, 1].cpu().data.numpy().squeeze() + self.v)
        original_shape = a.shape
        a_ = np.sort(a.ravel())
        useful_pixel_number = (a < 0).sum()
        if self.lowbound < useful_pixel_number and self.upbound > useful_pixel_number:
            self.s = ((a < 0) * 1.0).reshape(original_shape)
        elif useful_pixel_number <= self.lowbound:
            self.s = ((a <= a_[self.lowbound + 1]) * 1.0).reshape(original_shape)
        elif useful_pixel_number >= self.upbound:
            self.s = ((a <= a_[self.upbound - 1] * 1.0) * 1).reshape(original_shape)
        else:
            raise ('something wrong here.')
        assert self.s.shape.__len__() == 2
        LOGGER.debug('low_band:{},up_band:{},realsize:{}, new_S_size:{}'.format(self.lowbound,self.upbound,self.img_size,self.s.sum()))
        # assert self.lowbound <= self.s.sum() <= self.upbound

    def _update_theta(self, criterion):

        for i in range(self.optim_inner_loop_num):
            CE_loss = criterion(self.score, self.weak_gt.squeeze(1).long())
            unlabled_loss = self.p_v / 2 * (
                    F.softmax(self.score, dim=1)[:, 1] + torch.from_numpy(-self.s + self.v).float().to(
                device)).norm(p=2) ** 2

            unlabled_loss /= list(self.score.reshape(-1).size())[0]

            loss = CE_loss + unlabled_loss
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            self.forward_img(self.img, self.gt, self.weak_gt) ## update self.score

    def _update_v(self):
        new_v = self.v + (F.softmax(self.score, dim=1)[:, 1, :, :].cpu().data.numpy().squeeze() - self.s) * 0.1
        self.v = new_v


class AdmmGCSize(AdmmSize):
    size_hparam_keys = [] + AdmmSize.size_hparam_keys
    optim_hparam_keys = [] + AdmmSize.optim_hparam_keys
    gc_hparam_keys = ['lamda', 'sigma', 'kernelsize', 'dilation_level', 'stop_dilation_epoch']

    @classmethod
    def setup_arch_flags(cls):
        super().setup_arch_flags()
        flags.DEFINE_float('lamda', default=1,
                           help='balance between the unary and the neighor term')
        flags.DEFINE_float('sigma', default=0.02, help='Smooth the neigh term')
        flags.DEFINE_integer('kernelsize', default=5,
                             help='kernelsize of the gc')
        flags.DEFINE_integer('dilation_level', default=7,
                             help='iterations to execute the dilation operation')
        flags.DEFINE_integer('stop_dilation_epoch', default=100,
                             help='stop dilation operation at this epoch')

    def __init__(self, torchnet: nn.Module, hparams: dict) -> None:
        super().__init__(torchnet, hparams)
        gc_hparams = extract_from_big_dict(hparams, AdmmGCSize.gc_hparam_keys)
        for d, v in gc_hparams.items():
            setattr(self, d, v)
        self.is_dilation = True

    def reset(self, img, gt, wg):
        super().reset(img, gt, wg)
        self.gamma = np.zeros(img.squeeze().shape)
        self.u = np.zeros(img.squeeze().shape)

    def update(self, img_gt_weakgt, criterion):
        self.forward_img(*img_gt_weakgt)
        if self.initilize == False:
            self.initialize_dummy_variables(self.score)
        self._update_s()
        self._update_gamma()
        self._update_theta(criterion)
        self._update_u()
        self._update_v()

    def update_1(self, img_gt_weakgt):
        self.forward_img(*img_gt_weakgt)
        if self.initilize == False:
            self.initialize_dummy_variables(self.score)
        self._update_s()
        self._update_gamma()

    def update_2(self, criterion):
        self._update_theta(criterion)
        self._update_u()
        self._update_v()

    def initialize_dummy_variables(self, score):
        self.s = pred2segmentation(score).cpu().data.numpy().squeeze()  # b, w, h
        self.gamma = self.s
        self.initilize = True

    def _update_theta(self, criterion):
        for i in range(self.optim_inner_loop_num):
            self.torchnet.zero_grad()

            CE_loss = criterion(self.score, self.weak_gt.squeeze(1).long())
            unlabled_loss = self.p_v / 2 * (
                    F.softmax(self.score, dim=1)[:, 1] + torch.from_numpy(-self.s + self.v).float().to(
                device)).norm(p=2) ** 2 \
                            + self.p_u / 2 * (F.softmax(self.score, dim=1)[:, 1] + torch.from_numpy(
                -self.gamma + self.u).float().to(device)).norm(p=2) ** 2

            unlabled_loss /= list(self.score.reshape(-1).size())[0]

            loss = CE_loss + unlabled_loss
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            self.forward_img(self.img, self.gt, self.weak_gt)

    def _update_gamma(self):
        if self.weak_gt.sum() == 0 or self.gt.sum() == 0:
            self.gamma = np.zeros(self.img.squeeze().shape)
            return
        unary_term_gamma_1 = np.multiply(
            (0.5 - (F.softmax(self.score, dim=1).cpu().data.numpy()[:, 1, :, :].squeeze() + self.u)),
            1)
        unary_term_gamma_1[(self.weak_gt.squeeze().cpu().data.numpy() == 1).astype(bool)] = -np.inf

        weak_mask = self.weak_gt.cpu().squeeze().numpy()

        kernel = np.ones((5, 5), np.uint8)
        unary_term_gamma_0 = np.zeros(unary_term_gamma_1.shape)

        if self.is_dilation:
            dilation = cv2.dilate(weak_mask.astype(np.float32), kernel, iterations=self.dilation_level)
            unary_term_gamma_1[dilation != 1] = np.inf

        g = maxflow.Graph[float](0, 0)
        nodeids = g.add_grid_nodes(list(self.gamma.shape))
        g = self._set_boundary_term(g, nodeids, self.img, lumda=self.lamda, sigma=self.sigma)
        g.add_grid_tedges(nodeids, (unary_term_gamma_0).squeeze(),
                          (unary_term_gamma_1).squeeze())
        g.maxflow()
        sgm = g.get_grid_segments(nodeids) * 1
        new_gamma = np.int_(np.logical_not(sgm))
        if new_gamma.sum() > 0:
            self.gamma = new_gamma
        else:
            self.gamma = self.s
        assert self.gamma.shape.__len__() == 2

    def _set_boundary_term(self, g, nodeids, img, lumda, sigma):
        self.kernel = np.ones((self.kernelsize, self.kernelsize))
        self.kernel[int(self.kernel.shape[0] / 2), int(self.kernel.shape[1] / 2)] = 0
        kernel = self.kernel
        transfer_function = lambda pixel_difference: lumda * np.exp((-1 / sigma ** 2) * pixel_difference ** 2)

        img = img.squeeze().cpu().data.numpy()

        # =====new =========================================
        padding_size = int(max(kernel.shape) / 2)
        position = np.array(list(zip(*np.where(kernel != 0))))

        def shift_matrix(matrix, kernel):
            center_x, center_y = int(kernel.shape[0] / 2), int(kernel.shape[1] / 2)
            [kernel_x, kernel_y] = np.array(list(zip(*np.where(kernel == 1))))[0]
            dy, dx = kernel_x - center_x, kernel_y - center_y
            shifted_matrix = np.roll(matrix, -dy, axis=0)
            shifted_matrix = np.roll(shifted_matrix, -dx, axis=1)
            return shifted_matrix

        for p in position[:int(len(position) / 2)]:
            structure = np.zeros(kernel.shape)
            structure[p[0], p[1]] = kernel[p[0], p[1]]
            pad_im = np.pad(img, ((padding_size, padding_size), (padding_size, padding_size)), 'constant',
                            constant_values=0)
            shifted_im = shift_matrix(pad_im, structure)
            weights_ = transfer_function(
                np.abs(pad_im - shifted_im)[padding_size:-padding_size, padding_size:-padding_size])

            g.add_grid_edges(nodeids, structure=structure, weights=weights_, symmetric=True)

        return g

    def _update_u(self):
        new_u = self.u + (F.softmax(self.score, dim=1)[:, 1, :, :].cpu().data.numpy().squeeze() - self.gamma) * 0.01
        self.u = new_u
        assert self.u.shape.__len__() == 2
