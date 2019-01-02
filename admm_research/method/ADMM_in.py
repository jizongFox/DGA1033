from abc import ABC, abstractmethod
import torch, numpy as np
from admm_research import LOGGER
from torch import nn as nn
from torch.nn import functional as F
from admm_research import flags
import cv2, maxflow
import matplotlib.pyplot as plt
from .ADMM import AdmmBase
from admm_research.utils import extract_from_big_dict
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Base_constraint(ABC):
    @abstractmethod
    def setup_arch_flag(cls):
        pass

    def __init__(self) -> None:
        super().__init__()
        self.name = None

    @abstractmethod
    def update_Y(self):
        pass

    def reset(self, img, gt, weakgt):
        assert img.shape.__len__() == 4, "B=1, C=1, H, W should be assigned."
        assert img.shape[0] == 1
        assert img.shape[1] == 1
        self.Y = np.zeros(img.squeeze().shape)
        assert self.Y.shape.__len__() == 2
        self.s_p = np.zeros(self.Y.shape)
        self.s_n = np.zeros(self.Y.shape)
        self.U_p = np.zeros(self.Y.shape)
        self.U_n = np.zeros(self.Y.shape)
        self.img = img
        self.gt = gt
        self.weakgt = weakgt

    def update(self, S):
        self.update_S(S)
        self.update_Y()
        self.update_svariables()
        self.update_multipliers()
        # return self.return_L2_loss()

    def update_S(self, S):
        self.S = S
        self.S_proba = F.softmax(S/10, 1)[:, 1].data.cpu().numpy().squeeze()

    def update_svariables(self):
        assert self.eps is not None
        self.s_p = np.maximum(np.zeros(self.Y.shape),
                              (self.S_proba - self.Y + 0.5 - self.eps + self.U_n))
        self.s_n = np.maximum(np.zeros(self.Y.shape),
                              -(self.S_proba - self.Y - 0.5 + self.eps + self.U_p))
        assert self.s_p.min() >= 0
        assert self.s_n.min() >= 0

    def update_multipliers(self):
        self.U_p = self.U_p + (self.S_proba - (self.Y + 0.5 - self.eps - self.s_n)) * 1.0
        self.U_n = self.U_n + (self.S_proba - (self.Y - 0.5 + self.eps + self.s_p)) * 1.0

    def return_L2_loss(self):
        loss = self.p_p * (F.softmax(self.S, 1)[:, 1].squeeze() - torch.Tensor(
            self.Y + 0.5 - self.eps - self.s_n - self.U_p).float().to(device)).norm(2) \
               + self.p_n * (F.softmax(self.S, 1)[:, 1].squeeze() - torch.Tensor(
            self.Y - 0.5 + self.eps + self.s_p - self.U_n).float().to(device)).norm(2)
        return loss / self.Y.reshape(-1).size

    def show(self, name=None, fig_num=1):
        try:
            getattr(self, name)
        except Exception as e:
            # print(e)
            return
        plt.figure(fig_num, figsize=(5, 5))
        plt.clf()
        plt.subplot(1, 1, 1)
        plt.imshow(self.img[0].cpu().data.numpy().squeeze(), cmap='gray')

        plt.contour(self.weakgt.squeeze().cpu().data.numpy(), level=[0], colors="yellow", alpha=0.2, linewidth=0.001,
                    label='GT')
        plt.contour(self.gt.squeeze().cpu().data.numpy(), level=[0], colors="yellow", alpha=0.2, linewidth=0.001,
                    label='GT')
        plt.contour(self.S.max(1)[1].squeeze().cpu().data.numpy(), level=[0, 0, 5, 1],
                    colors="green", alpha=0.2, linewidth=0.001, label='CNN')
        if name is not None:
            plt.contour(getattr(self, name), level=[0], colors="red", alpha=0.2, linewidth=0.001, label=name)

        plt.title(name)
        plt.show(block=False)
        plt.pause(0.01)


class RegConstraint(Base_constraint):
    reg_hpara_keys = ['reg_eps', 'reg_p_p', 'reg_p_n', 'reg_lamda', 'reg_sigma', 'reg_kernelsize', 'reg_dilation_level','reg_kernelsize']

    @classmethod
    def setup_arch_flag(cls):
        flags.DEFINE_float('reg_eps', default=0.25, help='eps for inequality method')
        flags.DEFINE_float('reg_p_p', default=1, help='penalty for p_positive')
        flags.DEFINE_float('reg_p_n', default=1, help='penalty for p_negative')
        flags.DEFINE_float('reg_lamda', default=1, help='lamda for unary term and pairwise term')
        flags.DEFINE_float('reg_sigma', default=0.02, help='smoothness term for pairwise term')
        flags.DEFINE_integer('reg_dilation_level', default=10, help='dilation for the weak mask')
        flags.DEFINE_integer('reg_kernelsize', default=5, help='dilation for the weak mask')



    def __init__(self, hparam: dict) -> None:
        super().__init__()
        hparam = extract_from_big_dict(hparam, self.reg_hpara_keys)
        self.name = 'reg'
        assert isinstance(hparam, dict)
        for k in hparam.keys():
            assert k in self.reg_hpara_keys

        for k, v in hparam.items():
            setattr(self, k.replace('reg_', ''), v)
        self.__initial_kernel()
        self.is_dilation = True

    def __initial_kernel(self):
        self.kernel = np.ones((self.kernelsize, self.kernelsize))
        self.kernel[int(self.kernel.shape[0] / 2), int(self.kernel.shape[1] / 2)] = 0

    def update_Y(self):
        if self.weakgt.sum() <= 0 or self.gt.sum() < 0:
            self.Y = np.zeros(self.Y.shape)
            return

        unary_term_gamma_1 = np.multiply(
            (1 - self.S_proba - self.eps - self.s_n - self.U_p),
            self.p_p) + np.multiply(
            (- self.S_proba + self.eps + self.s_p - self.U_n),
            self.p_n)
        unary_term_gamma_1[(self.weakgt.data.cpu().numpy().squeeze() == 1).astype(bool)] = -np.inf

        if self.is_dilation:
            kernel = np.ones((5, 5), np.uint8)
            dilation = cv2.dilate(self.weakgt.data.cpu().numpy().squeeze().astype(np.float32), kernel, iterations=self.dilation_level)
            unary_term_gamma_0 = np.zeros(unary_term_gamma_1.shape)
            unary_term_gamma_1[dilation != 1] = np.inf

        g = maxflow.Graph[float](0, 0)
        # Add the nodes.
        nodeids = g.add_grid_nodes(self.Y.shape)
        # Add edges with the same capacities.

        # g.add_grid_edges(nodeids, neighbor_term)
        g = self.__set_boundary_term__(g, nodeids, self.img)

        # Add the terminal edges.
        g.add_grid_tedges(nodeids, (unary_term_gamma_0).squeeze(),
                          (unary_term_gamma_1).squeeze())
        g.maxflow()
        # Get the segments.
        sgm = g.get_grid_segments(nodeids) * 1

        # The labels should be 1 where sgm is False and 0 otherwise.
        new_Y = np.int_(np.logical_not(sgm))

        if new_Y.sum() > 0:
            self.Y = new_Y

    def __set_boundary_term__(self, g, nodeids, img):
        '''
        :param g:
        :param nodeids:
        :param img:
        :return:
        '''
        kernel = self.kernel
        sigma = self.sigma
        lumda = self.lamda
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


class SizeConstraint(Base_constraint):
    size_hpara_keys = ['size_eps', 'size_eps_size', 'size_p_p', 'size_p_n']

    @classmethod
    def setup_arch_flag(cls):
        flags.DEFINE_float('size_eps', default=0.25, help='eps for inequality method')
        flags.DEFINE_float('size_eps_size', default=0.2, help='eps_size for interval size band')
        flags.DEFINE_float('size_p_p', default=10, help='penalty for p_positive')
        flags.DEFINE_float('size_p_n', default=10, help='penalty for p_negative')

    def __init__(self, hparam: dict) -> None:
        super().__init__()
        self.individual_constrain = hparam['individual_size_constraint']
        if not self.individual_constrain:
            self.upbound = hparam['global_upbound']
            self.lowbound = hparam['global_lowbound']
        hparam = extract_from_big_dict(hparam, self.size_hpara_keys)
        self.name = 'size'
        assert isinstance(hparam, dict)
        for k in hparam.keys():
            assert k in self.size_hpara_keys
        self.eps_size = None
        for k, v in hparam.items():
            setattr(self, k.replace('size_', ''), v)

    def reset(self, img, gt, weakgt):
        super().reset(img, gt, weakgt)
        assert self.eps_size is not None
        if self.individual_constrain:
            self.lowbound = int(self.gt.sum().float() * (1 - self.eps_size))
            self.upbound = int(self.gt.sum().float() * (1 + self.eps_size))

    def update_Y(self):
        if self.weakgt.sum() <= 0:
            self.Y = np.zeros(self.Y.shape)
            return
        self.S_proba[self.weakgt.data.cpu().numpy().squeeze()==1]=1

        a = np.multiply(
            (1 - self.S_proba - self.eps - self.s_n - self.U_p),
            self.p_p) + np.multiply(
            (- self.S_proba + self.eps + self.s_p - self.U_n),
            self.p_n)
        original_shape = a.shape
        a_ = np.sort(a.ravel())
        useful_pixel_number = (a_ < 0).sum()
        if self.lowbound < useful_pixel_number and self.upbound > useful_pixel_number:
            self.Y = ((a < 0) * 1.0).reshape(original_shape)
        if useful_pixel_number < self.lowbound:
            self.Y = ((a <= a_[self.lowbound+1]) * 1.0).reshape(original_shape)
        if useful_pixel_number > self.upbound:
            self.Y = ((a <= a_[self.upbound-1]) * 1.0).reshape(original_shape)
        LOGGER.debug(
            'low_band:{},up_band:{},realsize:{}, new_S_size:{}'.format(self.lowbound, self.upbound, self.gt.sum(),
                                                                       self.Y.sum()))


class ADMM_size_inequality(AdmmBase):
    size_hparam_keys = ['individual_size_constraint', 'eps', 'global_upbound', 'global_lowbound']

    @classmethod
    def setup_arch_flags(cls):
        super().setup_arch_flags()
        flags.DEFINE_boolean('individual_size_constraint', default=True,
                             help='Individual size constraint for each input image')
        flags.DEFINE_float('eps', default=0.001,
                           help='Individual size eps')
        flags.DEFINE_integer('global_upbound', default=2000,
                             help='global upper bound if individual_size_constraint is False')
        flags.DEFINE_integer('global_lowbound', default=20,
                             help='global lower bound if individual_size_constraint is False')
        SizeConstraint.setup_arch_flag()

    def __init__(self, torchnet: nn.Module, hparams: dict) -> None:
        super().__init__(torchnet, hparams)
        self.size_constrain = SizeConstraint(hparams)

    def update(self, img_gt_weakgt, criterion):
        img, gt, weak = img_gt_weakgt
        # self.size_constrain.reset(img, gt, weak)
        self.size_constrain.update(self.torchnet(img))  # update Y based on S and slack variables
        self._update_theta(criterion)

    def update_1(self, img_gt_weakgt):
        img, gt, weak = img_gt_weakgt
        self.size_constrain.update(self.torchnet(img))  # update Y based on S and slack variables

    def update_2(self, criterion):
        self._update_theta(criterion)  ## update Networks

    def reset(self, img, gt=None, weak=None):
        self.img = img
        self.gt = gt
        self.weak_gt = weak
        self.size_constrain.reset(img, gt, weak)

    def forward_img(self, img, gt, weak_gt):
        pass

    @property
    def score(self):
        return self.size_constrain.S

    def show(self, name=None, fig_num=1):
        self.size_constrain.show(name=name, fig_num=fig_num)
        self.size_constrain.show(name='S_proba', fig_num=2)

    def _update_theta(self, criterion):
        for i in range(self.optim_inner_loop_num):
            CE_loss = criterion(self.score, self.weak_gt.squeeze(1).long())
            constraint_loss = self.size_constrain.return_L2_loss()  # return L2 loss based on current S and Y
            loss = constraint_loss + CE_loss
            print('loss: CEloss:{},Constraintloss:{}'.format(CE_loss.item(), constraint_loss.item()))
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            self.size_constrain.update_S(self.torchnet(self.img))  # update S so that it can converge rapidly.


class ADMM_reg_size_inequality(ADMM_size_inequality):
    reg_size_keys = ADMM_size_inequality.size_hparam_keys

    @classmethod
    def setup_arch_flags(cls):
        super().setup_arch_flags()
        RegConstraint.setup_arch_flag()
        flags.DEFINE_integer('stop_dilation_epoch',default=100, help='stop dilation')

    def __init__(self, torchnet: nn.Module, hparams: dict) -> None:
        super().__init__(torchnet, hparams)
        self.reg_constrain = RegConstraint(hparams)

    def update(self, img_gt_weakgt, criterion):
        img, gt, weak = img_gt_weakgt
        self.size_constrain.update(self.torchnet(img))  # update Y based on S and slack variables
        self.reg_constrain.update(self.torchnet(img))
        self._update_theta(criterion)

    def _update_theta(self, criterion):
        for i in range(self.optim_inner_loop_num):
            CE_loss = criterion(self.score, self.weak_gt.squeeze(1).long())
            constraint_loss = self.size_constrain.return_L2_loss()  # return L2 loss based on current S and Y
            reg_loss = self.reg_constrain.return_L2_loss()

            loss = constraint_loss + CE_loss + reg_loss
            # print('loss: CEloss:{},Constraintloss:{}'.format(CE_loss.item(), constraint_loss.item()))
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            pred = self.torchnet(self.img)
            self.size_constrain.update_S(pred)  # update S so that it can converge rapidly.
            self.reg_constrain.update_S(pred)

    def update_1(self, img_gt_weakgt):
        img, gt, weak = img_gt_weakgt
        pred = self.torchnet(img)
        self.size_constrain.update(pred)  # update Y based on S and slack variables
        self.reg_constrain.update(pred)

    def update_2(self, criterion):
        self._update_theta(criterion)  ## update Networks

    def reset(self, img, gt=None, weak=None):
        self.img = img
        self.gt = gt
        self.weak_gt = weak
        self.size_constrain.reset(img, gt, weak)
        self.reg_constrain.reset(img, gt, weak)

    def show(self, name=None, fig_num=1):
        self.size_constrain.show(name=name, fig_num=fig_num)
        self.reg_constrain.show(name=name,fig_num=fig_num+10)

