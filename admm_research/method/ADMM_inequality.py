# coding=utf-8
import os, sys, numpy as np
import matplotlib.pyplot as plt
import warnings
import torch
import torch.nn.functional as F
from admm_research.loss import CrossEntropyLoss2d
import maxflow, cv2
from admm_research.method import ModelMode
from admm_research.utils import AverageMeter, dice_loss

sys.path.insert(-1, os.getcwd())
warnings.filterwarnings('ignore')
use_gpu = True
device = torch.device('cuda') if torch.cuda.is_available() and use_gpu else torch.device('cpu')


class constraint:

    def __init__(self, name, image_fullmask_weakmask_S_pair, **kwargs) -> None:
        '''
        :param S: input image distribution with gradiant
        :param kwargs: parameter dict for instance initialization
        '''
        super().__init__()
        self.name = name
        [image, full_mask, weak_mask, S] = image_fullmask_weakmask_S_pair
        self.image = image
        self.weak_mask = weak_mask.cpu().squeeze().numpy()
        self.full_mask = full_mask.cpu().squeeze().numpy()
        self.S = S
        self.S_proba = F.softmax(S, 1)[:, 1].data.numpy().squeeze()
        self.__set_default_parametes()
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.__init_variables()
        if self.name == 'size':
            self.lowbound = int(self.full_mask.sum() * (1 - self.eps_size))
            self.upbound = int(self.full_mask.sum() * (1 + self.eps_size))

    def __set_default_parametes(self):
        if self.name == 'reg':
            self.eps = 0.5
            self.p_p = 10
            self.p_n = 10
            self.lamda = 5
            self.sigma = 0.02
            self.kernelsize = 5
            self.dilation_level = 10

        elif self.name == 'size':
            self.eps = 0.5
            self.eps_size = 0.1
            self.p_p = 10
            self.p_n = 10

        else:
            raise NotImplementedError

    def __init_variables(self):
        if self.name == 'reg':
            self.__initial_kernel()

        self.Y = self.S.max(1)[1].squeeze().numpy()
        self.s_p = np.zeros(self.Y.shape)
        self.s_n = np.zeros(self.Y.shape)
        self.U_p = np.zeros(self.Y.shape)
        self.U_n = np.zeros(self.Y.shape)

    def update_S(self, S):
        self.S = S
        self.S_proba = F.softmax(S, 1)[:, 1].data.numpy().squeeze()

    def update_Y(self):
        if self.name == 'reg':
            unary_term_gamma_1 = np.multiply(
                (1 - self.S_proba - self.eps - self.s_n - self.U_p),
                self.p_p) + np.multiply(
                (- self.S_proba + self.eps + self.s_p - self.U_n),
                self.p_n)
            unary_term_gamma_1[(self.weak_mask == 1).astype(bool)] = -np.inf

            kernel = np.ones((5, 5), np.uint8)
            dilation = cv2.dilate(self.weak_mask.astype(np.float32), kernel, iterations=self.dilation_level)
            unary_term_gamma_0 = np.zeros(unary_term_gamma_1.shape)
            unary_term_gamma_1[dilation != 1] = np.inf

            g = maxflow.Graph[float](0, 0)
            # Add the nodes.
            nodeids = g.add_grid_nodes(self.Y.shape)
            # Add edges with the same capacities.

            # g.add_grid_edges(nodeids, neighbor_term)
            g = self.__set_boundary_term__(g, nodeids, self.image)

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
            else:
                raise ValueError

        elif self.name == 'size':
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
                self.Y = ((a <= a_[self.lowbound]) * 1).reshape(original_shape)
            if useful_pixel_number > self.upbound:
                self.Y = ((a <= a_[self.upbound]) * 1).reshape(original_shape)

        else:
            raise NotImplementedError

    def update_svariables(self):
        self.s_p = np.maximum(np.zeros(self.Y.shape),
                              (self.S_proba - self.Y + 0.5 - self.eps + self.U_n))
        self.s_n = np.maximum(np.zeros(self.Y.shape),
                              -(self.S_proba - self.Y - 0.5 + self.eps + self.U_p))

    def update_multipliers(self):
        self.U_p = self.U_p + (self.S_proba - (self.Y + 0.5 - self.eps - self.s_n)) * 0.1
        self.U_n = self.U_n + (self.S_proba - (self.Y - 0.5 + self.eps + self.s_p)) * 0.1

    def return_L2_loss(self):
        loss = self.p_p * (F.softmax(self.S, 1)[:, 1].squeeze() - torch.Tensor(
            self.Y + 0.5 - self.eps - self.s_n - self.U_p).float()).norm(2) \
               + self.p_n * (F.softmax(self.S, 1)[:, 1].squeeze() - torch.Tensor(
            self.Y - 0.5 + self.eps + self.s_p - self.U_n).float()).norm(2)
        return loss / self.Y.reshape(-1).size

    def update(self, S):

        self.update_S(S)
        self.update_Y()
        self.update_svariables()
        self.update_multipliers()
        return self.return_L2_loss()

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

    def __initial_kernel(self):
        self.kernel = np.ones((self.kernelsize, self.kernelsize))
        self.kernel[int(self.kernel.shape[0] / 2), int(self.kernel.shape[1] / 2)] = 0

    def heatmap2segmentation(self, heatmap):
        return heatmap.max(1)[1]

    def show_S(self):
        plt.figure(1)
        plt.imshow(self.S_proba)
        plt.title('S')
        plt.show(block=False)

    def show_Y(self):
        plt.figure(2)
        plt.imshow(self.Y)
        plt.title('y')
        plt.show(block=False)

    def show(self, name=None, fig_num=1):
        try:
            getattr(self, name)
        except Exception as e:
            print(e)
            return
        plt.figure(fig_num, figsize=(5, 5))
        plt.clf()
        plt.subplot(1, 1, 1)
        plt.imshow(self.image[0].cpu().data.numpy().squeeze(), cmap='gray')

        plt.contour(self.weak_mask.squeeze(), level=[0], colors="yellow", alpha=0.2, linewidth=0.001,
                    label='GT')
        plt.contour(self.full_mask.squeeze(), level=[0], colors="yellow", alpha=0.2, linewidth=0.001,
                    label='GT')
        plt.contour(self.S.max(1)[1].squeeze().cpu().data.numpy(), level=[0],
                    colors="green", alpha=0.2, linewidth=0.001, label='CNN')
        if name is not None:
            plt.contour(getattr(self, name), level=[0], colors="red", alpha=0.2, linewidth=0.001, label=name)

        plt.title(name)
        plt.show(block=False)
        plt.pause(0.01)


class ADMM_inequality():
    '''
    this is a wrapper class using inequality constraint class
    '''

    def __init__(self, torchnet, list_of_constraints, optim_hparams) -> None:
        super().__init__()
        self.torchnet = torchnet
        self.constraints = list_of_constraints

    def set_mode(self, mode):
        assert mode in (ModelMode.TRAIN, ModelMode.EVAL)
        if mode == ModelMode.TRAIN:
            self.torchnet.train()
        else:
            self.torchnet.eval()

    def evaluate(self, dataloader):
        b_dice_meter = AverageMeter()
        f_dice_meter = AverageMeter()
        self.set_mode(ModelMode.EVAL)
        with torch.no_grad():

            for i, (image, mask, weak_mask, pathname) in enumerate(dataloader):
                if mask.sum() == 0 or weak_mask.sum() == 0:
                    continue
                image, mask, weak_mask = image.to(device), mask.to(device), weak_mask.to(device)
                proba = F.softmax(self.torchnet(image), dim=1)
                predicted_mask = proba.max(1)[1]
                [b_iou, f_iou] = dice_loss(predicted_mask, mask)
                b_dice_meter.update(b_iou, image.size(0))
                f_dice_meter.update(f_iou, image.size(0))

        self.set_mode(ModelMode.TRAIN)
        return b_dice_meter.avg, f_dice_meter.avg


class ADMM():
    '''

    '''

    def __init__(self, CNN) -> None:
        super().__init__()
        self.cnn = CNN
        self.innerloop_num = 50
        self.partialCE_criterion = CrossEntropyLoss2d(weight=torch.Tensor([0, 1]))
        self.optim = torch.optim.Adam(self.cnn.parameters(), lr=1e-3)

    def forward(self, image, full_mask, weak_mask):
        self.image = image
        self.full_mask = full_mask
        self.weak_mask = weak_mask
        self.S = self.cnn(image)
        self.constraint = constraint('size', [image, full_mask, weak_mask, self.S])
        self.constranit2 = constraint('reg', [image, full_mask, weak_mask, self.S])
        for i in range(self.innerloop_num):
            # update Y , U and s based on the current S
            self.constraint.update(self.S)
            self.constranit2.update(self.S)
            for j in range(3):
                self.optim.zero_grad()
                self.S = self.cnn(image)
                # the a fixed Y, U and s.
                self.constraint.update_S(self.S)
                self.constranit2.update_S(self.S)
                reg_loss = self.constraint.return_L2_loss()
                size_loss = self.constranit2.return_L2_loss()
                ce_loss = self.partialCE_criterion(self.S, weak_mask.squeeze(1).long())
                loss = ce_loss  + size_loss +reg_loss
                loss.backward()
                print('CE:', ce_loss.item(), 'Reg_loss:', reg_loss.item(), 'Size_loss:', size_loss.item())

                self.optim.step()

            self.constraint.show('S_proba',1)
            # self.constraint.show_Y()
            # self.constraint.show_U_p()
            # self.constraint.show_U_n()
            self.constraint.show('Y',2)
            self.constranit2.show('Y',3)
            plt.pause(0.001)


if __name__ == "__main__":
    S = torch.randn((1, 2, 200, 200), requires_grad=True)
    image = torch.randn((1, 200, 200))

    c = constraint('reg', [image, image, S])
    r = c.update(S)
    print()
