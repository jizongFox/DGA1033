import numpy as np
import cv2
import maxflow
from multiprocessing.dummy import Pool
from functools import partial
from itertools import repeat


def _update_gamma(img, probability, u, gt, weak_gt, lamda, sigma, kernelsize) -> np.ndarray:
    assert isinstance(img, np.ndarray)
    assert img.shape.__len__() == 2
    assert isinstance(gt, np.ndarray)
    assert gt.shape.__len__() == 2
    assert isinstance(weak_gt, np.ndarray)
    assert weak_gt.shape.__len__() == 2
    assert isinstance(probability, np.ndarray)
    c, h, w = probability.shape
    assert c == 2
    assert isinstance(u, np.ndarray)
    assert u.shape.__len__() == 2

    if gt.sum() == 0:
        gamma = np.zeros_like(img)
        return gamma

    unary_term_gamma_1 = np.multiply((0.5 - (probability[1].squeeze() + u)), 1)
    unary_term_gamma_1[(weak_gt == 1).astype(bool)] = -np.inf

    kernel = np.ones((5, 5), np.uint8)
    unary_term_gamma_0 = np.zeros(unary_term_gamma_1.shape)

    dilation = cv2.dilate(gt.astype(np.float32), kernel, iterations=1)
    unary_term_gamma_1[dilation != 1] = np.inf

    g = maxflow.Graph[float](0, 0)
    nodeids = g.add_grid_nodes(list(img.shape))
    g = _set_boundary_term(g, nodeids, img, lumda=lamda, sigma=sigma, kernelsize=kernelsize)
    g.add_grid_tedges(nodeids, (unary_term_gamma_0).squeeze(),
                      (unary_term_gamma_1).squeeze())
    g.maxflow()
    sgm = g.get_grid_segments(nodeids) * 1
    gamma = np.int_(np.logical_not(sgm))

    assert gamma.shape.__len__() == 2
    return gamma


def _set_boundary_term(g, nodeids, img, lumda, sigma, kernelsize):
    kernel = np.ones((kernelsize, kernelsize))
    kernel[int(kernel.shape[0] / 2), int(kernel.shape[1] / 2)] = 0
    transfer_function = lambda pixel_difference: lumda * np.exp((-1 / sigma ** 2) * pixel_difference ** 2)

    # =====new =========================================
    padding_size = int(max(kernel.shape) / 2)
    position = np.array(list(zip(*np.where(kernel != 0))))

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


def shift_matrix(matrix, kernel):
    center_x, center_y = int(kernel.shape[0] / 2), int(kernel.shape[1] / 2)
    [kernel_x, kernel_y] = np.array(list(zip(*np.where(kernel == 1))))[0]
    dy, dx = kernel_x - center_x, kernel_y - center_y
    shifted_matrix = np.roll(matrix, -dy, axis=0)
    shifted_matrix = np.roll(shifted_matrix, -dx, axis=1)
    return shifted_matrix


# helper function to call graphcut
import torch
def _multiprocess_Call(imgs, scores, us, gts, weak_gts, lamda, sigma, kernelsize):
    P = Pool()
    results = P.starmap(_update_gamma,
                        zip(imgs, scores, us, gts, weak_gts, repeat(lamda), repeat(sigma), repeat(kernelsize)))
    P.close()
    results = np.stack(results)
    return results
