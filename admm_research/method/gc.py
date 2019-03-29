import numpy as np
import cv2
import maxflow
from multiprocessing.dummy import Pool
from itertools import repeat
import matplotlib.pyplot as plt
import imutils
from scipy.special import softmax

def _update_gamma_CD_oracle(img, probability, u, gt, weak_gt, lamda, sigma, kernelsize, bounds) -> np.ndarray:
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

    kernel = np.ones((3, 3), np.uint8)

    unary_term_gamma_1 = (0.5 - (probability[1].squeeze() + u))  # here we only take the foreground probability
    weak_gt = cv2.erode(gt.astype(np.float32), kernel, iterations=2)

    unary_term_gamma_1[(weak_gt == 1).astype(bool)] = -np.inf

    unary_term_gamma_0 = np.zeros(unary_term_gamma_1.shape)

    dilation = cv2.dilate(gt.astype(np.float32), kernel, iterations=2)
    unary_term_gamma_1[dilation != 1] = np.inf

    # here is the trick to play with.

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


def _update_gamma_CD_method1(img, probability, u, gt, weak_gt, lamda, sigma, kernelsize, bounds) -> np.ndarray:
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

    assert isinstance(bounds, np.ndarray)
    # assert bounds.shape.__len__() == 2

    assert bounds[0] <= gt.sum() <= bounds[1]

    if gt.sum() == 0:
        gamma = np.zeros_like(img)
        return gamma

    # here is the trick to play with.
    kernel = np.ones((3, 3), np.uint8)
    unary_term_gamma_1 = (0.5 - (probability[1].squeeze() + u))  # here we only take the foreground probability

    fb = weak_gt.copy()
    if fb.sum() > bounds[0]:  # the weak_gt is more accurate than the low bound
        pass
        # weak_gt = cv2.erode(gt.astype(np.float32), kernel, iterations=3)
    elif fb.sum() < bounds[0]:  # we need to increase the weak_mask
        while fb.sum() < bounds[0]:
            fb = cv2.dilate(fb.astype(np.float32), kernel, iterations=1)

    unary_term_gamma_1[(fb == 1).astype(bool)] = -np.inf

    unary_term_gamma_0 = np.zeros_like(unary_term_gamma_1)
    dilation = weak_gt.copy()
    while dilation.sum() < bounds[1]:
        dilation = cv2.dilate(dilation.astype(np.float32), kernel, iterations=1)
    unary_term_gamma_1[dilation != 1] = np.inf
    # here is the trick to play with.

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


def _update_gamma_CD_method3(img, probability, u, gt, weak_gt, lamda, sigma, kernelsize, bounds) -> np.ndarray:
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

    assert isinstance(bounds, np.ndarray)
    # assert bounds.shape.__len__() == 2

    assert bounds[0] <= gt.sum() <= bounds[1]

    if gt.sum() == 0:
        gamma = np.zeros_like(img)
        return gamma

    # here is the trick to play with.
    kernel = np.ones((3, 3), np.uint8)
    unary_term_gamma_1 = (0.5 - (probability[1].squeeze() + u))  # here we only take the foreground probability

    fb = weak_gt.copy()
    if fb.sum() > bounds[0]:  # the weak_gt is more accurate than the low bound
        pass
        # weak_gt = cv2.erode(gt.astype(np.float32), kernel, iterations=3)
    elif fb.sum() < bounds[0]:  # we need to increase the weak_mask
        while fb.sum() < bounds[0]:
            fb = cv2.dilate(fb.astype(np.float32), kernel, iterations=1)

    unary_term_gamma_1[(fb == 1).astype(bool)] = -np.inf

    unary_term_gamma_0 = np.zeros_like(unary_term_gamma_1)
    dilation = weak_gt.copy()
    while dilation.sum() < bounds[1]:
        dilation = cv2.dilate(dilation.astype(np.float32), kernel, iterations=1)
    unary_term_gamma_1[dilation != 1] = np.inf

    ## calculate the momentum center of the FG.
    try:
        cn = cv2.findContours(weak_gt.copy().astype(np.uint8),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cn = imutils.grab_contours(cn)
        M = cv2.moments(cn[0])
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        xx,yy = np.meshgrid(np.linspace(0,255,256),np.linspace(0,255,256))
        distanceMap =np.sqrt((xx-cX)**2 +(yy-cY)**2)
        max_value = distanceMap[(unary_term_gamma_1!=np.inf) * (unary_term_gamma_1!=-np.inf)].max()
        min_value = distanceMap[(unary_term_gamma_1!=np.inf) * (unary_term_gamma_1!=-np.inf)].min()

        distanceMap = (distanceMap-min_value)/(max_value- min_value)
    except:
        distanceMap = np.zeros_like(unary_term_gamma_1)
    unary_term_gamma_1 += distanceMap



    # here is the trick to play with.

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


def _update_gamma(img, probability, u, gt, weak_gt, lamda, sigma, kernelsize, bounds) -> np.ndarray:
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

    unary_term_gamma_1 = (0.5 - (probability[1].squeeze() + u))  # here we only take the foreground probability
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

def _multiprocess_Call(imgs, scores, us, gts, weak_gts, lamda, sigma, kernelsize, bounds, method):
    assert method in ('oracle','method1','method3'),"method should be in 'oracle','method1','method3'"
    P = Pool(8)
    if method == 'oracle':
        results = P.starmap(_update_gamma_CD_oracle,
                        zip(imgs, scores, us, gts, weak_gts, repeat(lamda), repeat(sigma), repeat(kernelsize), bounds))
    elif method == 'method1':
        results = P.starmap(_update_gamma_CD_method1,
                            zip(imgs, scores, us, gts, weak_gts, repeat(lamda), repeat(sigma), repeat(kernelsize), bounds))
    else:
        results = P.starmap(_update_gamma_CD_method3,
                            zip(imgs, scores, us, gts, weak_gts, repeat(lamda), repeat(sigma), repeat(kernelsize), bounds))

    P.close()
    results = np.stack(results)
    return results
