#!/user/bin/env python
# _*_ coding: utf-8 _*_
# ==================================================
# @File_name: DoG&NDF -> DoG_NDF
# @Software: PyCharm
# @Author: 张福正
# @Time: 2023/4/12 19:01
# ==================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from timeit import default_timer as timer
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F


#---------------------------定义高斯核函数--------------------------#
def get_gaussian_kernel(kernel_size, sigma, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels,
                                bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


def Average_Filter(img, kernel_size):
    if kernel_size <= 0:
        raise ValueError("滤波器尺寸必须大于0！")
    elif kernel_size % 2 == 0:
        raise ValueError("滤波器尺寸必须是奇数！")

    padding = int((kernel_size - 1) / 2)
    kernel = torch.ones(kernel_size, kernel_size) / (kernel_size * kernel_size)

    average_filter = nn.Conv2d(1, 1, kernel_size=int(kernel_size), stride=1, padding=padding, bias=False)
    average_filter.weight.data = torch.tensor(kernel.reshape((1, 1, kernel.shape[0], kernel.shape[1])))
    average_filter.weight.requires_grad = False
    img_filtered = average_filter(img.reshape((1, 1, img.shape[1], img.shape[2])))

    return img_filtered


def atan2(x):
    y = (2/math.pi) * np.sign(x) * np.arctan(x * x)
    return y


def translateImage(f, di, dj):
    N = f.shape[0]
    M = f.shape[1]
    if di > 0:
        iind = list(range(di, N))
        iind.append(N-1)
        # print(iind)
    elif di < 0:
        iind = list(range(N+di))
        iind.insert(0,0)
        # print(iind)
    else:
        iind = list(range(N))
        # print(iind)

    if dj > 0:
        jind = list(range(dj, M))
        jind.append(M-1)
        # print(jind)
    elif dj < 0:
        jind = list(range(M+dj))
        jind.insert(0,0)
        # print(jind)
    else:
        jind = list(range(M))
        # print(jind)

    ftrans = f[iind, :]
    ftrans = ftrans[:, jind]

    return ftrans


def snldStep(L, c):
    cpc = translateImage(c, 1, 0)
    cmc = translateImage(c, -1, 0)
    ccp = translateImage(c, 0, 1)
    ccm = translateImage(c, 0, -1)
    Lpc = translateImage(L, 1, 0)
    Lmc = translateImage(L, -1, 0)
    Lcp = translateImage(L, 0, 1)
    Lcm = translateImage(L, 0, -1)
    r = ((cpc + c) * (Lpc - L) - (c + cmc) * (L - Lmc) +
         (ccp + c) * (Lcp - L) - (c + ccm) * (L - Lcm)) / 2
    return r


# 计算图像梯度
def gradient(x):
    # tf.image.image_gradients(image)
    h_x = x.size()[-2]
    w_x = x.size()[-1]
    # gradient step=1
    l = x
    r = F.pad(x, (0, 1, 0, 0))[:, :, 1:]
    t = x
    b = F.pad(x, (0, 0, 0, 1))[:, 1:, :]

    dx, dy = torch.abs(r - l), torch.abs(b - t)
    # dx will always have zeros in the last column, r-l
    # dy will always have zeros in the last row,    b-t
    dx[:, :, -1] = 0
    dy[:, -1, :] = 0

    return dx, dy


def NDF(e, eta, stepsize):
    Gx, Gy = gradient(e)
    grad = Gx * Gx + Gy * Gy
    c = 2 / (1 + np.exp(grad / (eta) ** 2))
    r = snldStep(e, c)
    e = e + stepsize * r
    return e


def generateInitialLSF(c, h, w, py):
    c = int(c)
    h = int(h)
    w = int(w)

    py = torch.floor(py)
    py = py.int()
    py = py.cpu()
    py = py.detach().numpy()
    print('py的形状：', py.shape)



    InitialLSF = torch.ones(c, h, w)
    mask = torch.ones(c, h, w)

    num_obj = int(py.shape[0])
    print(num_obj)
    roi_corners = []
    for i in range(num_obj):
        # mask = torch.ones(c, h, w)
        roi = []
        py_i = py[i, :, :]
        # py = py.cpu().numpy()
        for i in range(py_i.shape[0]):
            tup = tuple(py_i[i, :])
            roi.append(tup)

        roi_corners.append(np.array([roi], dtype=np.int32))

    channel = c
    ignore_mask_color = (0,) * channel
    for area in roi_corners:
        cv2.fillPoly(mask.squeeze(0).numpy(), area, ignore_mask_color)

        # print('掩码：', mask[0, 224, 336])

    # channel = c
    # ignore_mask_color = (0,) * channel
    # cv2.fillPoly(mask.squeeze(0).numpy(), roi_corners, ignore_mask_color)
    InitialLSF = cv2.bitwise_and(InitialLSF.squeeze(0).numpy(), mask.squeeze(0).numpy())
    InitialLSF[InitialLSF == 0] = -1
    InitialLSF = torch.tensor(InitialLSF).reshape((1, InitialLSF.shape[0], InitialLSF.shape[1]))
    # print('初始轮廓：', torch.nonzero(InitialLSF == -1))

    return InitialLSF

# 从等高线中提取坐标点和属性值
def get_contour_verts(cn):
    contours = []
    # for each contour line
    # print(cn.levels)
    for cc, vl in zip(cn.collections, cn.levels):
        # for each separate section of the contour line
        for pp in cc.get_paths():
            # paths = []
            # paths["id"] = idx
            # paths["type"] = 0
            # paths["value"] = float(vl)  # vl 是属性值
            # xy = []
            # for each segment of that section
            for vv in pp.iter_segments():
                contours.append([float(vv[0][0]), float(vv[0][1])])  # vv[0] 是等值线上一个点的坐标，是 1 个 形如 array[12.0,13.5] 的 ndarray。
            # paths["coords"] = xy
            # contours.append(paths)
    return contours


def ACM_mid(inp, py):
    # fig, ax = plt.subplots(1, figsize=(20, 10))
    # ax.imshow(inp)
    # plt.show()
    print(inp.shape)
    print(py.shape)
    # img = inp.permute(2, 0, 1)[0, :]
    img = inp[0, :]
    print(img.shape)
    img = img.reshape((1, img.shape[0], img.shape[1]))
    InitialLSF = generateInitialLSF(img.shape[0], img.shape[1], img.shape[2], py)

    alfa = -1
    w = 15
    sigma1 = 0.5
    sigma2 = 3.5
    k = 7
    iterNum = 10

    stepsize = 0.1  # NDF迭代步长
    timestep = 1  # 梯度下降流迭代步长
    epsilon = 1  # Heaviside函数参数

    # 实现高斯核并用于图像滤波
    G1 = get_gaussian_kernel(kernel_size=w, sigma=sigma1, channels=1)
    G2 = get_gaussian_kernel(kernel_size=w, sigma=sigma2, channels=1)
    f1 = G1(img)
    f2 = G2(img)

    Im = f1 - f2  # DoG算子

    e = Im

    eta = math.sqrt(torch.std(img))  # 自适应边界系数

    # ---------------------------实现NDF迭代---------------------------#
    for i in range(10):
        NDF(e, eta, stepsize)

    ex = alfa * atan2(e / (2 * eta))

    # -------------------------实现水平集函数迭代------------------------#
    # fig3 = plt.figure()
    # fig3.canvas.manager.set_window_title('Final contour')
    LSF = InitialLSF

    # tic = timer()  # 开始记录时间
    for i in range(1, iterNum):
        LSF1 = LSF
        Drc = (epsilon / math.pi) / (epsilon * epsilon + LSF * LSF)
        LSF = LSF + timestep * ex * Drc
        LSF = atan2(11 * LSF)
        LSF = Average_Filter(LSF, k).squeeze(0)
        if np.abs(LSF - LSF1).sum() < 0.001 * (LSF.shape[1] * LSF.shape[2]):
            break

    # print(LSF.shape)
    LSF = LSF.squeeze(0).cpu().detach().numpy()
    CS = plt.contour(LSF, [0])  # 绘制最终轮廓线

    # 获取最终轮廓线的像素点坐标
    zeroLS = get_contour_verts(CS)
    zeroLS = torch.tensor(zeroLS)
    zeroLS = zeroLS.reshape((1, zeroLS.shape[0], zeroLS.shape[1]))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    zeroLS = zeroLS.to(device)

    return zeroLS


def ACM_final(inp, py):
    # fig, ax = plt.subplots(1, figsize=(20, 10))
    # ax.imshow(inp)
    # plt.show()
    print(inp.shape)
    print(py.shape)
    img = inp.permute(2, 0, 1)[0, :]
    # img = inp
    # img = img.cpu()
    # img = img.detach().numpy()
    # img = torch.tensor(img)
    print(img)

    print(img.shape)
    img = img.reshape((1, img.shape[0], img.shape[1]))
    InitialLSF = generateInitialLSF(img.shape[0], img.shape[1], img.shape[2], py)

    alfa = -1
    w = 15
    sigma1 = 0.5
    sigma2 = 3.5
    k = 7
    iterNum = 5

    stepsize = 0.1  # NDF迭代步长
    timestep = 1  # 梯度下降流迭代步长
    epsilon = 1  # Heaviside函数参数

    # 实现高斯核并用于图像滤波
    G1 = get_gaussian_kernel(kernel_size=w, sigma=sigma1, channels=1)
    G2 = get_gaussian_kernel(kernel_size=w, sigma=sigma2, channels=1)
    f1 = G1(img)
    f2 = G2(img)

    Im = f1 - f2  # DoG算子

    e = Im

    eta = math.sqrt(torch.std(img))  # 自适应边界系数

    # ---------------------------实现NDF迭代---------------------------#
    for i in range(10):
        NDF(e, eta, stepsize)

    ex = alfa * atan2(e / (2 * eta))

    # -------------------------实现水平集函数迭代------------------------#
    # fig3 = plt.figure()
    # fig3.canvas.manager.set_window_title('Final contour')
    LSF = InitialLSF

    # tic = timer()  # 开始记录时间
    for i in range(1, iterNum):
        LSF1 = LSF
        Drc = (epsilon / math.pi) / (epsilon * epsilon + LSF * LSF)
        LSF = LSF + timestep * ex * Drc
        LSF = atan2(11 * LSF)
        LSF = Average_Filter(LSF, k).squeeze(0)
        if np.abs(LSF - LSF1).sum() < 0.001 * (LSF.shape[1] * LSF.shape[2]):
            break

    # print(LSF.shape)
    LSF = LSF.squeeze(0).cpu().detach().numpy()
    # CS = plt.contour(LSF, [0])  # 绘制最终轮廓线
    #
    # # 获取最终轮廓线的像素点坐标
    # zeroLS = get_contour_verts(CS)
    # zeroLS = torch.tensor(zeroLS)
    # zeroLS = zeroLS.reshape((1, zeroLS.shape[0], zeroLS.shape[1]))
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # zeroLS = zeroLS.to(device)

    return LSF









