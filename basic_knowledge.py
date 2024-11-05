#!/user/bin/env python
# _*_ coding: utf-8 _*_
# ==================================================
# @File_name: e2ec-main -> basic_knowledge
# @Software: PyCharm
# @Author: 张福正
# @Time: 2023/9/6 10:23
# ==================================================
import cv2
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import sys

# arr1 = np.array([[[1, 2, 3, 4, 5, 6, 7, 8, 7],
#                [1, 2, 3, 4, 5, 6, 7, 8, 8],
#                 [1, 2, 3, 4, 5, 6, 7, 8, 9]]])
#
# arr2 = np.array([1, 2, 3, 4, 5, 6])
# print(arr[:, 0])
# print(arr[:, 1])
# print(arr[:, 2])
# print(arr[0])
# print(len(arr2[2:]))
# print(arr2 // 2)
#
# heads = {'ct_hm': 20, 'wh': 128 * 2}
#
# for head in heads:
#     print(head)
#
# print(arr1[..., 0])
# print(arr1[:, 0, :])

def nms(heat, kernel=3):
    pad = (kernel - 1) // 2  # pad = 1

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = gather_feat(feat, ind)
    return feat

def topk(scores, K=3):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)
    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def decode_ct_hm(ct_hm, wh, reg=None, K=3, stride=1.):
    batch, cat, height, width = ct_hm.size()  # 获取特征图的尺寸，从左到右为batch_size，特征图通道数，图像高度，图像宽度
    ct_hm = nms(ct_hm)  # 对特征图进行非极大值抑制
    scores, inds, clses, ys, xs = topk(ct_hm, K=K)  # 对特征图中的像素点进行重排序，选取100个点作为初始轮廓点
    wh = transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, -1, 2)  # 对特征图形状进行调整

    if reg is not None:
        reg = transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1)
        ys = ys.view(batch, K, 1)

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    ct = torch.cat([xs, ys], dim=2)
    poly = ct.unsqueeze(2).expand(batch, K, wh.size(2), 2) + wh * stride
    detection = torch.cat([ct, scores, clses], dim=2)
    return poly, detection  # 返回选取的像素点坐标构成的序列


# input1 = torch.randn(1, 1, 3, 3)
# input2 = torch.randn(1, 4, 3, 3)
# print(input1)
# print(input2)

# input1 = torch.sigmoid(input1)
# input1 = nms(input1)

# scores, inds, clses, ys, xs = topk(input1, K=3)
# print(scores)
# print(inds)
# print(clses)
# print(ys)
# print(xs)


# poly_init, detection = decode_ct_hm(input1, input2, K=3, stride=1)
# print(poly_init)
# print(detection)
# print(detection[0, :, 2])
#
# valid = detection[0, :, 2] >= 0.05
# poly_init, detection = poly_init[0][valid], detection[0][valid]
# print(poly_init)
# print(detection)
#
#
# def img_poly_to_can_poly(img_poly):
#     if len(img_poly) == 0:
#         return torch.zeros_like(img_poly)
#     x_min = torch.min(img_poly[..., 0], dim=-1)[0]
#     y_min = torch.min(img_poly[..., 1], dim=-1)[0]
#     can_poly = img_poly.clone()
#     can_poly[..., 0] = can_poly[..., 0] - x_min[..., None]
#     can_poly[..., 1] = can_poly[..., 1] - y_min[..., None]
#
#     return (can_poly)
#
#
# scores = torch.randn(1, 3, 4, 4)
# scores = torch.sigmoid(scores) * 10
# scores = scores.view(1, 3, -1)
# print(scores)
#
# topk_scores, topk_inds = torch.topk(scores, 4)
# print("topk_scores:", topk_scores)
# print("topk_inds:", topk_inds)
#
# topk_inds = topk_inds % 16
# print(topk_inds)
#
# topk_ys = (topk_inds / 4).int().float()
# topk_xs = (topk_inds % 4).int().float()
#
# print("topk_xs:", topk_xs)
# print("topk_ys:", topk_ys)
#
# topk_score, topk_ind = torch.topk(topk_scores.view(1, -1), 4)
# print("topk_score:", topk_score)
# print("topk_ind:", topk_ind)
#
# topk_clses = (topk_ind / 4).int()
#
# feat = topk_inds.view(1, -1, 1)
# print("feat:", feat)
#
# dim = feat.size(2)
# ind = topk_ind.unsqueeze(2).expand(topk_ind.size(0), topk_ind.size(1), dim)
# print("dim:", dim)
# print("ind:", ind)
#
# feat = feat.gather(1, ind).view(1, 4)
# print(feat)
#
# topk_xs = gather_feat(topk_xs.view(1, -1, 1), topk_ind).view(1, 4)
# topk_ys = gather_feat(topk_ys.view(1, -1, 1), topk_ind).view(1, 4)
# print("topk_xs:", topk_xs)
# print("topk_ys:", topk_ys)
#
# xs = topk_xs.view(1, 4, 1)
# ys = topk_ys.view(1, 4, 1)
#
# print("xs:", xs)
# print("ys:", ys)
#
# ct = torch.cat([xs, ys], dim=2)
# print("ct:", ct)
#
#
# wh = torch.randn(1, 6, 4, 4)
# wh = torch.sigmoid(wh) * 10
# print("wh:", wh)
#
# wh = transpose_and_gather_feat(wh, feat)
# print("wh:", wh)
#
# poly = torch.randn(1, 2, 3, 3)
# print(poly[0].shape)
#
# wh = torch.randn(1, 2, 4, 2)
# wh = torch.sigmoid(wh) * 10
# print(wh)
#
# ct = torch.randn(1, 2, 2)
# ct = torch.sigmoid(ct) * 10
# print(ct)
# ct = ct.unsqueeze(2).expand(1, 2, 4, 2)
# print(ct)
#
# detection = torch.randn(1, 4, 4)
# detection = torch.sigmoid(detection) * 10
# valid = detection[0, :, 2]>=0.05

# print(detection)
# print(detection[0, :, 2])
# print(valid)

# detection = torch.randn(4, 4)
# detection = torch.sigmoid(detection) * 10
# print(detection)
# print(detection[:, :2].shape)
#
#
# init_polys = torch.randn(4, 4, 2)
# init_polys = torch.sigmoid(init_polys) * 10
#
# ct_polys = torch.randn(4, 2)
# ct_polys = torch.sigmoid(ct_polys) * 10
# print(ct_polys)
# ct_polys = ct_polys.unsqueeze(1).expand(init_polys.size(0), 1, init_polys.size(2))
# print(ct_polys)
#
# print(init_polys)
#
# points = torch.cat([ct_polys, init_polys], dim=1)
# print(points)
# print(points[..., 0])
#
# points[..., 0] = points[..., 0] / (10 / 2.) - 1
# print(points[..., 0])
#
# ct = torch.randn(1, 2, 2)
# ct = torch.sigmoid(ct) * 10
# print(ct)
#
# ct = ct.unsqueeze(2)
# print(ct)
#
# ct = ct.expand(1, 2, 5, 2)
# print(ct)

### global deformation的改进实现 ###

init_polys = torch.randn(1, 128, 2)
init_polys = torch.sigmoid(init_polys) * 10

points_features = torch.randn(1, 64, 129)
points_features = torch.sigmoid(points_features) * 10

points_features = points_features.repeat(1, 2, 1)
print(points_features.shape)
pad = torch.zeros(1, 1, 129)
points_features = torch.cat((points_features, pad), 1)
print(points_features.shape)

trans_feature = torch.nn.Sequential(torch.nn.Conv2d(1, 1, kernel_size=3,
                                                                 padding=0, stride=2, bias=True),
                                                 torch.nn.ReLU(inplace=True),  # 使用ReLU激活函数
                                                 torch.nn.Conv2d(1, 1, kernel_size=1,
                                                                 stride=3, padding=0, bias=True))

points_features = trans_feature(points_features).view(1, -1)
print(points_features.shape)

trans_poly = torch.nn.Linear(in_features=484, out_features=256, bias=True)

offsets = trans_poly(points_features).view(1, 128, 2)
print(offsets.shape)

coarse_polys = offsets * 4 + init_polys
print(coarse_polys.shape)

