#!/user/bin/env python
# _*_ coding: utf-8 _*_
# ==================================================
# @File_name: e2ec-main_ACM -> get_contour_ax
# @Software: PyCharm
# @Author: 张福正
# @Time: 2023/9/25 15:15
# ==================================================


import numpy as np
import matplotlib.pyplot as plt
import torch


# 从等值线中提取坐标点和属性值
def get_contour_verts(cn):
    contours = []
    idx = 0
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
            idx += 1
    return contours


# 等值线绘图及提取数据示例
# x = np.arange(-3.0, 3.0, delta)
# y = np.arange(-2.0, 2.0, delta)
# X, Y = np.meshgrid(x, y)
# Z = (X - Y) * 2
LSF = np.array([[3, 2, 1],
                [2, -2, 1],
                [-1, -2, 3]])
# fig, ax = plt.subplots()
CS = plt.contour(LSF, [0], linewidths = 2.0, linestyles = 'solid', colors='r')
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Simplest default with labels')
zeroLS = get_contour_verts(CS)
zeroLS = torch.tensor(zeroLS)
zeroLS1 = zeroLS.reshape((1, zeroLS.shape[0], zeroLS.shape[1]))
print(zeroLS1.shape)
plt.show()
# 打印提取的等值线坐标点和属性值
# 按每条线（段）提取，顺序标出 id
