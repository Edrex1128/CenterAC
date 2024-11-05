#!/user/bin/env python
# _*_ coding: utf-8 _*_
# ==================================================
# @File_name: e2ec-main_ACM -> txt_generate
# @Software: PyCharm
# @Author: 张福正
# @Time: 2024/4/17 15:00
# ==================================================

from glob import glob
import random

# 该目录存储图片数据
patch_fn_list = glob('L:/Deeplearning/e2ec-main_ACM/Mydataset/JPEGImages/*.jpg')
# 返回存储图片名的列表，不包含图片的后缀
patch_fn_list = [fn.split('\\')[-1][:-4] for fn in patch_fn_list]
# 将图片打乱顺序
random.shuffle(patch_fn_list)

# 按照7:3比例划分train和val
train_num = int(0.7 * len(patch_fn_list))
train_patch_list = patch_fn_list[:train_num]
valid_patch_list = patch_fn_list[train_num:]

# produce train/valid/trainval txt file
split = ['train_total', 'val_total']

for s in split:
    # 存储文本文件的地址
    save_path = 'L:/Deeplearning/e2ec-main_ACM/Mydataset/' + s + '.txt'

    if s == 'train_total':
        with open(save_path, 'w') as f:
            for fn in train_patch_list:
                # 将训练图像的地址写入train.txt文件
                f.write('%s\n' % fn)
    elif s == 'val_total':
        with open(save_path, 'w') as f:
            for fn in valid_patch_list:
                # 将验证图像的地址写入val.txt文件
                f.write('%s\n' % fn)
    print('Finish Producing %s txt file to %s' % (s, save_path))



