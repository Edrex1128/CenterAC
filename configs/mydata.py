#!/user/bin/env python
# _*_ coding: utf-8 _*_
# ==================================================
# @File_name: e2ec-main_ACM -> mydata
# @Software: PyCharm
# @Author: 张福正
# @Time: 2024/4/17 16:39
# ==================================================

from .base import commen, data, model, train, test

data.scale = None

model.heads['ct_hm'] = 1

train.batch_size = 12
train.epoch = 100
train.dataset = 'mydata_train'

test.dataset = 'mydata_test'

class config(object):
    commen = commen  # 等号左边的commen是变量名，右边的commen是从base中导入的commen对象。相当于创建一个commen对象的实例，该实例名为commen
    data = data
    model = model
    train = train
    test = test