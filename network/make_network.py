import torch.nn as nn
from .backbone.dla import DLASeg
from .detector_decode.refine_decode import Decode
from .evolve.evolve import Evolution
import torch


# E2EC模型的完整网络结构
class Network(nn.Module):
    def __init__(self, cfg=None):
        super(Network, self).__init__()  # 让Network继承nn.Module的方法，此时已经实现魔法函数

        # 在visualize.py文件中，cfg是coco.config的实例，其包含了base.py文件中的model，commen等对象
        num_layers = cfg.model.dla_layer  # 34
        head_conv = cfg.model.head_conv  # 256
        down_ratio = cfg.commen.down_ratio  # 4
        heads = cfg.model.heads  # {'ct_hm': 20, 'wh': commen.points_per_poly * 2}
        self.test_stage = cfg.test.test_stage  # 'final-dml'，即设置轮廓线演化的阶段，这里只看最终迭代出的轮廓线，无须对中间过程中的轮廓进行可视化

        # DLA-34是E2EC的backbone，此处创建backbone的网络实例。
        # 在visualize.py文件中，DLASeg('dla34', {'ct_hm': 20, 'wh': commen.points_per_poly * 2}, pretrained=True, down_ratio=4, final_kernel=1, last_level=5, head_conv=256, use_dcn=True)
        self.dla = DLASeg('dla{}'.format(num_layers), heads,
                          pretrained=True,
                          down_ratio=down_ratio,
                          final_kernel=1,
                          last_level=5,
                          head_conv=head_conv, use_dcn=cfg.model.use_dcn)

        # 创建解码器的网络实例
        # 在visualize.py文件中，Decode(num_point=128, init_stride=10, coarse_stride=4, down_sample=4, min_ct_score=0.05)
        self.train_decoder = Decode(num_point=cfg.commen.points_per_poly, init_stride=cfg.model.init_stride,
                                    coarse_stride=cfg.model.coarse_stride, down_sample=cfg.commen.down_ratio,
                                    min_ct_score=cfg.test.ct_score)  # 绘制出initial contour和coarse contour
        self.gcn = Evolution(evole_ietr_num=cfg.model.evolve_iters, evolve_stride=cfg.model.evolve_stride,
                             ro=cfg.commen.down_ratio)  # 创建曲线演化模块的网络实例，在visualize.py文件中，Evolution(evole_ietr_num=3, evolve_stride=1, ro=4)

    def forward(self, x, batch=None):
        x_ori = x
        output, cnn_feature = self.dla(x)
        if 'test' not in batch['meta']:
            self.train_decoder(batch, cnn_feature, output, is_training=True)
        else:
            with torch.no_grad():
                if self.test_stage == 'init':
                    ignore = True
                else:
                    ignore = False
                self.train_decoder(batch, cnn_feature, output, is_training=False, ignore_gloabal_deform=ignore)  # 此时output字典中包含两个特征图，粗糙轮廓点的像素值和detection
        output = self.gcn(output, cnn_feature, x_ori, batch, test_stage=self.test_stage)
        return output, cnn_feature


def get_network(cfg):
    network = Network(cfg)
    return network
