import torch
from .utils import decode_ct_hm, clip_to_image, get_gcn_feature
import argparse

# 导入设备序号，将新增创新点涉及到的tensor移动到cuda上
parser = argparse.ArgumentParser()
parser.add_argument("--device", default=0, type=int, help='device idx')  # 设置运行模型的设备

args = parser.parse_args()

device = args.device  # 关键点

class Refine(torch.nn.Module):
    def __init__(self, c_in=64, num_point=128, stride=4.):
        super(Refine, self).__init__()
        self.num_point = num_point
        self.stride = stride
        self.trans_feature = torch.nn.Sequential(torch.nn.Conv2d(c_in, 256, kernel_size=3,
                                                                 padding=1, bias=True),  # 输入通道数为64，输出通道数为256，卷积核大小3x3，图像周围填充1圈
                                                 torch.nn.ReLU(inplace=True),  # 使用ReLU激活函数
                                                 torch.nn.Conv2d(256, 64, kernel_size=1,
                                                                 stride=1, padding=0, bias=True))  # 输入通道数为256，输出通道数为64，卷积核大小1x1
        self.trans_poly = torch.nn.Linear(in_features=((num_point + 1) * 64),
                                          out_features=num_point * 4, bias=False)
        self.trans_fuse = torch.nn.Linear(in_features=num_point * 4,
                                          out_features=num_point * 2, bias=True)


        # self.trans_linear = torch.nn.Linear(in_features=484, out_features=num_point * 2, bias=True)

    # 论文中global deformation的实现
    def global_deform(self, points_features, init_polys):
        poly_num = init_polys.size(0)
        points_features = self.trans_poly(points_features)  # MLP隐藏层
        offsets = self.trans_fuse(points_features).view(poly_num, self.num_point, 2)  # MLP输出层
        coarse_polys = offsets * self.stride + init_polys.detach()
        return coarse_polys

    # 创新点2 -- 混合变形获取坐标偏移量
    # def blend_deform(self, points_features, init_polys):
    #     poly_num = init_polys.size(0)
    #     points_features = points_features.view(poly_num * 2, -1, self.num_point + 1)
    #     points_features = points_features.repeat(1, 4, 1)
    #     pad = torch.zeros(poly_num * 2, 1, self.num_point + 1)
    #     pad = pad.to(device)
    #
    #     points_features = torch.cat((points_features, pad), 1)
    #     points_features = points_features.to(device)
    #
    #     trans_conv = torch.nn.Sequential(torch.nn.Conv2d(poly_num * 2, 2, kernel_size=3,
    #                                                      padding=0, stride=2, bias=True),
    #                                      torch.nn.ReLU(inplace=True),  # 使用ReLU激活函数
    #                                      torch.nn.Conv2d(2, poly_num, kernel_size=1,
    #                                                      stride=3, padding=0, bias=True))
    #
    #     trans_conv = trans_conv.to(device)
    #
    #     points_features = trans_conv(points_features).view(poly_num, -1)
    #
    #     offsets = self.trans_linear(points_features).view(poly_num, self.num_point, 2)
    #     coarse_polys = offsets * self.stride + init_polys.detach()
    #
    #     return coarse_polys

    # self.refine(cnn_feature, detection[:, :2], poly_init, img_id, ignore=ignore_gloabal_deform)
    def forward(self, feature, ct_polys, init_polys, ct_img_idx, ignore=False):
        if ignore or len(init_polys) == 0:
            return init_polys
        h, w = feature.size(2), feature.size(3)  # 得到dla34网络输出的特征图的宽高，对于MyImg文件夹中的飞机图像来说，h=112，w=168
        poly_num = ct_polys.size(0)  # 获取目标个数
        print("目标个数：", poly_num)
    
        feature = self.trans_feature(feature)  # 对特征图进行通道数不变的resize

        ct_polys = ct_polys.unsqueeze(1).expand(init_polys.size(0), 1, init_polys.size(2))  # 将ct_polys的尺寸和init_polys的尺寸统一
        points = torch.cat([ct_polys, init_polys], dim=1)  # 将中心点和对应的初始轮廓点合并，torch.Size([4, 129, 2])
        feature_points = get_gcn_feature(feature, points, ct_img_idx, h, w).view(poly_num, -1)  # feature_points每个通道的每一行代表每个目标的每个轮廓点的特征信息
        print('feature_points的形状：', feature_points.shape)
        coarse_polys = self.global_deform(feature_points, init_polys)  # 将融合后的特征输入MLP网络得到新的预测偏移量，并与初始轮廓相加
        # coarse_polys = self.blend_deform(feature_points, init_polys)
        return coarse_polys

# 在visualize.py文件中，Decode(num_point=128, init_stride=10, coarse_stride=4, down_sample=4, min_ct_score=0.05)
class Decode(torch.nn.Module):
    def __init__(self, c_in=64, num_point=128, init_stride=10., coarse_stride=4., down_sample=4., min_ct_score=0.05):
        super(Decode, self).__init__()
        self.stride = init_stride
        self.down_sample = down_sample
        self.min_ct_score = min_ct_score
        self.refine = Refine(c_in=c_in, num_point=num_point, stride=coarse_stride)  # 深度特征提取器

    def train_decode(self, data_input, output, cnn_feature):
        wh_pred = output['wh']
        ct_01 = data_input['ct_01'].bool()
        ct_ind = data_input['ct_ind'][ct_01]
        ct_img_idx = data_input['ct_img_idx'][ct_01]
        _, _, height, width = data_input['ct_hm'].size()
        ct_x, ct_y = ct_ind % width, ct_ind // width

        if ct_x.size(0) == 0:
            ct_offset = wh_pred[ct_img_idx, :, ct_y, ct_x].view(ct_x.size(0), 1, 2)
        else:
            ct_offset = wh_pred[ct_img_idx, :, ct_y, ct_x].view(ct_x.size(0), -1, 2)

        ct_x, ct_y = ct_x[:, None].to(torch.float32), ct_y[:, None].to(torch.float32)
        ct = torch.cat([ct_x, ct_y], dim=1)

        init_polys = ct_offset * self.stride + ct.unsqueeze(1).expand(ct_offset.size(0),
                                                                      ct_offset.size(1), ct_offset.size(2))
        coarse_polys = self.refine(cnn_feature, ct, init_polys, ct_img_idx.clone())

        output.update({'poly_init': init_polys * self.down_sample})
        output.update({'poly_coarse': coarse_polys * self.down_sample})
        return

    # visualize中进行测试，采用test_decode
    def test_decode(self, cnn_feature, output, K=100, min_ct_score=0.05, ignore_gloabal_deform=False):
        hm_pred, wh_pred = output['ct_hm'], output['wh']  # 获取z字典中的两个特征图，ct_hm为包含物体中心点的heatmap，wh为包含中心点偏移量的特征图
        print("cnn_feature的形状：", cnn_feature.shape)
        print("heatmap的形状：", hm_pred.shape)
        print("偏移量shape的形状：", wh_pred.shape)
        poly_init, detection = decode_ct_hm(torch.sigmoid(hm_pred), wh_pred,
                                            K=K, stride=self.stride)  # 获得初始轮廓点的坐标和这些轮廓点信息构成的detection序列
        valid = detection[0, :, 2] >= min_ct_score  # 从detection中第1个通道的第3列筛选出大于0.05的值，相当于根据min_ct_score的值筛选detection中的scores
        print("valid的形状：", valid.shape)
        poly_init, detection = poly_init[0][valid], detection[0][valid]  # 将100个元素中灰度值>min_ct_score的元素保留，删除其余元素
        print("筛选后poly_init的形状：", poly_init.shape)  # torch.Size([4, 128, 2])，4代表当前图像中目标的个数，相当于给图像中每个目标都创建了一个由128个点构成的初始轮廓
        print("筛选后detection的形状：", detection.shape)  # torch.Size([4, 4])

        init_polys = clip_to_image(poly_init, cnn_feature.size(2), cnn_feature.size(3))  # 让绘制出的初始轮廓点的坐标不超过特征图的维度
        output.update({'poly_init': init_polys * self.down_sample})  # 将初始轮廓点还原到原图大小放入z字典

        img_id = torch.zeros((len(poly_init), ), dtype=torch.int64)  # 获得一个大小为目标个数的0矩阵，相当于给当前图像中的每个目标标上序号
        poly_coarse = self.refine(cnn_feature, detection[:, :2], poly_init, img_id, ignore=ignore_gloabal_deform)  # 对poly_init进行global deformation操作
        coarse_polys = clip_to_image(poly_coarse, cnn_feature.size(2), cnn_feature.size(3))  # 让绘制出的粗糙轮廓点的坐标不超过特征图的维度
        print('coarse_polys的形状：', coarse_polys.shape)
        output.update({'poly_coarse': coarse_polys * self.down_sample})  # 将粗糙轮廓点还原到原图大小放入z字典
        output.update({'detection': detection})  # 将detection放入z字典
        print(poly_coarse.shape)  # torch.Size([4, 128, 2])，4表示每个轮廓点有四个特征，128表示轮廓点的个数，2表示每个轮廓点的横纵坐标
        print(detection.shape)  # torch.Size([4, 4])
        return

    def forward(self, data_input, cnn_feature, output=None, is_training=True, ignore_gloabal_deform=False):
        if is_training:
            self.train_decode(data_input, output, cnn_feature)
        else:
            self.test_decode(cnn_feature, output, min_ct_score=self.min_ct_score,
                             ignore_gloabal_deform=ignore_gloabal_deform)

