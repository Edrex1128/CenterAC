import torch.nn as nn
from .snake import Snake
from .utils import prepare_training, prepare_testing_init, img_poly_to_can_poly, get_gcn_feature
import torch
import tqdm
import argparse
from visualize import bgr_to_rgb, unnormalize_img
from dataset.data_loader import make_demo_loader
from DoG_NDF_embed import *
from configs import coco
import torchvision.transforms as transforms

# 定义parser用于模型参数配置
parser = argparse.ArgumentParser()

parser.add_argument("--config_file", default='L:/Deeplearning/e2ec-main_ACM/configs/coco.py', help='/path/to/config_file.py')  # 选择coco数据集对应的训练参数
parser.add_argument("--image_dir", default='L:/Deeplearning/e2ec-main_ACM/MyImg', help='/path/to/images')  # 设置存放原图的文件夹
parser.add_argument("--checkpoint", default='L:/Deeplearning/e2ec-main_ACM/model/model_coco.pth', help='/path/to/model_weight.pth')  # 指定模型的权值文件
parser.add_argument("--ct_score", default=0.3, help='threshold to filter instances', type=float)  # 设置筛选实例的阈值为0.3
parser.add_argument("--with_nms", default=True, type=bool,
                    help='if True, will use nms post-process operation', choices=[True, False])  # 使用非极大值抑制
parser.add_argument("--with_post_process", default=False, type=bool,
                    help='if True, Will filter out some jaggies', choices=[True, False])  # 对预测轮廓进行平滑，抗锯齿
parser.add_argument("--stage", default='final-dml', help='which stage of the contour will be generated',
                    choices=['init', 'coarse', 'final', 'final-dml'])  # 设置生成轮廓的阶段，一共有4个阶段：初始轮廓，粗轮廓，最终轮廓，微调后的最终轮廓
parser.add_argument("--output_dir", default='L:/Deeplearning/e2ec-main/MyImg_result', help='/path/to/output_dir')  # 设置存放分割后图像的文件夹
parser.add_argument("--device", default=0, type=int, help='device idx')  # 设置运行模型的设备

args = parser.parse_args()

def get_cfg(args):
    # cfg = importlib.import_module('configs.' + args.config_file).config
    cfg = coco.config  # 创建coco.py文件中config对象的实例为cfg
    cfg.test.with_nms = bool(args.with_nms)
    cfg.test.test_stage = args.stage
    cfg.test.ct_score = args.ct_score
    return cfg

def nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


# 在visualize.py文件中，Evolution(evole_ietr_num=3, evolve_stride=1, ro=4)
class Evolution(nn.Module):
    def __init__(self, evole_ietr_num=3, evolve_stride=1., ro=4.):
        super(Evolution, self).__init__()
        assert evole_ietr_num >= 1
        self.evolve_stride = evolve_stride  # 1
        self.ro = ro  # 4
        self.evolve_gcn = Snake(state_dim=128, feature_dim=64+2, conv_type='dgrid')
        self.iter = evole_ietr_num - 1  # 2
        for i in range(self.iter):
            evolve_gcn = Snake(state_dim=128, feature_dim=64+2, conv_type='dgrid')
            self.__setattr__('evolve_gcn'+str(i), evolve_gcn)  # 创建两个Snake模型，分别命名为evolve_gcn1和evolve_gcn2

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def prepare_training(self, output, batch):
        init = prepare_training(output, batch, self.ro)
        return init

    def prepare_testing_init(self, output):
        init = prepare_testing_init(output['poly_coarse'], self.ro)
        return init

    def prepare_testing_evolve(self, output, h, w):
        img_init_polys = output['img_init_polys']  # 得到原始的coarse_polys
        img_init_polys[..., 0] = torch.clamp(img_init_polys[..., 0], min=0, max=w-1)  # 将coarse_polys每个通道的第1列的值限制到0-167
        img_init_polys[..., 1] = torch.clamp(img_init_polys[..., 1], min=0, max=h-1)  # 将coarse_polys每个通道的第2列的值限制到0-111
        output.update({'img_init_polys': img_init_polys})  # 更新img_init_polys
        return img_init_polys

    # evolve_poly(self, Snake, y[-1], img_init_polys, can_init_polys, [4, 4, 4, 4], stride=1., ignore=False)
    def evolve_poly(self, snake, cnn_feature, i_it_poly, c_it_poly, ind, stride=1., ignore=False):
        if ignore:
            return i_it_poly * self.ro
        if len(i_it_poly) == 0:
            return torch.zeros_like(i_it_poly)
        h, w = cnn_feature.size(2), cnn_feature.size(3)  # 获取特征图的高宽
        init_feature = get_gcn_feature(cnn_feature, i_it_poly, ind, h, w)
        c_it_poly = c_it_poly * self.ro
        init_input = torch.cat([init_feature, c_it_poly.permute(0, 2, 1)], dim=1)
        offset = snake(init_input).permute(0, 2, 1)
        print("MDA生成的坐标形状：", offset.shape)
        i_poly = i_it_poly * self.ro + offset * stride
        return i_poly

    def foward_train(self, output, batch, cnn_feature):
        ret = output
        init = self.prepare_training(output, batch)
        py_pred = self.evolve_poly(self.evolve_gcn, cnn_feature, init['img_init_polys'],
                                   init['can_init_polys'], init['py_ind'], stride=self.evolve_stride)
        py_preds = [py_pred]
        for i in range(self.iter):
            py_pred = py_pred / self.ro
            c_py_pred = img_poly_to_can_poly(py_pred)
            evolve_gcn = self.__getattr__('evolve_gcn' + str(i))
            py_pred = self.evolve_poly(evolve_gcn, cnn_feature, py_pred, c_py_pred,
                                       init['py_ind'], stride=self.evolve_stride)
            py_preds.append(py_pred)
        ret.update({'py_pred': py_preds, 'img_gt_polys': init['img_gt_polys'] * self.ro})
        return output

    # 在visualize.py文件中执行以下代码，此时output={'poly_coarse': coarse_polys * self.down_sample, 'detection': detection}，cnn_feature=y[-1]
    def foward_test(self, output, cnn_feature, inp, ignore):
        ret = output
        hm = output['ct_hm']
        hm = nms(hm)

        detection = output['detection']
        cls_inds = detection[:, 3]  # 获取每个目标对应的heatmap的通道索引
        cls_num = cls_inds.numel()

        with torch.no_grad():
            init = self.prepare_testing_init(output)  # init是一个字典，第1项是coarse_polys，第2项是处理后的coarse_polys，第三项是[4, 4, 4, 4]
            img_init_polys = self.prepare_testing_evolve(init, cnn_feature.size(2), cnn_feature.size(3))  # 在visualize.py文件中，cnn_feature.size(2)=112，cnn_feature.size(3)=168，本行代码的作用是将coarse_polys中每个点的坐标限制在特征图的图像域内，便于画图
            # img_init_polys = img_init_polys * 4
            pys = []  # 创建数组用于存放各个演化阶段的轮廓
            phi_list = []  # 创建数组用于存放每个目标的轮廓演化后的水平集函数
            pys.append(img_init_polys * 4)

            py = self.evolve_poly(self.evolve_gcn, cnn_feature, img_init_polys, init['can_init_polys'], init['py_ind'],
                                  ignore=ignore[0], stride=self.evolve_stride)  # 对粗糙轮廓进行第1次演化，evolve_poly调用Multi direction alignment模块
            # print(py.shape)
            # py = ACM_mid(inp, py)  # 调用ACM实现active correction

            pys.append(py)

            # 对粗糙轮廓进行第2和3次演化
            for i in range(self.iter):
                py = py / self.ro
                c_py = img_poly_to_can_poly(py)
                evolve_gcn = self.__getattr__('evolve_gcn' + str(i))
                py = self.evolve_poly(evolve_gcn, cnn_feature, py, c_py, init['py_ind'],
                                      ignore=ignore[i + 1], stride=self.evolve_stride)

            pys.append(py)

            # 创新点1
            # py = ACM_final(inp, py)  # 调用ACM实现active correction
            # cls_hm = int(cls_inds[0])
            # py = ACM_final(hm[0][cls_hm], py)  # 调用ACM实现active correction
            # py = py * self.ro
            # pys.append(py)


            ret.update({'py': pys})
            print('最终轮廓的数据形状：', output['py'][-1].shape)  # 查看最终轮廓的数据形状，torch.Size([4, 128, 2])，第一个数字代表图像中的目标个数，第二个数字代表构成轮廓线的像素点个数，第三个数字代表每个像素点的坐标数
        return output

    def forward(self, output, cnn_feature, x_ori, batch=None, test_stage='final-dml'):
        if batch is not None and 'test' not in batch['meta']:  # 在训练模式下
            self.foward_train(output, batch, cnn_feature)
        else:
            ignore = [False] * (self.iter + 1)
            if test_stage == 'coarse' or test_stage == 'init':  # 在测试模式下查看初始和粗糙轮廓线
                ignore = [True for _ in ignore]
            if test_stage == 'final':  # 在测试模式下查看最终轮廓线
                ignore[-1] = True

        cfg = get_cfg(args)
        inp = bgr_to_rgb(unnormalize_img(x_ori[0], cfg.data.mean, cfg.data.std).permute(1, 2, 0))
        self.foward_test(output, cnn_feature, inp, ignore=ignore)


        return output

