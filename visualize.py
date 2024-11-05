from network import make_network
import tqdm
import torch
import os
import nms
import post_process
from dataset.data_loader import make_demo_loader
from train.model_utils.utils import load_network
import argparse
import importlib
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from configs import coco
import random
from torchvision import transforms


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
parser.add_argument("--output_dir", default='L:/Deeplearning/e2ec-main_ACM/MyImg_result', help='/path/to/output_dir')  # 设置存放分割后图像的文件夹
# parser.add_argument("--output_dir", default=None, help='/path/to/output_dir')  # 设置存放分割后图像的文件夹
parser.add_argument("--device", default=0, type=int, help='device idx')  # 设置运行模型的设备

args = parser.parse_args()

# 定义参数配置函数
def get_cfg(args):
    # cfg = importlib.import_module('configs.' + args.config_file).config
    cfg = coco.config  # 创建coco.py文件中config对象的实例为cfg
    cfg.test.with_nms = bool(args.with_nms)
    cfg.test.test_stage = args.stage
    cfg.test.ct_score = args.ct_score
    return cfg

def bgr_to_rgb(img):
    return img[:, :, [2, 1, 0]]  # 将可能的BGR图像转换为RGB图像，即先取出图像的第三个通道，再取出第二个通道，最后取出第一个通道

# 利用标准差和均值对图像进行预处理
def unnormalize_img(img, mean, std):
    """
    img: [3, h, w]
    """
    img = img.detach().cpu().clone()  # 将原始的张量img分离一份出来，移动到cpu上，再克隆一份。意思是新的img和img的图像信息相同，但各自求梯度互不影响
    img *= torch.tensor(std).view(3, 1, 1)  # 将标准差转换为tensor并reshape为3x1x1的张量
    img += torch.tensor(mean).view(3, 1, 1)  # 将均值转换为tensor并reshape为3x1x1的张量
    min_v = torch.min(img)  # 找到图像中最大的像素值
    img = (img - min_v) / (torch.max(img) - min_v)
    return img

# 定义分割结果可视化函数
class Visualizer(object):
    def __init__(self, cfg):  # 将cfg设置为必须参数
        self.cfg = cfg

    # def visualize_ex(self, output, backbone_feature, batch, img_save_dir=None, feature_save_dir=None):
    def visualize_ex(self, output, batch, img_save_dir=None):
        inp = bgr_to_rgb(unnormalize_img(batch['inp'][0], self.cfg.data.mean,
                                         self.cfg.data.std).permute(1, 2, 0))  # 将预处理后的图像进行维度重设，原来第一维的维度数值变为第三维的，原来第二维的维度数值变为第一维的，原来第三维的维度数值变为第二维的
        print(inp.shape)  # [x, x, 3]
        ex = output['py']  # 模型的输出是一个字典，取出键py的值赋给ex，此时ex是包含三个轮廓的数组，其中最后一个元素是最终轮廓线
        ex = ex[-1]
        # ex = ex[-1] if isinstance(ex, list) else ex  # 如果ex是列表，取出其最后一个元素，否则返回ex本身
        ex = ex.detach().cpu().numpy()  # 将ex分离一份到cpu上，并转换为numpy数组，便于后续的可视化

        fig, ax = plt.subplots(1, figsize=(20, 10))  # 创建一块大小为20x10的画布，放一个图像
        fig.tight_layout()  # 自动调节画布中的参数显示的位置，避免文字重复
        ax.axis('off')
        ax.imshow(inp)  # 展示无轮廓的原始图像

        colors = np.array([  # 设置最终轮廓显示的颜色，这里定义一个np.array是为了使颜色显示具有随机性，我为了写论文全设置为黄色
            [255, 215, 0],
            [255, 215, 0],
            [255, 215, 0],
            [255, 215, 0],
            [255, 215, 0],
            [255, 215, 0],
            [255, 215, 0],
            [255, 215, 0],
            [255, 215, 0],
            [255, 215, 0],
        ]) / 255.
        np.random.shuffle(colors)  # 打乱colors数组的元素顺序
        colors = cycle(colors)  # 将colors数组转换为迭代器，可以使用next遍历其中的元素
        color = next(colors).tolist()  # 将colors数组中的一个元素作为列表返回

        # i = random.randint(0, 6)
        # plt_color = ['red', 'chocolate', 'gold', 'lime', 'blue', 'purple', 'hotpink']
        # plt.contour(ex, [0], linewidths=6, linestyles='solid', colors=plt_color[2])  # 绘制最终轮廓线
        for i in range(len(ex)):  # 遍历ex中的每一张图像
            poly = ex[i]  # 取出ex中的第i张图像的最终轮廓点坐标
            poly = np.append(poly, [poly[0]], axis=0)
            ax.plot(poly[:, 0], poly[:, 1], color=color, lw=6)  # 取poly的第0维和第1维的全部元素为横纵坐标绘图，即绘制第i张图像的轮廓


        if img_save_dir is not None:
            plt.savefig(fname=img_save_dir, bbox_inches='tight')  # 将展示的图像保存到指定文件夹
            plt.close()
        else:
            plt.show()

        # unloader = transforms.ToPILImage()
        # feature = backbone_feature.cpu().clone()
        # feature = feature.squeeze(0)
        # feature = feature[2:3, :, :]
        # image = unloader(feature)
        # image.save('feature_1.png')



    def visualize(self, output, backbone_feature, batch):
        if args.output_dir != 'None':
            img_file_name = os.path.join(args.output_dir, batch['meta']['img_name'][0])  # 设置输出图像的文件名
            # feature_file_name = os.path.join(args.output_dir, 'feature' + batch['meta']['img_name'][0])  # 设置输入图像对应的backbone特征图的文件名
        else:
            file_name = None
        self.visualize_ex(output, batch, img_save_dir=img_file_name)  # 如果有文件名，将输出结果保存到该文件夹中；如果没有，直接在屏幕上展示结果
        # self.visualize_ex(output, backbone_feature, batch, img_save_dir=img_file_name, feature_save_dir =feature_file_name)

def run_visualize(cfg):
    network = make_network.get_network(cfg).cuda()  # 生成模型网络，移动到cuda上
    load_network(network, args.checkpoint)  # 将权值文件导入网络
    network.eval()  # 将网络调为测试模式，不求梯度

    data_loader = make_demo_loader(args.image_dir, cfg=cfg)  # 将原始图像封装为迭代器，用于模型分割测试
    visualizer = Visualizer(cfg)  # 生成可视化器
    for batch in tqdm.tqdm(data_loader):  # 调用tqdm在终端显示模型分割进度,tqdm将data_loader转换为一个进度迭代器，对其进行迭代就可显示进度条
        '''
        由于在demo_dataset.py文件中，Dataset这个类存在__getitem__方法，而data_loader中的每一个batch都是Dataset类的一个实例，所以按照数组索引的方式就能调用该实例的__getitem__方法，生成一个名为ret的嵌套键值对。
        在这个键值对中有两个键名：'inp'和'meta'，'inp'对应的是待分割图像，'meta'对应一个字典
        '''
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()  # 将待分割图像移动到cuda上
        with torch.no_grad():  # 不计算梯度
            output, backbone_feature = network(batch['inp'], batch)  # batch['inp']是网络输入，batch表示告诉网络现在是测试模式，不训练和计算梯度
        if args.with_post_process:
            post_process.post_process(output)  # 对输出结果进行平滑
        if args.with_nms:
            # nms.post_process(output)  # 对输出结果进行非极大值抑制
            pass
        visualizer.visualize(output, backbone_feature, batch)  # 输出结果可视化

if __name__ == "__main__":  # 在当前文件中才运行以下代码
    cfg = get_cfg(args)  # 设置模型基本参数
    torch.cuda.set_device(args.device)  # 将模型和所有参数移动到cuda上
    run_visualize(cfg)  # 运行网络进行测试




