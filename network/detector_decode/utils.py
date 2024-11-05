import torch.nn as nn
import torch

def nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)  # 将feat中索引为ind的元素取出来构成新的feat
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

def topk(scores, K=100):
    batch, cat, height, width = scores.size()  # 获取heatmap的尺寸，batch=1；cat为通道数，即类别数；height和weight分别为heatmap的高宽

    # 第一次变形
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)  # 将heatmap的每个通道拉伸成向量，并从大到小取每个向量的前100个元素和元素在被取出前的索引，这里取出了80x100个元素

    topk_inds = topk_inds % (height * width)  # topk_inds < height * width，此步运算可以忽略
    topk_ys = (topk_inds / width).int().float()  # 得到目前取出元素在heatmap中的纵坐标
    topk_xs = (topk_inds % width).int().float()  # 得到目前取出元素在heatmap中的横坐标

    # 第二次变形
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)  # 将topk_scores中每行向量首尾连接形成一个新的向量，并从大到小取该向量的前100个元素和元素在被取出前的索引，这里就取出了100个元素
    topk_clses = (topk_ind / K).int()  # 得到取出的100个元素分别属于的类别编号
    # topk_clses = topk_clses.view(batch, K, 1).float()
    topk_inds = gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)  # 得到最新取出的100个元素在第一次变形后的tensor中的索引
    topk_ys = gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)  # 得到最新取出的100个元素在heatmap中的纵坐标
    topk_xs = gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)  # 得到最新取出的100个元素在heatmap中的横坐标
    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs  # 返回最终取出的100个元素值，它们在第一次变形后的tensor中的索引，每个元素对应的类别编号，每个元素的横纵坐标


# hm_pred, wh_pred = output['ct_hm'], output['wh']  # 获取z字典中的两个特征图
# poly_init, detection = decode_ct_hm(torch.sigmoid(hm_pred), wh_pred,K=K, stride=self.stride)  # 绘制初始的轮廓点和这些轮廓点坐标构成的detection序列
def decode_ct_hm(ct_hm, wh, reg=None, K=100, stride=10.):
    batch, cat, height, width = ct_hm.size()  # 获取heatmap的尺寸，从左到右为batch_size，特征图通道数，图像高度，图像宽度
    bs, ic, h, w = wh.size()
    print(batch, cat, height, width)  # heatmap的尺寸，以MyImg文件夹中的飞机图像为例，此处为1 80 112 168
    print(bs, ic, h, w)  # 包含横纵偏移量的特征图的尺寸，以MyImg文件夹中的飞机图像为例，此处为1 256 112 168
    ct_hm = nms(ct_hm)  # 对特征图进行maxpooling，将可能的中心点保留，其余点置0
    scores, inds, clses, ys, xs = topk(ct_hm, K=K)  # 对特征图中的像素点进行重排序，选取100个点作为初始轮廓点
    wh = transpose_and_gather_feat(wh, inds)  # 根据100个元素的索引取出各自元素的在x和y方向的偏移量
    print("gather后的wh形状：", wh.shape)
    wh = wh.view(batch, K, -1, 2)  # 对偏移量的tensor形状进行调整，以MyImg文件夹中的飞机图像为例，此处为1 100 128 2

    # reg表示中心点需要微调的距离
    if reg is not None:
        reg = transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1)
        ys = ys.view(batch, K, 1)
        print('横坐标的形状：', xs.shape)

    clses = clses.view(batch, K, 1).float()  # 获取100个元素各自的类别
    scores = scores.view(batch, K, 1)  # 获取100个元素各自的灰度值
    ct = torch.cat([xs, ys], dim=2)  # 获取100个元素各自的坐标，形状为1 100 2
    poly = ct.unsqueeze(2).expand(batch, K, wh.size(2), 2) + wh * stride  # 将100个元素的坐标和横纵偏移量相加得到初始轮廓
    detection = torch.cat([ct, scores, clses], dim=2)  # 存储100个元素的信息，一共4列，前两列为中心点横纵坐标，后两列分别为中心点灰度值和类别号
    print("poly的形状：", poly.shape)  # torch.Size([1, 100, 128, 2])
    print('detection的形状：', detection.shape)  # torch.Size([1, 100, 4])
    return poly, detection  # 返回选取的像素点坐标构成的序列

# 保证数据的稳定性
def clip_to_image(poly, h, w):
    poly[..., :2] = torch.clamp(poly[..., :2], min=0)
    poly[..., 0] = torch.clamp(poly[..., 0], max=w-1)
    poly[..., 1] = torch.clamp(poly[..., 1], max=h-1)
    return poly

# feature_points = get_gcn_feature(feature, points, ct_img_idx, h, w).view(poly_num, -1)
def get_gcn_feature(cnn_feature, img_poly, ind, h, w):
    img_poly = img_poly.clone()
    img_poly[..., 0] = img_poly[..., 0] / (w / 2.) - 1
    img_poly[..., 1] = img_poly[..., 1] / (h / 2.) - 1
    batch_size = cnn_feature.size(0)
    gcn_feature = torch.zeros([img_poly.size(0), cnn_feature.size(1), img_poly.size(1)]).to(img_poly.device)
    # print('ind:', ind)
    for i in range(batch_size):  # batch_size=1
        poly = img_poly[ind == i].unsqueeze(0)  # 取出img_poly中的所有元素赋值给poly，并扩展为四维张量
        feature = torch.nn.functional.grid_sample(cnn_feature[i:i+1], poly)[0].permute(1, 0, 2)  # 根据每个中心点和其对应的轮廓点坐标取出特征图上的对应特征
        print("grid_sample后的特征图形状：", feature.shape)
        gcn_feature[ind == i] = feature

    print('gcn_feature的形状：', gcn_feature.shape)
    return gcn_feature
