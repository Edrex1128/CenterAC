import torch
import torch.utils.data
from .collate_batch import collate_batch
from .info import DatasetInfo

def make_dataset(dataset_name, is_test, cfg):
    info = DatasetInfo.dataset_info[dataset_name]
    if is_test:
        from .test import coco, cityscapes, cityscapesCoco, sbd, kitti, mydata
        dataset_dict = {'coco': coco.CocoTestDataset, 'cityscapes': cityscapes.Dataset,
                        'cityscapesCoco': cityscapesCoco.CityscapesCocoTestDataset,
                        'kitti': kitti.KittiTestDataset, 'sbd': sbd.SbdTestDataset, 'mydata': mydata.MydataTestDataset}
        dataset = dataset_dict[info['name']]
    else:
        from .train import coco, cityscapes, cityscapesCoco, sbd, kitti, mydata
        dataset_dict = {'coco': coco.CocoDataset, 'cityscapes': cityscapes.Dataset,
                        'cityscapesCoco': cityscapesCoco.CityscapesCocoDataset,
                        'kitti': kitti.KittiDataset, 'sbd': sbd.SbdDataset, 'mydata': mydata.MydataDataset}
        dataset = dataset_dict[info['name']]
    dataset = dataset(info['anno_dir'], info['image_dir'], info['split'], cfg)
    return dataset


def make_data_sampler(dataset, shuffle):
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)  # 将文件夹中的图像打乱顺序后生成一个迭代器，每次迭代按顺序处理一张图像
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)  # dataset输入进SequentialSampler后会自动以dataset[int]的形式调用，从而返回ret字典。将文件夹中的图像按原顺序生成一个迭代器，每次迭代按顺序处理一张图像
    return sampler

def make_ddp_data_sampler(dataset, shuffle):
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    return sampler

def make_batch_data_sampler(sampler, batch_size, drop_last):
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last)
    return batch_sampler

def make_train_loader(cfg):
    batch_size = cfg.train.batch_size
    shuffle = True
    drop_last = False
    dataset_name = cfg.train.dataset

    dataset = make_dataset(dataset_name, is_test=False, cfg=cfg)
    sampler = make_data_sampler(dataset, shuffle)
    batch_sampler = make_batch_data_sampler(sampler, batch_size, drop_last)
    num_workers = cfg.train.num_workers
    collator = collate_batch
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collator
    )
    return data_loader

def make_test_loader(cfg, is_distributed=True):
    batch_size = 1
    shuffle = True if is_distributed else False
    drop_last = False
    dataset_name = cfg.test.dataset

    dataset = make_dataset(dataset_name, is_test=True, cfg=cfg)
    sampler = make_data_sampler(dataset, shuffle)
    batch_sampler = make_batch_data_sampler(sampler, batch_size, drop_last)
    num_workers = 1
    collator = collate_batch
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collator
    )
    return data_loader


def make_data_loader(is_train=True, is_distributed=False, cfg=None):
    if is_train:
        return make_train_loader(cfg), make_test_loader(cfg, is_distributed)
    else:
        return make_test_loader(cfg, is_distributed)

def make_demo_loader(data_root=None, cfg=None):
    from .demo_dataset import Dataset
    batch_size = 1  # 测试时图片数量少，故一个批次处理一张图像
    shuffle = False  # 不打乱图像顺序
    drop_last = False
    dataset = Dataset(data_root, cfg)  # 根据给定的图像路径创建Dataset实例
    sampler = make_data_sampler(dataset, shuffle)  # 将文件夹中的图像按原顺序生成一个迭代器，每次迭代按顺序处理一张图像
    batch_sampler = make_batch_data_sampler(sampler, batch_size, drop_last)  # 将sampler分为若干个batch sampler，每个batch sampler有一张图像
    num_workers = 1  # dataloader一次将一个batch的数据放入内存
    collator = collate_batch
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collator
    )  # 生成最终的dataloader
    return data_loader

def make_ddp_train_loader(cfg):
    batch_size = cfg.train.batch_size
    shuffle = True
    drop_last = False
    dataset_name = cfg.train.dataset

    dataset = make_dataset(dataset_name, is_test=False, cfg=cfg)
    sampler = make_ddp_data_sampler(dataset, shuffle)
    collator = collate_batch
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=batch_size,
        collate_fn=collator,
        pin_memory=False,
        drop_last=drop_last
    )
    return data_loader

def make_ddp_data_loader(is_train=True, is_distributed=False, cfg=None):
    if is_train:
        return make_ddp_train_loader(cfg), make_test_loader(cfg, is_distributed)
    else:
        return make_test_loader(cfg, is_distributed)

