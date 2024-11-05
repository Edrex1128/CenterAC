import torch
import os
import torch.nn.functional
from termcolor import colored

def load_model(net, optim, scheduler, recorder, model_path, map_location=None):
    strict = True

    if not os.path.exists(model_path):
        print(colored('WARNING: NO MODEL LOADED !!!', 'red'))
        return 0

    print('load model: {}'.format(model_path))
    if map_location is None:
        pretrained_model = torch.load(model_path, map_location={'cuda:0': 'cpu', 'cuda:1': 'cpu',
                                                                'cuda:2': 'cpu', 'cuda:3': 'cpu'})
    else:
        pretrained_model = torch.load(model_path, map_location=map_location)
    net.load_state_dict(pretrained_model['net'], strict=strict)
    optim.load_state_dict(pretrained_model['optim'])
    scheduler.load_state_dict(pretrained_model['scheduler'])
    recorder.load_state_dict(pretrained_model['recorder'])
    return pretrained_model['epoch'] + 1

def save_model(net, optim, scheduler, recorder, epoch, model_dir):
    os.system('mkdir -p {}'.format(model_dir))
    torch.save({
        'net': net.state_dict(),
        'optim': optim.state_dict(),
        'scheduler': scheduler.state_dict(),
        'recorder': recorder.state_dict(),
        'epoch': epoch
    }, os.path.join(model_dir, '{}.pth'.format(epoch)))
    return

def save_weight(net, model_dir):
    os.system('mkdir -p {}'.format(model_dir))
    torch.save({
        'net': net.state_dict(),
    }, os.path.join(model_dir, '{}.pth'.format('final')))
    return

def load_network(net, model_dir, strict=True, map_location=None):

    if not os.path.exists(model_dir):  # 如果存放权重文件的文件夹不存在则报错
        print(colored('WARNING: NO MODEL LOADED !!!', 'red'))
        return 0

    print('load model: {}'.format(model_dir))
    if map_location is None:
        pretrained_model = torch.load(model_dir, map_location={'cuda:0': 'cpu', 'cuda:1': 'cpu',
                                                               'cuda:2': 'cpu', 'cuda:3': 'cpu'})  # pretrained_model是一个字典，其中有两个键分别是'net'和'epoch'，分别存放网络主体和训练世代数
    else:
        pretrained_model = torch.load(model_dir, map_location=map_location)
    if 'epoch' in pretrained_model.keys():
        epoch = pretrained_model['epoch'] + 1
    else:
        epoch = 0
    pretrained_model = pretrained_model['net']

    net_weight = net.state_dict()  # 获取模型中所有参数的数据结构，以字典的形式存储在net_weight中
    # for key in net_weight.keys():
    #     net_weight.update({key: pretrained_model[key]})
    for key in net_weight.keys():  # 对参数顺序进行调整
        key1 = key
        key2 = key
        if key[-18:] == "conv_offset.weight":
            key1 = key[:-18] + "conv_offset.weight"
            key2 = key[:-18] + "conv_offset_mask.weight"
        if key[-16:] == "conv_offset.bias":
            key1 = key[:-16] + "conv_offset.bias"
            key2 = key[:-16] + "conv_offset_mask.bias"
        net_weight.update({key1: pretrained_model[key2]})

    net.load_state_dict(net_weight, strict=strict)  # 将权重参数按照net_weight的结构导入net中
    return epoch
