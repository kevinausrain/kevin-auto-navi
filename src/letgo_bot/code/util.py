import torch
import torch.nn as nn
import numpy as np
from ViT import VisionTransformer

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_weight(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)

def add_vision_transformer(image_size, patch_size, num_classes, dim, mlp_dim, channels, depth, heads):
    return VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        mlp_dim=mlp_dim,
        channels=channels,
        depth=depth,
        heads=heads
    )

def add_convs(max_layer_num, in_out_channels, kernel_size, stride):
    convs = []
    for i in range(max_layer_num):
        convs.append(nn.Conv2d(in_out_channels[i][0], in_out_channels[i][1], kernel_size=kernel_size, stride=stride))
    return convs

def add_full_conns(max_layer_num, in_out_features):
    fcs = []
    for i in range(max_layer_num):
        fcs.append(nn.Linear(in_out_features[i][0], in_out_features[i][1]))
    return fcs

