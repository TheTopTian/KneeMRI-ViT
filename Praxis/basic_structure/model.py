import torch
import torch.nn as nn
from functools import partial
import numpy as np
from torchvision import models
import timm
import math
import os
import torch.nn.functional as F
from transformers import ViTForImageClassification


"""
AlexNet
"""
class MRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.alexnet(weights=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, 4)

    def forward(self, x):
        x = torch.squeeze(x, dim=0) # only batch size 1 supported
        x = self.model.features(x)
        x = self.gap(x).view(x.size(0), -1)
        x = torch.max(x, 0, keepdim=True)[0]
        x = self.classifier(x)
        return x

"""
MaxViT
"""
# class MRNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = timm.create_model(
#             'hf-hub:timm/maxxvit_rmlp_nano_rw_256.sw_in1k',  # 'hf-hub:timm/convnext_xxlarge.clip_laion2b_soup_ft_in1k'
#             pretrained=True)
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.classifier = nn.Linear(512, 4) #512

#     def forward(self, x):
#         x = torch.squeeze(x, dim=0) # only batch size 1 supported
#         x = self.model.forward_features(x)
#         x = self.gap(x).view(x.size(0), -1)
#         x = torch.max(x, 0, keepdim=True)[0]
#         x = self.classifier(x)
#         return x

# class MRNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = timm.create_model(
#             'hf-hub:timm/maxxvit_rmlp_nano_rw_256.sw_in1k',  # 'hf-hub:timm/convnext_xxlarge.clip_laion2b_soup_ft_in1k'
#             pretrained=True)
#         self.conv = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
#         self.soft = nn.Softmax(2)
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.classifier = nn.Linear(512, 4) #512
    
#     def tile(a, dim, n_tile):
#         init_dim = a.size(dim)
#         repeat_idx = [1] * a.dim()
#         repeat_idx[dim] = n_tile
#         a = a.repeat(*(repeat_idx))
#         order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
#         if torch.cuda.is_available():
#             a = a.cuda()
#             order_index = order_index.cuda()
#         return torch.index_select(a, dim, order_index)

#     def forward(self, x):
#         x = torch.squeeze(x, dim=0) # only batch size 1 supported
#         x = self.model.forward_features(x) # ([n,512,8,8])
#         attention = self.conv(x) # ([n, 512, 8, 8])
#         attention =  self.soft(attention.view(*attention.size()[:2], -1)).view_as(attention) # ([n, 512, 16, 16])
#         maximum = torch.max(attention.flatten(2), 2).values # ([n,512])
#         maximum = MRNet.tile(maximum, 1, attention.shape[2]*attention.shape[3]) # ([n, 131072])
#         attention_norm = attention.flatten(2).flatten(1) / maximum # ([n, 131072])
#         attention_norm= torch.reshape(attention_norm, (attention.shape[0],attention.shape[1],attention.shape[2],attention.shape[3])) # ([n, 512, 16, 16])
#         o = x*attention_norm
#         x = self.gap(o).view(x.size(0), -1)
#         x = torch.max(x, 0, keepdim=True)[0]
#         x = self.classifier(x)
#         return x

'''
DINO v2
'''
# class MRNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
#         self.gap = nn.AdaptiveAvgPool1d(1)
#         self.classifier = nn.Linear(256, 4) #512

#     def forward(self, x):
#         x = torch.squeeze(x, dim=0) # only batch size 1 supported
#         x = self.model.forward_features(x) # ['x_norm_clstoken', 'x_norm_patchtokens', 'x_prenorm', 'masks']

#         # print(x['x_norm_clstoken'].shape) # ([n, 384])
#         # print(x['x_norm_patchtokens'].shape) # ([n, 256, 384])
#         # print(x['x_prenorm'].shape) # ([n, 257, 384])

#         x = self.gap(x['x_norm_patchtokens']).view(x['x_norm_patchtokens'].size(0), -1)
#         x = torch.max(x, 0, keepdim=True)[0]
#         # print(x.shape)
#         x = self.classifier(x)
#         return x

# class MRNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
#         self.gap = nn.AdaptiveAvgPool1d(1)
#         self.classifier = nn.Linear(768, 4) #512

#     def forward(self, x):
#         x = torch.squeeze(x, dim=0)
#         x = self.model(x)
#         print(f"shape: {x.shape}")
#         x = self.gap(x.unsqueeze(0).transpose(1,2)).squeeze(0).permute(1,0)
#         x = self.classifier(x)
#         return x


"""
DINO
"""
# class MRNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
#         # self.gap = nn.AdaptiveAvgPool2d(1)
#         self.classifier = nn.Linear(768, 4)

#     def forward(self, x):
#         x = torch.squeeze(x, dim=0) # only batch size 1 supported
#         self.model.head = self.classifier
#         x = self.model(x)
#         # x = self.gap(x).view(x.size(0), -1)
#         x = torch.max(x, 0, keepdim=True)[0]
#         x = self.classifier(x)
#         return x