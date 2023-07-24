#!/usr/bin/env python3.6

import torch
import torch.nn as nn
import numpy as np
from torchvision import models
import timm

# class MRNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = models.alexnet(weights=True)
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.classifier = nn.Linear(256, 3)

#     def forward(self, x):
#         x = torch.squeeze(x, dim=0) # only batch size 1 supported
#         x = self.model.features(x)
#         x = self.gap(x).view(x.size(0), -1)
#         x = torch.max(x, 0, keepdim=True)[0]
#         x = self.classifier(x)
#         return x

'''
MaxViT
'''
class MRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            'hf-hub:timm/maxxvit_rmlp_nano_rw_256.sw_in1k',  # hf-hub:timm/maxxvit_rmlp_nano_rw_256.sw_in1k
            pretrained=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, 3)

    def forward(self, x):
        x = torch.squeeze(x, dim=0) # only batch size 1 supported
        x = self.model.forward_features(x)
        x = self.gap(x).view(x.size(0), -1)
        x = torch.max(x, 0, keepdim=True)[0]
        x = self.classifier(x)
        return x

# class MRNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = timm.create_model(
#             'hf-hub:timm/maxxvit_rmlp_nano_rw_256.sw_in1k',  # 'hf-hub:timm/convnext_xxlarge.clip_laion2b_soup_ft_in1k'
#             pretrained=True)
#         self.conv = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
#         self.soft = nn.Softmax(2)
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.classifier = nn.Linear(512,3) #512
#         # self.classifer = nn.Linear(1000, 4)
    
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
#         # o = x*attention_norm
#         # out = self.model.head.global_pool(o)
#         # out = self.model.head.fc(out.squeeze())
#         # x = torch.max(out, 0, keepdim=True)[0]
#         # x = self.classifier(x)
#         o = x*attention_norm
#         x = self.gap(o).view(x.size(0), -1)
#         x = torch.max(x, 0, keepdim=True)[0]
#         x = self.classifier(x)
#         return x

'''
Dino V2
'''
# class MRNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
#         self.classifier = nn.Linear(768, 3)

#     def forward(self, x):
#         x = torch.squeeze(x, dim=0) # only batch size 1 supported
#         x = self.model(x)
#         x = torch.max(x, 0, keepdim=True)[0]
#         x = self.classifier(x)
#         return x