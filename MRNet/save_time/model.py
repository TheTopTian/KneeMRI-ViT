#!/usr/bin/env python3.6

import torch
import torch.nn as nn
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


class MRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            'hf-hub:timm/maxxvit_rmlp_nano_rw_256.sw_in1k',  # maxvit_rmlp_tiny_rw_256.sw_in1k
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

