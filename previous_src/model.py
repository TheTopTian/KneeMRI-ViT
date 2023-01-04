import os
import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from einops import rearrange, repeat 

ROOT_DIR = '.model'

class BranchViT(nn.Module):
    def __init__(self, 
                input_size, 
                n_classes=8, 
                n_channels=1, 
                n_branch=3, 
                n_layers=4, 
                n_head=16,  
                n_hidden=16, 
                patch_size=(4,4,4)):
        super().__init__()
        # input_size [D, H, W]
        self.n_branch = n_branch
        self.patch_size = patch_size
        
        patch_dim = np.prod(patch_size) * n_channels
        n_patches = np.prod([i // j for i,j in zip(input_size, patch_size)])
        self.n_patches = n_patches 
        self.path_dim  = patch_dim
        self.branches = nn.ModuleList([
            nn.Linear(patch_dim, n_hidden) for i in range(self.n_branch)
            ])
        
        self.pos_embedding = nn.Parameter(torch.rand(1, n_patches + 1, n_hidden))
        self.cls_token     = nn.Parameter(torch.rand(1, 1, n_hidden))
        
        self.transformers = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model = n_hidden,
                    nhead = n_head,
                ), 
                n_layers)
            for i in range(self.n_branch)
        ])
      
        self.fc = nn.Sequential(
            nn.LayerNorm(n_hidden * n_branch),
            nn.Linear(n_hidden *  n_branch, n_classes)
        )

    def forward(self, xs):
        # reshape to patches
        if isinstance(xs, dict):
            xs = list(xs.values())
        xs = [
            rearrange(x, 'b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)', 
            p1 = self.patch_size[0], 
            p2 = self.patch_size[1],
            p3 = self.patch_size[2]) 
            for x in xs]
        # project 
        xs = [branch(x) for (branch,x) in zip(self.branches, xs)]
            
        # add cls token 
        batch_size = xs[0].shape[0]
        xs = [
            torch.cat(
                (repeat(
                    self.cls_token,
                    '1 1 d -> b 1 d', 
                    b = batch_size
                    ), x), 
                dim = 1)
            for x in xs
            ]

        # add position embedding 
        xs = [x + self.pos_embedding[:, :(self.n_patches + 1)] for x in xs]

        # transform
        xs = [transform(x) for (transform, x) in zip(self.transformers, xs)]

        # concat
        xs = torch.cat([x.mean(dim = 1) for x in xs], -1)

        # fully connected
        return self.fc(xs)

    def save(self,prefix='',affix=''):
        if not os.path.exists(ROOT_DIR):
            os.mkdir(ROOT_DIR)
        torch.save(self.state_dict(), f"{prefix}{self.__class__.__name__}{affix}.pt")
        return self 
    
    def load(self,prefix='',affix=''):
        state_dict = torch.load(os.path.join(ROOT_DIR,  f"{prefix}{self.__class__.__name__}{affix}.pt"))
        self.load_state_dict(state_dict)
        return self 

if __name__ == '__main__':
    n_channels = 16 
    n_classes  = 8
    n_branch   = 3
    size = (8, 16, 16)
    xs = [torch.rand(4,n_channels,*size) for i in range(n_branch)]

    model = BranchViT(size,n_classes,n_channels=n_channels,n_branch=n_branch)

    print(model(xs).shape)