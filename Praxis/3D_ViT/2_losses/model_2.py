import torch
import math
from torch import nn
import pandas as pd
import numpy as np
import torchio as tio
import ast
import copy

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from utils import preprocess_data

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, 
                qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads # 96
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False, return_x_and_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_x_and_attention:
            return x, attn
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, image_patch_size, frames, frame_patch_size, 
                 dim, depth, heads, mlp_ratio, 
                 pool = 'cls', channels = 1, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, 0., depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=dim, num_heads=heads, mlp_ratio=mlp_ratio, qkv_bias=0., qk_scale=None,
                drop=dropout, attn_drop=0., drop_path=dpr[i], norm_layer=nn.LayerNorm)
            for i in range(depth)])

        self.norm = nn.LayerNorm(dim)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.last_lin = nn.Linear(dim,1)

    '''I write from here'''
    def interpolate_pos_encoding(self, x, d, w, h):
        # x.shape=[1,4097,1024]
        # print(f"x.shape: {x.shape}")
        npatch = x.shape[1] - 1 # npatch=4096
        N = self.pos_embedding.shape[1] - 1  # pos_embedding: [1,4097,1024], N=4096
        if npatch == N and w == h: 
            return self.pos_embedding
        
        class_pos_embed = self.pos_embedding[:, 0]
        patch_pos_embed = self.pos_embedding[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(
                math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(
            w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, d, w, h = x.shape
        x = self.to_patch_embedding(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, d, w, h)

        return self.dropout(x)
    
    def forward(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i == 0: # First layer attention
            # if i == len(self.blocks) - 1: # Last layer attention
                x, attn = blk(x, return_x_and_attention=True)
            x = blk(x)
        x = self.norm(x)[:, 0]
        x = self.last_lin(x)
        return x, attn
    
    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)
    
    def get_first_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            return blk(x, return_attention=True)

                
    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

if __name__ == "__main__":
    def case_names(disease_num=1,case_num=1,view="sagittal"):
        # Find out the cases have ACL location
        labels_dir = "../../new_data/ACL.csv"
        labels_df = pd.read_csv(labels_dir)
        pathologys = labels_df.columns.tolist()
        pathology = pathologys[disease_num]
        location_csv_pathology = labels_df[~labels_df[pathology].isnull()]
        names = location_csv_pathology["StudyUID"].tolist()[:case_num]
        labels = location_csv_pathology.loc[location_csv_pathology["StudyUID"]== names[0], pathology].tolist()
        case_path = f'../../previous_dataset/Preprocessed_dataset_2/{names[0]}/{view.upper()}_PROTON.nii'
        case_tensor = preprocess_data(case_path).unsqueeze(0)
        return case_path, case_tensor, labels[0]

    def gaussian_ball(case_path,label):
        physical_location = ast.literal_eval(label)

        img = tio.ScalarImage(case_path) # img.shape:[1,512,512,n]
        sitk_img = img.as_sitk()

        # Get the real coordinate of pixel
        pixel_location = sitk_img.TransformPhysicalPointToIndex(physical_location)

        '''Previous Gaussian label'''
        STD = [6.0, 6.0, 6.0, 6.0, 1.0, 1.0]
        pixel_map = copy.deepcopy(img)
        pixel_map_data = torch.zeros_like(pixel_map.data)
        pixel_map_data[0, pixel_location[0],
                        pixel_location[1], pixel_location[2]] = 1.0
        pixel_map.set_data(pixel_map_data.type(torch.float32))
        pixel_map = tio.transforms.RandomBlur(
            std=STD)(pixel_map)
        pixel_map = pixel_map.data

        pixel_map = pixel_map.squeeze(0)

        # pixel_map = zoom(pixel_map, (0.5,0.5,1))
        depth = pixel_map.shape[2]
        padding = (0,32-depth)
        pixel_map = torch.Tensor(pixel_map)
        pixel_map = torch.nn.functional.pad(pixel_map,padding,"constant",0).permute(2,0,1)

        return pixel_map
    
    v = ViT(
        image_size = 512,          # image size
        frames = 32,               # number of frames
        image_patch_size = 32,     # image patch size
        frame_patch_size = 2,      # frame patch size
        dim = 1024,
        depth = 6,
        heads = 8,
        mlp_ratio = 4.,
        dropout = 0.0,
        emb_dropout = 0.0
    )

    case_path, x, label = case_names() #(1, 1, 32, 512, 512)
    pixel_map = gaussian_ball(case_path, label)
    np.save("check/gaussian_ball.npy",pixel_map)
    np.save("check/dataset.npy",x.squeeze(0).squeeze(0))

    preds = v(x) # [1,1024] (cls_token)
    print(preds.shape)

    attentions = v.get_last_selfattention(x) # [1,4097,1024]
    print(f"attention_first_output: {attentions.shape}")
    nh = attentions.shape[1]  # number of head
    # keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    print(f"attentions_reshape: {attentions.shape}")

    d=w=h=16
    attentions = attentions.reshape(nh,d,w,h)
    print(attentions.shape)
    attentions = torch.nn.functional.interpolate(
        attentions.unsqueeze(0), size=(32, 512, 512), mode='trilinear', align_corners=False
        )[0].cpu().detach().numpy()
    print(attentions.shape)

    mean_attention = np.mean(attentions,0)
    print(mean_attention.shape)
    np.save("check/new_attention.npy",mean_attention)
    

    