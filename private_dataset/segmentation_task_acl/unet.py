import torch
import torch.nn as nn
import numpy as np
import nibabel as nib

def conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        activation,)


def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        activation,)


def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


def conv_block_2_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim, activation),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),)

class UNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters):
        super(UNet, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.LeakyReLU(0.2, inplace=True)
        
        # Down sampling
        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filters, activation)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_2_3d(self.num_filters, self.num_filters * 2, activation)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 4, activation)
        self.pool_3 = max_pooling_3d()
        self.down_4 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 8, activation)
        self.pool_4 = max_pooling_3d()
        self.down_5 = conv_block_2_3d(self.num_filters * 8, self.num_filters * 16, activation)
        self.pool_5 = max_pooling_3d()
        
        # Bridge
        self.bridge = conv_block_2_3d(self.num_filters * 16, self.num_filters * 32, activation)
        
        # Up sampling
        self.trans_1 = conv_trans_block_3d(self.num_filters * 32, self.num_filters * 32, activation)
        self.up_1 = conv_block_2_3d(self.num_filters * 48, self.num_filters * 16, activation)
        self.trans_2 = conv_trans_block_3d(self.num_filters * 16, self.num_filters * 16, activation)
        self.up_2 = conv_block_2_3d(self.num_filters * 24, self.num_filters * 8, activation)
        self.trans_3 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation)
        self.up_3 = conv_block_2_3d(self.num_filters * 12, self.num_filters * 4, activation)
        self.trans_4 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, activation)
        self.up_4 = conv_block_2_3d(self.num_filters * 6, self.num_filters * 2, activation)
        self.trans_5 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2, activation)
        self.up_5 = conv_block_2_3d(self.num_filters * 3, self.num_filters * 1, activation)
        
        # Output
        self.out = conv_block_3d(self.num_filters, out_dim, activation)
    
    def forward(self, x):
        # Down sampling
        down_1 = self.down_1(x) # -> [1, 4, 512, 512, 32]
        pool_1 = self.pool_1(down_1) # -> [1, 4, 256, 256, 16]
        
        down_2 = self.down_2(pool_1) # -> [1, 8, 256, 256, 16]
        pool_2 = self.pool_2(down_2) # -> [1, 8, 128, 128, 8]
        
        down_3 = self.down_3(pool_2) # -> [1, 16, 128, 128, 8]
        pool_3 = self.pool_3(down_3) # -> [1, 16, 64, 64, 4]
        
        down_4 = self.down_4(pool_3) # -> [1, 32, 64, 64, 4]
        pool_4 = self.pool_4(down_4) # -> [1, 32, 32, 32, 2]
        
        down_5 = self.down_5(pool_4) # -> [1, 64, 32, 32, 2]
        pool_5 = self.pool_5(down_5) # -> [1, 64, 16, 16, 1]
        
        # Bridge
        bridge = self.bridge(pool_5) # -> [1, 128, 16, 16, 1]
        
        # Up sampling
        trans_1 = self.trans_1(bridge) # -> [1, 128, 32, 32, 2]
        concat_1 = torch.cat([trans_1, down_5], dim=1) # -> [1, 192, 32, 32, 2]
        up_1 = self.up_1(concat_1) # -> [1, 64, 32, 32, 2]
        
        trans_2 = self.trans_2(up_1) # -> [1, 64, 64, 64, 4]
        concat_2 = torch.cat([trans_2, down_4], dim=1) # -> [1, 96, 64, 64, 4]
        up_2 = self.up_2(concat_2) # -> [1, 32, 64, 64, 4]
        
        trans_3 = self.trans_3(up_2) # -> [1, 32, 128, 128, 8]
        concat_3 = torch.cat([trans_3, down_3], dim=1) # -> [1, 48, 128, 128, 8]
        up_3 = self.up_3(concat_3) # -> [1, 16, 128, 128, 8]
        
        trans_4 = self.trans_4(up_3) # -> [1, 16, 256, 256, 16]
        concat_4 = torch.cat([trans_4, down_2], dim=1) # -> [1, 24, 256, 256, 16]
        up_4 = self.up_4(concat_4) # -> [1, 8, 256, 256, 16]
        
        trans_5 = self.trans_5(up_4) # -> [1, 8, 512, 512, 32]
        concat_5 = torch.cat([trans_5, down_1], dim=1) # -> [1, 12, 512, 512, 32]
        up_5 = self.up_5(concat_5) # -> [1, 4, 512, 512, 32]
        
        # Output
        out = self.out(up_5) # -> [1, 3, 128, 128, 128]

        # unpadding = (0, self.depth-32)
        # out_unpadded = torch.nn.functional.pad(out,unpadding,"constant",0)
        
        # # Change tensor into numpy
        # nonzero_indices = torch.nonzero(out_unpadded)
        # print(nonzero_indices.shape)
        # stacked_tensor = torch.cat((tensor for tensor in nonzero_indices), dim=-1)
        # center_of_mass = stacked_tensor.mean(dim=1)
        # result = center_of_mass[-3:]
        return out

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 512
    x = torch.Tensor(1, 3, image_size, image_size, 25)

    # Add 7 slices on the z axis
    padding = (0, 7)  
    x_padded = torch.nn.functional.pad(x,padding,"constant",0)
    x_padded.to(device)
    print("input size: {}".format(x.size()))

    model = UNet(in_dim=3, out_dim=3, num_filters=4, depth = 25)
    out = model(x_padded)
    
    # # Delete 7 slices on the z axis
    # unpadding = (0,-7)
    # out_unpadded = torch.nn.functional.pad(out,unpadding,"constant",0)
    print("output size: {}".format(out.shape))



















    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # # image_size = 512

    # series = nib.load("../Examples_for_output/resize_input.nii").get_fdata().astype(np.float32)
    # series = torch.tensor(np.stack((series,)*3, axis=0))
    # series = series.permute(3,0,1,2)

    # # x = torch.Tensor(25, 3, image_size, image_size)
    # x = series
    # x.to(device)
    # print("x size: {}".format(x.size()))

    # model = UNet(3, 3)

    # out = model(x)
    # print("out size: {}".format(out.size()))

    # # out = out.permute(1, 2, 3, 0)
    # # assume the output tensor is named 'output_tensor'
    # output_array = out.detach().numpy()
    # print(output_array.shape)

    # # create a NIFTI header
    # header = nib.Nifti1Header()
    # header.set_data_shape(output_array.shape)
    # header.set_xyzt_units('mm', 'sec')
    # # set other header properties as needed

    # # create a new image object
    # img = nib.Nifti1Image(output_array, None, header=header)

    # # save the image to a NIFTI file
    # nib.save(img, '../Examples_for_output/output_seg.nii')
