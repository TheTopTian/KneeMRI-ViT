import os
from glob import glob
from PIL import Image
import ast
import math
import torch
import torchio as tio
import torch.nn.functional as F
import numpy as np
import copy
import pandas as pd
from torch.utils.data import Dataset
from scipy.ndimage import zoom

from utils import preprocess_data

def create_solid_ball(img,pixel_location):
    radius = [12,12,1]
    zoom_rate = 0.5

    ''' A : numpy.ndarray of shape size*size*size. '''
    pixel_map = copy.deepcopy(img)
    pixel_map_data = torch.zeros_like(pixel_map.data)
    # pixel_map_data = torch.permute(pixel_map_data,(0,3,1,2))
    pixel_map_data = zoom(pixel_map_data, (1,zoom_rate,zoom_rate,1))

    ''' (x0, y0, z0) : coordinates of center of circle inside A. '''
    x0, y0, z0 = pixel_location
    x0 = int(x0*zoom_rate)
    y0 = int(y0*zoom_rate)
    z0 = int(z0*zoom_rate)

    for x in range(x0-radius[0], x0+radius[0]+1):
        for y in range(y0-radius[1], y0+radius[1]+1):
            for z in range(z0-radius[2], z0+radius[2]+1):
                distance = radius[0]-math.sqrt((x0-x)**2+(y0-y)**2)
                z1 = radius[2]-abs(z0-z)
                # deb = radius - abs(x0-x) - abs(y0-y) - abs(z0-z) 
                if distance>=0.0 and z1>=0.0:#(deb)>=0: 
                    pixel_map_data[0,x,y,z] = 1
    
    pixel_map.set_data(pixel_map_data.astype(np.float32))
    pixel_map = pixel_map.data
    # print(pixel_map.shape)
    return pixel_map.squeeze(0)

class MRNetDataset(Dataset):
    def __init__(self, paths, labels_path, pathology,transform=None, device=None):
        self.case_paths = paths
        self.labels_df = pd.read_csv(labels_path)
        self.pathology = pathology
        self.transform = transform
        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __len__(self):
        return len(self.case_paths)

    def __getitem__(self, idx):
        case_path = self.case_paths[idx]
        series = preprocess_data(case_path,self.transform)

        dir_path, file_name = os.path.split(case_path)
        case_id = os.path.basename(dir_path)
        location_label = self.labels_df.loc[self.labels_df['StudyUID'] == case_id][self.pathology].tolist()
        if isinstance(location_label[0], str):
            physical_location = ast.literal_eval(location_label[0])

            img = tio.ScalarImage(case_path) # img.shape:[1,512,512,n]
            sitk_img = img.as_sitk()

            # Get the real coordinate of pixel
            pixel_location = sitk_img.TransformPhysicalPointToIndex(physical_location)

            pixel_map = create_solid_ball(img,pixel_location)

            depth = pixel_map.shape[2]
            padding = (0,32-depth)
            pixel_map = torch.Tensor(pixel_map)
            pixel_map = torch.nn.functional.pad(pixel_map,padding,"constant",0).permute(2,0,1)

            # normalize the gaussian value [0,1]
            max_value = torch.max(pixel_map)
            normalized_pixel_map = pixel_map/max_value    
            label = normalized_pixel_map

        else: 
            label = 0.
        
        return (series, label)

def make_dataset(data_dir, plane, labels_path, pathology, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    location_csv = pd.read_csv(labels_path)
    names = location_csv["StudyUID"].tolist()

    case_paths = []
    for name in names:
        case_path = f'{data_dir}/{name}/{plane.upper()}_PROTON.nii'
        case_paths.append(case_path)

    return case_paths