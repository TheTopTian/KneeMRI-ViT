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

def create_gaussian_ball(tensor_shape, center, radius, depth_rad=1):
    # Create coordinates grid
    coords = torch.meshgrid(torch.arange(tensor_shape[0]), torch.arange(tensor_shape[1]), torch.arange(tensor_shape[2]))
    coords = torch.stack(coords, dim=-1).float()

    # Compute squared Euclidean distance from the center
    distance = torch.sum(torch.square(coords - center), dim=-1)

    # Create the Gaussian ball
    gaussian_ball = torch.exp(-distance / (2 * radius**2))
    gaussian_ball = F.normalize(gaussian_ball, p=2, dim=(0, 1, 2))  # Normalize to sum up to 1

    # Apply depth range mask
    depth_mask = torch.zeros(tensor_shape[2])
    depth_range = (center[2]-depth_rad,center[2]+depth_rad)
    depth_mask[depth_range[0]:depth_range[1] + 1] = 1
    gaussian_ball *= depth_mask.view(1, 1, -1)

    return gaussian_ball

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
            label = 1.

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