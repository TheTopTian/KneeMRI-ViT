import os
from glob import glob
from PIL import Image
import ast
import torch
import torchio as tio
import numpy as np
import copy
import pandas as pd
from torch.utils.data import Dataset

from utils import preprocess_data


class MRNetDataset(Dataset):
    def __init__(self, paths, labels_path, pathology, depth, transform=None, device=None):
        self.case_paths = paths
        self.labels_df = pd.read_csv(labels_path)
        self.pathology = pathology
        self.depth = depth
        self.transform = transform
        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __len__(self):
        return len(self.case_paths)

    def __getitem__(self, idx):
        case_path = self.case_paths[idx]
        series = preprocess_data(case_path, self.depth, self.transform)

        dir_path, file_name = os.path.split(case_path)
        case_id = os.path.basename(dir_path)
        location_label = self.labels_df.loc[self.labels_df['StudyUID'] == case_id][self.pathology].tolist()
        physical_location = ast.literal_eval(location_label[0])

        img = tio.ScalarImage(case_path)
        sitk_img = img.as_sitk()
        # Get the real coordinate of pixel
        pixel_location = sitk_img.TransformPhysicalPointToIndex(physical_location)

        STD = [6.0, 6.0, 6.0, 6.0, 1.0, 1.0]

        pixel_map = copy.deepcopy(img)
        pixel_map_data = torch.zeros_like(pixel_map.data)
        pixel_map_data[0, pixel_location[0],
                        pixel_location[1], pixel_location[2]] = 1.0
        pixel_map.set_data(pixel_map_data.type(torch.float32))
        pixel_map = tio.transforms.RandomBlur(
                    std=STD)(pixel_map)
        pixel_map = pixel_map.data
        labels = pixel_map
        padding = (0,32-self.depth)
        labels = torch.nn.functional.pad(labels,padding,"constant",0)
        
        return (series, labels)

def make_dataset(data_dir, plane, pathology, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    location_csv = pd.read_csv("../new_location.csv")
    location_csv_pathology = location_csv[~location_csv[pathology].isnull()]
    names = location_csv_pathology["StudyUID"].tolist()

    case_paths = []
    for name in names:
        case_path = f'{data_dir}/{name}/{plane.upper()}_PROTON.nii'
        case_paths.append(case_path)

    return case_paths