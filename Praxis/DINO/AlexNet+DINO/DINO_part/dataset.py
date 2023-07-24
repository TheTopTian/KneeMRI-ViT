import os
from glob import glob
from PIL import Image

import torch
import ast
import copy
import torchio as tio
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from utils import preprocess_data


class MRNetDataset(Dataset):
    def __init__(self, paths, labels_path, transform=None, device=None):
        self.case_paths = paths
        self.labels_df = pd.read_csv(labels_path)
        self.transform = transform
        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __len__(self):
        return len(self.case_paths)

    def __getitem__(self, idx):
        case_path = self.case_paths[idx]
        dir_path, file_name = os.path.split(case_path)
        case_id = os.path.basename(dir_path)
        predict_slice = self.labels_df.loc[self.labels_df['StudyUID'] == case_id]["predict_slice"].tolist()[0]
        location =  self.labels_df.loc[self.labels_df['StudyUID'] == case_id]["sagittal_location"].tolist()[0]
        series = preprocess_data(case_path, predict_slice, self.transform)

        
        if isinstance(location,str):
            location = ast.literal_eval(location)
            labels = torch.tensor((float(location[0]),float(location[1])))
        else:
            labels = torch.tensor([0.])

        return (series, labels)


def make_dataset(data_dir, plane,labels_path, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    location_csv = pd.read_csv(labels_path)
    names = location_csv["StudyUID"].tolist()
 
    for name in names:
        if not os.path.exists(f"{data_dir}/{name}"):
            names.remove(name)
    
    case_paths = []
    for name in names:
        case_path = f'{data_dir}/{name}/{plane.upper()}_PROTON.nii'
        case_paths.append(case_path)

    print(f"There are {len(case_paths)} need to be trained!")
    return case_paths