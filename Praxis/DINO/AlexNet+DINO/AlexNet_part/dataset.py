import os
from glob import glob
from PIL import Image

import torch
import ast
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
        series = preprocess_data(case_path, self.transform)

        dir_path, file_name = os.path.split(case_path)
        case_id = os.path.basename(dir_path)
        location_label = self.labels_df.loc[self.labels_df['StudyUID'] == case_id]["ACL"].tolist()
        physical_location = ast.literal_eval(location_label[0])

        img = tio.ScalarImage(case_path) # img.shape:[1,512,512,n]
        sitk_img = img.as_sitk()

        # Get the real coordinate of pixel
        pixel_location = sitk_img.TransformPhysicalPointToIndex(physical_location)
        labels = torch.tensor([float(pixel_location[2])])

        return (series, labels)


def make_dataset(data_dir, plane,labels_path, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    location_csv = pd.read_csv(labels_path)
    names = location_csv["StudyUID"].tolist()

    # remove the too thick dataset
    if data_dir == '../../previous_dataset/only_crop_2':
        problem_csv = pd.read_csv("../../previous_dataset/problem_dataset_from_PraxisData/too_large.csv")
        problem_names = problem_csv["slice>50"].tolist()
        for problem_name in problem_names:
            if problem_name in names:
                names.remove(problem_name)
 
    for name in names:
        if not os.path.exists(f"{data_dir}/{name}"):
            names.remove(name)
    
    case_paths = []
    for name in names:
        case_path = f'{data_dir}/{name}/{plane.upper()}_PROTON.nii'
        case_paths.append(case_path)

    print(f"There are {len(case_paths)} need to be trained!")
    return case_paths