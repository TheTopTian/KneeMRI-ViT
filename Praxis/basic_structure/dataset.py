import os
from glob import glob
from PIL import Image

import torch
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
        case_row = self.labels_df[self.labels_df.StudyUID == case_id]
        diagnoses = case_row.values[0,1:].astype(np.float32)
        labels = torch.tensor(diagnoses)

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