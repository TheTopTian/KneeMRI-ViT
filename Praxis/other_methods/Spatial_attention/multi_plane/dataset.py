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
        multi_path = self.case_paths[idx]
        series = []
        for case_path in multi_path:
            serie = preprocess_data(case_path, self.transform)
            series.append(serie)

        dir_path, file_name = os.path.split(case_path)
        case_id = os.path.basename(dir_path)
        case_row = self.labels_df[self.labels_df.StudyUID == case_id]
        diagnoses = case_row.values[0,1:].astype(np.float32)
        labels = torch.tensor(diagnoses)

        return (series, labels)


def make_dataset(data_dir, labels_path, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    location_csv = pd.read_csv(labels_path)
    names = location_csv["StudyUID"].tolist()

    for name in names:
        if not os.path.exists(f"{data_dir}/{name}"):
            names.remove(name)
    
    case_paths = []
    planes = ["sagittal","coronal","transversal"]
    for name in names:
        multi_path = []
        for plane in planes:
            case_path = f'{data_dir}/{name}/{plane.upper()}_PROTON.nii'
            multi_path.append(case_path)
        case_paths.append(multi_path)

    print(f"There are {len(case_paths)} need to be trained!")
    return case_paths