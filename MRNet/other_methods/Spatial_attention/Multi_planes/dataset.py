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
        # self.case_paths = sorted(glob(f'{dataset_dir}/{plane}/**.npy'))
        self.case_paths = paths
        self.labels_df = pd.read_csv(labels_path)
        self.transform = transform
        # self.window = 7
        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __len__(self):
        return len(self.case_paths)

    def __getitem__(self, idx):
        case_paths = self.case_paths[idx]
        series = []
        for case_path in case_paths:
            serie = preprocess_data(case_path, self.transform)
            series.append(serie)

        case_id = int(os.path.splitext(os.path.basename(case_paths[0]))[0])
        case_row = self.labels_df[self.labels_df.case == case_id]
        diagnoses = case_row.values[0,1:].astype(np.float32)
        labels = torch.tensor(diagnoses)

        return (series, labels)


def make_dataset(data_dir, dataset_type, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    planes = ['axial','coronal','sagittal']
    if dataset_type == 'train':
        dataset_dir = f'{data_dir}/train'
        # get the case names:
        paths = sorted(glob(f'{dataset_dir}/{planes[0]}/**.npy'))
        names = []
        for path in paths:
            file_name = os.path.basename(path)
            name = os.path.splitext(file_name)[0]
            names.append(name)
        
    elif dataset_type == 'test':
        dataset_dir = f'{data_dir}/valid'
        # get the case names:
        paths = sorted(glob(f'{dataset_dir}/{planes[0]}/**.npy'))
        names = []
        for path in paths:
            file_name = os.path.basename(path)
            name = os.path.splitext(file_name)[0]
            names.append(name)
    
    case_paths = []
    for name in names:
        case_path = []
        for plane in planes:
            if dataset_type == 'train':
                dataset_dir = f'{data_dir}/train'
                path = f'{dataset_dir}/{plane}/{name}.npy'
                case_path.append(path)
            elif dataset_type == 'test':
                dataset_dir = f'{data_dir}/valid'
                path = f'{dataset_dir}/{plane}/{name}.npy'
                case_path.append(path)
        case_paths.append(case_path)
    return case_paths