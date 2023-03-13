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
        case_path = self.case_paths[idx]
        series = preprocess_data(case_path, self.transform)

        case_id = int(os.path.splitext(os.path.basename(case_path))[0])
        case_row = self.labels_df[self.labels_df.case == case_id]
        diagnoses = case_row.values[0,1:].astype(np.float32)
        labels = torch.tensor(diagnoses)

        return (series, labels)


def make_dataset(data_dir, dataset_type, plane, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if dataset_type == 'train':
        dataset_dir = f'{data_dir}/train'
        # labels_path = f'{data_dir}/train_labels.csv'
        case_paths = sorted(glob(f'{dataset_dir}/{plane}/**.npy'))
        
    elif dataset_type == 'test':
        dataset_dir = f'{data_dir}/valid'
        # labels_path = f'{data_dir}/valid_labels.csv'
        case_paths = sorted(glob(f'{dataset_dir}/{plane}/**.npy'))
    
    return case_paths
