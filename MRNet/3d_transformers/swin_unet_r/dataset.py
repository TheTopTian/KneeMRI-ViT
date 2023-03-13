import os
from glob import glob

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms

from monai.data import DataLoader, ImageDataset
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    Resize,
    ScaleIntensity,
)



class MRNetDataset(Dataset):
    def __init__(self, dataset_dir, labels_path, plane, device=None):
        self.case_paths = sorted(glob(f'{dataset_dir}/{plane}/**.npy'))
        self.labels_df = pd.read_csv(labels_path)
        self.window = 7
        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __len__(self):
        return len(self.case_paths)

    def __getitem__(self, idx):
        case_path = self.case_paths[idx]
        # series = preprocess_data(case_path, self.transform)

        case_id = int(os.path.splitext(os.path.basename(case_path))[0])
        case_row = self.labels_df[self.labels_df.case == case_id]
        diagnoses = case_row.values[0,1:].astype(np.float32)
        
        labels = torch.tensor(diagnoses)
        return case_path, labels


def make_dataset(data_dir, dataset_type, plane, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset_dir = f'{data_dir}/{dataset_type}'
    labels_path = f'{data_dir}/{dataset_type}_labels.csv'

    if dataset_type == 'train':
        transform = Compose([
            ScaleIntensity(), 
            EnsureChannelFirst(), 
            Resize((32, 256, 256))
            ])
    elif dataset_type == 'valid':
        transform = Compose([
            ScaleIntensity(), 
            EnsureChannelFirst(), 
            Resize((32, 256, 256))
            ])
    else:
        raise ValueError('Dataset needs to be train or valid.')

    outputs = MRNetDataset(dataset_dir, labels_path, plane, device=device)
    case_path = []
    labels = []
    for i in outputs:
        case_path.append(i[0])
        labels.append(i[1])
    dataset = ImageDataset(image_files=case_path, labels=labels, transform=transform)

    return dataset


if __name__ == "__main__":
    dataset = make_dataset(data_dir = "../../MRNet/MRNet-v1.0", dataset_type = "train", plane = "axial", device= "cuda")
