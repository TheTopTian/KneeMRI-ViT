import os 
import numpy as np
import pandas as pd
import torch 
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader


ROOT_DIR = '../MRNet-v1.0'
PLANE = ['axial', 'coronal', 'sagittal']

class MyKnees2D(Dataset):
    def __init__(self, mode):
        assert mode in ['train', 'valid']
        pass 
    def __len__(self):
        pass 
    def __getitem__(self, index):
        pass


class MyKnees3D(Dataset):
    size = (32, 256, 256)
    n_channels = 1
    n_classes = 8
    n_branch = 3
    def __init__(self, mode = 'train'):
        assert mode in ['train', 'valid']
        self.mode = mode

        # read 3 different diseases from csv files
        self.labels = []
        df = pd.read_csv(os.path.join(ROOT_DIR, f'{self.mode}-abnormal.csv'), header=None)
        self.ids    = df[0]
        labels_1 = df[1]

        df = pd.read_csv(os.path.join(ROOT_DIR, f'{self.mode}-acl.csv'), header=None)
        labels_2 = df[1]

        df = pd.read_csv(os.path.join(ROOT_DIR, f'{self.mode}-meniscus.csv'), header=None)
        labels_3 = df[1]

        for i,n in enumerate(labels_1):
            self.labels.append(str(n) + str(labels_2[i])+str(labels_3[i]))

        # 8 different classes for the combination of 3 different diseases
        for i,n in enumerate(self.labels):
            self.labels[i] = int(n,2) # 二进制转为十进制，表示8种不同情况
        
        self.paths = {}
        for plane in PLANE:
            self.paths[plane] = []
            for _id in self.ids:
                self.paths[plane].append(os.path.join(ROOT_DIR,self.mode, plane, f'{_id}'.rjust(4,'0') + '.npy'))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        feats = {}
        for plane in PLANE:
            feats[plane] = np.load(self.paths[plane][index]).astype(np.float32)
        return {
            'feats':feats, 
            'label':self.labels[index]
            }
    
    @classmethod
    def loader(cls, mode='train', batch_size=1, shuffle=False, shape=None):
        if shape is None:
            shape = cls.size
        assert len(shape) == 3 
        assert shape[1] == shape[2]

        def reshape(image):

            """Reshape the Image to get desired size
            Args:
                image : np.ndarray 3D of shape [C1, H1, H1]
            Returns:
                np.ndarray 3D of shape [C2, H2, H2]
            """
            assert image.shape[1] == image.shape[2]
            if image.shape[0] >= shape[0]:
                image = image[:shape[0]]
            else:
                pad_left  = int((shape[0] - image.shape[0]) // 2)
                pad_right = shape[0] - image.shape[0] - pad_left
                image = np.pad(image, [(pad_left, pad_right), (0,0), (0,0)])
            image = torch.tensor(image)
            image = F.interpolate(image[None,...], shape[1:], mode='bilinear', align_corners=True)[0]
            return image 
            
        def collate_fn(data):
            feats = dict(zip(PLANE,[[],[],[]]))
            labels = []
            for item in data:
                
                # record feats
                for plane in PLANE:
                    feats[plane].append(reshape(item['feats'][plane]))        
                
                # record label
                labels.append(item['label'])

            for plane in PLANE:
                feats[plane] = torch.stack(feats[plane], 0)[:,None,...]

            labels = torch.tensor(labels)
            return feats,labels

        return DataLoader(cls(mode), batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


if __name__ == '__main__':
    l = MyKnees3D.loader(batch_size=5)
    feat, label =next(iter(l))
    print(f"""
    |feat|:{len(feat)}
    feat.axial.shape:{feat['axial'].shape}
    feat.coronal.shape:{feat['coronal'].shape}
    feat.sagittal.shape:{feat['sagittal'].shape}
    label.shape:{label.shape}
    """)