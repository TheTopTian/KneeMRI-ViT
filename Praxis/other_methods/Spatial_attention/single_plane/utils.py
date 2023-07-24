import os
import csv

import numpy as np
import pandas as pd
import math
import SimpleITK as sitk
from scipy.ndimage import zoom
import torch
import torchio as tio
from sklearn import metrics
from torchvision import transforms
import torch.nn.functional as F

location_csv = pd.read_csv("../../../new_data/new_acl_meniscus(May).csv")
column_name = location_csv.columns.tolist()[1:]

def preprocess_data(case_path, transform=None):
    img = sitk.ReadImage(case_path)
    series = sitk.GetArrayFromImage(img).astype(np.float32)
    # series = zoom(series, (1,0.4375,0.4375)) # shrimp to 224: (1,0.4375,0.4375)
    series = torch.tensor(np.stack((series,)*3, axis=0))
    
    series = series.permute(1, 0, 2, 3)

    if transform is not None:
        for i, slice in enumerate(series.split(1)):
            series[i] = transform(slice.squeeze())

    return series



def create_auc_dir(exp, plane):
    out_dir = f'./models/{exp}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    auc_path = create_auc_csv(out_dir,plane)

    return out_dir, auc_path

def create_loss_dir(out_dir, plane, Fold_num):
    losses_path = create_losses_csv(out_dir, plane, Fold_num)

    return losses_path


def create_auc_csv(out_dir,plane):
    auc_path = f'{out_dir}/highest_auc_{plane}.csv'

    with open(f'{auc_path}', mode='w') as auc_csv:
        fields = column_name
        writer = csv.DictWriter(auc_csv, fieldnames=fields)
        writer.writeheader()
    
    return auc_path


def create_losses_csv(out_dir, plane, Fold_num):
    losses_path = f'{out_dir}/Fold{Fold_num}_losses_{plane}.csv'

    with open(f'{losses_path}', mode='w') as losses_csv:
        fields = ['training_loss', 'validation_loss']
        for name in column_name:
            fields.append(f'auc_{name}')
        writer = csv.DictWriter(losses_csv, fieldnames=fields)
        writer.writeheader()

    return losses_path


def calculate_aucs(all_labels, all_preds):
    all_labels = np.array(all_labels).transpose()
    all_preds =  np.array(all_preds).transpose()

    aucs = [metrics.roc_auc_score(labels, preds) for \
            labels, preds in zip(all_labels, all_preds)]

    return aucs


def print_stats(batch_train_losses, batch_valid_losses,
                valid_labels, valid_preds):
    aucs_valid = calculate_aucs(valid_labels, valid_preds)
    # aucs_test = calculate_aucs(test_labels,test_preds)

    i = 0
    print(f'Train losses: {batch_train_losses[0]:.3f},',
          f'Valid losses: {batch_valid_losses[0]:.3f},')
    while i < len(column_name):
        print(f'\nValid AUCs - {column_name[i]}: {aucs_valid[i]:.3f},')
        i += 1
    return aucs_valid


def save_losses(train_losses , valid_losses, aucs, losses_path):
    with open(f'{losses_path}', mode='a') as losses_csv:
        writer = csv.writer(losses_csv)
        x = np.append(train_losses, valid_losses)
        y = np.append(x, aucs)
        writer.writerow(y)


def save_auc(max_auc,auc_path):
    with open(f'{auc_path}', mode='a') as auc_csv:
        writer = csv.writer(auc_csv)
        writer.writerow(max_auc)


def save_checkpoint(Fold_num, epoch, plane, model, optimizer, out_dir):
    print(f'Min valid loss for epoch{epoch}, saving the checkpoint...')

    checkpoint = {
        'epoch': epoch,
        'plane': plane,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    chkpt = f'ELNet_Fold{Fold_num}_{plane}.pt'
    torch.save(checkpoint, f'{out_dir}/{chkpt}')

def add_mean_auc(auc_path,plane):
    df = pd.read_csv(auc_path)
    i = 0
    dictionary = {}

    while i < len(column_name):
        dictionary[column_name[i]] = df[column_name[i]].mean()
        i += 1
    
    dir_path, file_name = os.path.split(auc_path)
    df_new = pd.DataFrame(dictionary,index=[0])
    df_new.to_csv(f"{dir_path}/mean_auc_{plane}.csv",index=False)