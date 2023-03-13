import os
import csv

import numpy as np
import pandas as pd
import nibabel as nib
import torch
from sklearn import metrics
from torchvision import transforms
import torch.nn.functional as F

# MAX_PIXEL_VAL = 255
# MEAN = 58.09
# STD = 49.73


def preprocess_data(case_path, depth, transform=None):
    series = nib.load(case_path).get_fdata().astype(np.float32)
    series = torch.tensor(np.stack((series,)*1, axis=0))
    # series = torch.tensor(series)
    # series = series.permute(3, 0, 1, 2)
    
    # if transform is not None:
    #     for i, slice in enumerate(series.split(1)):
    #         series[i] = transform(slice.squeeze())

    padding = (0,32-depth)
    series = torch.nn.functional.pad(series,padding,"constant",0)

    # series = (series - series.min()) / (series.max() - series.min()) * MAX_PIXEL_VAL
    # series = (series - MEAN) / STD
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
    
    # Read the column names from the label csv
    location_csv = pd.read_csv("../new_labels.csv",header=0)
    column_name = location_csv.columns.tolist()
    column_name.pop(0)

    with open(f'{auc_path}', mode='w') as auc_csv:
        fields = column_name
        writer = csv.DictWriter(auc_csv, fieldnames=fields)
        writer.writeheader()
    
    return auc_path


def create_losses_csv(out_dir, plane, Fold_num):
    losses_path = f'{out_dir}/Fold{Fold_num}_losses_{plane}.csv'

    with open(f'{losses_path}', mode='w') as losses_csv:
        fields = ['training_loss', 'validation_loss']
        writer = csv.DictWriter(losses_csv, fieldnames=fields)
        writer.writeheader()

    return losses_path


def calculate_aucs(all_labels, all_preds):
    all_labels = np.array(all_labels).transpose()
    all_preds =  np.array(all_preds).transpose()

    aucs = [metrics.roc_auc_score(labels, preds) for \
            labels, preds in zip(all_labels, all_preds)]

    return aucs


def print_stats(batch_train_losses, batch_valid_losses):
    # aucs_valid = calculate_aucs(valid_labels, valid_preds)
    # aucs_test = calculate_aucs(test_labels,test_preds)

    # Read the column names from the label csv
    location_csv = pd.read_csv("../new_labels.csv",header=0)
    column_name = location_csv.columns.tolist()
    column_name.pop(0)

    print(f'Train losses: {batch_train_losses[0]:.3f},',
          f'Valid losses: {batch_valid_losses[0]:.3f}')
    # return aucs_valid


def save_losses(train_losses , valid_losses, losses_path):
    with open(f'{losses_path}', mode='a') as losses_csv:
        writer = csv.writer(losses_csv)
        x = np.append(train_losses, valid_losses)
        writer.writerow(x)


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

    chkpt = f'seg_Fold{Fold_num}_{plane}.pt'
    torch.save(checkpoint, f'{out_dir}/{chkpt}')

def add_mean_auc(auc_path,plane):
    # Read the column names from the label csv
    location_csv = pd.read_csv("../new_labels.csv",header=0)
    column_name = location_csv.columns.tolist()
    column_name.pop(0)

    df = pd.read_csv(auc_path)
    i = 0
    dictionary = {}

    while i < len(column_name):
        dictionary[column_name[i]] = df[column_name[i]].mean()
        i += 1
    
    dir_path, file_name = os.path.split(auc_path)
    df_new = pd.DataFrame(dictionary)
    df_new.to_csv(f"{dir_path}/mean_auc_{plane}.csv",index=False)