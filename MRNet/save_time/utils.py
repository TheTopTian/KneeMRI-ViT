import os
import csv

import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from torchvision import transforms
import torch.nn.functional as F

MAX_PIXEL_VAL = 255
MEAN = 58.09
STD = 49.73


def preprocess_data(case_path, transform=None):
    series =np.load(case_path).astype(np.float32)
    series = torch.tensor(np.stack((series,)*3, axis=1))
    # series = torch.tensor(series)

    if transform is not None:
        for i, slice in enumerate(series.split(1)):
            series[i] = transform(slice.squeeze())

    # array = series.cpu().detach().numpy()
    # np.save("../../out.npy",array)    

    series = (series - series.min()) / (series.max() - series.min()) * MAX_PIXEL_VAL
    series = (series - MEAN) / STD
    # print(f"series: {series.shape}")
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
        fields = ['abnormal', 'acl', 'meniscus']
        writer = csv.DictWriter(auc_csv, fieldnames=fields)
        writer.writeheader()
    
    return auc_path


def create_losses_csv(out_dir, plane, Fold_num):
    losses_path = f'{out_dir}/Fold{Fold_num}_losses_{plane}.csv'

    with open(f'{losses_path}', mode='w') as losses_csv:
        fields = ['training_loss', 'validation_loss', 'test_loss']
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
                batch_test_losses, valid_labels, valid_preds, test_labels, test_preds):
    aucs_valid = calculate_aucs(valid_labels, valid_preds)
    aucs_test = calculate_aucs(test_labels,test_preds)

    print(f'Train losses: {batch_train_losses[0]:.3f},',
          f'\nValid losses: {batch_valid_losses[0]:.3f},',
          f'\nTest losses: {batch_test_losses[0]:.3f},',
          f'\nValid AUCs - abnormal: {aucs_valid[0]:.3f},',
          f'acl: {aucs_valid[1]:.3f},',
          f'meniscus: {aucs_valid[2]:.3f}',
          f'\nTest AUCs - abnormal: {aucs_test[0]:.3f},',
          f'acl: {aucs_test[1]:.3f},',
          f'meniscus: {aucs_test[2]:.3f}')
    return aucs_test


def save_losses(train_losses , valid_losses, test_losses, losses_path):
    with open(f'{losses_path}', mode='a') as losses_csv:
        writer = csv.writer(losses_csv)
        x = np.append(train_losses, valid_losses)
        y = np.append(x, test_losses)
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

    chkpt = f'cnn_Fold{Fold_num}_{plane}_{epoch:02d}.pt'
    torch.save(checkpoint, f'{out_dir}/{chkpt}')

def add_mean_auc(auc_path):
    file_name = auc_path
    read = pd.read_csv(file_name)

    abnormal_mean = read["abnormal"].mean()
    acl_mean = read["acl"].mean()
    meniscus_mean = read["meniscus"].mean()
    x = np.append(abnormal_mean,acl_mean)
    y = np.append(x,meniscus_mean)

    empty_row = ["mean_auc"]

    with open(file_name, mode='a') as auc_csv:
        writer = csv.writer(auc_csv)
        writer.writerow(empty_row)
        writer.writerow(y)