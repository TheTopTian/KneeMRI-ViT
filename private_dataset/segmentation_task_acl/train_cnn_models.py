import sys
import numpy as np
import pandas as pd
from docopt import docopt
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torchvision import transforms
from dataset import make_dataset,    \
                    MRNetDataset
from torch.utils.data import DataLoader
from unet import UNet
# from tqdm import tqdm
from utils import create_auc_dir,    \
                  create_loss_dir,   \
                  print_stats,       \
                  save_losses,       \
                  save_checkpoint


def calculate_weights(data_dir, dataset_type, device):
    # Read the column names from the label csv
    location_csv = pd.read_csv("../new_labels.csv",header=0)
    column_name = location_csv.columns.tolist()
    column_name.pop(0)

    labels_path = "../new_labels.csv"
    labels_df = pd.read_csv(labels_path)

    sum = 0.0
    for diagnosis in column_name:
        neg_count, pos_count = labels_df[diagnosis].value_counts().sort_index()
        weight = torch.tensor([neg_count / pos_count])
        weight = weight.to(device)
        sum += weight

    weights = sum/13

    return weights


def make_adam_optimizer(model, lr, weight_decay):
    return optim.Adam(model.parameters(), lr, weight_decay=weight_decay)


def make_lr_scheduler(optimizer,
                      mode='min',
                      factor=0.3,
                      patience=1,
                      verbose=False):
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                mode=mode,
                                                factor=factor,
                                                patience=patience,
                                                verbose=verbose)


def batch_forward_backprop(models, inputs, labels, criterions, optimizers):
    models.train()
    optimizers.zero_grad()

    out = models(inputs)
    # print(out.squeeze(0).squeeze(0).shape)
    # Unpadding the output

    # print(f"out:{out.shape}")
    # print(f"labels:{labels.shape}")
    loss = criterions(out, labels)
    # print(loss)
    loss.backward()
    optimizers.step()

    # save the .npy files as the output
    # out_unet = out.squeeze(0).squeeze(0).cpu().detach().numpy()
    # np.save("../out_unet.npy",out_unet)

    # label = labels.squeeze(0).squeeze(0).cpu().detach().numpy()
    # np.save("../label.npy",label)
    # print("Saved!")

    return np.array(loss.item())


def batch_forward(models, inputs, labels, criterions):
    models.eval()

    out = models(inputs)
    # preds = out.squeeze(0).tolist()
    loss = criterions(out, labels)

    return np.array(loss.item())


def update_lr_schedulers(lr_schedulers, batch_valid_losses):
    lr_schedulers.step(batch_valid_losses)


def main(data_dir, plane, pathology, epochs, lr, weight_decay, patience = 5, device=None):

    exp = f'{datetime.now():%Y-%m-%d_%H-%M}'
    out_dir, auc_path = create_auc_dir(exp, plane)

    if plane == "transversal":
        depth = 27
    else:
        depth = 25

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Creating data loaders...')

    paths = make_dataset(data_dir, plane, pathology, device)
    labels_path = f'../new_location.csv'
    kf = KFold(n_splits=5, shuffle = False)

    # min_valid_losses = [np.inf, np.inf, np.inf]

    for i,(train_index,valid_index) in enumerate(kf.split(paths)):
        Fold_num = i
        losses_path = create_loss_dir(out_dir, plane, Fold_num)
        train_paths, valid_paths = np.array(paths)[train_index],np.array(paths)[valid_index]

        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(25, translate=(0.1, 0.1)),
            transforms.ToTensor()
        ])
        train_dataset = MRNetDataset(train_paths, labels_path, pathology, depth, transform=None, device=device)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

        transform_valid = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])
        valid_dataset = MRNetDataset(valid_paths, labels_path, pathology, depth, transform=None, device=device)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)

        print(f'Creating models...')

        # Create a model for each diagnosis

        models = UNet(in_dim=1, out_dim=1, num_filters=4).to(device)

        # Calculate loss weights based on the prevalences in train set

        # pos_weights = calculate_weights(data_dir, 'train', device)
        # criterions = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        criterions = nn.BCEWithLogitsLoss()
        optimizers = make_adam_optimizer(models, lr, weight_decay)

        lr_schedulers = make_lr_scheduler(optimizers)

        iteration_change_loss = 0 # Used to stop the training

        print(f'Training a model using {plane} series...')
        print(f'Checkpoints and losses will be save to {out_dir}')

        min_valid_losses = [np.inf]

        for epoch, _ in enumerate(range(epochs), 1):
            print(f'=== Fold {Fold_num} Epoch {epoch}/{epochs} ===')

            batch_train_losses = np.array([0.0])
            batch_valid_losses = np.array([0.0])

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                batch_loss = batch_forward_backprop(models, inputs, labels,
                                                    criterions, optimizers)
                batch_train_losses += batch_loss

            # valid_preds = []
            valid_labels = []

            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                batch_loss = \
                    batch_forward(models, inputs, labels, criterions)
                batch_valid_losses += batch_loss

                valid_labels.append(labels.detach().cpu().numpy().squeeze())
                # valid_preds.append(batch_preds)

            batch_train_losses /= len(train_loader)
            batch_valid_losses /= len(valid_loader)

            print_stats(batch_train_losses, batch_valid_losses)
            save_losses(batch_train_losses, batch_valid_losses, losses_path)

            update_lr_schedulers(lr_schedulers, batch_valid_losses)
            iteration_change_loss += 1
            
            if batch_valid_losses[0] < min_valid_losses[0]:
                save_checkpoint(Fold_num, epoch, plane, models,
                                optimizers, out_dir)
                iteration_change_loss = 0
                min_valid_losses[0] = batch_valid_losses[0]

            if iteration_change_loss == patience:
                print('Early stopping after {0} iterations without the decrease of the val loss'.
                    format(iteration_change_loss))
                break

    #         for i, (aucs_train, max_auc_train) in \
    #                 enumerate(zip(aucs, max_auc)):

    #             if aucs_train > max_auc_train:
    #                 max_auc[i] = aucs_train

    #     save_auc(max_auc, auc_path)

    # add_mean_auc(auc_path,plane)

if __name__ == '__main__':
    print('Parsing arguments...')

    main(data_dir = '../previous_dataset/Preprocessed_dataset_2', 
        plane = "sagittal", 
        pathology = 'ACL',
        epochs = 100, 
        lr = 0.00001, 
        weight_decay = 0.01
        )
