import sys
import numpy as np
import pandas as pd
from docopt import docopt
from datetime import datetime
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torchvision import transforms
from dataset import make_dataset,    \
                    MRNetDataset
from torch.utils.data import DataLoader
from model_maxvit import MRNet
from utils import create_auc_dir,    \
                  create_loss_dir,   \
                  print_stats,       \
                  save_losses,       \
                  save_checkpoint,   \
                  save_auc,          \
                  add_mean_auc


def calculate_weights(data_dir, dataset_type, device):
    diagnoses = ['abnormal', 'acl', 'meniscus']

    labels_path = f'{data_dir}/{dataset_type}_labels.csv'
    labels_df = pd.read_csv(labels_path)

    weights = []

    for diagnosis in diagnoses:
        neg_count, pos_count = labels_df[diagnosis].value_counts().sort_index()
        weight = torch.tensor([neg_count / pos_count])
        weight = weight.to(device)
        weights.append(weight)

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
    losses = []

    for i, (model, label, criterion, optimizer) in \
            enumerate(zip(models, labels[0], criterions, optimizers)):
        model.train()
        optimizer.zero_grad()

        out = model(inputs)
        # print(out.squeeze(0))
        # print(label.unsqueeze(0))
        loss = criterion(out.squeeze(0), label.unsqueeze(0))
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return np.array(losses)


def batch_forward(models, inputs, labels, criterions):
    preds = []
    losses = []

    for i, (model, label, criterion) in \
            enumerate(zip(models, labels[0], criterions)):
        model.eval()

        out = model(inputs)
        preds.append(out.item())
        loss = criterion(out.squeeze(0), label.unsqueeze(0))
        losses.append(loss.item())

    return np.array(preds), np.array(losses)


def update_lr_schedulers(lr_schedulers, batch_valid_losses):
    for scheduler, v_loss in zip(lr_schedulers, batch_valid_losses):
        scheduler.step(v_loss)


def main(data_dir, plane, epochs, lr, weight_decay, patience = 5, device=None):
    diagnoses = ['abnormal', 'acl', 'meniscus']

    exp = f'{datetime.now():%Y-%m-%d_%H-%M}'
    out_dir, auc_path = create_auc_dir(exp, plane)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Creating data loaders...')

    test_paths = make_dataset(data_dir, 'test', plane, device)
    labels_path = f'{data_dir}/valid_labels.csv'
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    test_dataset = MRNetDataset(test_paths, labels_path, transform=transform_test, device=device)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle = False)

    paths = make_dataset(data_dir, 'train', plane, device)
    labels_path = f'{data_dir}/train_labels.csv'
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
        train_dataset = MRNetDataset(train_paths, labels_path, transform=transform_train, device=device)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

        transform_valid = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])
        valid_dataset = MRNetDataset(valid_paths, labels_path, transform=transform_valid, device=device)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)

        # train_loader = make_data_loader(data_dir, 'train', plane, device, shuffle=True)
        # valid_loader = make_data_loader(data_dir, 'valid', plane, device)

        print(f'Creating models...')

        # Create a model for each diagnosis

        models = [MRNet().to(device), MRNet().to(device), MRNet().to(device)]

        # Calculate loss weights based on the prevalences in train set

        pos_weights = calculate_weights(data_dir, 'train', device)
        criterions = [nn.BCEWithLogitsLoss(pos_weight=weight) \
                    for weight in pos_weights]

        optimizers = [make_adam_optimizer(model, lr, weight_decay) \
                    for model in models]

        lr_schedulers = [make_lr_scheduler(optimizer) for optimizer in optimizers]

        # min_valid_losses = [np.inf, np.inf, np.inf]
        # max_auc = [0.0, 0.0, 0.0]
        iteration_change_loss = 0 # Used to stop the training

        print(f'Training a model using {plane} series...')
        print(f'Checkpoints and losses will be save to {out_dir}')

        # Saving the best AUC for each folder
        max_auc = [0.0, 0.0, 0.0]
        min_valid_losses = [np.inf, np.inf, np.inf]

        for epoch, _ in enumerate(range(epochs), 1):
            print(f'=== Fold {Fold_num} Epoch {epoch}/{epochs} ===')

            batch_train_losses = np.array([0.0, 0.0, 0.0])
            batch_valid_losses = np.array([0.0, 0.0, 0.0])
            batch_test_losses  = np.array([0.0, 0.0, 0.0])

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                batch_loss = batch_forward_backprop(models, inputs, labels,
                                                    criterions, optimizers)
                batch_train_losses += batch_loss

            valid_preds = []
            valid_labels = []

            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                batch_preds, batch_loss = \
                    batch_forward(models, inputs, labels, criterions)
                batch_valid_losses += batch_loss

                valid_labels.append(labels.detach().cpu().numpy().squeeze())
                valid_preds.append(batch_preds)
            
            test_preds = []
            test_labels = []

            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                batch_preds, batch_loss = \
                    batch_forward(models, inputs, labels, criterions)
                batch_test_losses += batch_loss

                test_labels.append(labels.detach().cpu().numpy().squeeze())
                test_preds.append(batch_preds)

            batch_train_losses /= len(train_loader)
            batch_valid_losses /= len(valid_loader)
            batch_test_losses /= len(test_loader)

            aucs = print_stats(batch_train_losses, batch_valid_losses, 
                        batch_test_losses, valid_labels, valid_preds, test_labels, test_preds)
            save_losses(batch_train_losses, batch_valid_losses, batch_test_losses, losses_path)

            update_lr_schedulers(lr_schedulers, batch_valid_losses)
            iteration_change_loss += 1

            for i, (batch_v_loss, min_v_loss) in \
                    enumerate(zip(batch_valid_losses, min_valid_losses)):

                if batch_v_loss < min_v_loss:
                    save_checkpoint(Fold_num, epoch, plane, diagnoses[i], models[i],
                                    optimizers[i], out_dir)
                    iteration_change_loss = 0
                    min_valid_losses[i] = batch_v_loss
            
            # When the Validation loss doesn't decrease for a long time, it will stop the progress
            if iteration_change_loss == patience:
                print('Early stopping after {0} iterations without the decrease of the val loss'.
                    format(iteration_change_loss))
                break

            for i, (aucs_train, max_auc_train) in \
                    enumerate(zip(aucs, max_auc)):

                if aucs_train > max_auc_train:
                    max_auc[i] = aucs_train

        save_auc(max_auc, auc_path)

    add_mean_auc(auc_path)

if __name__ == '__main__':
    # arguments = docopt(__doc__)

    print('Parsing arguments...')

    main(data_dir = '../../MRNet/MRNet-v1.0', 
        plane = "sagittal", 
        epochs = 50, 
        lr = 0.00001, 
        weight_decay = 0.01
        )
