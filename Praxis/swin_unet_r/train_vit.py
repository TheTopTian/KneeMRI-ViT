import sys
import numpy as np
import pandas as pd
from docopt import docopt
from datetime import datetime

import torch
import wandb
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
import torch.nn.functional as F
from torchvision import transforms
from monai.losses.dice import *
from monai.losses.dice import DiceLoss
from dataset import make_dataset,    \
                    MRNetDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR
from utils import create_auc_dir,    \
                  create_loss_dir,   \
                  print_stats,       \
                  save_losses,       \
                  save_checkpoint,   \
                  save_auc,          \
                  add_mean_auc

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


def batch_forward_backprop(models, inputs, labels, optimizers, criterions):
    models.train()
    optimizers.zero_grad()
    out = models(inputs)
    if labels.shape != (1,):
        loss = criterions(out, torch.tensor([1.0]).to("cuda"))
    else: 
        loss = criterions(out,labels)

    # out, seg = models(inputs)

    # if labels.shape != (1,):
    #     loss1 = criterions(out, torch.tensor([1.0]).to("cuda"))
    #     diceloss = DiceLoss(reduction='mean')
    #     loss2 = diceloss(F.sigmoid(seg),labels.unsqueeze(0))
    #     loss = loss1 + loss2
    # else: 
    #     loss = criterions(out,labels)
    loss.backward()
    optimizers.step()

    return np.array(loss.item())


def batch_forward(models, inputs, labels, criterions):
    models.eval()
    out = models(inputs)
    if labels.shape != (1,):
        loss = criterions(out, torch.tensor([1.0]).to("cuda"))
        valid_labels = [1.0]
    else:
        loss = criterions(out, labels)
        valid_labels = [0.0]

    # out,seg = models(inputs)
    # if labels.shape != (1,):
    #     loss1 = criterions(out, torch.tensor([1.0]).to("cuda"))
    #     valid_labels = [1.0]
    #     diceloss = DiceLoss(reduction='mean')
    #     loss2 = diceloss(F.sigmoid(seg),labels.unsqueeze(0))
    #     loss = loss1 + loss2
    # else:
    #     loss = criterions(out, labels)
    #     valid_labels = [0.0]
    preds = out.tolist()

    
    return preds,valid_labels, np.array(loss.item())


def update_lr_schedulers(lr_schedulers, batch_valid_losses):
    lr_schedulers.step(batch_valid_losses)


def main(data_dir, plane, disease_num, epochs, lr, weight_decay, patience=10, device=None):
    labels_path = '../../new_location/json_for_new/new_grouped_whatever_location.csv'
    location_csv = pd.read_csv(labels_path)
    pathologys = location_csv.columns.tolist()
    pathology = pathologys[disease_num]
    print(f'We are training {pathology} now!')

    exp = f'disease{disease_num}_new'
    out_dir, auc_path = create_auc_dir(exp,plane)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # torch.cuda.set_device(1)

    print('Creating data loaders...')

    paths = make_dataset(data_dir, plane, labels_path, pathology, device)
    kf = KFold(n_splits=5, shuffle = False)

    for i,(train_index,valid_index) in enumerate(kf.split(paths)):
        Fold_num = i
        losses_path = create_loss_dir(out_dir, plane, Fold_num)
        train_paths, valid_paths = np.array(paths)[train_index],np.array(paths)[valid_index]
        
        # I didn't add any transform methods here
        train_dataset = MRNetDataset(train_paths, labels_path, pathology, transform=None, device=device)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

        valid_dataset = MRNetDataset(valid_paths, labels_path, pathology, transform=None, device=device)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)

        models = SwinUNETR(
                img_size=(32,256,256),
                in_channels=1,
                out_channels=1,
                feature_size=24,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                dropout_path_rate=0.0,
                use_checkpoint=True,
            ).to(device)
   
        criterions = nn.BCEWithLogitsLoss()        
        optimizers = make_adam_optimizer(models, lr, weight_decay)
        lr_schedulers = make_lr_scheduler(optimizers)

        iteration_change_loss = 0
        max_auc = [0.0]
        min_valid_losses = [np.inf]

        print(f'Training a model using {plane} series...')
        print(f'Checkpoints and losses will be save to {out_dir}')

        for epoch, _ in enumerate(range(epochs), 1):
            print(f'=== Fold {Fold_num} Epoch {epoch}/{epochs} ===')

            batch_train_losses = np.array([0.0])
            batch_valid_losses = np.array([0.0])

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                batch_loss = batch_forward_backprop(models, inputs, labels,
                                                    optimizers, criterions) 
                batch_train_losses += batch_loss

            valid_preds = []
            valid_labels = []

            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                batch_preds,batch_labels,batch_loss = \
                    batch_forward(models, inputs, labels, criterions)
                batch_valid_losses += batch_loss

                valid_labels.append(batch_labels)
                valid_preds.append(batch_preds)

            batch_train_losses /= len(train_loader)
            batch_valid_losses /= len(valid_loader)

            aucs = print_stats(batch_train_losses, batch_valid_losses,
                               valid_labels, valid_preds)
            save_losses(batch_train_losses, batch_valid_losses,aucs,losses_path)
            
            # wandb.log({
            #     f"train_loss_fold{Fold_num}": batch_train_losses[0],
            #     f"valid_loss_fold{Fold_num}": batch_valid_losses[0]
            # })

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

            for i, (aucs_train, max_auc_train) in \
                    enumerate(zip(aucs, max_auc)):

                if aucs_train > max_auc_train:
                    max_auc[i] = aucs_train

        save_auc(max_auc, auc_path)

    add_mean_auc(auc_path,plane)

if __name__ == '__main__':
    # arguments = docopt(__doc__)

    print('Parsing arguments...')

    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="Swin unet localization for meniscus",
        
    #     # track hyperparameters and run metadata
    #     config={
    #     "learning_rate": 0.00001,
    #     "architecture": "Swin_Unet",
    #     "dataset": "preprocessed_dataset_2_praxisdata",
    #     "epochs": 500,
    #     }
    # )

    main(data_dir = '../../previous_dataset/Preprocessed_dataset_2', 
         plane = "sagittal", 
         disease_num = 1, # ACL
         epochs = 500, 
         lr = 0.00001, 
         weight_decay = 0.01)
