# Basic structure

## dataset
This file is used to reading read the images and labels from the dataset.

## model
In this model file, I put 4 different models inside and do comparison. The baseline method AlexNet, MaxViT, DINO and DINO v2.

## train_models
This file includes the setting of learning rate, loss function, optimizer, cross-validation and other basic settings. I set the patience equals to 5 so that if the validation loss didn't keep reducing from 5 iterations, the model will stop automatically.

## utils
