# Knee MRI Classification & Location
This is my master thesis project. We aim to use the vision transformers to classify different diseases in knee MRIs and use the attension inside to locate the disease position.

## Dataset
This time we are going to use the private dataset in Uniklinik Aachen, which contains both the disease and its location. We decided to seperate this task into two different parts: **classification** and **location**. First we need to know whether this patient gets this kind of disease, second find the location of the disease. If the patient doesn't have this disease then the second part will be skipped directly.

Since we need to first classify the diseases, there is already a really famous public knee MRI dataset called **MRNet** which was made by Standford University. It consists of 1370 knee MRI with 1104 (80.6%) **abnormal** cases, 319 (23.3%) **ACL** tears and 508 (37.1%) **meniscal** tears. Each case has 3 different sides which is also the same with the private dataset.

<p align="center">
  <img src="./images/Different_sides_MRNet.png" alt="Different_sides_MRNet" width="700" height="auto">
</p>

## Methods with Results
### Baseline


### Swin Unet R
Swin Transformer is compatible for a broad range of vision tasks, including image classification and dense prediction tasks such as object detection. It is also included in MONAI and shows a promising result in Brain MRI segmentation. Since my job is to do a classification of diseases, I can just use the downsampling part and output the parameter at the bottom of Unet, and input them again in a linear layer and classify the diseases I want. But the result with this method doesn't end up good:

|Diseases|Accuracy|
|Abnormal|0.79 (95/120)|
|ACL|0.55 (66/120)|
|Meniscus|0.57 (68/120)|

### MaxViT

