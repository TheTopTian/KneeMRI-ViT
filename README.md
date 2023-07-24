# Knee MRI Classification & Location
This is my master thesis project. We aim to use the vision transformers to classify different diseases in knee MRIs and use the attension inside to locate the disease position.

## Dataset
This time we are going to use the private dataset in Uniklinik Aachen, which contains both the disease and its location. We decided to seperate this task into two different parts: **classification** and **location**. First we need to know whether this patient gets this kind of disease, second find the location of the disease. If the patient doesn't have this disease then the second part will be skipped directly.

Since we need to first classify the diseases, there is already a really famous public knee MRI dataset called **MRNet** which was made by Standford University. It consists of 1370 knee MRI with 1104 (80.6%) **abnormal** cases, 319 (23.3%) **ACL** tears and 508 (37.1%) **meniscal** tears. Each case has 3 different sides which is also the same with the private dataset.

<p align="center">
  <img src="./images/Different_sides_MRNet.png" alt="Different_sides_MRNet" width="auto" height="auto">
</p>

## Methods with Results
According to the baseline method, every single disease had a separate trained model. This would take a lot of time, for example in the MRNet dataset, there were 3 different views and 3 diseases, so 9 models had to be trained to get all the diseases' AUC on every views. Since simply change the last linear layer inside the model could make it output 3 values together. Here was a comparison between using the 1 output and 3 outputs with MaxViT on MRNet dataset. The abnormal and meniscus had similar results, but the AUC of ACL had a small improvement. It could prove that had 3 outputs at the same time wouldn't affect the performance of model. In order to saving time, all of the other models were also designed to output the diseases values together.

|Methods|Views|Abnormal|ACL|Meniscus|
| --- | --- | --- | --- | --- |
|MaxViT with 1 output|Axial|0.924|0.885|0.806|
||Sagittal|**0.949**|0.883|0.801|
||Coronal|0.865|0.865|**0.833**|
|MaxViT with 3 outputs|Axial|0.930|0.900|0.816|
||Sagittal|0.945|0.903|0.782|
||Coronal|0.902|**0.931**|0.823|

Here were the results from different methods trained on MRNet dataset:

|Methods|Views|Abnormal|ACL|Meniscus|
| --- | --- | --- | --- | --- |
|Baseline|Axial|0.921|0.883|0.778|
||Sagittal|0.929|0.860|0.766|
||Coronal|0.802|0.804|0.788|
|ResNet+spatial attention

