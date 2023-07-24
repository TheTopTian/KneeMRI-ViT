# Knee MRI Classification & Location
This is my master thesis project. We aim to use the vision transformers to classify different diseases in knee MRIs and use the attension inside to locate the disease position.

## Dataset
Two different datasets were used in this project:
### MRNet
**MRNet** was a famous public dataset which was made by Standford University. It consists of 1370 knee MRI with 1104 (80.6%) **abnormal** cases, 319 (23.3%) **ACL** tears and 508 (37.1%) **meniscal** tears. Each case has 3 different sides which is also the same with the private dataset.

<p align="center">
  <img src="./images/Different_sides_MRNet.png" alt="Different_sides_MRNet" width="auto" height="auto">
</p>

### Praxis
The private dataset Praxis has a really similar structure compared with the MRNet. Each case within Praxis consists of three distinct views: coronal, sagittal, and transversal. However, in stark contrast to the MRNet dataset, Praxis boasts a significantly larger number of cases, totally 3794. Moreover, the labels associated with Praxis differ substantially from those of the MRNet dataset. In Praxis, the labels provide detailed information in the form of coordinates for numerous diseases.

Praxis had an old label from 2 years before, that one only had 0 or 1 labels similar with the MRNet dataset. Some diseases on the old label and the new label were the same, so they could be compared with each other, for example the most common disease: ACL and meniscus. But some diseases didn't match up with each other some of the cases. Combined with the latest label got from radiologists in May, 4 different versions of labels were made for the following training:
> label 1: The label from 2 years ago, only had 0 or 1,
> label 2: The label extracted from the first json file,
> label 3: The overlap cases on ACL and meniscus from label 1 and label 2,
> label 4: The latest label received at the end of May from the radiologists’ group.

The disease distribution in Praxis:
|Label number (cases)|ACL|PCL|Inner Meniscus|Outer Meniscus|
| --- | --- | --- | --- | --- |
|Label 1 (3794)|ACL: 251(6.62%)||Menicus: 1300 (34.26%)||
|Label 2 (3511)|218 (6.24%)|17 (0.48%)|1015 (28.91%)|309 (8.80%)|
|Label 3 (3317)|180 (5.43%)|16 (0.48%)|943 (28.43%)|286 (8.62%)|
|Label 4 (3503)|211 (6.03%)|16 (0.48%)|933 (26.62%)|281 (8.02%)|

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

