# Preprocess of Dataset

## Proprocess of images
A python package called TorchIO was used to facilitate the preprocessing stage. TorchIO is an open-source library specifically designed for efficient loading, preprocessing, augmentation and patch-based sampling of 3D medical images in deep learning, following the design principles of PyTorch. The whole process could be seen in the <span style="color: red;">text</preprocess.ipynb> file.

### 1. Calculate the mean spacing
Since the sequences of the MRIs were unknown, it was possible that the spacing between pixels varied across the dataset. This variability in spacing could pose challenged when directly inputting the original dataset into the model. Upon investigation, it was discovered that some cases had more than 200 slices, while the average number of slices was around 30. This significant difference in the number of slices created a memory overload on CUDA platform each time the model was trained. (The table below was the mean spacing of every view of Praxis dataset)

|Views|X-direction|Y-direction|Z-direction|
| --- | --- | --- | --- |
|Coronal|0.358|0.358|3.727|
|Sagittal|0.357|0.357|3.632|
|Transversal|0.343|0.343|3.990|

### 2. Calculate the mean shaping


<p align="center">
  <img src="../../images/preprocessed_private_dataset.png" alt="preprocessed_private_dataset" width="700" height="auto">
</p>
