# Knee MRI Classification & Location
This is my master thesis project. We aim to use the vision transformers to classify different diseases in knee MRIs and use the attension inside to locate the disease position.

## Dataset
This time we are going to use the private dataset in Uniklinik Aachen, which contains both the disease and its location. We decided to seperate this task into two different parts: **classification** and **location**. First we need to know whether this patient gets this kind of disease, second find the location of the disease. If the patient doesn't have this disease then the second part will be skipped directly.

Since we need to first classify the diseases, there is already a really famous public knee MRI dataset called **MRNet** which was made by Standford University.

<p align="center">
  <img src="./images/Different_sides_MRNet.png" alt="Different_sides_MRNet" width="700" height="auto">
</p>
