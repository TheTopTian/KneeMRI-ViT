# KneeMRI-ViT
Master thesis

The file of MRNet is the structure of the dataset, which includes the train and valid, and both of them have 3 different views of a same patient's knee.

There are 3 different kind of diseases included inside. The goal is to use the transformers to do the classification of different diseases.

The folder of ViT includes my first try of vision transformers, but the main problem is that it couldn't read the .npy file directly.

I also created a dataset that saves the knee pictures of each layer, which is to turn the npy file into a lot of pictures. But it is very difficult to give a unified label to a bunch of photos.

The previous_src folder contains the previous codes you wrote.
