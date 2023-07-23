# Exract the histograms from the dataset
import torch
import csv
import numpy as np
from pathlib import Path
from glob import glob
import torchio as tio
import os
from torchio.transforms import HistogramStandardization

# New preprocessed dataset
dataset_dir_current = f'../only_crop'
case_paths_current = sorted(glob(f"{dataset_dir_current}/**"))

coronal_paths = []
sagittal_paths = []
transversal_paths = []
for case_path in case_paths_current:
    coronal_paths.append(f"{case_path}/CORONAL_PROTON.nii")
    sagittal_paths.append(f"{case_path}/SAGITTAL_PROTON.nii")
    transversal_paths.append(f"{case_path}/TRANSVERSAL_PROTON.nii")

coronal_landmarks_path = Path('./npy_files/coronal_landmarks.npy')
sagittal_landmarks_path = Path('./npy_files/sagittal_landmarks.npy')
transversal_landmarks_path = Path('./npy_files/transversal_landmarks.npy')

coronal_landmarks = (
    coronal_landmarks_path
    if coronal_landmarks_path.is_file()
    else HistogramStandardization.train(coronal_paths)    
)
np.save(coronal_landmarks_path, coronal_landmarks)

sagittal_landmarks = (
    sagittal_landmarks_path
    if sagittal_landmarks_path.is_file()
    else HistogramStandardization.train(sagittal_paths)    
)
np.save(sagittal_landmarks_path, sagittal_landmarks)

transversal_landmarks = (
    transversal_landmarks_path
    if transversal_landmarks_path.is_file()
    else HistogramStandardization.train(transversal_paths)    
)
np.save(transversal_landmarks_path, transversal_landmarks)

file = open('./csv/his_stand.csv',"r")
data = list(csv.reader(file, delimiter=","))
file.close()
coronal_his = np.array([float(x) for x in data[1]])
sagittal_his = np.array([float(x) for x in data[2]])
transversal_his = np.array([float(x) for x in data[3]])

landmarks_dict = {
    'coronal': coronal_his,
    'sagittal': sagittal_his,
    'transversal': transversal_his,
    }

transform = tio.HistogramStandardization(landmarks_dict)

output_dataset = '../only_crop_2'

i = 0
while i < len(case_paths_current):
    case_path = case_paths_current[i]
    coronal_paths = f"{case_path}/CORONAL_PROTON.nii"
    sagittal_paths = f"{case_path}/SAGITTAL_PROTON.nii"
    transversal_paths = f"{case_path}/TRANSVERSAL_PROTON.nii"
    data_coronal = tio.ScalarImage(coronal_paths)
    data_sagittal = tio.ScalarImage(sagittal_paths)
    data_transversal = tio.ScalarImage(transversal_paths)
    
    subject = tio.Subject(
        coronal = data_coronal,
        sagittal = data_sagittal,
        transversal = data_transversal
    )

    output = transform(subject)

    planes = ['coronal','sagittal','transversal']
    for plane in planes:
        path = f"{case_path}/{plane.upper()}_PROTON.nii"

        dir_path, file_name = os.path.split(path)
        folder_name = os.path.basename(dir_path)
        directory = f'{output_dataset}/{folder_name}'
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        output[plane].save(f'{directory}/{file_name}')
    print(f"Case number: {i}")
    i += 1

