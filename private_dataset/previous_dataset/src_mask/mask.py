import pandas as pd
import torchio as tio
import os
import copy
import torch
import ast
import matplotlib.pyplot as plt
import numpy as np
import cv2
import seaborn as sns

location_csv = pd.read_csv('../../new_location.csv',header=0)
pathology = location_csv.columns.tolist()
# print(pathology)

views = ['coronal','sagittal','transversal']

def case_name(disease_num): # should be int but couldn't be 0
    location_csv_pathology = location_csv[~location_csv[pathology[disease_num]].isnull()]
    names = location_csv_pathology['StudyUID'].tolist()
    print(f"disease{disease_num} has {len(names)} cases")
    return names

def pixel_location(view,names,disease_num):
    pixel_locations = []
    for name in names:
        location_label = location_csv.loc[location_csv['StudyUID'] == name][pathology[disease_num]].tolist()
        root_path = '../Preprocessed_dataset_2'
        study_path = os.path.join(root_path, name, f'{view.upper()}_PROTON.nii')
        img = tio.ScalarImage(study_path)
        physical_location = ast.literal_eval(location_label[0])
        sitk_img = img.as_sitk()
        pixel_location = sitk_img.TransformPhysicalPointToIndex(physical_location)
        pixel_locations.append(pixel_location)

    return pixel_locations


if __name__ == "__main__":
    i = 2 # 0 is the index
    while i < len(pathology):
        csv_dir = f"./location_diseases/disease{i}_location.csv"
        if not os.path.exists(csv_dir):
            dictionary = {}
            for view in views:
                dictionary[view] = pixel_location(view,case_name(i),i)
                print(f"{view} of {pathology[i]} is finished!")       
            df = pd.DataFrame(dictionary)
            df.to_csv(csv_dir,index=False)
            print(f"{pathology[i]} is finished!")
            i += 1
        else:
            i += 1