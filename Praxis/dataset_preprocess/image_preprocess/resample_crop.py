from glob import glob
import pandas as pd
import torchio as tio
import os
import math

dataset_dir = f'../PraxisData'
case_paths = sorted(glob(f"{dataset_dir}/**"))

# Read the average spacing from csv
spacing_path = "./csv/mean_spacing.csv"
df = pd.read_csv(spacing_path, header=None)
lst = df.values.tolist()
new_lst = [tuple(x) for x in lst]

coronal_spacing = tuple([float(x) for x in new_lst[1]])
sagittal_spacing = tuple([float(x) for x in new_lst[2]])
transversal_spacing = tuple([float(x) for x in new_lst[3]])

def resample_crop(plane,case_paths,space):
    for case_path in case_paths:
        path = f"{case_path}/{plane.upper()}_PROTON.nii"

        test = tio.ScalarImage(path)
        preprocess_1 = tio.Resample(space)
        result_1 = preprocess_1(test)
        depth = math.ceil(result_1.shape[3])
        preprocess_2 = tio.CropOrPad(target_shape=(512,512,depth))
        result_2 = preprocess_2(result_1)


        dir_path, file_name = os.path.split(path)
        folder_name = os.path.basename(dir_path)
        directory = f'../Preprocessed_dataset/{folder_name}'
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        result_2.save(f'{directory}/{file_name}')



if __name__ == "__main__":
    resample_crop("coronal",case_paths,coronal_spacing)
    resample_crop("sagittal",case_paths,sagittal_spacing)
    resample_crop("transversal",case_paths,transversal_spacing)