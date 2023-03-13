from glob import glob
import pandas as pd
import torchio as tio
import os

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

# After the calculation of previous step
coronal_shape = (512,512,25)
sagittal_shape = (512,512,25)
transversal_shape = (512,512,27)

def resample_crop(plane,case_paths,space,shape):
    for case_path in case_paths:
        path = f"{case_path}/{plane.upper()}_PROTON.nii"

        test = tio.ScalarImage(path)
        preprocess = tio.Compose([tio.Resample(space),tio.CropOrPad(target_shape=shape)])
        preprocess_test = preprocess(test)

        dir_path, file_name = os.path.split(path)
        folder_name = os.path.basename(dir_path)
        directory = f'../Preprocessed_dataset/{folder_name}'
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        preprocess_test.save(f'{directory}/{file_name}')



if __name__ == "__main__":
    resample_crop("coronal",case_paths,coronal_spacing,coronal_shape)
    resample_crop("sagittal",case_paths,sagittal_spacing,sagittal_shape)
    resample_crop("transversal",case_paths,transversal_spacing,transversal_shape)