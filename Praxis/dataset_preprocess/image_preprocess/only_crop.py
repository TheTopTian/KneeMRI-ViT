from glob import glob
import pandas as pd
import torchio as tio
import os
import shutil

dataset_dir = f'../PraxisData'
case_paths = sorted(glob(f"{dataset_dir}/**"))

def only_crop(plane,case_paths):
    current_dataset = "../only_crop"
    count = 0
    for case_path in case_paths:
        path = f"{case_path}/{plane.upper()}_PROTON.nii"
        dir_path, file_name = os.path.split(path)
        folder_name = os.path.basename(dir_path)

        test = tio.ScalarImage(path)
        length = test.shape[1]
        width = test.shape[2]
        depth = test.shape[3]

        if length == 512 and width == 512:
            if not os.path.exists(f"{current_dataset}/{folder_name}"):
                os.mkdir(f"{current_dataset}/{folder_name}")
            shutil.copyfile(
                f"{case_path}/{file_name}",
                f"{current_dataset}/{folder_name}/{file_name}"
                )
        else:
            count += 1
            preprocess = tio.CropOrPad(target_shape=(512,512,depth))
            result = preprocess(test)

            directory = f'{current_dataset}/{folder_name}'
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            result.save(f'{directory}/{file_name}')
    
    print(f"In the view {plane}, there are {count} cases need to cropped")



if __name__ == "__main__":
    views = ['coronal','sagittal','transversal']
    for view in views:
        only_crop(view,case_paths)