import torchio as tio
import pandas as pd
from glob import glob
import os

dataset_dir = f'../only_crop_2'
original_dataset = f'..//PraxisData'
case_paths = sorted(glob(f"{dataset_dir}/**"))

def check(case_paths,view):
    for case_path in case_paths:
        depth = tio.ScalarImage(f"{case_path}/{view.upper()}_PROTON.nii").shape[3]
        if depth > 40:
            case_id = os.path.basename(case_path)
            original_depth = tio.ScalarImage(f"{original_dataset}/{case_id}/{view.upper()}_PROTON.nii").shape[3]
            print(f"In view {view}, case{case_id}, depth={depth}, original depth={original_depth}")

if __name__ == '__main__':
    views = ['coronal','sagittal','transversal']
    for view in views:
        check(case_paths,view)
    