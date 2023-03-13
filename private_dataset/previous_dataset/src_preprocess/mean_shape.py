from glob import glob
import pandas as pd
from utils import mean_shape,        \
                  create_shape_csv
                

dataset_dir = f'../PraxisData'
case_paths = sorted(glob(f"{dataset_dir}/**"))
case_num = len(case_paths)
# Got 3794 cases here

out_dir = './csv'

if __name__ == '__main__':
    # Read the average spacing from csv
    spacing_path = "./csv/mean_spacing.csv"
    df = pd.read_csv(spacing_path, header=None)
    lst = df.values.tolist()
    new_lst = [tuple(x) for x in lst]

    coronal_spacing = tuple([float(x) for x in new_lst[1]])
    sagittal_spacing = tuple([float(x) for x in new_lst[2]])
    transversal_spacing = tuple([float(x) for x in new_lst[3]])
    
    # Calculate the avergae shape and record it
    shape_path = create_shape_csv(out_dir)
    coronal_shape = mean_shape("coronal",case_paths,case_num,coronal_spacing,shape_path)
    sagittal_shape = mean_shape("sagittal",case_paths,case_num,sagittal_spacing,shape_path)
    transversal_shape = mean_shape("transversal",case_paths,case_num,transversal_spacing,shape_path)
    