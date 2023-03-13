from glob import glob
from utils import mean_spacing,        \
                  create_spacing_csv
                

dataset_dir = f'../PraxisData'
case_paths = sorted(glob(f"{dataset_dir}/**"))
case_num = len(case_paths)
# Got 3794 cases here

out_dir = './csv'

if __name__ == '__main__':
    # Calculate the avergae spacing and record it
    spacing_path = create_spacing_csv(out_dir)
    coronal_spacing = mean_spacing("coronal", case_paths, case_num, spacing_path)
    sagittal_spacing = mean_spacing("sagittal", case_paths, case_num, spacing_path)
    transversal_spacing = mean_spacing("transversal", case_paths, case_num, spacing_path)