import torchio as tio
import csv

def mean_spacing(plane,case_paths,case_num,spacing_path):
    shape_1 = 0.0
    shape_2 = 0.0
    shape_3 = 0.0
    for case_path in case_paths:
        path = f"{case_path}/{plane.upper()}_PROTON.nii"
        test = tio.ScalarImage(path)
        shape_1 += test.spacing[0]
        shape_2 += test.spacing[1]
        shape_3 += test.spacing[2]
    mean_spacing = (shape_1/case_num, shape_2/case_num, shape_3/case_num)
    print(f"{plane}_spacing: {mean_spacing}")
    save_spacing(mean_spacing,spacing_path)
    return mean_spacing

def create_spacing_csv(out_dir):
    spacing_path = f'{out_dir}/mean_spacing.csv'
    with open(f'{spacing_path}', mode='w') as spacing_csv:
        fields = ['x', 'y', 'z']
        writer = csv.DictWriter(spacing_csv, fieldnames=fields)
        writer.writeheader()
    
    return spacing_path

def save_spacing(mean_spacing,spacing_path):
    with open(f'{spacing_path}', mode='a') as spacing_csv:
        writer = csv.writer(spacing_csv)
        writer.writerow(mean_spacing)

def mean_shape(plane,case_paths,case_num,space,shape_path):
    shape_1 = 0.0
    shape_2 = 0.0
    shape_3 = 0.0
    for case_path in case_paths:
        path = f"{case_path}/{plane.upper()}_PROTON.nii"
        test = tio.ScalarImage(path)
        preprocess = tio.Compose([tio.Resample(space)])
        preprocess_test = preprocess(test)
        shape_1 += preprocess_test.shape[1]
        shape_2 += preprocess_test.shape[2]
        shape_3 += preprocess_test.shape[3]
    
    mean_shape = (shape_1/case_num, shape_2/case_num, shape_3/case_num)
    print(f"{plane}_shape: {mean_shape}")
    save_shape(mean_shape,shape_path)
    return mean_shape

def create_shape_csv(out_dir):
    shape_path = f'{out_dir}/mean_shape.csv'
    with open(f'{shape_path}', mode='w') as shape_csv:
        fields = ['x', 'y', 'z']
        writer = csv.DictWriter(shape_csv, fieldnames=fields)
        writer.writeheader()
    
    return shape_path

def save_shape(mean_shape,shape_path):
    with open(f'{shape_path}', mode='a') as shape_csv:
        writer = csv.writer(shape_csv)
        writer.writerow(mean_shape)
