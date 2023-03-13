import torch
import numpy as np
import SimpleITK as sitk
from torchvision.transforms import Compose
import pandas as pd
import torchio as tio
import ast
import copy
import os
import shutil
from unet import UNet
from utils import preprocess_data

def inference(name,plane,depth,fold_num,device=None):
    output_dir = f'./inference/output/{name}'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    output_plane_dir = f'{output_dir}/{plane}_output.nii'
    if not os.path.exists(output_plane_dir):
        # Define the transforms to be applied to the input data
        # transforms = Compose([
        #     # Add any required preprocessing steps here
        # ])
        
        # Load the saved model
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = UNet(in_dim=1, out_dim=1, num_filters=4).to(device)
        checkpoint = torch.load(f'./models/2023-03-06_10-19/seg_Fold{fold_num}_{plane}.pt')
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        # Load the test data and apply transforms
        case_path = f'./inference/example/{name}/{plane.upper()}_PROTON.nii'
        case_tensor = preprocess_data(case_path,depth).to(device)
        # test_data = sitk.ReadImage(f'./inference/example/{name}/{plane.upper()}_PROTON.nii')
        # test_data_array = sitk.GetArrayFromImage(test_data).astype(np.float32)
        # test_data_tensor = torch.from_numpy(test_data_array)
        # test_data_tensor = transforms(test_data_tensor)

        # Make predictions on the test data
        with torch.no_grad():
            predictions = model(case_tensor.unsqueeze(0)).to(device)
            predictions = predictions.squeeze(0).squeeze(0).cpu().detach().numpy()

        # Save the predictions as a Nifti image
        predictions_image = sitk.GetImageFromArray(predictions)
        sitk.WriteImage(predictions_image, output_plane_dir)
        print("Output saved!")

def label(plane,depth,pathology='ACL'):
    # Find out the cases have ACL location
    labels_dir = "../new_location.csv"
    labels_df = pd.read_csv(labels_dir)
    location_csv_pathology = labels_df[~labels_df[pathology].isnull()]
    names = location_csv_pathology["StudyUID"].tolist()

    # Read ACL location of the first case
    location_label = labels_df.loc[labels_df['StudyUID'] == names[0]][pathology].tolist()
    physical_location = ast.literal_eval(location_label[0])

    # Copy the case from the dataset to inference
    dataset_dir = f"../previous_dataset/Preprocessed_dataset_2/{names[0]}"
    copy_dir = f"./inference/example/{names[0]}"
    if not os.path.exists(copy_dir):
        shutil.copytree(dataset_dir, copy_dir)
    
    labels_output_dir = f"./inference/labels/{names[0]}"
    if not os.path.exists(labels_output_dir):
        os.mkdir(labels_output_dir)

    labels_output_plane_dir = f"{labels_output_dir}/{plane}_output.nii"
    if not os.path.exists(labels_output_plane_dir):
        # Get the real coordinate of pixel
        case_path = f'{copy_dir}/{plane.upper()}_PROTON.nii'
        img = tio.ScalarImage(case_path)
        sitk_img = img.as_sitk()
        pixel_location = sitk_img.TransformPhysicalPointToIndex(physical_location)

        STD = [6.0, 6.0, 6.0, 6.0, 1.0, 1.0]
        pixel_map = copy.deepcopy(img)
        pixel_map_data = torch.zeros_like(pixel_map.data)
        pixel_map_data[0, pixel_location[0],
                        pixel_location[1], pixel_location[2]] = 1.0
        pixel_map.set_data(pixel_map_data.type(torch.float32))
        pixel_map = tio.transforms.RandomBlur(std=STD)(pixel_map)
        pixel_map = pixel_map.data
        labels = pixel_map

        padding = (0,32-depth)
        labels = torch.nn.functional.pad(labels,padding,"constant",0)
        label = labels.squeeze(0).squeeze(0).cpu().detach().numpy()

        # Save the label as a Nifti image
        label_output = sitk.GetImageFromArray(label)
        print(label_output.shape)
        sitk.WriteImage(label_output, labels_output_plane_dir)
        print("Label saved!")

    return names[0]


if __name__ == "__main__":
    plane = "coronal"
    if plane == "transversal":
        depth = 27
    else:
        depth = 25
    file_name = label(plane, depth)
    print(file_name)
    # inference(file_name,"sagittal",depth,fold_num=0)
