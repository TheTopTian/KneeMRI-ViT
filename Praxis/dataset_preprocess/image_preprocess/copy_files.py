import os 
import shutil
import pandas as pd

# Read the folder's name
location_csv = pd.read_csv("../../new_location.csv")
names = location_csv["StudyUID"].tolist()

previous_dataset = "../Preprocessed_dataset_2"
current_dataset = "../../dataset"

for i,name in enumerate(names):
    if not os.path.exists(f"{previous_dataset}/{name}"):
        print(f"In case{i}: {name} does't exist")
        continue
    else:
        if os.path.exists(f"{current_dataset}/{name}"):
            continue
        else:
            shutil.copytree(f"{previous_dataset}/{name}",f"{current_dataset}/{name}")