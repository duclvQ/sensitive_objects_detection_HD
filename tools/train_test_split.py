import os
import random
import shutil

import os
import random
import shutil

import os
import random
import shutil
from tqdm import tqdm
# Paths to the source folders
image_source_folder =  r"E:\HD_VNese_map\co3soc\dataset\images"
annotation_source_folder =  r"E:\HD_VNese_map\co3soc\dataset\txt_anns"

# Paths to the destination folders
image_train_folder = r"E:\HD_VNese_map\co3soc\train\img"
annotation_train_folder = r"E:\HD_VNese_map\co3soc\train\ann"
image_val_folder =  r"E:\HD_VNese_map\co3soc\test\img"
annotation_val_folder =  r"E:\HD_VNese_map\co3soc\test\ann"






# Create destination folders
os.makedirs(image_train_folder, exist_ok=True)
os.makedirs(annotation_train_folder, exist_ok=True)
os.makedirs(image_val_folder, exist_ok=True)
os.makedirs(annotation_val_folder, exist_ok=True)

# List all the images in the image folder
image_files = os.listdir(image_source_folder)
random.shuffle(image_files)  # Shuffle the list randomly

# Determine the split ratio (e.g., 80% training, 20% validation)
split_ratio = 0.8
split_index = int(len(image_files) * split_ratio)

# Split images
image_train_files = image_files[:split_index]
image_val_files = image_files[split_index:]
count=0
# Copy images to the appropriate folders
for image_file in tqdm(image_train_files):
    src_image_path = os.path.join(image_source_folder, image_file)
    dst_image_path = os.path.join(image_train_folder, image_file)
    
    try:
        # Copy corresponding annotations
        annotation_file = os.path.splitext(image_file)[0] + ".txt"  # Assuming annotations have the same name with a different extension
        src_annotation_path = os.path.join(annotation_source_folder, annotation_file)
        dst_annotation_path = os.path.join(annotation_train_folder, annotation_file)
        shutil.copy(src_annotation_path, dst_annotation_path)
        shutil.copy(src_image_path, dst_image_path)
    except:
        count+=1
        continue

for image_file in tqdm(image_val_files):
    src_image_path = os.path.join(image_source_folder, image_file)
    dst_image_path = os.path.join(image_val_folder, image_file)
    
    try:
        # Copy corresponding annotations
        annotation_file = os.path.splitext(image_file)[0] + ".txt"  # Assuming annotations have the same name with a different extension
        src_annotation_path = os.path.join(annotation_source_folder, annotation_file)
        dst_annotation_path = os.path.join(annotation_val_folder, annotation_file)
        shutil.copy(src_annotation_path, dst_annotation_path)
        shutil.copy(src_image_path, dst_image_path)
    except:
        count+=1
        continue
print(count)


