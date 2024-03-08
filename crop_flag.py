# crop image with given bbox and save to new folder, bbox is yolo format


import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

IMAGE_FOLDER = r"E:\HD_VNese_map\dataset\training_data\images\train"
LABEL_FOLDER = r"E:\HD_VNese_map\dataset\training_data\labels\train"
CROP_FOLDER = r"E:\HD_VNese_map\dataset\flag_classify"
train_crop_folder = r"E:\HD_VNese_map\dataset\flag_classify\train\flag"
test_crop_folder = r"E:\HD_VNese_map\dataset\flag_classify\test\flag"
if not os.path.exists(CROP_FOLDER):
    os.makedirs(CROP_FOLDER)
_image_list = os.listdir(IMAGE_FOLDER)
image_list = []
for filename in _image_list:
    if filename.split("_")[0] != "co3soc":
        continue
    image_list.append(filename)
# train_test_split
train_ratio = 0.8
train_num = int(len(image_list) * train_ratio)
train_list = image_list[:train_num]
test_list = image_list[train_num:]
for filename in tqdm(image_list[:]):
    if filename.split("_")[0] != "co3soc":
        continue
    img_path = os.path.join(IMAGE_FOLDER, filename)
    label_path = os.path.join(LABEL_FOLDER, filename.replace(".jpg", ".txt"))
    

    img = cv2.imread(img_path)
    img_h, img_w, _ = img.shape

    with open(label_path, "r") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            line = line.strip().split()
            cls_id = line[0]
            x = float(line[1])
            y = float(line[2])
            w = float(line[3])
            h = float(line[4])
            x1 = int((x - w / 2) * img_w)
            y1 = int((y - h / 2) * img_h)
            x2 = int((x + w / 2) * img_w)
            y2 = int((y + h / 2) * img_h)
            # in crease bbox size 
            x1 = max(0, x1 - (x2 - x1) // 4)
            y1 = max(0, y1 - (y2 - y1) // 4)
            x2 = min(img_w, x2 + (x2 - x1) // 8)
            y2 = min(img_h, y2 + (y2 - y1) // 8)


            crop_img = img[y1:y2, x1:x2]
            
            if filename in train_list:
                crop_path = os.path.join(train_crop_folder, filename)
            else:
                crop_path = os.path.join(test_crop_folder, filename)
            cv2.imwrite(crop_path, crop_img)