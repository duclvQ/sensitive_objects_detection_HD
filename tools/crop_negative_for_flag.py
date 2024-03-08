# random crop a region from image and save to new folder

import os
import cv2
import numpy as np
import random
from tqdm import tqdm
IMAGE_FOLDER = r"C:\Users\Administrator\Downloads\4"
CROP_FOLDER = r"E:\HD_VNese_map\dataset\flag_classify\train\N"

image_list = os.listdir(IMAGE_FOLDER)
num_images = len(image_list)
# each image crop 10 times

for idx, filename in tqdm(enumerate(image_list[:])):
    
    try:
        img_path = os.path.join(IMAGE_FOLDER, filename)
        img = cv2.imread(img_path)
        img_h, img_w, _ = img.shape
        if img_h < 224 or img_w < 224:
            continue
        if idx < num_images * 0.8:
            CROP_FOLDER = r"E:\HD_VNese_map\dataset\flag_classify\train\N"
        else:
            CROP_FOLDER = r"E:\HD_VNese_map\dataset\flag_classify\test\N"
        for i in range(10):
            x1 = random.randint(0, img_w - 224)
            y1 = random.randint(0, img_h - 224)
            x2 = x1 + 224
            y2 = y1 + 224
            crop_img = img[y1:y2, x1:x2]
            crop_path = os.path.join(CROP_FOLDER, f"n1_{filename}_{i}.jpg")
            cv2.imwrite(crop_path, crop_img)
        cv2.imwrite(crop_path, crop_img)
    except:
        continue