import numpy as np 
import cv2
from PIL import Image
import imagehash
import pickle as pkl 
import os
from tqdm import tqdm

###########-HASHING-##############
def hashing_image(imge_path):
	hashed_img = imagehash.average_hash(Image.open(imge_path)) 
	return hashed_img

folder_path = "../dataset/training_data/images/train/"
hashing_list = list()
for filename in tqdm(os.listdir(folder_path)):
	if filename.startswith("."):continue
	if filename.endswith("svg") or filename.endswith("webm"):continue
	file_path = folder_path+filename
	hashed_img = hashing_image(file_path)
	hashing_list.append(hashed_img)


with open('../dataset/training_data/images/training_hashed_image.pkl', 'wb') as f:
   pkl.dump(hashing_list, f)

folder_path = "../dataset/training_data/images/val/"
hashing_list = list()
for filename in tqdm(os.listdir(folder_path)):
	if filename.startswith("."):continue
	if filename.endswith("svg") or filename.endswith("webm"):continue
	file_path = folder_path+filename
	hashed_img = hashing_image(file_path)
	hashing_list.append(hashed_img)


with open('../dataset/training_data/images/validating_hashed_image.pkl', 'wb') as f:
   pkl.dump(hashing_list, f)



