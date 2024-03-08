import numpy as np 
import cv2
from PIL import Image
import imagehash
import pickle as pkl 
import os
from tqdm import tqdm
import argparse
# Initialize parser
parser = argparse.ArgumentParser()

# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-p", "--folder_path", help = "raw_image_folder_path")


 

# Read arguments from command line
args = parser.parse_args()

if args.folder_path:
    folder_path  = args.folder_path + "\\"
###########-HASHING-##############
def hashing_image(imge_path):
	hashed_img = imagehash.average_hash(Image.open(imge_path)) 
	return hashed_img
def are_2images_similar(hash0, hash1, cutoff):
	if hash0 - hash1 < cutoff:
  		return True
	else:
  		return False

hashing_list = list()
print("starting ...")
count = 0
for filename in tqdm(os.listdir(folder_path)):
	count+=1
	if filename.startswith("."):continue
	if filename.endswith("svg") or filename.endswith("webm")  or filename.endswith("webp"):continue
	file_path = folder_path+filename
	try:
		hashed_img = hashing_image(file_path)
	except:
		os.remove(file_path)
	if count<5:continue
	for h in hashing_list:
		if are_2images_similar(hashed_img, h, 5):
			try:
				os.remove(file_path)
			except:
				print('cannot rm!')
			print("dupl")
			continue

	hashing_list.append(hashed_img)

print(len(hashing_list))
#with open('training_hashed_image.pkl', 'wb') as f:
#   pkl.dump(hashing_list, f)




