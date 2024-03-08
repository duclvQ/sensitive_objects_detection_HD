from utils import hashing_image, are_2images_similar

import os
import argparse
import pickle
import pickle as pkl

TRAINING_IMAGE_FOLDER = "../dataset/training_data/images/val/"
TRAINING_HASHED_IMAGE = "../dataset/training_data/images/validating_hashed_image.pkl"
RAW_IMAGE_FOLDER = None


with open(TRAINING_HASHED_IMAGE, 'rb') as f:
 	training_hashed_image_list = pickle.load(f)

def check_folder(raw_image_folder_path):
	
	print(f"Number of images in folder:{len(os.listdir(raw_image_folder_path))}")

	for filename in os.listdir(raw_image_folder_path):

		file_path = os.path.join(raw_image_folder_path, filename)
		if filename.endswith('svg'):
			os.remove(file_path)
			continue
		raw_hashed_image = hashing_image(file_path)
		for training_ahshed_image in training_hashed_image_list:
			if are_2images_similar(training_ahshed_image, raw_hashed_image, 5):
				os.remove(file_path)
				break
			else:
				continue
	print(f"keeping {len(os.listdir(raw_image_folder_path))} images")














# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-p", "--folder_path", help = "raw_image_folder_path")
 
# Read arguments from command line
args = parser.parse_args()

if args.folder_path:
    RAW_IMGAE_FOLDER  = args.folder_path
    check_folder(RAW_IMGAE_FOLDER)
else:
	print("give me a folder path")
