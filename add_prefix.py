import argparse
import os
from utils import hashing_image, are_2images_similar
# Initialize parser
parser = argparse.ArgumentParser()

# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-p", "--folder_path", help = "raw_image_folder_path")

parser.add_argument("-f", "--prefix", help = "prefix")
 

# Read arguments from command line
args = parser.parse_args()

if args.folder_path:
    folder_path  = args.folder_path
    

    prefix = args.prefix

    # List all files in the folder
    files = os.listdir(folder_path)

    # Iterate through the files
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.txt')):
             old_path = os.path.join(folder_path, file)
             new_name = prefix + file
             new_path = os.path.join(folder_path, new_name)

             # Rename the file
             os.rename(old_path, new_path)

print("Prefix added to all image names.")