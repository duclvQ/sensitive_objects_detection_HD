import os
from PIL import Image
import shutil

# Paths to source and destination folders
source_folder = r'C:\Users\Administrator\Downloads\2508'
destination_folder = r'C:\Users\Administrator\Downloads\h_larger_w'

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Loop through all files in the source folder
for filename in os.listdir(source_folder):
    file_path = os.path.join(source_folder, filename)
    
    # Check if the file is an image
    if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                img.close()
            
                
                # Compare height and width
                if height > width:
                    # Move the image to the destination folder
                    destination_path = os.path.join(destination_folder, filename)
                    shutil.move(file_path, destination_path)
                    print(f"Moved {filename} to {destination_folder}")
        except:
                continue   