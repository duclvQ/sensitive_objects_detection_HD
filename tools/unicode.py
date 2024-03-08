import os
from unidecode import unidecode
import argparse
import os
from utils import hashing_image, are_2images_similar
# Initialize parser
parser = argparse.ArgumentParser()

# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-p", "--folder_path", help = "raw_image_folder_path")


 

# Read arguments from command line
args = parser.parse_args()

if args.folder_path:
    folder_path  = args.folder_path
    
root_folder = folder_path

# Duyệt qua tất cả các thư mục và tệp tin bên trong
for folder_path, _, file_list in os.walk(root_folder):
    for old_name in file_list:
        new_name = unidecode(old_name)
        old_path = os.path.join(folder_path, old_name)
        new_path = os.path.join(folder_path, new_name)
        
        # Kiểm tra xem tên mới có khác tên cũ hay không trước khi đổi tên
        if new_name != old_name:
            os.rename(old_path, new_path)
            print(f"Đổi tên: {old_name} -> {new_name}")
