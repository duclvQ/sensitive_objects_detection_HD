import os

# Specify the folder path where your files are located
folder_path = r'C:\Users\Administrator\Downloads\Imageye - Official Map 9 Dash Line - Bing images'

# Get a list of all files in the folder
files = os.listdir(folder_path)

# Sort the files alphabetically, which will order them by their current names
files.sort()

# Define a starting order number
order_number = 1

# Iterate through the sorted files and rename them with the order number
for file_name in files:
    # Construct the new file name with the order number and the original file's extension
    new_file_name = f"9d_1006_{order_number:04d}_{file_name}"
    
    # Build the full paths for the old and new file names
    old_file_path = os.path.join(folder_path, file_name)
    new_file_path = os.path.join(folder_path, new_file_name)
    
    # Rename the file
    os.rename(old_file_path, new_file_path)
    # Check if the file is not a directory and its extension is not .png or .jpg
    if not os.path.isfile(new_file_path) or (not new_file_name.endswith('.png') and not new_file_name.endswith('.jpg')):
        # Remove the file
        os.remove(new_file_path)
        print(f"Removed: {file_name}")
    # Increment the order number for the next file
    order_number += 1