from ultralytics import YOLO
import cv2 
import shutil
import os
import pickle
import numpy as np
#-------------SETTING PHASE-----------------

IMAGE_FOLDER = r"D:\vietmap_detection\raw_data\27_07_3th_vietnamese"
#### Load model
model = YOLO(r"D:\vietmap_detection\yolov8n\runs\detect\train7\weights\best.pt")


#-----------RELATIVE PATH--------------------

WRONG_PREDICTION_FOLDER = r"D:\vietmap_detection\wrong_prediction_images"
TRAINING_IMAGE_FOLDER = r"D:\vietmap_detection\training_data\images\train"
TRAINING_LABEL_FOLDER = r"D:\vietmap_detection\training_data\labels\train"
TRAINING_FOLDER = r"D:\vietmap_detection\training_data"
PREDICTED_LABEL_FOLDER = r"D:\vietmap_detection\yolov8n\runs\detect\predict\labels"
prefix_name = IMAGE_FOLDER.split("\\")[-1] + "_"
#-------before create new annotation, we save all the previous version of data---
def get_file_paths(folder_path):
    file_paths = []
    for root, directories, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_paths.append(file_path)
    return file_paths
folder_path = TRAINING_FOLDER
file_paths = get_file_paths(folder_path)

pickle_file_path = f"../data_version_control/before_adding_{prefix_name}.pickle"
with open(pickle_file_path, 'wb') as file:
    pickle.dump(file_paths, file)

print("File paths saved to pickle:", pickle_file_path)

#---------annotation phase---------
print("Starting to predict then anotate images")
print(f"folder has {len(os.listdir(IMAGE_FOLDER))} images")

def remove_folders_starting_with(directory, prefix):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path) and item.startswith(prefix):
            try:
                shutil.rmtree(item_path)
                print(f"Removed folder: {item_path}")
            except OSError as e:
                print(f"Error: {e}")

# Replace "path_to_directory" with the actual path of the directory.
predict_directory_path = r"D:\vietmap_detection\yolov8n\runs\detect"
prefix_to_remove = "predict"

remove_folders_starting_with(predict_directory_path, prefix_to_remove)

def moving_image(destination_directory, image_filename,file_dir,  move=True):
	# Full path to the source and destination files.
	#destination_directory = r"D:\vietmap_detection\wrong_prediction_images"

	destination_file = os.path.join(destination_directory, prefix_name+image_filename)
	if move==True:
		# Move the image.
		try:
		    shutil.move(file_dir, destination_file)
		    print(f"Image '{image_filename}' moved successfully from to '{destination_directory}'.")
		except FileNotFoundError:
		    print("Error: The specified file or directories were not found.")
		except shutil.Error as e:
		    print(f"Error: {e}")
	else:
		# Move the image.
		try:
		    shutil.copy(file_dir, destination_file)
		    print(f"Image '{image_filename}' moved successfully from to '{destination_directory}'.")
		except FileNotFoundError:
		    print("Error: The specified file or directories were not found.")
		except shutil.Error as e:
		    print(f"Error: {e}")



image_list = os.listdir(IMAGE_FOLDER)
image_dir_list = list()

for img_idx, image_filename in enumerate(image_list):
	if img_idx%50 ==0:
		print(f"Remain {len(image_list)-img_idx} images!")
	image_dir = os.path.join(IMAGE_FOLDER, image_filename)
	image_dir_list.append(image_dir)
	img = cv2.imread(image_dir)
	try:
		# Get the current image dimensions
		height, width = img.shape[:2]
	except:
		print(image_dir)
		continue
		

	# Check if the image size is larger than 960x960
	if width > 960 or height > 960:
	    # Resize the image to fit within 960x960 while maintaining aspect ratio
	    scale = min(960 / width, 960 / height)
	    new_width, new_height = int(width * scale), int(height * scale)
	    img = cv2.resize(img, (new_width, new_height))
	    cv2.imwrite(image_dir, img)
	else:
	    new_width, new_height = width, height

	results = model.predict(source=image_dir, save_txt=False, conf=0. , iou=0.2, show=True)	
	print(f"Do  you want to save the image with name {image_filename}")
	saving_command = input()
	cv2.destroyAllWindows()

	label_filename = image_filename.split(".")[0]+".txt"
	lable_dir = os.path.join(PREDICTED_LABEL_FOLDER ,label_filename)
	# save the image to training images set
	if saving_command == 's':
		results = model.predict(source=image_dir, save_txt=True, conf=0.1, iou=0.2, show=False)	
		
		moving_image(TRAINING_IMAGE_FOLDER, image_filename, image_dir, move=False)
		moving_image(TRAINING_LABEL_FOLDER, label_filename, lable_dir, move=False)

	# remove image if got a wrong prediction and label this image
	elif saving_command == 'f':
		moving_image(WRONG_PREDICTION_FOLDER, image_filename, image_dir, move=True)
	else:
		# image with no label, just move to training image folder
		moving_image(TRAINING_IMAGE_FOLDER, image_filename, image_dir, move=False)
		continue
# Run inference

# Process results list
#for result in results:
    #boxes = result.boxes  # Boxes object for bbox outputs
    #print(boxes.xywh)
#    break
	# Print image.jpg results in JSON format
	
	#bbox = results[0].tojson()
	#print(bbox)
	#print(results)
	#plot_image(image_dir, results[0])
	