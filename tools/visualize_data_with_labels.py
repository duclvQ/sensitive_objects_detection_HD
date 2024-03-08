import cv2
import os

# Paths to the image and label folders
#image_folder = r'E:\HD_VNese_map\dataset\training_data\images\train'
#label_folder = r'E:\HD_VNese_map\dataset\training_data\labels\train'

image_folder = r'E:\HD_VNese_map\dataset\training_data\images\val'
label_folder = r'E:\HD_VNese_map\dataset\training_data\labels\val'

# Create a list of image and label files
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])
label_files = sorted([f for f in os.listdir(label_folder) if f.endswith('.txt')])

total_images = len(image_files)
current_index = 0
for img_filename in (image_files ):
    # Load the image
    #if not img_filename.startswith('transformed_1707'):continue
    
    image_path = os.path.join(image_folder, img_filename)
    image = cv2.imread(image_path)
    img_type = img_filename.split('.')[-1]
    label_filename = img_filename.replace(img_type, 'txt')
    # Load the label file
    label_path = os.path.join(label_folder, label_filename)
    print(image_path)
    print(label_path)
    if not os.path.exists(label_path):
        os.remove(image_path)
        continue
    print('_')
    with open(label_path, 'r') as label_file:
        for line in label_file:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            
            # Convert YOLO coordinates to image coordinates
            img_height, img_width, _ = image.shape
            x1 = int((x_center - width / 2) * img_width)
            y1 = int((y_center - height / 2) * img_height)
            x2 = int((x_center + width / 2) * img_width)
            y2 = int((y_center + height / 2) * img_height)
            
            # Draw bounding box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'Class {int(class_id)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    image  = cv2.resize(image, (640,640))
    cv2.imshow('Image Viewer', image)
    key = cv2.waitKey(0)
    # Display the image
    if key == ord('d'):
        os.remove(image_path)
        os.remove(label_path)
        total_images -= 1

    if key == 13:  # Enter key
        current_index += 1
        continue
    cv2.destroyAllWindows()

print("All images processed.")

# Close all windows
#cv2.destroyAllWindows() 