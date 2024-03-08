from ultralytics import YOLO
import os
import time
# Load a model
model = YOLO("yolov8l.yaml")  # build a new model from scratch
model = YOLO(f"./best_weights/best_large.pt")
image_folder = r"D:\HD_VNese_map\dataset\training_data\images\val"
count = 0
# Measure inference time
total_time = 0
count_time = 0
for file in os.listdir(image_folder):
	try:
		image_dir = os.path.join(image_folder, file)
		start_time = time.time()
		results = model.predict(source=image_dir, save_txt=False, conf=0.1, iou=0.1, show=False, stream=False)	
		end_time = time.time()
		total_time += end_time - start_time
		#print(end_time - start_time,"______________________________")
		count_time+=1
	except:
		count+=1



inference_time = total_time/count_time

print(f"Inference Time: {inference_time:.4f} seconds")
print("error:", count)