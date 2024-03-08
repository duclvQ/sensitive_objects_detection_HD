from ultralytics import YOLO
import os 
import cv2
# Load a model
import argparse
import tqdm
#from utils import hashing_image, are_2images_similar
# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-n", "--number", help = "number_of_path")
 
# Read arguments from command line
args = parser.parse_args()

if args.number:
    model_number  = int(args.number)
model = YOLO(f'runs/detect/train{model_number}/weights/best.pt')  # pretrained YOLOv8n model


folder = r"E:\HD_VNese_map\dataset\training_data\images\val"
folder = r"C:\Users\Administrator\Downloads"
label_folder = r"E:\HD_VNese_map\main_src\yolov8n\missed_vn_label"
#folder =  r"C:\Users\Administrator\Downloads\15_08_1"
image_list = [folder + "/"+a for a in os.listdir(folder)]
# Run batched inference on a list of images

# Define the output video filename and the codec to use
output_video_filename = "predicted_vide02902.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec for your desired video format

# Create a video writer object
video_writer = cv2.VideoWriter(output_video_filename, fourcc, 1, (640, 640))
# Process results list
for img in tqdm.tqdm(image_list):
    #result = model.predict(img, device=1)  # return a list of Results objects
    label_path = label_folder + "/" + img.split("/")[-1].split(".")[0] + ".txt"
    include_vn = False
    if  os.path.exists(label_path):
        # read the label file
        label = open(label_path, "r").readlines()
        
        for i in label:
            i = i.split(" ")[0]
            if i in ["0","1","2","3","4"]:
                include_vn = True
    #if include_vn:continue
    #if not os.path.exists(label_path):continue
    #(label_path)
    try:
        filename = img.split("/")[-1]
        #if  filename.startswith("co3soc"):continue
        #if filename.startswith("background"):continue
        result = model.predict(img, device="cpu", conf=0.25, verbose=False)  # return a list of Results objects
        #if len(result[0].boxes.cls.tolist())!=0:continue
        #if float(1) not in result[0].boxes.cls.tolist():continue
        #if float(0) not in result[0].boxes.cls.tolist():continue
        
    except:
        continue
    num_1_pre = 0
    include_vn = False
    for i in result[0].boxes.cls.tolist():
        if i in [0,1,2,3,4]:
            include_vn = True
    if not include_vn:
        continue
    if len(result[0].boxes.cls.tolist()) == 0:
        continue
    print(result[0].boxes.cls.tolist())
    # Plot the results
    res_plotted = cv2.resize(result[0].plot(), (640,640))
    
    plotted_folder = "./plotted"
    if not os.path.exists(plotted_folder):
        os.makedirs(plotted_folder)
    #cv2.imwrite(f"plotted/{filename}", res_plotted)
    cv2.imshow('o', res_plotted)
    cv2.waitKey(0)
    #if enter is pressed, do the following, space to skip
    if cv2.waitKey(0) == 13:
        continue
    else:
        for i in result[0].boxes.cls.tolist():
            if i == 1: 
                num_1_pre += 1
            if i in [0,1,2,3,4]:
                # move the image to missed_vn folder
                out_folder = "./missed_vn"
                if not os.path.exists(out_folder):
                    os.makedirs(out_folder)
                cv2.imwrite(out_folder + "/" + filename, cv2.imread(img))
                # create yolo label for the image by prediction
                out_label = "./missed_vn_label"
                if not os.path.exists(out_label):
                    os.makedirs(out_label)
                with open(out_label + "/" + filename.split(".")[0] + ".txt", "a") as f:
                    # write i and box norm xywh to the file

                    for box in result[0].boxes.xywhn.tolist():
                        f.write(f"{i} " + " ".join([str(i) for i in box]) + "\n")
                

    #boxes = result.boxes  # Boxes object for bbox outputs
    #masks = result.masks  # Masks object for segmentation masks outputs
    #keypoints = result.keypoints  # Keypoints object for pose outputs
    #probs = result.probs  # Class probabilities for classification outputs
    # if 5 in result[0].boxes.cls.tolist() :
    #     continue
    # if len(result[0].boxes.cls.tolist()) == 0:
    #     continue
    # print(result[0].boxes.cls.tolist())
    # # Plot the results
    # res_plotted = cv2.resize(result[0].plot(), (640,640))
    # cv2.imshow('o', res_plotted)
    # cv2.waitKey(0)
    # if not os.path.exists(label_path): continue
    # gt_boxes = open(label_path, "r").readlines()
    # if not os.path.exists(img):continue
    # org_img = cv2.imread(img)
    # # if the image is empty in cv2 continue
    # if org_img is None:continue
    # number_1 = 0
    # for box in gt_boxes:
        
    #     box = box.split(" ")
    #     if box[0]=='1':
    #         number_1 += 1
    #     try:
    #         x1 = int(float(box[1])*640)
    #         y1 = int(float(box[2])*640)
    #         x2 = int(float(box[3])*640)
    #         y2 = int(float(box[4])*640)
    #     except:
    #         print(label_path)
    #     # create from orginal image
    #     _x2 = x1 + x2
    #     _y2 = y1 + y2
    #     x1 = x1 - x2//2
    #     y1 = y1 - y2//2
    #     x2 = _x2
    #     y2 = _y2
    #     org_img = cv2.resize(org_img, (640,640))
    #     #draw in yolo format
    #     cv2.rectangle(org_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # if org_img.shape != res_plotted.shape:
    #     res_plotted = cv2.resize(res_plotted, (org_img.shape[1], org_img.shape[0]))
    # # concatenate the original image and the result image
    # video_writer.write(res_plotted)
    # res_plotted = cv2.hconcat([org_img, res_plotted])
    
    # #video_writer.write(res_plotted)
    # print("num_1_true",number_1)
    # print("num_1_pred",num_1_pre)
    # out_folder = "./diff"
    # if not os.path.exists(out_folder):
    #     os.makedirs(out_folder)
    # # if number_1 < num_1_pre:
    # cv2.imwrite(out_folder + "/" + filename, res_plotted)
        
    # Display the result
        
    #cv2.imshow("result", res_plotted)
    
video_writer.release()   
# Release the video capture and close the OpenCV windows
# cap.release()
cv2.destroyAllWindows()