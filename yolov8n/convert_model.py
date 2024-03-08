
import timeit

from ultralytics import YOLO

# Load a model
#model = YOLO("yolov8m.yaml")  # build a new model from scratch
model = YOLO("runs/detect/train12/weights/best.pt")  # load a pretrained model (recommended for training)
model.fuse()
# Export the model
model.export(format='engine', half=True, nms= True, device=0)
#model = YOLO("runs/detect/train7/weights/best.torchscript")
import cv2
import numpy as np
"""
# Open the video file
video_path = '22.mp4'
cap = cv2.VideoCapture(video_path)
init = timeit.default_timer()
# Frame sampling rate (5 frames per second)
frame_rate = 5
frame_interval = int(cap.get(cv2.CAP_PROP_FPS) / frame_rate)
count = 0
frames = []
final_results = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
    if len(frames)==15: 
        start = timeit.default_timer()
        results = model.predict(source = frames,  stream=False, save=False, show=True, device=0, half=True)
        for idx, r in enumerate(results):
            if r.boxes.cls.size()[0]>0:
                #print(r.boxes.cls.size()[0])
            
                #print(count )
                final_results.append(count+idx)


        stop = timeit.default_timer()
        print("time:", stop -start )
        frames = []
    
    count +=1
    # Move to the next frame
    for _ in range(frame_interval - 1):
        cap.read()

cap.release()

end = timeit.default_timer()
print(count)

print("total time: ", end-init)
with open(r'22.txt', 'w') as fp:
    for item in final_results:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Done')
"""

if __name__ == '__main__':
    
    #model.train(data="dataset.yaml", epochs=20, batch=32, pretrained=True, lr0=0.01)  # train the model
    #model.val(data="dataset.yaml",batch=16, device=0, iou=0.01, conf=0.2)  # evaluate model performance on the validation set
    #print(len(metrics[0].boxes.cls.tolist()))
    pass
    #metrics..map    # map50-95
    #metrics.box.map50  # map50
    #metrics.box.map75  # map75
    #metrics.box.maps   # a list contains map50-95 of each category
    #results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    #path = model.export(format="onnx")  # export the model to ONNX format

