
import timeit
import os
from ultralytics import YOLO

# Load a model
#model = YOLO("yolov8m.yaml")  # build a new model from scratch

if __name__ == '__main__':    
    
    model = YOLO("runs/detect/train41/weights/best.pt")  # load a pretrained model (recommended for training)
    validation_results = model.val(data = 'dataset.yaml', batch=8,
                                   conf=0.2,
                                   iou=0.4,
                                   device='0')
#model.fuse()
# Export the model
#model.export(format='engine', nms= True,dynamic=True,simplify = True, device=0, half=False)
#model = YOLO("runs/detect/train12/weights/best.engine", task = "detect")
#model = YOLO("best.engine")
# import cv2
# import numpy as np
# import torch
# # Open the video file
# video_path = '22.mp4'
# model.predict(source = video_path, verbose=True, stream=False, save=True, show=True, device=0, half=True)
# #for r in results:
# #            boxes = r.boxes  # Boxes object for bbox outputs
# stride = 1
"""
init = timeit.default_timer()
filename_list = os.listdir(video_path)
print("num file", len(filename_list))
for idx, file in enumerate(filename_list):
    filename_list[idx] = os.path.join(video_path, file)

for i in range(0,len(filename_list)- stride, stride):
    print("+++++++++++",i)
    start = timeit.default_timer()
    results = model.predict(source = filename_list[i:i+stride], verbose=True, stream=False, save=False, show=False, device=1, half=True)
    stop = timeit.default_timer()
    print("time: ",stop - start)
end = timeit.default_timer()
print(end-init)

cap = cv2.VideoCapture(video_path)
init = timeit.default_timer()
# Frame sampling rate (5 frames per second)
frame_rate = 5
frame_interval = int(cap.get(cv2.CAP_PROP_FPS) / frame_rate)
count = 0
frames = []
final_results = []
start, stop =0,0
frame_count = 0
frame_number = 0
while True:
    # Seek to the next frame that is divisible by 5
    frame_number += 5
    #cap.set(5, frame_number)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
    ret, frame = cap.read()
    
    if not ret:
        break
    frame_count+=1
    '''
    frames.append(frame)
    if len(frames)==stride: 
        start = timeit.default_timer()
        #source = torch.rand(32, 3, 640, 640, dtype=torch.float16)
        results = model.predict(source = frames, verbose=True, stream=False, save=False, show=False, device=1, half=True)
        for idx, r in enumerate(results):
                if r.boxes.cls.size()[0]>0:
                    final_results.append(count+idx)

        stop = timeit.default_timer()
        print(f"at frame number {frame_number}, inf time:", stop -start )
        frames = []
    '''  
    print(f"at frame number {frame_number})")
    count +=1
        
cap.release()

end = timeit.default_timer()
print(count)

print("total time: ", end-init)
with open(r'22.txt', 'w') as fp:
    for item in final_results:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Done')


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
"""