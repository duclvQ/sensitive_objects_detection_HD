
import timeit
import os
from ultralytics import YOLO
import multiprocessing
import cv2
model1 = YOLO("../yolov8n/runs/detect/train12/weights/best.engine", task = "detect")
model2 = YOLO("../yolov8n/runs/detect/train12/weights/best.engine", task = "detect")





# Open the video file
video_path = '../yolov8n/22.mp4'
stride = 1

#init = timeit.default_timer()
#filename_list = os.listdir(video_path)
#print("num file", len(filename_list))
#for idx, file in enumerate(filename_list):
#    filename_list[idx] = os.path.join(video_path, file)



every = 5
# Define a function to be executed by each process
def worker_function(final_results, number):
    capture = cv2.VideoCapture(video_path)  # open the video using OpenCV
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # get its total frame count
    #print(f"Worker {number}: {result}")
    stride = 1
    
    if number == 0:
        #_start_idx = 0
        #_end_idx = int(len(filename_list) /2)
        gpu_num = 0
        frame_start = 0
        frame_end = int(total/2)
        model = model1
    else:
        #_start_idx = int(len(filename_list) /2)
        #_end_idx = int(len(filename_list) /1)
        gpu_num = 1
        frame_start= int(total/2)
        frame_end = total
        model = model2
    frame_num = frame_start
    capture.set(1, frame_start)  # set the starting frame of the capture
    while frame_num < frame_end:  # lets loop through the frames until the end

        _, image = capture.read()  # read an image from the capture


        if frame_num % every == 0:  # if this is a frame we want to write out based on the 'every' argument
            results = model.predict(source = image, verbose=True, stream=False, save=False, show=False, device=gpu_num, half=True)
            print(results)
            for idx, r in enumerate(results):
                        if r.boxes.cls.size()[0]>0:
                            #print(filename_list[i+idx])
                            final_results.append(frame_num)

        frame_num += 1  # increment our frame count

    capture.release()  # after the while has finished close the capture
    """
    for i in range(_start_idx,_end_idx - stride, stride):
            #print("+++++++++++",i)
            #start = timeit.default_timer()
            
            results = model.predict(source = filename_list[i:i+stride], verbose=True, stream=False, save=False, show=False, device=gpu_num, half=True)
            print(filename_list[i:i+stride])
            for idx, r in enumerate(results):
                        if r.boxes.cls.size()[0]>0:
                            print(filename_list[i+idx])
                            final_results.append(filename_list[i+idx])
                            # Specify the file path where you want to save the list
                            file_path = f"results/frame_list{idx+i}.txt"

                            # Open the file in write mode
                            with open(file_path, "w") as file:
                                
                                    file.write(str(1) + "\n")
    """

    
if __name__ == "__main__":
    
    #results = model1.predict(source = filename_list[0], verbose=True, stream=False, save=False, show=True, device=0, half=True)
    #print(results)
    # Open the video file

    # Create a list of numbers
    numbers = [0,1]
    # Create a multiprocessing Manager
    with multiprocessing.Manager() as manager:
        #create a managed list
        shared_list = manager.list()
        # Create a multiprocessing Pool with a specified number of processes
        with multiprocessing.Pool(processes=2) as pool:
            
            # Use the map function to apply the worker_function to each number in parallel
            pool.starmap(worker_function, [(shared_list, item) for item in numbers])

        # Specify the file path where you want to save the list
        file_path = f"frame_list__.txt"

        # Open the file in write mode
        with open(file_path, "w") as file:
            # Iterate through the list and write each item to the file
            for item in shared_list:
                file.write(str(item) + "\n")


