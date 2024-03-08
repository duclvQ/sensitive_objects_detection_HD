import cv2
import threading
import queue
import time
from ultralytics import YOLO
# Create a shared variable to track the current frame number
#current_frame = 0
lock = threading.Lock()
shared_var_changed_event = threading.Event()
# Shared variable
percent_var = 0
def predict(model, frame_list):
    results = model.predict(source = frame_list, verbose=False,  stream=False, save=False, show=False, device=0, half=True)
    #print('results:',results)       
    final_results = list()
    for idx, r in enumerate(results):
        if r.boxes.cls.size()[0]>0:
            #print(filename_list[i+idx])
            final_results.append(1)
            
    return final_results

# Function to capture frames from a video and put them in a queue
def capture_frames(video_path, frame_queue, thread_id):
    cap = cv2.VideoCapture(video_path)
    global percent_var
    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if thread_id == 0:
        start_frame = 0
        stop_frame = int(total_frames/3)-1
    if thread_id == 1:
        start_frame = int(total_frames/3)
        stop_frame = int((total_frames*2)/3)-1
    if thread_id == 2:
        start_frame = int((total_frames*2)/3)
        stop_frame = int((total_frames*3)/3) -1
        #stop_frame = total_frames-1
    #if thread_id == 3:
    #    start_frame = int((total_frames*3)/4)
    #    stop_frame = total_frames-1


    #print("start_frame:", start_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_num = start_frame
    
    while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break   
            frame_num+=1
            if thread_id==0:
                with lock:
                    percent_var = (frame_num-start_frame)/(stop_frame-start_frame)
                    #time.sleep(0.001)
            if frame_num%6==0: 
                #if frame_num%2==thread_id:
                frame_queue.put(frame)
            #print("current id:", thread_id)
            if frame_queue.qsize()>256: # Wait for 10ms to prevent running out of memory.
                #print(f"sleep in thread {frame_queue.qsize()}")
                #print(frame_num)
                time.sleep(0.050)
            if cap.get(cv2.CAP_PROP_POS_FRAMES) >= stop_frame:
                break
    cap.release()
    frame_queue.put(None)  # Signal end of frames
    #print(f"#####put the thread {thread_id}########")

# Function to process frames for prediction
def process_frames(model, frame_queue):
    frame_list = list()
    none_list = list()
    while True:
        #print('qsize:', frame_queue.qsize())
        
        frame = frame_queue.get() 
        
        if frame is None:
            #print("coming to the end")
            none_list.append(frame)
            if len(none_list)==3: 
                #print(none_list)
                break
            else:
                continue

        if (frame_queue.qsize()==0):
            #print("wait 100ms in predict")
            time.sleep(0.100)
        frame_list.append(frame)
        #print("qsize:", frame_queue.qsize())
        #print(len(frame_list))

        if len(frame_list)==16:
            #print('p')
            results = predict(model, frame_list)
            #print(results)
            frame_list = []
        
# Event for signaling changes


#if __name__ == "__main__":
import timeit
def process_video(video_path="../yolov8n/22.mp4"):
    start_time = timeit.default_timer()
    
    model = YOLO("../yolov8n/runs/detect/train12/weights/best.pt", task = "detect")

    frame_queue = queue.Queue()

    thread1 = threading.Thread(target=capture_frames, args=(video_path, frame_queue, 0))
    thread2 = threading.Thread(target=capture_frames, args=(video_path, frame_queue, 1))
    thread3 = threading.Thread(target=capture_frames, args=(video_path, frame_queue, 2))
    #thread4 = threading.Thread(target=capture_frames, args=(video_path, frame_queue, 3))

    predict_thread = threading.Thread(target=process_frames, args=(model, frame_queue,))

    thread1.daemon = True
    thread1.start()
    thread2.start()
    thread3.start()
    #thread4.start()
    predict_thread.start()
    shared_var_changed_event.set()
    # Main thread
    #while True:
    #    with lock:
    #        current_value = percent_var
    #    if current_value==1.0: break
    #    print("Current value of the variable:", current_value)
    #    time.sleep(2)  # Query the variable every 2 seconds

    thread1.join()
    thread2.join()
    thread3.join()
    #thread4.join()
    predict_thread.join()
#process_video()


#stop_time = timeit.default_timer()

#print('inference_time:', stop_time-start_time)
