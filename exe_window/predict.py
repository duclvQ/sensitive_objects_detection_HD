import os
from typing import Any
import cv2
import numpy as np
import threading
import queue
import time
from ultralytics import YOLO
import sys
import argparse
import warnings
import torch
import GPUtil
import logging
from datetime import datetime
import re
from PIL import Image
import re
from collections import defaultdict

from resnet_inference import ResNetClassifier
from dataclasses import dataclass
HOME = os.getcwd()
sys.path.append(f"{HOME}/ByteTrack/")

import yolox

#from yolox.tracker.byte_tracker import BYTETracker, STrack
# Ignore specific warning categories
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image")
warnings.filterwarnings("ignore", category=UserWarning, module="torch._jit_internal")
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Process some args.')

# Add arguments
parser.add_argument('video_path', metavar='video_path', type=str, help='path to input video')
parser.add_argument('--model', type=str, default='model.pt', help='path to model.pt file')
parser.add_argument('--model_resnet', type=str, default='flag_resnet18.pth', help='path to double check model')
parser.add_argument('--conf', type=float, default=0.3, help='minimum confidence threshold for detection')
parser.add_argument('--stride', type=int, default=5, help='frame stride for detection')
parser.add_argument('--run_on', type=int, default=0, help='choose which GPU to run on')
parser.add_argument('--timing_inspection', type=int, default=0, help='print inference time for each batch')
parser.add_argument('--skip_similarity', type=int, default=0, help='skip similars frames by histogram comparison')
# Parse arguments
args = parser.parse_args()

label_dict = {0: "china", 1:"vietnam", 2:"malaysia", 3:"nine_dash_line", 5:"flag"}
@dataclass(frozen=True )
class BYTETrackerArgs:
    track_thresh = 0.25
    track_buffer = 30
    match_thresh = 0.5
    aspect_ratio_thresh = 0.25
    min_box_area = 1.0
    mot20 = False

class HD_Detection:
    """
    Detection class.

    Attributes:
        video_path (str): The path to the video file to process.
        model (str): The path to the trained model for object detection.
        model_reset (str): The path to the trained model for flag classification.
        conf (float): The confidence threshold for object detection.
        stride (int): The stride size for the sliding window in the object detection algorithm.
        run_on (int): The device to run the object detection on ('cpu' or 'gpu').
        timing_inspection (bool): Whether to print timing information for debugging purposes.
        skip_similarity (bool): Whether to skip frames that are similar to the previous frame to speed up processing.

    Methods:
        __call__(): Starts the video processing and object detection.
        capture_frames(cap, frame_queue): Captures frames from the video and puts them in a queue.
        process_frames(model, frame_queue, conf): Processes frames from the queue and performs object detection.
        start_predictor(): Starts the frame capture and processing threads.
    """
    def __init__(self, video_path: str, model:str,  conf: float, stride:int, run_on: int, timing_inspection: int, similarity_comparision: int) -> None:
        
        ##### checking phase
        
       
        # print("available memory: ", torch.cuda.get_device_properties(0).total_memory)
        if not self.check_if_video_exists(video_path):
            raise Exception("Video not found")
            sys.exit(1)
        else:
            #print("Video found")
            pass
        
        if not self.is_conf_number(conf):
            raise Exception("Confidence must be a float number")
            sys.exit(1)
        else:
            if not self.check_if_conf_is_valid(conf):
                raise Exception("Confidence must be in range [0,1]")
                sys.exit(1)
            else:
                #print("Confidence is valid")
                pass
        if not self.is_stride_number(stride):
            raise Exception("Stride must be an integer number")
            sys.exit(1)
        else:
            if not self.check_if_stride_is_valid(stride):
                raise Exception("Stride must be greater than 0")
                sys.exit(1)
            else:
                #print("Stride is valid")
                pass
        if not self.check_if_model_exists(f"{model}"):
            raise Exception("Model not found")
            sys.exit(1)
        else:
            #print("Model found")
            pass
    
        #print("Checking phase finished")
        ##### end checking phase
        #########################################################

        
        # init phase
        ### init tracker
        #self.byte_tracker = BYTETracker(BYTETrackerArgs())

        self.width = 0
        self.height = 0
        self.frames_need_to_be_recheck = list()
        self.frames_need_to_be_recheck.append([0,None])
        self.sensitive_frames = list()
        self.label_of_sensitive_frames = list()
        self.confidence_of_sensitive_frames = list()
        self.flag_box_of_sensitive_frames = list()
        
        self.previous_percent = 0
        self.timing_inspection = False
        if timing_inspection==1:
            self.timing_inspection = True
        self.capture_done = threading.Event() # event to signal predict thread to stop
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.video_FPS = self.cap.get(cv2.CAP_PROP_FPS)
        print("Video FPS:",self.video_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.conf = conf
        self.stride = stride
        self.similarity_comparision = similarity_comparision
        self.model_path = model
        num_gpus = torch.cuda.device_count()
        if run_on>=num_gpus or run_on<-1:
            raise Exception("Device number is out of range, it should be in range [-1, {num_gpus}-1]")
            sys.exit(1)
        if run_on >=0:
            if self.get_size_of_free_memory(run_on) < 2000:
                raise Exception("Not enough memory to run the model")
                sys.exit(1)
        if run_on == -1:
            self.device = torch.device("cpu")
            print("Running on CPU")
        else:
            if not torch.cuda.is_available():
                raise Exception("CUDA is not available")
                sys.exit(1)
            self.device = torch.device(f"cuda:{run_on}")
            print(f"Running on GPU {run_on}")
        self.log_path = self.setup_logging(video_path)  
        self.Predictor = YOLO(model= self.model_path, task = "detect")
        #self.ResNet = self.load_resnet(model_path = model_resnet, device = self.device)
        # Initialize SORT
        #print('init MOT_Tracker')
        #self.mot_tracker = Sort()
        
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        print("checking phase finished")
        print("init phase finished")
        print("Video processing started")
        self.start_predictor()
        
    def setup_logging(self, video_path: str) -> None:
        # Set up logging
        os.makedirs('det_logs', exist_ok=True)  # setup log dir
        filename = os.path.basename(video_path)
        video_name_without_extension = os.path.splitext(filename)[0]
        # Get the current date and time
        now = datetime.now()
        # Format the current date and time as a string
        now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        log_path = f'det_logs/{now_str}_____{video_name_without_extension}.txt'
        logging.basicConfig(filename=log_path, level=logging.INFO)
        return log_path
    @staticmethod
    def compute_duration(time1, time2):
        # Convert the times to datetime objects
        time1 = datetime.strptime(time1, "%H:%M:%S:%f")
        time2 = datetime.strptime(time2, "%H:%M:%S:%f")

        # Compute the duration
        duration = time2 - time1

        # Convert the duration to seconds
        duration_in_s = duration.total_seconds()

        return duration_in_s
    def seconds_to_timecode(self, seconds: float) -> str:
        #print(seconds)
        hours = int(seconds // 3600)

        minutes = int((seconds % 3600) // 60)

        _seconds = int(seconds%60)
        miliseconds = self.get_first_two_digits((seconds - int(seconds))*100)
        #print("miliseconds", (seconds - int(seconds))*100)
        return f"{hours:02d}:{minutes:02d}:{_seconds:02d}:{miliseconds}"
    def zeros_to_timecode(self, ms: float) -> str:
        return f"00:00:00:{(str(int(ms*100))).zfill(2)}"
    def post_process_log(self, log_path: str) -> None:
        pattern = pattern = r"INFO:root:\[DETECTED\]type=\[(.*?)\];timecode=(\d{2}:\d{2}:\d{2}:\d{2});position=([0-9.,]+);confidence=([0-9.]+);"
        dict_results = dict()
        #self.get_frames_need_to_be_recheck()
        #wrong_frames = self.recheck_frames()
        #print(wrong_frames)
        #wrong_frames_timecode = [self.frame_to_timecode(frame_num, self.video_FPS) for frame_num in wrong_frames]
        with open(log_path, 'r') as f:
            for line in f:
                match = re.search(pattern, line)
                #print(match)
                if match:
                    # Extract the type, timecode, position, and confidence
                    type_ = match.group(1)
                    timecode = match.group(2)
                    #if timecode in wrong_frames_timecode:
                    #    continue
                        #### rewrite this
                    position = match.group(3)
                    confidence = match.group(4)
                    if type_ not in list(dict_results.keys()):
                        dict_results[type_] = dict()
                        dict_results[type_]['timecode'] = list()
                        dict_results[type_]['position'] = list()
                        dict_results[type_]['confidence'] = list()
                        
                    dict_results[type_]['timecode'].append(timecode)
                    dict_results[type_]['position'].append(position)
                    dict_results[type_]['confidence'].append(confidence)
                    
        for type_ in list(dict_results.keys()):
            start_time = dict_results[type_]['timecode'][0]
            previous_time = start_time
            miscs_list = list()
            for i in range(len(dict_results[type_]['timecode'])):
                next_time = dict_results[type_]['timecode'][i]
                gap_time = self.compute_duration(previous_time, next_time)
                _position = dict_results[type_]['position'][i]
                x, y, w, h = _position.split(',')
                _confidence = dict_results[type_]['confidence'][i]
                #print(position)
                miscs_sample = {
                    "TimeCode": f"{next_time}",
                    "XCenterCoordinates": f"{x}",
                    "YCenterCoordinates": f"{y}",
                    "Width": f"{w}",
                    "Height": f"{h}",
                    "Confidence": str(_confidence)
                }
                
                if gap_time > 2:
                    _duration = self.compute_duration(start_time, previous_time)
                    
                    if _duration ==0:
                        _duration = self.zeros_to_timecode(1/self.video_FPS)
                    else:
                        _duration = self.seconds_to_timecode(_duration)
                    if type_=="flag":
                        frame_num_ = self.timecode_to_frame(start_time, self.video_FPS)
                        s = miscs_list[0]
                        
                        bbox = []
                        #if self.recheck_first_frame(frame_num_,  ) ==False:
                        #    continue
                    
                    print(f"[DETECTED]type=[{type_}];timecode={start_time};duration={_duration};position={miscs_list}")
                    miscs_list = list()
                    start_time = next_time
                
                miscs_list.append(miscs_sample)
                previous_time = next_time
            #if 
            next_time = dict_results[type_]['timecode'][-1]
            
            duration = self.compute_duration(start_time, next_time)
            #print(duration)
            #print(duration)
            _duration = self.seconds_to_timecode(duration)
            if duration ==0:
                duration = self.zeros_to_timecode(1/self.video_FPS)
                _duration = self.seconds_to_timecode(duration)
                #print('end')
            print(f"[DETECTED]type=[{type_}];timecode={start_time};duration={_duration};position={miscs_list}")
    def timecode_to_frame(self, timecode: str, fps: float) -> int:
        hours, minutes, seconds, miliseconds = timecode.split(':')
        hours = int(hours)
        minutes = int(minutes)
        seconds = int(seconds)
        miliseconds = int(miliseconds)
        total_seconds = hours*3600 + minutes*60 + seconds + miliseconds
        frame_num = int(total_seconds*fps)
        #print(fps)
        return frame_num
    def get_frames_need_to_be_recheck(self, ) -> None:
        for idx, frame_num in enumerate(self.sensitive_frames):
            if self.label_of_sensitive_frames[idx] == "flag" and idx>2 and idx < len(self.label_of_sensitive_frames)-2:
                if self.label_of_sensitive_frames[idx-2] != "flag" and self.label_of_sensitive_frames[idx+2] != "flag":
                    self.frames_need_to_be_recheck.append(frame_num)
                elif self.confidence_of_sensitive_frames[idx] < 0.4:
                    self.frames_need_to_be_recheck.append(frame_num)
                
                else:
                    continue
    def recheck_frames(self, ) -> list:
        cap = cv2.VideoCapture(self.video_path)
        wrong_frames = list()
        for frame_num, bbox in self.frames_need_to_be_recheck[1:]:
            #seek to frame
            frame_list = list()
            for i in range(frame_num-1, frame_num+2):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                #print(bbox)
                crop_img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]  
                #crop_img = np.array(crop_img)
                #crop_img = torch.tensor(crop_img)
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

                frame_list.append(crop_img)

            #print(frame_num)
            frame_list = [Image.fromarray(image) for image in frame_list]
            results = self.ResNet.predict_a_batch(frame_list)
            #print(results)
            if results.count("flag")<=1:
                wrong_frames.append(frame_num)
            
            
        return wrong_frames
          
    def recheck_first_frame(self, frame_num, bbox) -> list:
        cap = cv2.VideoCapture(self.video_path)
        #seek to frame
        frame_list = list()
        for i in range(frame_num-1, frame_num+2):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            #print(bbox)
            crop_img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]  
            crop_img = np.array(crop_img)
            #crop_img = torch.tensor(crop_img)
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

            frame_list.append(crop_img)

        #print(frame_num)
        frame_list = [Image.fromarray(image) for image in frame_list]
        results = self.ResNet.predict_a_batch(frame_list)
        #print(results)
        if results.count("flag")<=1:
                return False
        return True    
                
            
    # classifying flag
    def classify_flag(self, image_list) -> list:
        results = self.ResNet.predict_a_batch(image_list)
        return results
    
    # load resnet to classify flag
    def load_resnet(self, model_path: str, device: int) -> Any:
        # load model
        model = ResNetClassifier(model_path=model_path, device=device)
        return model        
    def get_size_of_free_memory(self, device) -> int:
        gpus = GPUtil.getGPUs()
        gpu = gpus[device]     
        return gpu.memoryFree
    def check_if_model_exists(self, model_path: str) -> bool:
        return os.path.exists(model_path)
    def check_if_conf_is_valid(self, conf: float) -> bool:
        return conf>=0 and conf<=1
    def check_if_stride_is_valid(self, stride: int) -> bool:
        return stride>0
    def is_conf_number(self, conf: float) -> bool:
        return isinstance(conf, float)
    def is_stride_number(self, stride: int) -> bool:
        return isinstance(stride, int)
    def check_if_video_exists(self, video_path: str) -> bool:
        return os.path.exists(video_path)
    @staticmethod
    def get_first_two_digits(num: int) -> str:
        if num>=100:
            num/=10
        num_str = str(num)
        #print(num_str)
        if '.' in num_str:
            num_str = num_str.replace('.', '')
        #print(num_str[:2].zfill(2))
        return num_str[:2].zfill(2)
    
    def frame_to_timecode(self, frame_num, fps) -> str:
        total_seconds = frame_num / fps
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        miliseconds = frame_num%(int(fps))
        
        miliseconds = self.get_first_two_digits(miliseconds)
        
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{miliseconds}"

    def similarity_mesure(self, img1, img2, inference_time_visible=False) -> float:
        if inference_time_visible:
            start_time = time.time()
            # histogram comparison
            hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
            similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            end_time = time.time()
            print(f"Similarity Inference time: {end_time - start_time}")
        else:
            # histogram comparison
            hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
            similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return similarity
    
    def predict(self, model, frame_list, frame_num_list, conf) -> None:
        if self.device == torch.device("cpu"):
            haft = False
        else:
            haft = True
        frame_shape = frame_list[0].shape
        results = model.predict(source = frame_list, verbose=self.timing_inspection,  stream=False, save=False, show=False, device=self.device, half=haft, conf = conf)   
        # get width and height of frame
        width_height = f"{self.width},{self.height}"
        for idx, r in enumerate(results):
            percent = int(frame_num_list[idx]/self.total_frames*100)

            if r.boxes.cls.size()[0]>0:
                          
                tensor_data = r.boxes.xywhn 
                box_list = r.boxes.xyxy
                id_cls_list = r.boxes.cls
                for i in range(tensor_data.size(0)):
                    # Get bounding boxes and confidences
                    bbox_xyxy = r.boxes.xyxy.cpu().numpy()
                    confidences = r.boxes.conf.cpu().numpy()
                    dets = np.hstack((bbox_xyxy, confidences[:, np.newaxis])).astype(np.float32, copy=False)
                    
                   
                    
                    
                    
                    if int(id_cls_list[i].item()) == 0 and r.boxes.conf[i].item() < 0.4: continue
                    if int(id_cls_list[i].item()) == 2 and r.boxes.conf[i].item() < 0.4: continue
                    x, y, w, h = tensor_data[i].tolist()
                    position = f"{x},{y},{w},{h}"
                    predicted_label = label_dict[int(id_cls_list[i].item())]
                    if predicted_label == "flag" and frame_num_list[idx]>self.stride:
                        if (frame_num_list[idx]!=self.frames_need_to_be_recheck[-1][0]) and (frame_num_list[idx]-self.frames_need_to_be_recheck[-1][0]!=self.stride):
                            
                            if r.boxes.conf[i].item() < 0.5: 
                                self.frames_need_to_be_recheck.append((frame_num_list[idx], box_list[i].tolist()))
                                
                            elif frame_num_list[idx] - self.stride != self.frames_need_to_be_recheck[-1]:
                                self.frames_need_to_be_recheck.append((frame_num_list[idx], box_list[i].tolist()))
                             
                    timecode = self.frame_to_timecode(frame_num_list[idx], self.video_FPS) 
                    msg = f"[DETECTED]type=[{predicted_label}];timecode={timecode};position={position};confidence={round(r.boxes.conf[i].item(), 2)};frame={frame_num_list[idx]};w_h={width_height}"
                    #print(msg)  
                    logging.info(msg)   
           
            if percent!=self.previous_percent:
                print(f"Progress: {percent}%")
            self.previous_percent = percent

                #timecode = self.frame_to_timecode(frame_num_list[idx], self.video_FPS)
                #msg = f"[DETECTED]type=[none];timecode={timecode};position=none;confidence=none;"
                #print(msg)
                
                    

    def capture_frames(self, cap, frame_queue) -> None:
        
        cap = cap
        frame_count = 0
        frame_num = 0
        ret, frame = cap.read()
        self.height, self.width, _ = frame.shape
        while True:
            read_start_time = time.time()
            ret = cap.grab()
            frame_count+=1
            if ret == True:
                if frame_count%self.stride!=0: 
                    continue
                status, frame =cap.retrieve()  # Decode processing frame
                if self.timing_inspection:
                    print(f"Read {self.stride} frames in {(time.time() - read_start_time)*1000} miliseconds")
                frame_num = frame_count
                frame_queue.put([frame, frame_num])
                #print(frame_num)
                if frame_queue.qsize()>32: # Wait for 10ms to prevent running out of memory.
                    time.sleep(0.030)
            else:
                cap.release()
                time.sleep(0.1) # wait for 100ms to make sure all frames are processed.
                       
                break # Break the loop when video is completed
        self.capture_done.set() 
        

    # Function to process frames for prediction
    def process_frames(self, model, frame_queue, conf) -> None:
        frame_list = list()
        frame_num_list = list()
        none_list = list()
        
        while True:
            frame, frame_num = frame_queue.get() 
            if (frame_queue.qsize()==0):
                time.sleep(0.020) # wait for 20ms to prevent running out of memory.
                
                if frame_queue.qsize()==0:
                    time.sleep(0.5) # wait for 500ms to make sure all frames are processed.
                    
            if self.similarity_comparision:
                # check if frame is similar to previous frame by histogram comparison
                if len(frame_list)>1 and len(frame_list)<16:
                    if self.similarity_mesure(frame_list[-1], frame, inference_time_visible=False)>0.9995:
                        continue
            frame_list.append(frame)
            frame_num_list.append(frame_num)
            if len(frame_list)==16:
                self.predict(model, frame_list, frame_num_list, conf)
                frame_list = []
                frame_num_list = []

            if  self.capture_done.is_set() and frame_queue.qsize()==0:
                break
            
            
            

    def start_predictor(self, ) -> None:
        frame_queue = queue.Queue()
        capture_thread = threading.Thread(target=self.capture_frames, args=(self.cap, frame_queue))
        predict_thread = threading.Thread(target=self.process_frames, args=(self.Predictor, frame_queue,self.conf))
                                          
        capture_thread.start()
        predict_thread.start()
        capture_thread.join()
        predict_thread.join()
        print(f"Progress: {int(100)}%")
        self.post_process_log(self.log_path)
import time
start_time = time.time()

#Detector = HD_Detection(args.video_path, args.model, args.model_resnet, args.conf, args.stride, args.run_on, args.timing_inspection, args.skip_similarity)
Detector = HD_Detection(args.video_path, args.model, args.conf, args.stride, args.run_on, args.timing_inspection, args.skip_similarity)

Detector()
print(f"Total inference time: {round(time.time() - start_time, 2)} seconds for a video of {Detector.total_frames} frames")



