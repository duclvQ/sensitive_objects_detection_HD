# Copyright (C) 2022 The Qt Company Ltd.
# SPDX-License-Identifier: LicenseRef-Qt-Commercial OR BSD-3-Clause

"""PySide6 Multimedia player example"""
import subprocess
try: 
    import PySide6
except:
    subprocess.check_call(["pip", "install", "PySide6"])
import pandas as pd
import sys
#import QScrollArea

from PySide6.QtCore import QStandardPaths, Qt, Slot, QTimer
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import (QApplication, QDialog, QFileDialog, 
    QMainWindow, QSlider, QStyle, QToolBar, QFrame)
from PySide6.QtMultimedia import ( QMediaFormat,
                                  QMediaPlayer, QMediaPlayer)
from PySide6.QtMultimediaWidgets import QVideoWidget, QGraphicsVideoItem
from PySide6.QtWidgets import QApplication,QGraphicsScene,  QMainWindow,QDoubleSpinBox, QLineEdit, QProgressBar,QSpinBox, QMessageBox, QSplitter, QTextEdit, QGridLayout,QLabel, QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QGraphicsRectItem, QGraphicsLineItem
from PySide6.QtGui import QColor, QPalette, QPixmap, QImage, QKeyEvent, QPainter, QPen, QImage, QPainter, QPen, QColor, QWindow
from PySide6.QtCore import QMetaObject,  Qt, QThread, Signal,  QObject
from PySide6 import QtWidgets, QtMultimediaWidgets, QtCore, QtGui, QtMultimedia
from PySide6.QtWidgets import QScrollArea
from PySide6.QtWidgets import QGroupBox
#############import predict module########
#from test_predictor import process_video, lock, percent_var, shared_var_changed_event
import threading
import time
import logging
import cv2
import numpy as np
import os
import timeit
from PIL import Image
from resnet_inference import ResNetClassifier

NUMBER_THREADS = 1 
AVI = "video/x-msvideo"  # AVI


MP4 = 'video/mp4'

import cv2
import threading
import queue
import time
from ultralytics import YOLO
# Create a shared variable to track the current frame number
#current_frame = 0
lock = threading.Lock()
# Shared variable
percent_var = 0
PERCENT_PATH = "percent.txt"
CROPPED_FOLDER = "cropped_frames"
CROPPED_IMADE_dict = dict()
def predict(model, saving_dir, frame_list, frame_num_list, conf):
    results = model.predict(source = frame_list, verbose=False,  stream=False, save=False, show=False, device=0, half=True, conf = conf)
    #print('results:',results)       
    final_results = list()
    for idx, r in enumerate(results):
        if r.boxes.cls.size()[0]>0:
            
            tensor_data = r.boxes.xyxyn
            id_cls_list = r.boxes.cls
            
            #print(r.boxes.conf)
            #print(id_cls_list[0].item())
            # Calculate min and max values for each dimension
            for i in range(tensor_data.size(0)):
                if int(id_cls_list[i].item()) == 0 and r.boxes.conf[i].item() < 0.4: continue
                if int(id_cls_list[i].item()) == 2 and r.boxes.conf[i].item() < 0.4: continue
                min_x = tensor_data[i, 0].item()
                min_y = tensor_data[i, 1].item()
                max_x = tensor_data[i, 2].item()
                max_y = tensor_data[i, 3].item()
                # extend the bounding box
                min_x = max(0, min_x-(max_x-min_x)/2)
                min_y = max(0, min_y-(max_y-min_y)/8)
                max_x = min(1, max_x+(max_x-min_x)/2)
                max_y = min(1, max_y+(max_y-min_y)/8)

                if int(id_cls_list[i].item()) != 5:
                    if int(id_cls_list[i].item()) not in list(CROPPED_IMADE_dict.keys()):
                        CROPPED_IMADE_dict[int(id_cls_list[i].item())] = list()
                        CROPPED_IMADE_dict[int(id_cls_list[i].item())].append(frame_list[idx])
                        # save cropped frame
                        frame = frame_list[idx]
                        height,width, _ = frame.shape
                        cropped_frame = frame[int(min_y*height):int(max_y*height), int(min_x*width):int(max_x*width)]
                        cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                        #save cropped frame
                        cv2.imwrite(f"{CROPPED_FOLDER}/{int(id_cls_list[i].item())}_{len(CROPPED_IMADE_dict[int(id_cls_list[i].item())])}.jpg", cropped_frame)

                    else:
                        print(frame_num_list[idx])
                        # if np.any(frame_num_list[idx]) - np.any(CROPPED_IMADE_dict[int(id_cls_list[i].item())][-1]) >=6*2:
                        # save cropped frame
                        frame = frame_list[idx]
                        height,width, _ = frame.shape
                        cropped_frame = frame[int(min_y*height):int(max_y*height), int(min_x*width):int(max_x*width)]
                        cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                        #save cropped frame
                        cv2.imwrite(f"{CROPPED_FOLDER}/{int(id_cls_list[i].item())}_{len(CROPPED_IMADE_dict[int(id_cls_list[i].item())])}.jpg", cropped_frame)
                        
                        CROPPED_IMADE_dict[int(id_cls_list[i].item())].append(frame_list[idx])
                       
                    

                       
                    frame = frame_list[idx] 

                    height,width, _ = frame.shape
                    cropped_frame = frame[int(min_y*height):int(max_y*height), int(min_x*width):int(max_x*width)]
                    cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                    #save cropped frame
                   


                    
                # Create and write to a plain text file
                with open(saving_dir, 'a') as file:
                    file.write(f"{frame_num_list[idx]} {int(id_cls_list[i].item())} {min_x} {min_y} {max_x} {max_y}\n")
                    file.close()
            
    return final_results

# Function to capture frames from a video and put them in a queue
def capture_frames(video_path, frame_queue, thread_id):
    #cap = cv2.VideoCapture(video_path )
    cap = cv2.VideoCapture(video_path, cv2.CAP_ANY, [ cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY ])
    global percent_var
    global PERCENT_PATH
    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    start_frame = int((thread_id)*total_frames/NUMBER_THREADS)
    stop_frame = total_frames
    #print("start_frame:", start_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_num = start_frame
    
    while cap.isOpened():
        ret = cap.grab()
        if ret:
            ret, frame = cap.retrieve()
            frame_num+=1
            # write percent to file
            with open(PERCENT_PATH, 'w') as file:
                # remove all the content in the file
                file.truncate(0)

                percent_var = int((frame_num-start_frame)/(stop_frame-start_frame)*100)
                file.write(f"{percent_var}")
                file.close()
            if thread_id==0:
                
                with lock:
                    percent_var = (frame_num-start_frame)/(stop_frame-start_frame)
                    #time.sleep(0.001)
            if frame_num%6==0: 
                #if frame_num%2==thread_id:
                frame_queue.put([frame, frame_num])
            #print("current id:", thread_id)
            if frame_queue.qsize()>64: # Wait for 10ms to prevent running out of memory.
                #print(f"sleep in thread {frame_queue.qsize()}")
                #print(frame_num)
                time.sleep(0.050)
            if cap.get(cv2.CAP_PROP_POS_FRAMES) >= stop_frame:
                break
        else:
            break
    cap.release()
    frame_queue.put([None, -1])  # Signal end of frames
    #print(f"#####put the thread {thread_id}########")

# Function to process frames for prediction
def process_frames(model,saving_dir, frame_queue, conf):
    frame_list = list()
    frame_num_list = list()
    none_list = list()
    while True:
        frame, frame_num = frame_queue.get() 
        if frame is None:
            none_list.append(frame)
            if len(none_list)==NUMBER_THREADS: 
                break
            else:
                continue
        if (frame_queue.qsize()==0):
            #print("wait 10ms in predict")
            time.sleep(0.010)
        frame_list.append(frame)
        frame_num_list.append(frame_num)
        if len(frame_list)==16:
            results = predict(model,saving_dir, frame_list, frame_num_list, conf)
            frame_list = []
            frame_num_list = []


#if __name__ == "__main__":
import timeit
def process_video(video_path="../yolov8n/22.mp4", saving_dir="", conf = 0.25):
    start_time = timeit.default_timer()
    
    model = YOLO("../yolov8n/runs/detect/train41/weights/best.pt", task = "detect")
    
    frame_queue = queue.Queue()
    
    thread_list = list()
    for idx in range(NUMBER_THREADS):
        thread_list.append(threading.Thread(target=capture_frames, args=(video_path, frame_queue, idx)))
    
    predict_thread = threading.Thread(target=process_frames, args=(model,saving_dir, frame_queue,conf))

    for i in range(NUMBER_THREADS):
        thread_list[i].start()
    predict_thread.start()
    for i in range(NUMBER_THREADS):
        thread_list[i].join()
    predict_thread.join()
   




def get_supported_mime_types():
    result = []
    for f in QMediaFormat().supportedFileFormats(QMediaFormat.Decode):
        mime_type = QMediaFormat(f).mimeType()
        result.append(mime_type.name())
    return result

class ProgressThread(QThread):
    progress_updated = Signal(float)  # Signal to notify the main thread of progress updates

    def run(self):
        while True:
            time.sleep(0.1)
            # Access the shared variable while holding the lock
            with lock:
                percent = percent_var
                self.progress_updated.emit(percent)
            if int(percent) == 1:
                break

class PredictThread(QThread):
    # Define a custom signal to communicate with the main thread
    update_signal = Signal(str)
    def __init__(self, url, saving_dir, conf):
        super().__init__()
        self.url = url
        self.saving_dir = saving_dir
        self.conf = conf
    def run(self):
        process_video(self.url, self.saving_dir, self.conf)
        self.update_signal.emit("done")


class CustomHLine(QFrame):
    def __init__(self, txt_url, total_frames):
        super(CustomHLine, self).__init__()

        self.prediction_saving_dir = txt_url
        self.total_frames = total_frames
        #self.marked_frames = marked_frames
        self.setFrameShape(QFrame.HLine)
        self.lineWidth = 10
        self.pointPosition = 0.5  # Position along the line (0 to 1)
    def read_the_marked_frames(self):

        with open(self.prediction_saving_dir,'r') as file:
            frame_list = list()
            for line in file:
                # Remove leading and trailing whitespace (e.g., newline characters)
                line = line.strip()
                frame_num = int(line.split(" ")[0])
                frame_list.append(frame_num)
        return frame_list
    def reset(self):

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen()
        pen.setColor(QColor(0, 0, 0))  # Set line color (black)
        pen.setWidth(self.lineWidth)  # Set line width
        painter.setPen(pen)
        painter.drawLine(0, self.height() / 2, self.width(), self.height() / 2)

    def paintEvent(self, event):
        #print(self.total_frames)
        frame_list = self.read_the_marked_frames()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        # Create a QPen for drawing the line
        pen = QPen()
        pen.setColor(QColor(0, 0, 0))  # Set line color (black)
        pen.setWidth(self.lineWidth)  # Set line width
        painter.setPen(pen)
        painter.drawLine(0, self.height() / 2, self.width(), self.height() / 2)

        if len(frame_list)>0:
            for frame_num in frame_list:
                if self.total_frames==0:
                    break
                # Calculate the position where red should start
                red_start = self.width() * (frame_num/self.total_frames)
                # Draw the line in black
                
                # Draw the red portion
                red_pen = QPen()
                red_pen.setColor(QColor(255, 0, 0))  # Set red color
                red_pen.setWidth(self.lineWidth)
                painter.setPen(red_pen)
                painter.drawLine(red_start, self.height() , red_start+0.005, self.height() )


class SliderFrameGroup(QWidget):
    def __init__(self, txt_url, total_frames):
        super().__init__()

        # Create a layout for the group
        self.layout = QGridLayout()  
        # Create sliders and frames
        #self.slider = QSlider(Qt.Horizontal)
        self.hline = CustomHLine(txt_url, total_frames)
        # Add the slider and CustomHLine to the layout
        #self.layout.addWidget(self.slider, 0, 0, 1, 1)
        self.layout.addWidget(self.hline, 1, 0, 1, 1)
        
        # Set the layout for the group
        self.setLayout(self.layout)
class CustomVideoWidget(QVideoWidget):
    def __init__(self):
        super().__init__()
        self.bounding_box = None


    def setBoundingBox(self, xmin, ymin, xmax, ymax):
        self.bounding_box = (xmin, ymin, xmax, ymax)
        self.repaint()

    def clearBoundingBox(self):
        self.bounding_box = None
        self.repaint()

    def paintEvent(self, event):
        painter = QPainter(self)
        
        super().paintEvent(event)
        # Create a QPainter for custom painting
        if self.bounding_box:
            # Set the rectangle color and style
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))

            # Retrieve the rectangle coordinates from self.bounding_box
            xmin, ymin, xmax, ymax = self.bounding_box

            # Draw the rectangle
            painter.drawRect(xmin, ymin, xmax - xmin, ymax - ymin)
        """
        # Paint the bounding box on top of the video frame
        if self.bounding_box:
            print('drawing')
            painter = QPainter(self)
            xmin, ymin, xmax, ymax = self.bounding_box
            x, y, width, height = xmin, ymin, xmax - xmin, ymax - ymin
            painter.setPen(QPen(Qt.red, 20))
            painter.drawRect(x, y, width, height)
        """


class VideoThread(QThread):
    #stop_signal = Signal()
    image_data = Signal(QImage)
    position_changed = Signal(int)
    paused = False
    is_holding = False
    
    def __init__(self, video_source=0, prediction_saving_dir = ""):
        super().__init__()
        self.video_source = video_source
        self.get_recheck_changed = False
        self.stride = 6
        self.prediction_saving_dir = prediction_saving_dir
        print("reading txt",self.prediction_saving_dir)
        self.running = True
        self.cap = cv2.VideoCapture(self.video_source, cv2.CAP_ANY, [ cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY ])
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_position = 0
        self.seek_position = 0
        self.release_pause = False
        self.read_only_text_box = None
        self.not_read = True
        self.saving_folder = ""
        self.classificator = ResNetClassifier(r"E:\HD_VNese_map\dataset\flag_classify\28_02_2024_474757.0_resnet18_0.9952073197298672.pth")
        self.end_check = False
    def post_process(self, filename, stride=6):
        names = ["frame", "class_id", "x", "y", "x_max", "y_max"]
        data = pd.read_csv(filename, sep=" ", comment="#", na_values="Nothing", names=names)
        uniques = data["class_id"].unique()
        data_dict = {}
        recheck_dict = {}
        class_id_ = 5
        _key = 0
        if class_id_ not in uniques:
            return recheck_dict
        flag_data = data[data["class_id"] == class_id_]
        flag_data.reset_index(drop=True, inplace=True)
        continuous_list = []
        first_i = 0
        range_ = len(flag_data['frame'])
        print('range', range_)
        for i in range(len(flag_data['frame'])):
            if len(continuous_list) == 0:
                first_i = i
                _key = flag_data['frame'][i]
                continuous_list.append(flag_data['frame'][i])
                if len(flag_data['frame']) == 1:
                    recheck_dict[_key] = {
                    'frames':continuous_list,
                    'position_of_middle_frame': (flag_data['x'][first_i], flag_data['y'][first_i], flag_data['x_max'][first_i], flag_data['y_max'][first_i])
                }
                continue
                
            if flag_data['frame'][i] - continuous_list[-1] <= stride*3:
                continuous_list.append(flag_data['frame'][i])
            else:
                recheck_dict[_key] = {
                    'frames':continuous_list,
                    'position_of_middle_frame': (flag_data['x'][first_i], flag_data['y'][first_i], flag_data['x_max'][first_i], flag_data['y_max'][first_i])
                }
                continuous_list = []
                continuous_list.append(flag_data['frame'][i])
                first_i = i
                _key = flag_data['frame'][i]
        recheck_dict[_key] = {
                    'frames':continuous_list,
                    'position_of_middle_frame': (flag_data['x'][first_i], flag_data['y'][first_i], flag_data['x_max'][first_i], flag_data['y_max'][first_i])
                }    
            
        # 
        # close file after reading the file

        return recheck_dict

    def read_the_marked_frames(self, prediction_saving_dir ):
        #print("--------------------------------------------------------------------")
        removing_list = list()
        print(self.get_recheck_changed)
        # recheck the flag
        if self.get_recheck_changed:
            # write message to the text box
            print("recheck the flag!!!!!")
            # text_to_append = "---"+ f"Recheck the flag" + "\n"
            # cursor = self.read_only_text_box.textCursor()
            # cursor.setPosition(0)  # Move cursor to the beginning
            # cursor.insertText(text_to_append)
            # # recheck the flag
            os.makedirs(self.saving_folder, exist_ok=True)
            recheck_dict_flag = self.post_process(prediction_saving_dir, self.stride)
            print(type(self.cap.read()[1]))
            height,width, _ = self.cap.read()[1].shape
            print(width, height)
            print(len(list(recheck_dict_flag.keys())))
            if len(list(recheck_dict_flag.keys()))>0:
                print(f"found {len(list(recheck_dict_flag.keys()))} flags to recheck")
                
                for frame_num in list(recheck_dict_flag.keys()):
                    #print(frame_num)
                    frame_list = list()
                    # recheck only middle frame, middle frame + 1, middle frame - 1
                    if frame_num==0:
                        frame_num+=2
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num-1)
                    _resize_crop = None
                    for i in range(15):
                        
                        ret = self.cap.grab()
                        #print(ret)
                        if not ret:
                            continue
                        ret, frame = self.cap.retrieve()
                        cropped_frame = frame[int(recheck_dict_flag[frame_num]['position_of_middle_frame'][1]*height):int(recheck_dict_flag[frame_num]['position_of_middle_frame'][3]*height), int(recheck_dict_flag[frame_num]['position_of_middle_frame'][0]*width):int(recheck_dict_flag[frame_num]['position_of_middle_frame'][2]*width)]
                        if i ==1:
                            _resize_crop = cv2.resize(cropped_frame, (100, 100))
                        cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                        #save cropped frame
                       
                        cropped_frame = Image.fromarray(cropped_frame)
                        frame_list.append(cropped_frame)
                    #print(frame_num)
                    #frame_list = [Image.fromarray(image) for image in frame_list]
                        
                    _result = self.classificator.predict_a_batch(frame_list)
                    print(_result)

                    is_right = 1
                    if _result.count('flag')<=1:
                        is_right = 0
                        removing_list += recheck_dict_flag[frame_num]['frames']
                        # remove the middle frame + 1
                    # add a tex box right side of the _resize_crop
                    txt = "yes" if is_right==1 else "no"
                    if txt == "_yes":
                        # Convert the image to the HSV color space
                        hsv_image = cv2.cvtColor(_resize_crop, cv2.COLOR_BGR2HSV)

                        # Define the lower and upper bounds for the orange color
                        lower_orange = np.array([5, 50, 50])
                        upper_orange = np.array([200, 255, 255])

                        # Create a mask that selects only the orange pixels
                        mask = cv2.inRange(hsv_image, lower_orange, upper_orange)

                        # Count the number of orange pixels
                        orange_pixels = np.sum(mask > 0)

                        # Calculate the percentage of orange pixels
                        total_pixels = _resize_crop.shape[0] * _resize_crop.shape[1]
                        percentage_orange = (orange_pixels / total_pixels) * 100
                        print("percentage_orange", percentage_orange)   
                        if percentage_orange <10:
                            txt = "no"

                    # create a white cv2 image
                    white_img = np.zeros((100, 100, 3), np.uint8)
                    # concatenate the white image and the _resize_crop
                    _resize_crop = np.concatenate((white_img, _resize_crop), axis=1)
                    # add text to the image
                    cv2.putText(_resize_crop, txt, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    # add frame number belowthe text
                    cv2.putText(_resize_crop, str(frame_num), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    # convert to rgb
                    


                    cv2.imwrite(f"{self.saving_folder}/cropped_frame_{frame_num}_{is_right}.jpg", _resize_crop)
            with open(PERCENT_PATH, "w") as file:
                file.truncate(0)
                file.write("100")
                file.close()


            self.end_check = True
            print("end recheck")
            #self.cap.release()

                        

            
        frame_num_and_bounding_box_dict = dict()
        with open(prediction_saving_dir, 'r') as file:
            
            for i in range(self.frame_count):
                frame_num_and_bounding_box_dict[i] = list()
            for line in file:
                # Remove leading and trailing whitespace (e.g., newline characters)
                line = line.strip()
                #print(line)

                parts = line.split()
                if len(parts) > 0 :
                    frame_num, id_cls, xmin, ymin, xmax, ymax = map(float, parts)
                    frame_num = int(frame_num)
                    #print((xmin, ymin, xmax, ymax))
                    frame_num_and_bounding_box_dict[frame_num].append((id_cls,xmin, ymin, xmax, ymax))
                    for i in range(1,3):
                        frame_num_and_bounding_box_dict[frame_num+i].append((id_cls,xmin, ymin, xmax, ymax))
                        frame_num_and_bounding_box_dict[frame_num-i].append((id_cls,xmin, ymin, xmax, ymax))
            
              
        return frame_num_and_bounding_box_dict


    def run(self):
        while self.running:
            if self.not_read:
                self.predicted_results = self.read_the_marked_frames(self.prediction_saving_dir)
                
                self.marked_frames = sorted(list(self.predicted_results.keys()))
                
                self.not_read = False
            if self.is_holding:
                    #print("seek____", self.seek_position)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.seek_position)
            if not self.paused:
                
                ret, frame = self.cap.read()
                if not ret:
                    self.current_position = 0
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (1080, 720))
                frame = cv2.putText(frame, f"frame: {self.current_position}", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                if self.current_position in self.marked_frames and (self.current_position+6) in self.marked_frames :

                    #print(self.predicted_results[self.current_position])
                    for i in range(len(self.predicted_results[self.current_position])):
                        id_cls,xmin, ymin, xmax, ymax = self.predicted_results[self.current_position][i]
                        xmin, ymin, xmax, ymax = int(xmin*1080), int(ymin*720), int(xmax*1080), int(ymax*720) 
                        color = (255, 0, 0)  # Green color (BGR) 
                        thickness = 4
                        if id_cls<5: color = (0, 255, 0)
                        frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness)
                    #self.marked_frames.remove(self.current_position)
                
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.image_data.emit(q_image)

                self.current_position = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                
                self.position_changed.emit(self.current_position)
                if self.current_position == self.frame_count:
                    self.toggle_pause()
                if self.release_pause:
                    self.paused=True
                # using waitkey to control the speed of the video
                cv2.waitKey(30)
                #   

                #time.sleep(30 / 1000.0)
                # Check if the stop signal has been emitted
                #if self.stop_signal.isSet():
                #    break
    

    @Slot()
    def seek(self, frame):
        #pass

        if 0 <= frame < self.frame_count:
            self.seek_position = frame
            #self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            self.current_position = frame
            self.position_changed.emit(self.current_position)
    @Slot()
    def back3f(self):
        if self.current_position>=3:
            self.seek_position-=3
            self.set_release_pause()
    @Slot()
    def forward3f(self):
        if self.current_position<=self.frame_count-3:
            self.seek_position+=3
            self.set_release_pause()

    @Slot()
    def toggle_pause(self):
        self.paused = not self.paused
        self.release_pause = False
    @Slot()
    def set_pause(self):
        self.paused = True

    @Slot()
    def set_release_pause(self):
        self.paused = False
        self.release_pause = True


    @Slot()
    def toggle_is_holding(self):
        self.is_holding = not self.is_holding
        #self.release_pause = False
    @Slot()
    def stop(self):
        self.running = False
        self.current_position = 0
        print("stopping...")
        self.cap.release()
        self.wait()
        print("stopped...")
class VideoDisplay(QLabel):
    def __init__(self): 
        super().__init__()

    def update_image(self, q_image):
        pixmap = QPixmap.fromImage(q_image)
        self.setPixmap(pixmap)

class Widget(QtWidgets.QWidget):
    def __init__(self, source, prediction_saving_dir):
        super(Widget, self).__init__()

        self.video_display = VideoDisplay()
        self.video_thread = VideoThread(source, prediction_saving_dir )  # Set your video source path here
        self.video_thread.image_data.connect(self.update_display)
        #self.video_thread.start()
        self.slider = QSlider(Qt.Horizontal)
        #self.slider.setMaximum(self.video_thread.frame_count - 1)
        self.slider.valueChanged.connect(self.seek_video)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.video_display)
        self._rectangle_item = None
        self.bounding_box =  None

    def update_display(self, q_image):
        self.video_display.update_image(q_image)
    
    def seek_video(self, position):
        #print("seek",position)
        self.video_thread.seek(int(position))

    def setBoundingBox(self, xmin, ymin, xmax, ymax, frame_num):
        self.frame_num = frame_num
        self.bounding_box = (xmin, ymin, xmax, ymax)
        
    def clearBoundingBox(self):
        self.bounding_box = None
        self.repaint()



    
           
import datetime

class MainWindow(QMainWindow):
    all_threads_done = Signal()
    def __init__(self):
        super().__init__()
        self.is_first_run = True
        #self.setWindowState(Qt.WindowFullScreen) 
        self.setWindowState(Qt.WindowMaximized)
        self._playlist = []  # FIXME 6.3: Replace by QMediaPlaylist?
        self._playlist_index = -1
        self._mime_types = []
        
        self._player = QMediaPlayer()
        self.setFocusPolicy(Qt.StrongFocus) # listen to the input with more focus
        self._player.errorOccurred.connect(self._player_error)
        self.url = None
        self.percent = 0
        self.total_frames = 0
        self.confidence = 0.3
        self.stride = 6
        #self.recheck_changed = False
        # Get the current date and time
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        self.prediction_saving_dir = f"results/{formatted_datetime}.txt"
        txtfile = open(self.prediction_saving_dir, 'w')
        txtfile.close()
        # Create an instance of the progress thread
        self.progress_thread = ProgressThread()

        # Connect the progress_updated signal to the update_progress_bar method
        self.progress_thread.progress_updated.connect(self.update_progress_bar)
        # add a check box for recheck or not on toolbar
        #self.recheck_changed = False
        self.recheck_state = False
        self.recheck_checkbox = QtWidgets.QCheckBox("Double check")
        self.recheck_checkbox.setChecked(False)
        self.recheck_checkbox.stateChanged.connect(self.recheck_changed)
        self.recheck_checkbox.setStyleSheet("background-color: lightblue;")
        self.recheck_checkbox.setEnabled(False)
        self.recheck_checkbox.setToolTip("Recheck the flag")
        

        
        # Create a layout for the central widget
        layout = QVBoxLayout()
        tool_bar = QToolBar()
        self.addToolBar(tool_bar)
       
        self.available_width = self.screen().availableGeometry().width()
        self.available_height = self.screen().availableGeometry().height()
        icon = QIcon.fromTheme("document-open")
        file_menu = self.menuBar().addMenu("&File")
        icon = QIcon.fromTheme("application-exit")
        exit_action = QAction(icon, "E&xit", self,
                              shortcut="Ctrl+Q", triggered=self.close)
        file_menu.addAction(exit_action)
        ####### OPEN ICON ########
        self.open_action = QAction(icon, "&Open...", self, shortcut=QKeySequence.Open, triggered=self.open)
        self.open_button = QPushButton("Open", self)
        self.open_button.setStyleSheet("background-color: pink;")
        self.open_button.clicked.connect(self.open)
        
        ##### menu ###########
        file_menu.addAction(self.open_action)
        tool_bar.addWidget(self.open_button)
        tool_bar.addWidget(self.recheck_checkbox)
        #### CONFIDENCE SPIN BOX ######
        # Create a spin box
        self.spin_box = QDoubleSpinBox()
        self.spin_box.setSingleStep(0.05)
        #initial_confidence_score = 0.20  # Change this to your desired initial value
        self.spin_box.setValue(self.confidence)
        self.spin_box.valueChanged.connect(self.on_spin_box_value_changed)
        # Create a label for description
        description_label = QLabel("conf:")
        tool_bar.addWidget(description_label)
        tool_bar.addWidget(self.spin_box)
        #### STARTING BUTTON ##########
        # Create a "Start" button
        self.start_button = QPushButton("Scan now", self)
        self.start_button.setStyleSheet("background-color: lightblue;")
        self.start_button.setEnabled(False)
        # Connect the button's clicked signal to a custom slot
        self.start_button.clicked.connect(self.on_start_button_clicked)
        
        tool_bar.addWidget(self.start_button)
        #### PROGRESS BAR #########
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(self.available_width/10)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setMinimum(0)

        tool_bar.addWidget(self.progress_bar)


        
        ##### play/pause icon #####
        play_menu = self.menuBar().addMenu("&Play")
        style = self.style()
        icon = QIcon.fromTheme("media-playback-start.png",
                               style.standardIcon(QStyle.SP_MediaPlay))
        self._play_action = tool_bar.addAction(icon, "Play")
        #self._play_action.triggered.connect(self.play_video)
        play_menu.addAction(self._play_action)

        icon = QIcon.fromTheme("media-skip-backward-symbolic.svg",
                               style.standardIcon(QStyle.SP_MediaSkipBackward))
        self._previous_action = tool_bar.addAction(icon, "Previous")
        self._previous_action.triggered.connect(self.previous_clicked)
        play_menu.addAction(self._previous_action)

        icon = QIcon.fromTheme("media-playback-pause.png",
                               style.standardIcon(QStyle.SP_MediaPause))
        self._pause_action = tool_bar.addAction(icon, "Pause")
        play_menu.addAction(self._pause_action)

        #icon = QIcon.fromTheme("media-skip-forward-symbolic.svg",
        #                       style.standardIcon(QStyle.SP_MediaSkipForward))
        #self._next_action = tool_bar.addAction(icon, "Next")
        #self._next_action.triggered.connect(self.next_clicked)
        #play_menu.addAction(self._next_action)

        #icon = QIcon.fromTheme("media-playback-stop.png",
        #                       style.standardIcon(QStyle.SP_MediaStop))
        #self._stop_action = tool_bar.addAction(icon, "Stop")
        #self._stop_action.triggered.connect(self._ensure_stopped)
        #play_menu.addAction(self._stop_action)

        self.slider_frame_group = SliderFrameGroup(self.prediction_saving_dir, self.total_frames)
        self.hline = self.slider_frame_group.hline
        #self.slider = self.slider_frame_group.slider
        #self.slider.setOrientation(Qt.Horizontal)
        #self.slider.valueChanged.connect(self.on_slider_value_changed)
        #self.slider.valueChanged.connect(self.seek_video)

        #QTimer.singleShot(1000, self.hline.update)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.hline.update)
        self.timer.start(25)  # Adjust the interval as needed
        
        
        #
         # Connect slider signals to control video playback
        

        about_menu = self.menuBar().addMenu("&About")
        about_qt_act = QAction("About &Qt", self, triggered=qApp.aboutQt)
        about_menu.addAction(about_qt_act)

        tool_bar.addWidget(self.slider_frame_group)
        
        
        ###### VIDEO WIDGET #########
        #splitter = QSplitter(self)
        

         # Create a QTextEdit widget for text
        
        central_widget = QWidget(self)
        self.main_layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        # Create a splitter widget
        self.splitter = QSplitter(central_widget)
        self.main_layout.addWidget(self.splitter)


        
        
        # create a group box for the read only text box and detected images
        self.group_box = QGroupBox()
        self.group_box.setFixedWidth(self.available_width/10)
        self.group_box.setFixedHeight(self.available_height)
        self.group_box.setStyleSheet("background-color: lightblue;")
        

        self.read_only_text_box = QTextEdit()
        #self.read_only_text_box.setPlaceholderText("This is a read-only text box")
        self.read_only_text_box.setReadOnly(True)  # Make it read-only
        # Create buttons to draw and clear bounding boxes
        self.draw_button = QPushButton("Draw Bounding Box")
        self.clear_button = QPushButton("Clear Bounding Box")
        #tool_bar.addWidget(self.draw_button)
        #tool_bar.addWidget(self.clear_button)
        #create a region beside the read only text box to show list of detected images
        # Create a layout
        self.detected_images_region = QScrollArea()
        
        self.detected_images_region.setWidgetResizable(True)
        self.detected_images_region.setFixedWidth(self.available_width/10)
        self.detected_images_region.setFixedHeight(self.available_height)
        self.detected_images_region.setStyleSheet("background-color: lightblue;")
        self.detected_images_region.setWidget(QWidget())
        self.detected_images_region.widget().setLayout(QVBoxLayout())
        self.detected_images_region.widget().layout().setAlignment(Qt.AlignTop)
        self.detected_images_region.widget().layout().setContentsMargins(0, 0, 0, 0)
        self.detected_images_region.widget().layout().setSpacing(0)

        
        
        # add the region to the main layout
        #self.main_layout.addWidget(self.detected_images_region)

        #self.draw_button.clicked.connect(self.read_the_marked_frames)
        self.clear_button.clicked.connect(self.clear_bounding_box)
        self.predict_thread = None
        self.frame_num_and_bounding_box_dict = dict()
        self.saving_folder = ""
    @Slot()
    def recheck_changed(self, state):
        if QtCore.Qt.Checked:
            if self.recheck_state == False:
                self.recheck_state = True
                print("recheck")
            else:
                self.recheck_state = False
                print("not recheck")
       
    def get_recheck_state(self):
        return self.recheck_state
    

    def setup_before_show_video(self):
        try:
            #self.slider_frame_group.layout.removeWidget(self.slider_frame_group.slider)
            self.slider.setParent(None)
            self.video_widget.setParent(None)
            self.splitter.removeWidget(self.video_widget)
            self.splitter.removeWidget(self.read_only_text_box)
            self.splitter.removeWidget(self.detected_images_region)
            print("remove")
            #self.setLayout(layout)
        except:
            pass    
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        self.prediction_saving_dir = f"results/{formatted_datetime}.txt"
        self.video_widget = Widget(self.url.toLocalFile(), self.prediction_saving_dir)
        

        #self.slider_frame_group.layout.removeWidget(self.slider)
        self.slider = self.video_widget.slider 
        self.slider_frame_group.layout.addWidget(self.slider, 0, 0, 1, 1)    
        self.video_display = self.video_widget.video_display
        self.video_thread = self.video_widget.video_thread
        self.video_thread.position_changed.connect(self.update_slider)
        self.open_action.triggered.connect(self.video_thread.stop)
        self._pause_action.triggered.connect(self.video_thread.toggle_pause)
        self._play_action.triggered.connect(self.video_thread.toggle_pause)
        self.slider.sliderPressed.connect(self.video_thread.set_pause)
        self.slider.sliderReleased.connect(self.video_thread.set_release_pause)
        self.slider.sliderPressed.connect(self.video_thread.toggle_is_holding)
        self.slider.sliderReleased.connect(self.video_thread.toggle_is_holding)
        self.open_button.clicked.connect(self.video_thread.stop)
        self.slider.setRange(0, self.total_frames)
        self.slider.setValue(0)

        #self.splitter.addWidget(self.group_box)
        self.splitter.addWidget(self.read_only_text_box)
        self.splitter.addWidget(self.detected_images_region)
        self.splitter.addWidget(self.video_widget)
        
        
        initial_sizes = [1, 10]
        fixed_size = 200
        #set sizes for 3 regions
        self.splitter.setSizes([fixed_size,2*fixed_size, self.splitter.size().width() - 3*fixed_size])
        
        #self.splitter.setSizes([fixed_size,fixsize, self.splitter.size().width() - 2*fixed_size])
        #self.slider.valueChanged.connect(self.video_thread.seek)
        #self._player.setVideoOutput(self.video_widget.video_display)
        #self.all_threads_done.connect(self.on_thread_finished())

        #self.setCentralWidget(self.video_widget)
    def plot_detected_images(self):
        
        
            image_list = os.listdir(self.saving_folder)
            image_list.sort(key=lambda x: int(x.split("_")[2].split(".")[0]), reverse=False)
            for image in image_list:
                if image.endswith(".jpg"):
                    image_path = os.path.join(self.saving_folder, image)
                    pixmap = QPixmap(image_path)
                    label = QLabel(self)
                    label.setPixmap(pixmap)
                    self.detected_images_region.widget().layout().addWidget(label)
            

    
                
    def read_the_marked_frames(self):
        
        #print("--------------------------------------------------------------------")
        with open(self.prediction_saving_dir, 'r') as file:
            for line in file:
                # Remove leading and trailing whitespace (e.g., newline characters)
                line = line.strip()
                #print(line)

                parts = line.split()
                if len(parts) == 5:
                    frame_num, xmin, ymin, xmax, ymax = map(float, parts)
                    frame_num = int(frame_num)
                    #print((xmin, ymin, xmax, ymax))
                    self.frame_num_and_bounding_box_dict[frame_num] = (xmin, ymin, xmax, ymax)
                    for i in range(1,3):
                        self.frame_num_and_bounding_box_dict[frame_num+i] = (xmin, ymin, xmax, ymax)
                        self.frame_num_and_bounding_box_dict[frame_num-i] = (xmin, ymin, xmax, ymax)
            self.marked_frames = list(self.frame_num_and_bounding_box_dict.keys())
            #text_to_append = "---"+ f"Found {len(self.marked_frames)} frames that can contain potential suspects" + "\n"
            #cursor = self.read_only_text_box.textCursor()
            #cursor.setPosition(0)  # Move cursor to the beginning
            #cursor.insertText(text_to_append)
            file.close()
    @Slot(int)
    def on_slider_value_changed(self, position):
        # This slot is called when the slider value changes
        # You can seek the video to the specified frame
        #self.video_thread.seek(position)
        pass
     
    @Slot()   
    def update_slider(self, position):
        self.slider.setValue(position)

    @Slot()
    def on_positionChanged(self, position):
        #print(self.frame_num_and_bounding_box_dict)
        # Calculate the frame number based on the position and frame rate (e.g., 30 frames per second)
        frame_rate = 30  # Adjust this value based on your video's frame rate
        current_frame = int(position / 1000 * frame_rate)  # Convert milliseconds to seconds
        if current_frame  in list(self.frame_num_and_bounding_box_dict.keys()):
            bbox = self.frame_num_and_bounding_box_dict[current_frame]
            self.draw_bounding_box(bbox, current_frame)
        else:
            print(current_frame)
            bbox = (0,0,0,0)
        #self.clear_bounding_box()
        
    @Slot()
    def on_spin_box_value_changed(self, value):
        self.confidence = value
    @Slot()
    def draw_bounding_box(self, bbox, frame_num):
        # Example: Set bounding box coordinates
        xmin, ymin, xmax, ymax = bbox
        self.video_widget.setBoundingBox(xmin, ymin, xmax, ymax, frame_num)
    @Slot()
    def clear_bounding_box(self):
        self.video_widget.clearBoundingBox()
   
    #def closeEvent(self, event):
    #    self._ensure_stopped()
    #    event.accept()
    @Slot()
    def update_progress_bar(self):
        global PERCENT_PATH
        with open(PERCENT_PATH, 'r') as file:
            #print(file.read())
            try:
                self.percent = float(file.read())        
                file.close()
                self.progress_bar.setValue(self.percent)
            except:
                pass

    def reset_4_new_prediction(self):
        self.video_thread.video_source = self.url.toLocalFile()
        self.video_thread.not_read = True
        self.video_thread.get_recheck_changed = self.get_recheck_state()
        self.video_thread.running = False
        self.video_thread.end_check = False
        #self.video_thread.read_only_text_box = self.read_only_text_box
        cap = cv2.VideoCapture(self.video_thread.video_source)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        # create a percent log file
        global PERCENT_PATH
        PERCENT_PATH = f"results/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_percent.txt"
        # write 0 to the percent log file
        with open(PERCENT_PATH, 'w') as file:
            file.truncate(0)
            file.write("0")
            file.close()
        self.slider.setRange(0,self.total_frames)
        self.slider.setValue(0)
        self.progress_bar.setValue(0)
        self.recheck_checkbox.setEnabled(True)
        #os.remove(self.prediction_saving_dir)
        base_url = os.path.basename(self.url.toLocalFile())
        base_url = base_url.split(".")[0]
        current_datetime = datetime.datetime.now()
        
        self.saving_folder = os.path.join("cropped_images", f"{base_url}_{current_datetime.strftime('%Y-%m-%d_%H-%M-%S')}")
        self.video_thread.saving_folder = self.saving_folder
        os.makedirs(self.saving_folder, exist_ok=True)
        txtfile = open(self.prediction_saving_dir, 'w')
        txtfile.close()
        self.hline.total_frames = self.total_frames
        self.hline.prediction_saving_dir = self.prediction_saving_dir

        try:
            # remove all images in detected images region
            for i in reversed(range(self.detected_images_region.widget().layout().count())):
                self.detected_images_region.widget().layout().itemAt(i).widget().setParent(None)
            #self.video_thread.terminate()
            #self.video_thread.wait()
            #self.predict_thread.terminate()
            #self.predict_thread.wait()
            
            print("wait-quit")
        except:
            print("threads finished")

    def on_start_button_clicked(self):

        print('recheck state',self.get_recheck_state())
        self.video_thread.get_recheck_changed = self.get_recheck_state()
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        print(formatted_datetime)
        print("...Scanning....\n")
        print("video url:",self.url.toLocalFile(), "\n")
        text_to_append = "---"+ f"Scanning video: {self.url.toLocalFile()}" + "\n"
        cursor = self.read_only_text_box.textCursor()
        cursor.setPosition(0)  # Move cursor to the beginning
        cursor.insertText(text_to_append)
        self.start_inference_time = timeit.default_timer()
        
        
        self.reset_4_new_prediction()
        # Start the progress thread
        self.progress_thread.start()
        #done_event = threading.Event()
        self.predict_thread = PredictThread(self.url.toLocalFile(), self.prediction_saving_dir, self.confidence)
        
        print(f'confidence threshold: {self.confidence} \n')
        self.predict_thread.update_signal.connect(self.done_signal)
        self.predict_thread.started.connect(self.on_thread_started)
        self.predict_thread.finished.connect(self.on_thread_finished)
        self.predict_thread.start()
        self.open_button.setEnabled(False)
        
        




        
        #self._player.play
        #self._player.pause
    #@Slot()
    def play_video(self):
        #print(self.url)
        if self.url is not None:
            print("playing")
            self.start_button.setEnabled(False)
            print(self.video_thread.video_source)
            self.video_thread.start()

    def count_lines(self, ):
        file_path = self.prediction_saving_dir
        with open(file_path, 'r') as file:
            lines = file.readlines()
        return len(lines)
    def done_signal(self):
        self.stop_inference_time = timeit.default_timer()
        self.video_thread.running = True
        self.video_thread.not_read = True
        self.read_the_marked_frames()
        self.play_video()
        self.open_button.setEnabled(True)
        self.progress_bar.setValue(0)
        num_lines = self.count_lines()
        #time.sleep(2)
        if self.video_thread.end_check:
            self.plot_detected_images()
        
        else:
            QTimer.singleShot(2000, self.plot_detected_images)
        #self.plot_detected_images()
        self.append_msg(f"inference time: {self.stop_inference_time-self.start_inference_time}")
        
        #self.video_widget.video_thread.running = True
        #print(self.frame_num_and_bounding_box_dict)
        

        

    def on_thread_started(self):
        self.start_button.setEnabled(False)
        print("Thread started.")

    def on_thread_finished(self):
        self.play_video()
        
        self.start_button.setEnabled(True)
        
        print("Thread finished.")
    

    #def update_progress_bar(self, percent):
    #    self.progress_bar.setValue(int(percent * 100))

    def closeEvent(self, event):
        # Ensure that the progress thread is stopped when the application is closed
        self.progress_thread.quit()
        self.progress_thread.wait()
        self._ensure_stopped()
        event.accept()      
            
    @Slot()
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            try:
                self.video_thread.toggle_pause()
            except:
                pass
        elif event.key() == Qt.Key_Left:
            try:
                self.video_thread.is_holding
                self.video_thread.back3f()
                self.video_thread.is_holding
            except:
                pass 
        elif event.key() == Qt.Key_Right:
            try:
                self.video_thread.is_holding
                self.video_thread.forward3f()
                self.video_thread.is_holding
            except:
                pass
        else: 
            # Pass the event to the parent class for other key events
            super().keyPressEvent(event)
   
    def append_msg(self, text_to_append):
        text_to_append =  f"---{text_to_append}\n" 
        cursor = self.read_only_text_box.textCursor()
        cursor.setPosition(0)  # Move cursor to the beginning
        cursor.insertText(text_to_append)
    @Slot()
    def pause_video(self):
        self._player.pause()
    
    @Slot()
    def toggle_video_playback(self):
        if self._player.state() == QMediaPlayer.PlayingState:
            self._player.pause()
        else:
            self._player.play()
    @Slot()
    def duration_changed(self, duration):
        self.slider.setRange(0, duration)
    @Slot()
    def set_position(self, position):
        self._player.setPosition(position)
    @Slot()
    def position_changed(self, position):
        #print(position)
        self.slider.setValue(position)
    @Slot()
    def open(self):
        self._ensure_stopped()
        file_dialog = QFileDialog(self)
        
        
        is_windows = sys.platform == 'win32'
        # set for all video formats
        list_of_supported_format = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv', '*.webm', '*.vob', '*.ogv', '*.ogg', '*.drc', '*.gif', '*.gifv', '*.mng', '*.mts', '*.m2ts', '*.ts', '*.mov', '*.qt', '*.yuv', '*.rm', '*.rmvb', '*.viv', '*.asf', '*.amv', '*.mp4', '*.m4p', '*.m4v', '*.mpg', '*.mp2', '*.mpeg', '*.mpe', '*.mpv', '*.m2v', '*.m4v', '*.svi', '*.3gp', '*.3g2', '*.mxf', '*.roq', '*.nsv', '*.flv', '*.f4v', '*.f4p', '*.f4a', '*.f4b']
        if not self._mime_types:
            self._mime_types = get_supported_mime_types()
            if (is_windows and AVI not in self._mime_types):
                self._mime_types.append(AVI)
            elif MP4 not in self._mime_types:
                self._mime_types.append(MP4)

        file_dialog.setMimeTypeFilters(list_of_supported_format)

        default_mimetype = MP4 if is_windows else AVI
        if default_mimetype in self._mime_types:
            file_dialog.selectMimeTypeFilter(default_mimetype)
        

        
        movies_location = QStandardPaths.writableLocation(QStandardPaths.MoviesLocation)
        file_dialog.setDirectory(movies_location)
        if file_dialog.exec() == QDialog.Accepted:
            url = file_dialog.selectedUrls()[0]
            self._playlist.append(url)
            self._playlist_index = len(self._playlist) - 1
            #self._player.setSource(url)
            self.url = url
            text_to_append = "---"+ f"Video {self.url.toLocalFile()} loaded successfully" + "\n"
            cursor = self.read_only_text_box.textCursor()
            cursor.setPosition(0)  # Move cursor to the beginning
            cursor.insertText(text_to_append)
            #self.reset_4_new_prediction()



            self.start_button.setEnabled(True)
            #self.video_thread.start()
            self.setup_before_show_video()
            self.reset_4_new_prediction()
            if self.is_first_run==True:
                self.is_first_run=False
            
            #self.video_widget.setvideo(url)
            #process_video(url)
            #self._player.play()

    


    @Slot()
    def _ensure_stopped(self):
        if self._player.playbackState() != QMediaPlayer.StoppedState:
            self._player.stop()

    @Slot()
    def previous_clicked(self):
        # Go to previous track if we are within the first 5 seconds of playback
        # Otherwise, seek to the beginning.
        if self._player.position() <= 5000 and self._playlist_index > 0:
            self._playlist_index -= 1
            self._playlist.previous()
            self._player.setSource(self._playlist[self._playlist_index])
        else:
            self._player.setPosition(0)

    @Slot()
    def next_clicked(self):
        if self._playlist_index < len(self._playlist) - 1:
            self._playlist_index += 1
            self._player.setSource(self._playlist[self._playlist_index])

    @Slot("QMediaPlayer::PlaybackState")
    def update_buttons(self, state):
        media_count = len(self._playlist)
        self._play_action.setEnabled(media_count > 0
            and state != QMediaPlayer.PlayingState)
        self._pause_action.setEnabled(state == QMediaPlayer.PlayingState)
        #self._stop_action.setEnabled(state != QMediaPlayer.StoppedState)
        self._previous_action.setEnabled(self._player.position() > 0)
        #self._next_action.setEnabled(media_count > 1)

    def show_status_message(self, message):
        self.statusBar().showMessage(message, 5000)

    @Slot("QMediaPlayer::Error", str)
    def _player_error(self, error, error_string):
        print(error_string, file=sys.stderr)
        self.show_status_message(error_string)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    available_geometry = main_win.screen().availableGeometry()
    main_win.resize(available_geometry.width() / 3,
                    available_geometry.height() / 2)
    main_win.show()
    sys.exit(app.exec())