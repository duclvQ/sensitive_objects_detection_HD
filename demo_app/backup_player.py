# Copyright (C) 2022 The Qt Company Ltd.
# SPDX-License-Identifier: LicenseRef-Qt-Commercial OR BSD-3-Clause

"""PySide6 Multimedia player example"""
import subprocess
try: 
    import PySide6
except:
    subprocess.check_call(["pip", "install", "PySide6"])

import sys
from PySide6.QtCore import QStandardPaths, Qt, Slot, QTimer
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import (QApplication, QDialog, QFileDialog, 
    QMainWindow, QSlider, QStyle, QToolBar, QFrame)
from PySide6.QtMultimedia import (QAudioOutput, QMediaFormat,
                                  QMediaPlayer, QMediaPlayer)
from PySide6.QtMultimediaWidgets import QVideoWidget, QGraphicsVideoItem
from PySide6.QtWidgets import QApplication,QGraphicsScene,  QMainWindow,QDoubleSpinBox, QLineEdit, QProgressBar,QSpinBox, QMessageBox, QSplitter, QTextEdit, QGridLayout,QLabel, QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QGraphicsRectItem, QGraphicsLineItem
from PySide6.QtGui import QColor, QPalette, QPixmap, QImage, QKeyEvent, QPainter, QPen, QImage, QPainter, QPen, QColor, QWindow
from PySide6.QtCore import QMetaObject,  Qt, QThread, Signal,  QObject
from PySide6 import QtWidgets, QtMultimediaWidgets, QtCore, QtGui
#############import predict module########
#from test_predictor import process_video, lock, percent_var, shared_var_changed_event
import threading
import time
import logging
import cv2
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

def predict(model, saving_dir, frame_list, frame_num_list, conf):
    results = model.predict(source = frame_list, verbose=False,  stream=False, save=False, show=False, device=0, half=True, conf = conf)
    #print('results:',results)       
    final_results = list()
    for idx, r in enumerate(results):
        if r.boxes.cls.size()[0]>0:
            
            tensor_data = r.boxes.xyxy
            # Calculate min and max values for each dimension
            min_x = tensor_data[:, 0].min().item()
            min_y = tensor_data[:, 1].min().item()
            max_x = tensor_data[:, 2].max().item()
            max_y = tensor_data[:, 3].max().item()
            # Create and write to a plain text file
            with open(saving_dir, 'a') as file:
                file.write(f"{frame_num_list[idx]} {min_x} {min_y} {max_x} {max_y}\n")
                file.close()
            
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
                frame_queue.put([frame, frame_num])
            #print("current id:", thread_id)
            if frame_queue.qsize()>256: # Wait for 10ms to prevent running out of memory.
                #print(f"sleep in thread {frame_queue.qsize()}")
                #print(frame_num)
                time.sleep(0.050)
            if cap.get(cv2.CAP_PROP_POS_FRAMES) >= stop_frame:
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
        #print('qsize:', frame_queue.qsize())
        
        frame, frame_num = frame_queue.get() 
        
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
        frame_num_list.append(frame_num)
        if len(frame_list)==16:
            results = predict(model,saving_dir, frame_list, frame_num_list, conf)
            #print(results)
            frame_list = []
            frame_num_list = []
        
# Event for signaling changes


#if __name__ == "__main__":
import timeit
def process_video(video_path="../yolov8n/22.mp4", saving_dir="", conf = 0.35):
    start_time = timeit.default_timer()
    
    model = YOLO("../yolov8n/runs/detect/train12/weights/best.pt", task = "detect")

    frame_queue = queue.Queue()

    thread1 = threading.Thread(target=capture_frames, args=(video_path, frame_queue, 0))
    thread2 = threading.Thread(target=capture_frames, args=(video_path, frame_queue, 1))
    thread3 = threading.Thread(target=capture_frames, args=(video_path, frame_queue, 2))
    #thread4 = threading.Thread(target=capture_frames, args=(video_path, frame_queue, 3))

    predict_thread = threading.Thread(target=process_frames, args=(model,saving_dir, frame_queue,conf))

    thread1.daemon = True
    thread1.start()
    thread2.start()
    thread3.start()
    #thread4.start()
    predict_thread.start()
    

    thread1.join()
    thread2.join()
    thread3.join()
    #thread4.join()
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
            time.sleep(1)
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
                painter.drawLine(red_start-0.01, self.height() , red_start+0.01, self.height() )


class SliderFrameGroup(QWidget):
    def __init__(self, txt_url, total_frames):
        super().__init__()

        # Create a layout for the group
        self.layout = QGridLayout()  
        # Create sliders and frames
        self.slider = QSlider(Qt.Horizontal)
        self.hline = CustomHLine(txt_url, total_frames)
        # Add the slider and CustomHLine to the layout
        self.layout.addWidget(self.slider, 0, 0, 1, 1)
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
        super().paintEvent(event)
        # Create a QPainter for custom painting
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Set the rectangle color and style
        painter.setPen(QColor(255, 0, 0))  # Red color
        #painter.setBrush(QBrush(QColor(255, 0, 0, 50)))  # Semi-transparent red fill

        # Define the rectangle's position and size
        rect_x = 50
        rect_y = 50
        rect_width = 200
        rect_height = 100

        # Draw the rectangle
        painter.drawRect(rect_x, rect_y, rect_width, rect_height)
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
class VideoWidget(QVideoWidget):
    def getCurrentFrame(self):
        # Replace this with your video frame retrieval logic.
        # Return a QImage representing the current frame.
        # Example: Read frames from a video file using OpenCV
        return None
import datetime
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowState(Qt.WindowMaximized)
        self._playlist = []  # FIXME 6.3: Replace by QMediaPlaylist?
        self._playlist_index = -1
        self._player = QMediaPlayer()
        self.setFocusPolicy(Qt.StrongFocus) # listen to the input with more focus
        self._player.errorOccurred.connect(self._player_error)
        self.url = None
        self.percent = 0
        self.total_frames = 0
        self.confidence = 0.35
        # Get the current date and time
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        self.prediction_saving_dir = f"results_backup/{formatted_datetime}.txt"
        txtfile = open(self.prediction_saving_dir, 'w')
        txtfile.close()
        # Create an instance of the progress thread
        self.progress_thread = ProgressThread()

        # Connect the progress_updated signal to the update_progress_bar method
        self.progress_thread.progress_updated.connect(self.update_progress_bar)

        
        # Create a layout for the central widget
        layout = QVBoxLayout()
        tool_bar = QToolBar()
        self.addToolBar(tool_bar)

        self.available_width = self.screen().availableGeometry().width()
        icon = QIcon.fromTheme("document-open")
        file_menu = self.menuBar().addMenu("&File")
        icon = QIcon.fromTheme("application-exit")
        exit_action = QAction(icon, "E&xit", self,
                              shortcut="Ctrl+Q", triggered=self.close)
        file_menu.addAction(exit_action)
        ####### OPEN ICON ########
        open_action = QAction(icon, "&Open...", self, shortcut=QKeySequence.Open, triggered=self.open)
        
        ##### menu ###########
        file_menu.addAction(open_action)
        tool_bar.addAction(open_action)
        #### CONFIDENCE SPIN BOX ######
        # Create a spin box
        self.spin_box = QDoubleSpinBox()
        self.spin_box.setSingleStep(0.05)
        initial_confidence_score = 0.20  # Change this to your desired initial value
        self.spin_box.setValue(initial_confidence_score)
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
        self._play_action.triggered.connect(self._player.play)
        self._play_action.triggered.connect(self.play_video)
        play_menu.addAction(self._play_action)

        icon = QIcon.fromTheme("media-skip-backward-symbolic.svg",
                               style.standardIcon(QStyle.SP_MediaSkipBackward))
        self._previous_action = tool_bar.addAction(icon, "Previous")
        self._previous_action.triggered.connect(self.previous_clicked)
        play_menu.addAction(self._previous_action)

        icon = QIcon.fromTheme("media-playback-pause.png",
                               style.standardIcon(QStyle.SP_MediaPause))
        self._pause_action = tool_bar.addAction(icon, "Pause")
        self._pause_action.triggered.connect(self._player.pause)
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

        slider_frame_group = SliderFrameGroup(self.prediction_saving_dir, self.total_frames)
        self.hline = slider_frame_group.hline
        self.slider = slider_frame_group.slider
        self.slider.setOrientation(Qt.Horizontal)
        #QTimer.singleShot(1000, self.hline.update)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.hline.update)
        self.timer.start(30)  # Adjust the interval as needed
        
        self.slider.valueChanged.connect(self.set_position)
        self.slider.sliderMoved.connect(self.set_position)
         # Connect slider signals to control video playback
        self.slider.sliderPressed.connect(self.pause_video)
        self.slider.sliderReleased.connect(self.pause_video)

        about_menu = self.menuBar().addMenu("&About")
        about_qt_act = QAction("About &Qt", self, triggered=qApp.aboutQt)
        about_menu.addAction(about_qt_act)

        tool_bar.addWidget(slider_frame_group)
        
        ###### VIDEO WIDGET #########
        splitter = QSplitter(self)
        

         # Create a QTextEdit widget for text
        self.read_only_text_box = QTextEdit(self)
        #self.read_only_text_box.textCursor().insertText(" " * 5)
        self.read_only_text_box.setPlaceholderText("This is a read-only text box")
        self.read_only_text_box.setReadOnly(True)  # Make it read-only



        # Add the widgets to the splitter
        splitter.addWidget(self.read_only_text_box)
        splitter.setStretchFactor(1, 15)
        self.video_widget = CustomVideoWidget()
        splitter.addWidget(self.video_widget)
        self.setCentralWidget(splitter)
        

        # Create buttons to draw and clear bounding boxes
        self.draw_button = QPushButton("Draw Bounding Box")
        self.clear_button = QPushButton("Clear Bounding Box")
        tool_bar.addWidget(self.draw_button)
        #tool_bar.addWidget(self.clear_button)

        self.draw_button.clicked.connect(self.draw_bounding_box)
        self.clear_button.clicked.connect(self.clear_bounding_box)

        ## Set up a timer to periodically update the frame and overlay
        #self.timer = QTimer(self)
        #self.timer.timeout.connect(self.update_video_frame)
        #self.timer.start(30)  # Adjust the interval as needed


        ###### connect to button ##############
        self._player.playbackStateChanged.connect(self.update_buttons)
        self._player.setVideoOutput(self.video_widget)
        self._player.positionChanged.connect(self.position_changed)
        self._player.durationChanged.connect(self.duration_changed)
        #self._player.stateChanged.connect(self.mediastate_changed)
        self.update_buttons(self._player.playbackState())
        self._mime_types = []
        self.process_thread_done = threading.Event()


        

    @Slot()
    def on_spin_box_value_changed(self, value):
        self.confidence = value
    @Slot()
    def draw_bounding_box(self):
        # Example: Set bounding box coordinates
        xmin, ymin, xmax, ymax = 0, 0, 3000, 250
        self.video_widget.setBoundingBox(xmin, ymin, xmax, ymax)
    @Slot()
    def clear_bounding_box(self):
        self.video_widget.clearBoundingBox()
    @Slot()
    def update_video_frame(self):
        # Replace this with your video frame retrieval logic.
        # Retrieve the current frame and set it in the video widget.
        # Example: Read frames from a video file using OpenCV
        frame = None  # Replace None with your frame retrieval logic
        if frame:
            image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
            self.video_widget.videoSurface().present(image)
       
    #def closeEvent(self, event):
    #    self._ensure_stopped()
    #    event.accept()
    def update_progress_bar(self):
        self.progress_bar.setValue(int(self.percent * 100))
    def on_start_button_clicked(self):
        print("...Scanning....\n")
        print("video url:",self.url.toLocalFile(), "\n")
        cap = cv2.VideoCapture(self.url.toLocalFile())

        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.hline.total_frames = self.total_frames
        # Start the progress thread
        self.progress_thread.start()
        #done_event = threading.Event()
        self.predict_thread = PredictThread(self.url.toLocalFile(), self.prediction_saving_dir, self.confidence)
        print(f'confidence threshold: {self.confidence} \n')
        self.predict_thread.update_signal.connect(self.done_sigal)
        self.predict_thread.started.connect(self.on_thread_started)
        self.predict_thread.finished.connect(self.on_thread_finished)
        self.predict_thread.start()


        print("-----processing-----")
        #self._player.play
        #self._player.pause
    def done_sigal(self):
        self.play_video()
        

    def on_thread_started(self):
        self.start_button.setEnabled(False)
        print("Thread started.")

    def on_thread_finished(self):
        self.start_button.setEnabled(True)
        print("Thread finished.")
    

    def update_progress_bar(self, percent):
        self.progress_bar.setValue(int(percent * 100))

    def closeEvent(self, event):
        # Ensure that the progress thread is stopped when the application is closed
        self.progress_thread.quit()
        self.progress_thread.wait()
        self._ensure_stopped()
        event.accept()

    
    @Slot()
    def play_video(self):
        if self.url is not None:
            self.start_button.setEnabled(False)
            self._player.setSource(self.url)
            self._player.play()    
    @Slot()
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            # Pause or play the video when the Space key is pressed
            if self._player.playbackState() == QMediaPlayer.PlayingState:
                self._player.pause()
            else:
                self._player.play()
        elif event.key() == Qt.Key_Left:
            position = self._player.position()
            self._player.setPosition(position - 100)  
        elif event.key() == Qt.Key_Right:
            position = self._player.position()
            self._player.setPosition(position + 100)  
        else: 
            # Pass the event to the parent class for other key events
            super().keyPressEvent(event)
   

    @Slot()
    def pause_video(self):
        self._player.pause()
    @Slot()
    def play_video(self):
        self._player.play()
    @Slot()
    def toggle_video_playback(self):
        if self._player.state() == QMediaPlayer.PlayingState:
            self._player.pause()
        else:
            self._player.play()
    @Slot()
    def duration_changed(self, duration):
        print("duration",duration)
        self.slider.setRange(0, duration)
    @Slot()
    def set_position(self, position):
        self._player.setPosition(position)
    @Slot()
    def position_changed(self, position):
        print(position)
        self.slider.setValue(position)
    @Slot()
    def open(self):
        self._ensure_stopped()
        file_dialog = QFileDialog(self)

        is_windows = sys.platform == 'win32'
        if not self._mime_types:
            self._mime_types = get_supported_mime_types()
            if (is_windows and AVI not in self._mime_types):
                self._mime_types.append(AVI)
            elif MP4 not in self._mime_types:
                self._mime_types.append(MP4)

        file_dialog.setMimeTypeFilters(self._mime_types)

        default_mimetype = MP4  if is_windows else AVI
        if default_mimetype in self._mime_types:
            file_dialog.selectMimeTypeFilter(default_mimetype)

        movies_location = QStandardPaths.writableLocation(QStandardPaths.MoviesLocation)
        file_dialog.setDirectory(movies_location)
        if file_dialog.exec() == QDialog.Accepted:
            url = file_dialog.selectedUrls()[0]
            self._playlist.append(url)
            self._playlist_index = len(self._playlist) - 1
            self._player.setSource(url)
            self.url = url
            self.start_button.setEnabled(True)
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