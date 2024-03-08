import os
from typing import Any
import numpy as np
import time
import sys
import argparse
import warnings
from datetime import datetime
from HD_Detection import HD_Detection
import time


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

start_time = time.time()

#Detector = HD_Detection(args.video_path, args.model, args.model_resnet, args.conf, args.stride, args.run_on, args.timing_inspection, args.skip_similarity)
Detector = HD_Detection(video_path = args.video_path, \
                        model_path = args.model_path,     \
                        conf = args.conf,      \
                        stride = args.stride,      \
                        run_on = args.run_on,      \
                        timming_inspection = args.timing_inspection, \
                        similarity_comparision = args.skip_similarity,  \
                        label_dict = label_dict
                        )
# starting the detection
Detector()
# post processing
print(f"Total inference time: {round(time.time() - start_time, 2)} seconds for a video of {Detector.total_frames} frames")



