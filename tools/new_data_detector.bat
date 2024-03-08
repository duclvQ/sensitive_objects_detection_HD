@echo off
setlocal

set common_arg=%1

python hashing_image.py 
python new_data_detector.py -p %common_arg% 


endlocal