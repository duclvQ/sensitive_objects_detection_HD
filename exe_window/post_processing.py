import time 
import os
import numpy as np 




class Post_Processing:
    """
    Post processing class for HD_Detection
    
    Attributes:
    -----------
    log_path: str
        path to log file
    
    Methods:
    --------
    __init__(self, log_path: str)
        Constructor
    Converter(self, log_path: str)
        Convert the log file to a numpy file
    Tracker(self, numpy: file)
        Track the detected objects
    """
    
    def __init__(self, log_path: str):
        """
        Constructor
        
        Parameters:
        -----------
        log_path: str
            path to log file
        """
        self.log_path = log_path
        