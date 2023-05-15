import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from moviepy.video.io.VideoFileClip import VideoFileClip 

# %% Get the first digit in string

def first_digit(s):
    return re.search(r"\d", s).start()

# %% Get the video segment ID in string

def get_video_ID(s):
    return s[first_digit(s):-4]

# %% Get the video length

def get_video_length(file_path):   
    clip = VideoFileClip(file_path))
    video_length.append(int(clip.duration))
    
    hrs, mins, secs = dur//60//60, dur//60%60, dur%60 
    hrs = "0"+str(hrs) if(hrs<10) else str(hrs) 
    mins = "0"+str(mins) if(mins<10) else str(mins) 
    secs = "0"+str(secs) if(secs<10) else str(secs) 
    return [hrs, mins, secs]



# %% Main function

if __name__ == "__main__":
    video_ID, video_length, y_test = [], [], []
        
    data_dir = r'D:/Testing/Segments'

    # NOTE: Class 0 = NonPorn, Class 1 = Porn
    # We can use the one-hot vector technique to represent the video content
    # For example: [1. 0.] means NonPorn, [0. 1.] means Porn
    
    for file in os.listdir(data_dir):
        if file.endswith('.avi'):
            s = file[1:first_digit(file)]
                
            if (s == 'NonPorn'):
                y_test.append(0)
            else:
                y_test.append(1)