import cv2
import numpy as np
import os
from multiprocessing import Process, Manager
from datetime import date
from ultralytics import YOLO
import supervision as sv
from screeninfo import get_monitors
import torch
import sys
from numpy.random import choice

image_list = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ''
direc = ''
if sys.argv[1]=='1':
    model = YOLO('/home/ubuntu/Downloads/yolov8n.pt')  # Person Detection
    direc = '/home/ubuntu/seleced_images/person'   # Person Images

if sys.argv[1]=='2':
    direc = '/home/ubuntu/seleced_images/Faces'     # Face Images
    model = YOLO("/home/ubuntu/Akshay/Project/YOLO_Face_detection.pt") # Face Detection

cap = cv2.VideoCapture('/home/ubuntu/Akshay/video.AVI')
frame_count = 0
total_count = 0

while True:
    is_frame_read, frame = cap.read()

    if not is_frame_read:
        break

    ''' if total_count == 30*60*10: # FPS * SEC per Minute *  Minutes
        break   
    '''
    frame_count += 1

    total_count += 1
    
    if frame_count == 10:
        frame_count = 0

    if frame_count == 0:
        result = model(source = frame, conf = 0.7, classes = 0 )[0]
        k = 0#choice([0,20,40,60,80])
        for i , bb in enumerate(result.boxes.xyxy.cpu().numpy()):
            if sys.argv[1]=='1':
                roi = frame[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2])]
            else:
                x1,y1 = max(int(bb[1])-k,0), min(int(bb[3])+k,frame.shape[0])
                x2,y2 = max(int(bb[0])-k,0), min(int(bb[2])+k,frame.shape[1])
                roi = frame[x1:y1, x2:y2]
            
            image_list.append(roi)

save_path = '/home/ubuntu/Akshay/faces'
for i, image in enumerate(image_list):
    img_path = f"img_{i}.jpg"
    path = os.path.join(save_path,img_path)
    cv2.imwrite(path,image)
