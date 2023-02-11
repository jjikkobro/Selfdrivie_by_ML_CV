#Data Collectiong For General Driving by using bluethooth
import serial
import os
import sys
import time
import numpy as np
import cv2
import csv


def normalize(values, actual_bounds, desired_bounds):
    return round(desired_bounds[0] + (values - actual_bounds[0]) * (desired_bounds[1] - desired_bounds[0]) / (actual_bounds[1] - actual_bounds[0]),2)

##############Variable#############
t_now = time.time()
t_prev = time.time()
cnt_frame = 0
cnt_frame_total = 0
g_rl = 0
dirname = "data.%f" %(time.time())
count = 0
###################################

os.mkdir(dirname)
os.mkdir(os.path.join(dirname, "cam1"))

f_csv = open( os.path.join(dirname, "0_road_labels.csv"), 'w', newline='\n' )
wr = csv.writer(f_csv)
wr.writerow(['cam1', 'throttle', 'steering'])
ch1 = cv2.VideoCapture(cv2.CAP_DSHOW + 1)
        

ser = serial.Serial("COM3", 9600)
while True:
    tdefinition = -1
    _, frame1 = ch1.read()
    value = ser.readline().decode()
    throttle = int(value[value.find('e:')+3:value.find('St')].replace('\t',''))
    steering = int(value[value.find('g:')+3:value.find('Ul')].replace('\t',''))
    if throttle > 50:
        road_file = "%f.png" %(time.time())
        if steering <= 30:
            steering = normalize(steering, (30, 0),(-0.1,-1.0))
        elif steering > 30 and steering < 65:
            steering = 0.0
        elif steering >= 65:
            steering = normalize(steering, (65,99), (0.1, 1.0))
        cv2.imwrite(os.path.join(os.path.join(dirname, "cam1"),road_file), frame1)
        print('save')
        wr.writerow([os.path.join('cam1', road_file), tdefinition, steering])
        f_csv.flush()
        cnt_frame_total += 1
        cnt_frame += 1
        t_now = time.time()
        if t_now - t_prev >= 1.0:
            t_prev = t_now
            print("frame count : %d" %cnt_frame, \
                      "total count : %d" %cnt_frame_total)
            cnt_frame = 0

 

        
        