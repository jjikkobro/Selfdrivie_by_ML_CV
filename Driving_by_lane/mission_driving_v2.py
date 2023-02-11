import cv2
import numpy as np
import serial
import time
from Function_mission_driving import *


#####################################################
ser = serial.Serial("COM3", 9600)
lineport = 1
objport = 2
######################################################




linecam = cv2.VideoCapture(cv2.CAP_DSHOW + lineport)
start = input('Give 1 : ')

# 'til arriving at the crosswalk
while True:
    _, line = linecam.read()
    preprocess = img_predict(line)
    prediction = model.predict(preprocess)
    index = np.argmax(prediction)
    confidence_score = prediction[0][index]
    if index == 2 and int(str(np.round(confidence_score * 100))[:-2]) > 80:
        ser.write('5'.encode('utf-8') )
        height, width = line.shape[:2]
        vertices = np.array([[(0,height),(0,150), (width, 150), (width,height)]], dtype=np.int32)
        roi_img = region_of_interest(line, vertices)
        roi_img = roi_img[150:450,:]
        mark = np.copy(roi_img)
        marked = mark_img(roi_img, mark)
        steer = center_of_frame(marked)
        ser.write(str(steer).encode('utf-8'))
        print(int(str(np.round(confidence_score * 100))[:-2]))
        print('crosswalk')
        break
        
    height, width = line.shape[:2]
    vertices = np.array([[(0,height),(0,150), (width, 150), (width,height)]], dtype=np.int32)
    roi_img = region_of_interest(line, vertices)
    roi_img = roi_img[150:450,:]
    mark = np.copy(roi_img)
    marked = mark_img(roi_img, mark)
    try:
        steer = center_of_frame(marked)
    except:
        continue
    cv2.imshow('line',marked)
    cv2.imshow('line1',line)
    ser.write(str(steer).encode('utf-8'))
    print(steer)
    time.sleep(0.1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Waiting for greenlight
objcam = cv2.VideoCapture(cv2.CAP_DSHOW + objport)
time.sleep(0.1)
while True:
    _,obj = objcam.read()
    cv2.imshow('obj', obj)
    preprocess = color_predict(obj)
    color = predict(preprocess)
    if index == 0 and int(str(np.round(confidence_score * 100))[:-2]) > 99:
        print('go')
        ser.write('6'.encode('utf-8'))
        time.sleep(0.5)
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Passing throught out the crosswalk
while True:
    _, line = linecam.read()
    height, width = line.shape[:2]
    vertices = np.array([[(0,height),(0,150), (width, 150), (width,height)]], dtype=np.int32)
    roi_img = region_of_interest(line, vertices)
    roi_img = roi_img[150:450,:]
    mark = np.copy(roi_img)
    marked = mark_img(roi_img, mark)
    try:
        steer = center_of_frame(marked)
    except:
        continue
    cv2.imshow('line',marked)
    cv2.imshow('line1',line)
    ser.write(str(steer).encode('utf-8'))
    print(steer)
    time.sleep(0.1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
linecam.release()
objcam.release()
cv2.destroyAllWindows()
