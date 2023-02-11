import cv2
import numpy as np
import serial
import time
from tensorflow.keras.models import load_model
from Function_general_driving import *

######################################################
ser = serial.Serial("COM3", 9600)
model = load_model('Models/cross_walk_model.h5')
lineport = 1
objport = 2
#######################################################


linecam = cv2.VideoCapture(cv2.CAP_DSHOW + lineport)

start = input('Give 1 : ')

while True:
    _, line = linecam.read()
    preprocess = img_predict(line)
    prediction = model.predict(preprocess)
    index = np.argmax(prediction)
    confidence_score = prediction[0][index]
    if index == 2 and int(str(np.round(confidence_score * 100))[:-2]) > 90:
        ser.write('5'.encode('utf-8'))
        print('crosswalk')
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
cv2.destroyAllWindows()


