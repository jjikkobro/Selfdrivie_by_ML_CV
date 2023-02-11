from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
import serial
import os
import tensorflow as tf
import time


def img_preprocess(img):
    img = cv2.resize(img, (160,120))
    img = img.astype('int8')/255
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = np.clip(img, 0, 1)
    return img

################ Preset ################ 
ser = serial.Serial("COM3", 9600)
model1 = load_model('model.h5')
lineport = 1
objport = 2
#########################################


#%%
linecam = cv2.VideoCapture(cv2.CAP_DSHOW + lineport)

start = input('Give 1 : ')

while True:
    _, line = linecam.read()
    time.sleep(0.1)
    cv2.imshow('frame',line)
    lineimg = img_preprocess(line)
    image_tensor = tf.convert_to_tensor(lineimg, dtype=tf.float32)
    image_tensor = tf.expand_dims(image_tensor, 0)
    y_predict = model1.predict( image_tensor ) 
    y_predict = np.argmax(y_predict, axis =1)[0]
    print(y_predict)
    ser.write(str(y_predict).encode('utf-8'))
    time.sleep(0.1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

linecam.release()
cv2.destroyAllWindows()
