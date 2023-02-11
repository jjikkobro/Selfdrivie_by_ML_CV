from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
import serial
import os
import tensorflow as tf
import time


def test_an_image(img, model):
    desired_dim=(32,32)
    img_resized = cv2.resize(img, desired_dim, interpolation=cv2.INTER_LINEAR)
    img_ = np.expand_dims(np.array(img_resized), axis=0)

    predicted_state = model.predict(img_)

    return predicted_state

def img_preprocess(img):
    img = cv2.resize(img, (160,120))
    img = img.astype('int8')/255
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = np.clip(img, 0, 1)
    return img

################ Preset ################ 
ser = serial.Serial("COM3", 9600)
model1 = load_model('mission_model.h5')
model=load_model('traffic_model.h5')
lineport = 1
objport = 2
########################################

linecam = cv2.VideoCapture(cv2.CAP_DSHOW + lineport)
objcam = cv2.VideoCapture(cv2.CAP_DSHOW + objport)

start = input('Give 1 : ')

while True:
    _, line = linecam.read()
    time.sleep(0.1)
    cv2.imshow('line', line)
    lineimg = img_preprocess(line)
    image_tensor = tf.convert_to_tensor(lineimg, dtype=tf.float32)
    image_tensor = tf.expand_dims(image_tensor, 0)
    y_predict = model1.predict( image_tensor ) 
    y_predict = np.argmax(y_predict, axis =1)[0]
    if y_predict == 10:
        y_predict = 'a'
        time.sleep(2)
        ser.write(y_predict.encode('utf-8'))
        _, obj = objcam.read()
        states = ['red', 'yellow', 'green', 'off']
        if True:
            predicted_state = test_an_image( obj, model )
            try:
                idx = np.where(predicted_state[0] == 1.0)[0][0]
                color = states[idx]
                print( color )
            except:
                time.sleep(0.5)
            if color == 'green':
                ser.write('b'.encode('utf-8'))
                break
    else:
        print(y_predict)
        ser.write(str(y_predict).encode('utf-8'))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
linecam.release()
objcam.release()
cv2.destroyAllWindows()
    