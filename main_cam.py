import Function_Library_cam as fl
import cv2
import time

EPOCH = 500000

if __name__ == "__main__":
    # Exercise Environment Setting
    env = fl.libCAMERA()

    """ Exercise 1: RGB Color Value Extracting """
    ############## YOU MUST EDIT ONLY HERE ##############
# =============================================================================
#     example = env.file_read("./Example Image.jpg")
#     R, G, B = env.extract_rgb(example, print_enable=True)
#     quit()
# =============================================================================
    #####################################################

    # Camera Initial Setting
    ch0, ch1 = env.initial_setting(capnum=1)
    count = 0
    linecam = cv2.VideoCapture(cv2.CAP_DSHOW + 1)
   # objcam = cv2.VideoCapture(cv2.CAP_DSHOW + 2)
    # Camera Reading..
    for i in range(EPOCH):
    #    _, obj = objcam.read()
        _, line = linecam.read()
        #cv2.imshow('obj', obj)
        #cv2.imshow('line', line)
        """ Exercise 2: Webcam Real-time Reading """
        ############## YOU MUST EDIT ONLY HERE ##############
        env.image_show( line)
        #####################################################

        """ Exercise 3: Object Detection (Traffic Light Circle) """
        #################### YOU MUST EDIT ONLY HERE ####################
        color = env.object_detection(line, sample=16, print_enable=True)
        #################################################################

        """ Exercise 4: Specific Edge Detection (Traffic Line) """
        #################### YOU MUST EDIT ONLY HERE ####################
        #direction = env.edge_detection(frame1, width=300, height=120,
         #                              gap=40, threshold=150, print_enable=True)
        #################################################################


        # Process Termination (If you input the 'q', camera scanning is ended.)
        if env.loop_break():    
            break
    linecam.release()
    cv2.destroyAllWindows()
# =============================================================================
# img_path = "C:/image sample.png"
# import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import load_model
# 
# img_color = cv.imread(img_path)
# img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
# img_checker = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 191, 15)
# plt.imshow(img_checker)
# ret, img_binary = cv.threshold(img_gray, 127, 255, 0)
# contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
# 
# rects = [cv.boundingRect(contour) for contour in contours]
# 
# for rect in rects:
#     if rect[2]*rect[3] < 1000:
#         continue
#     print(rect)
#     cv.rectangle(img_color, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
#     if rect[2] < 50:
#         margin = 100
#     else:
#         margin = 30
#         
#     roi = img_checker[rect[1]-margin:rect[1]+rect[3]+margin, rect[0]-margin:rect[0]+rect[2]+margin]
#     try:
#        roi = cv.resize(roi, (28, 28),  cv.INTER_AREA)
#     except Exception as e:
#        print(str(e))
#     roi = roi/255.0
#     img_input = roi.reshape(1, 28, 28, 1)
#     prediction = model.predict(img_input)
#     num = np.argmax(prediction)
#     print(num)
#     location = (rect[0]+rect[2], rect[1] + 20)
#     cv.putText(img_color, str(num), location, cv.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 0), 2)
#  
# cv.imshow("Resulting Image", img_color)
# cv.imwrite("R10.jpg", img_color)
# cv.waitKey()
# =============================================================================
  









# =============================================================================
#     cv.rectangle(img_color, (x,y), (x+w, y+h),(0,0,0), 2)
#     area = cv.contourArea(cnt)
# 
#     print(area)
# 
# cv.imshow("result", img_color)
# cv.waitKey(0)
# =============================================================================


