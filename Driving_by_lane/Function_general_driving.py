import cv2
import numpy as np
import time


def img_predict(img):
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
    img = (img / 127.5) - 1
    return img


def mark_img(img, mark, blue_threshold=200, green_threshold=200, red_threshold=200):

    bgr_threshold = [blue_threshold, green_threshold, red_threshold]

    thresholds = (img[:,:,0] < bgr_threshold[0]) \
                | (img[:,:,1] < bgr_threshold[1]) \
                | (img[:,:,2] < bgr_threshold[2])
    mark[thresholds] = [0,0,0]
    return mark

def region_of_interest(img, vertices, color3=(255,255,255), color1=255):

    mask = np.zeros_like(img)
    
    if len(img.shape) > 2: 
        color = color3
    else: 
        color = color1
        

    cv2.fillPoly(mask, vertices, color)

    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image


def get_fitline(img, f_lines): 
    lines = np.squeeze(f_lines)
    lines = lines.reshape(lines.shape[0]*2,2)
    rows,cols = img.shape[:2]
    output = cv2.fitLine(lines,cv2.DIST_L2,0, 0.01, 0.01)
    vx, vy, x, y = output[0], output[1], output[2], output[3]
    x1, y1 = int(((img.shape[0]-1)-y)/vy*vx + x) , img.shape[0]-1
    x2, y2 = int(((img.shape[0]/2+100)-y)/vy*vx + x) , int(img.shape[0]/2+100)
    
    result = [x1,y1,x2,y2]
    return result

def draw_fit_line(img, lines, color=[255, 0, 0], thickness=10):
        cv2.line(img, (lines[0], lines[1]), (lines[2], lines[3]), color, thickness)

def make_coordinates(img, line_parameters):
    slope, intercept = line_parameters
    y1 = img.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]

def weighted_img(img, initial_img, α=1, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)

def center_of_frame(img):
    global edges, lines, left_line, right_line, left_fit, right_fit
    edges = cv2.Canny(img, 85,85, apertureSize = 3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 10, minLineLength=5, maxLineGap=10)
    
    if lines is not None:
        left_fit = []
        right_fit = []

        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]

            if slope < -0.8:
                if x1 > (img.shape[1] / 2)+100:
                    right_fit.append((slope, intercept))
                    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),3)
                else:
                    left_fit.append((slope, intercept))
                    cv2.line(img,(x1,y1),(x2,y2),(255,0,0),3)
            elif slope < 0.8 and slope > -0.8:
                continue
            else:
                if x1 < (img.shape[1] / 2)-100:
                    left_fit.append((slope, intercept))
                    cv2.line(img,(x1,y1),(x2,y2),(255,0,0),3)
                else:
                    right_fit.append((slope, intercept))
                    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),3)

    if len(left_fit) and len(right_fit):
        if len(left_fit) > 50 and len(right_fit) > 50:
            p = 2
        else:
            left_fit_average = np.average(left_fit, axis=0)
            right_fit_average = np.average(right_fit, axis=0)
            left_line = make_coordinates(img, left_fit_average)
            right_line = make_coordinates(img, right_fit_average)
    
            left_line_bottom = left_line[0][0]
            right_line_bottom = right_line[0][0]
            center_of_frame = (img.shape[1] / 2)
            distance_left = center_of_frame - left_line_bottom
            distance_right = abs(right_line_bottom - center_of_frame)
            print( distance_left, "\n")
            print( distance_right, "\n")
            if distance_left - distance_right > 10:
                p = 0
                                
            elif distance_left - distance_right < -10:
                p = 1
                
            else:
                p = 2

    elif len(left_fit) > 1 and len(right_fit) == 0:
        p = 4
        time.sleep(0.1)
    elif len(right_fit) > 1 and len(left_fit) == 0:
        p = 3
        time.sleep(0.1)
    else:
        p = 2
    
    return p