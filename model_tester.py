# test the model which detect the color of traffic light and crosswalk
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import random
from keras.models import load_model  

# Load frames
dirr = glob(r"C:\PythonPrograming\selfdriving\data.1675950870.971505\cam1\*.png")

np.set_printoptions(suppress=True)

# Load the model
model = load_model("Models/crosswalk_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

random.shuffle(dirr)

for i,im in enumerate(dirr):
    i+=1
    _ = cv2.imread(im)
    image = cv2.resize(_, (224, 224), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 172.5)-1
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    plt.subplot(5,5, i)
    plt.title(str(index) + str(np.round(confidence_score * 100))[:-2],fontdict = {'fontsize' : 8})
    plt.xticks([])
    plt.yticks([])
    plt.imshow(_)
    if i == 25:
        break
plt.show()
    
 
# =============================================================================
#     print("Class:", class_name[2:], end="")
#     print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
# =============================================================================

