from tensorflow.keras.preprocessing import image as keras_image
import os
import numpy as np
from tqdm import tqdm
from PIL import ImageFile
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection  import train_test_split
import tensorflow as tf
from imgaug import augmenters as iaa
import random
import glob

dirname = "C:\PythonPrograming\selfdriving\data.1674100831.013040/"

def image_to_tensor(img_path):
    img = keras_image.load_img(
        os.path.join(dirname, img_path), 
        target_size = (120, 160))
    x = keras_image.img_to_array(img)
    return np.expand_dims(x, axis = 0)

def data_to_tensor(img_paths):
    list_of_tensors = [
        image_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

ImageFile.LOAD_TRUNCATED_IMAGES = True

data = pd.read_csv(os.path.join(dirname, "0_road_labels.csv"))

files = data["cam1"]
targets = data['steering'].values

tensors = data_to_tensor(files)


#%%
def img_preprocess(img):
    img = img.astype('int8')/255
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = np.clip(img, 0, 1)
    return img

def zoom(image):
  zoom = iaa.Affine(scale=(1, 1.3))
  image = zoom.augment_image(image)
  return image

def pan(image):
  pan = iaa.Affine(translate_percent= {"x" : (-0.1, 0.1), "y": (-0.1, 0.1)})
  image = pan.augment_image(image)
  return image

def img_random_brightness(image):
    brightness = iaa.Multiply((0.2, 1.2))
    image = brightness.augment_image(image)
    return image

def normalize(values, actual_bounds, desired_bounds):
    return round(desired_bounds[0] + (values - actual_bounds[0]) * (desired_bounds[1] - desired_bounds[0]) / (actual_bounds[1] - actual_bounds[0]),2)


def random_augment(image):
    if np.random.rand() < 0.5:
      image = pan(image)
    if np.random.rand() < 0.5:
      image = zoom(image)
    if np.random.rand() < 0.5:
      image = img_random_brightness(image)
    image = img_preprocess(image)
    return image

#%%

targets = to_categorical(targets/10, 10)
x_train, x_test, y_train, y_test = train_test_split(
    
    tensors,
    targets,
    test_size = 0.2,
    random_state =1     
    
    )

#%%

x_train = np.array(list(map(random_augment, x_train)))
x_test = np.array(list(map(random_augment, x_test)))


#%%
model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (5, 5), strides = (2, 2), padding = "same", activation ='elu', input_shape = x_train.shape[1:]),
        tf.keras.layers.Dropout(0.5),   
        tf.keras.layers.Conv2D(64, (5,5), strides = (2,2), padding = "same", activation ='elu'),
        tf.keras.layers.Conv2D(64, (3,3), padding = "same", activation ='elu'), 
        tf.keras.layers.Conv2D(64, (3,3), activation ='elu'),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='elu', input_dim = 3),
        tf.keras.layers.Dropout(0.5),      
        tf.keras.layers.Dense(10, activation='softmax'),
    ])

model.summary()
#%%

sgd = tf.keras.optimizers.legacy.SGD(lr=0.001, decay=1e-6, momentum = 0.9, nesterov=True)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics= ['accuracy'])

history = model.fit(x_train, y_train, epochs = 60,
                    validation_data = (x_test, y_test))

#%%
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss)+1)

plt.plot(epochs, val_loss, 'g', label= 'valid loss')
plt.plot(epochs, loss, 'r', label = 'loss')
plt.plot(epochs, history.history['accuracy'], 'b', label = 'accuracy')
plt.plot(epochs, history.history['val_accuracy'], 'y', label = 'val_accuracy')
plt.title('Tuning loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

model.save("model.h5")

#%%
from tensorflow.keras.models import load_model
import glob
import numpy as np
import tensorflow as tf 
import cv2
import matplotlib.pyplot as plt

def img_preprocess(img):
    img = cv2.resize(img, (160, 120))
    img = img.astype('int8')/255
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = np.clip(img, 0, 1)
    return img

dirname = "C:\PythonPrograming\selfdriving/data.1674100831.013040/cam1/"
imagedir = glob.glob(dirname+'*.png', recursive = True)


model1 = load_model('model.h5')
for i,image in enumerate(imagedir):
    i += 1
    image = cv2.imread(image)
    image1 = img_preprocess(image)
    image_tensor = tf.convert_to_tensor(image1, dtype=tf.float32)
    image_tensor = tf.expand_dims(image_tensor, 0)
    y_predict = model1.predict( image1 ) 
    y_predict = np.argmax(y_predict, axis =1)[0]*10
    ttitle = f'{y_predict}'    
    plt.subplot(4,5, i)
    plt.title(ttitle)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image1)
    if i == 20:
        break
plt.show()

#%%

