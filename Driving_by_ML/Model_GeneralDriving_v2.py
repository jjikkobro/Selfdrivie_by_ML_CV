import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import random
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras import layers
from keras.models import load_model
import ntpath
from keras.applications import VGG16

dirname = "data.1674100831.013040/"
data = pd.read_csv(os.path.join(dirname, "0_road_labels.csv"))
pd.set_option('display.max_colwidth', None)
print(data.head())

#%%
num_bins = 10
hist, bins = np.histogram(data['steering'], num_bins)
x = np.arange(len(bins)-1)
plt.bar(x, hist, width= 0.4)
plt.xticks(x, [0,10,20,30,40,50,60,70,80,90])

plt.show()
#%%

def  load_img_steering(datadir, df):
    img_path = []
    steering = []
    for i in range(len(df)):
        indexed_data = df.iloc[i]
        cam1 = indexed_data[0]
        img_path.append(os.path.join(datadir, cam1.strip()))
        steering.append(float(indexed_data[2]))
    img_paths = np.asarray(img_path)
    steerings = np.asarray(steering)
    return img_paths, steerings





#%%
img_paths, steerings = load_img_steering( dirname, data )
X_train, X_valid, y_train, y_valid = train_test_split(img_paths, steerings,
                                                      test_size = 0.2, random_state = 6 )

print('Training Samples: {}\nValid Samples: {}'.format(len(X_train), len(X_valid)))
fig, axis = plt.subplots(1, 2, figsize=(12, 4))
axis[0].hist(y_train, bins=num_bins, width=0.4, color='blue')
axis[0].set_title('Training set')
axis[1].hist(y_valid, bins=num_bins, width=0.4, color='red')
axis[1].set_title('Validation set')

#%%
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

def img_random_flip(image, steering_angle):
    image = cv2.flip(image,1)
    steering_angle = float(normalize(int(steering_angle), (0,90), (90,0)))
    return image, steering_angle

def random_augment(image, steering_angle):
    image = cv2.imread(image)
    if np.random.rand() < 0.5:
      image = pan(image)
    if np.random.rand() < 0.5:
      image = zoom(image)
    if np.random.rand() < 0.5:
      image = img_random_brightness(image)
    if np.random.rand() < 0.5:
      image, steering_angle = img_random_flip(image, steering_angle)   
    return image, steering_angle


def img_preprocess(img):
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img

X_train = np.array(list(map(img_preprocess, X_train)))
X_valid = np.array(list(map(img_preprocess, X_valid)))



#%%
num_classes = 10

model = Sequential([
  layers.Rescaling(1./255, input_shape=(66, 200, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes, activation='relu'),
  layers.Dense(1)
])
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train, validation_data=(X_valid,y_valid),epochs=30)

#%%

loss = history.history['loss']
val_loss = history.history['val_loss']


plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
model.save('model.h5')

#%%
model = load_model('model10.h5')
image = mpimg.imread("data.1673539814.568109/_0_right/1673539820.050267.png")
image = img_preprocess(image)
image = np.array([image])
angle = float(model.predict(image))