# -*- coding: utf-8 -*-
"""Untitled4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dtmEWYDkYI3X1JvzcnXjq8SM3lqk7PcJ
"""

# from tensorflow.keras.optimizers import RMSprop, Adam
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Input
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import to_categorical
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
from PIL import Image
from numpy import array

zip_address = '/content/USPS_images.zip'
zip_ref = zipfile.ZipFile(zip_address, 'r')
zip_ref.extractall('/content/')
train_dir = '/content/USPS_images/train'
validation_dir = '/content/USPS_images/test'

y_train = []
x_train = []
y_test = []
x_test = []



for path in os.listdir(train_dir):
    if os.path.isfile(os.path.join(train_dir, path)):
        y_train.append(int(path[0]))

for path in os.listdir(validation_dir):
  if os.path.isfile(os.path.join(validation_dir, path)):
      y_test.append(int(path[0]))


for path in os.listdir(train_dir):
  if os.path.isfile(os.path.join(train_dir,path)):
    toArr = cv2.imread(f"{train_dir}/{path}")
    x_train.append(cv2.cvtColor(toArr, cv2.COLOR_RGB2GRAY))

for path in os.listdir(validation_dir):
  if os.path.isfile(os.path.join(validation_dir,path)):
    toArr = cv2.imread(f"{validation_dir}/{path}")
    x_test.append(cv2.cvtColor(toArr, cv2.COLOR_RGB2GRAY))

print(x_train[0])

x_train = np.array(x_train)
x_test = np.array(x_test)

y_train_cat = to_categorical(np.array(y_train), 10)
y_test_cat = to_categorical(np.array(y_test), 10)

print(y_train_cat[0])
print(x_train[0])

print(x_train.shape)

x_train_final = x_train.reshape(-1 ,16*16) / 255
x_test_final = x_test.reshape(-1 ,16*16) / 255

x_train_final.shape

print(y_train_cat.shape)
print(y_test_cat.shape)

print(y_train[0]) # 5 >>>> [0,0,0,0,0,1,0,0,0,0]
print(y_train_cat[0])

from keras.layers import Dense, Input
from keras.models import Sequential

model = Sequential()
model.add(Input(shape = (16*16)))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(10 , activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam' , metrics = ['accuracy'])

batch_size = 128
epochs = 30
model.fit(x_train_final, y_train_cat,
          batch_size= batch_size ,
          epochs=epochs, verbose= 1,
          validation_data=(x_test_final,y_test_cat))

random_number = np.random.randint(0,2007)

x = x_test[random_number]
expected = y_test[random_number]

print(f"expected is {expected}")

x = np.expand_dims(x, axis=0) / 255.
x = x.reshape(-1,16*16)
classes = model.predict(x, batch_size=10, verbose = 0)

print(f"result is {np.argmax(classes)}")