# -*- coding: utf-8 -*-
"""
Created 

@author: amey sharma
"""

import os
import csv
from sklearn.utils import shuffle

# Data Collection

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


import cv2
import numpy as np
import sklearn

center_angles=[]
right_angles=[]
left_angles=[]
center_images=[]
right_images=[]
left_images=[]

def generator(images, batch_size=128):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                filename=batch_sample[0].split('/')[-1]
                current_path='data/IMG/'+filename
                name = current_path
                image = cv2.imread(name,cv2.IMREAD_COLOR)
                image=cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
                angle = float(batch_sample[3])
                images.append(image)
                angles.append(angle)
                images.append(cv2.flip(image,1))
                angles.append(-1*angle)
                #left image data with angle
                filename=batch_sample[1].split('/')[-1]
                current_path='data/IMG/'+filename
                name = current_path
                image = cv2.imread(name,cv2.IMREAD_COLOR)
                image=cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
                angle = float(batch_sample[3])
                images.append(image)
                angles.append(angle+0.22)
                images.append(cv2.flip(image,1))
                angles.append(-1*(angle+0.22))
                #Right image data
                filename=batch_sample[2].split('/')[-1]
                current_path='data/IMG/'+filename
                name = current_path
                image = cv2.imread(name,cv2.IMREAD_COLOR)
                image=cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
                angle = float(batch_sample[3])
                images.append(image)
                angles.append(angle-0.22)
                images.append(cv2.flip(image,1))
                angles.append(-1*(angle-0.22))
             

                

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            yield sklearn.utils.shuffle(X_train, y_train)
             # compile and train the model using the generator function

train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)

#Learning Model


from keras.models import Sequential,Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras import optimizers
model=Sequential()
model.add(Lambda(lambda x: x/127.5 - 1,
        input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,20), (0,0))))
model.add(Convolution2D(3,5,5,activation='relu'))
model.add(MaxPooling2D((2,2),strides=(2,2)))
model.add(Convolution2D(18,5,5,activation='relu'))
model.add(MaxPooling2D((2,2),strides=(2,2)))
model.add(Convolution2D(36,3,3,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(1))
#Setting the learning rate for the model
adam=optimizers.adam(lr=0.0008)
model.compile(loss='mse',optimizer=adam)
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,nb_val_samples=len(validation_samples), nb_epoch=20)
model.save('model.h5')
model.save('model.json')