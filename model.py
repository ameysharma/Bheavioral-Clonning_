# -*- coding: utf-8 -*-
"""
Created 

@author: amey sharma
"""
 
import os
import csv
from sklearn.utils import shuffle

# Data Collection
# Reading Total number of Lines in an excel file 
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

# Using Generators and collecting steering measurements of the images
 
def generator(images, batch_size=128):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                
                # Collecting steering angles and image data for central camera
                # Moreover, Also flipping the images vertically .
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
                
                # Collecting left image data with angle improvement of +0.22 and
                # Also flipping the image and adjusting the angles accordingly
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
                
                #Collecting Right image data with angle improvement of -0.22
                #Also flipping the image and adjusting angle accordingly.
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
             

                
            
            # Generating Training Datasets 
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            yield sklearn.utils.shuffle(X_train, y_train)

# compiling  and training 

train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)

#Learning Model


from keras.models import Sequential,Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras import optimizers

model=Sequential()
# Adjusting Image  data values to -1 to 1 .
model.add(Lambda(lambda x: x/127.5 - 1,
        input_shape=(160, 320, 3)))
#Cropping 70px from the Top and 20px from bottom
model.add(Cropping2D(cropping=((70,20), (0,0))))
#Using Convolution Layers of 5X5 Kernals and Filters 3 folllowed by 18.
# Moreover Using Maxpool Function with 2 strides
model.add(Convolution2D(3,5,5,activation='relu'))
model.add(MaxPooling2D((2,2),strides=(2,2)))
model.add(Convolution2D(18,5,5,activation='relu'))
model.add(MaxPooling2D((2,2),strides=(2,2)))
#Using Convolution Layers of 3X3 Kernals and Filters 36 folllowed by 64.
# Moreover Using Maxpool Function.
model.add(Convolution2D(36,3,3,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(MaxPooling2D())
#Using A Dropout Rate of 0.5
model.add(Dropout(0.5))
#Flattening The 2d Dimensional image.
model.add(Flatten())
#Uing Dense model to Reduce the calues finally to 1
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(1))
#Setting the learning rate for the model
adam=optimizers.adam(lr=0.0008)
#Compiling the model
model.compile(loss='mse',optimizer=adam)
#Calling Generators Function Fo Training and Validation
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,nb_val_samples=len(validation_samples), nb_epoch=20)
#After 20 epoches Saving the model in "model.h5" file and "model.json" file.
model.save('model.h5')
model.save('model.json')