import os
import cv2
import numpy as np
from keras.models import load_model
from keras.callbacks import Callback, ModelCheckpoint
import sklearn
from sklearn.model_selection import train_test_split
import csv

samples = []
data = 'data2/driving_log.csv'

print (data)
#load the lines in driving log file
with open(data) as csvfile:
    reader = csv.reader(csvfile, skipinitialspace=True)
    for line in reader:
        samples.append(line)
#Delete the header
del samples[0]

#Split dataset into train and validation samples
train_samples, validation_samples = train_test_split(samples, test_size=0.4)

import math
from math import ceil
import random
from random import shuffle


# Set our batch size
batch_size=8
# Set our train and validations steps per epoch (80% and 20% of total training samples).
num_train_steps = ceil(192.6/(batch_size))
num_validation_steps = ceil(128.4/(batch_size))

#Generator function to yield train and validation samples
def generator(samples, train, batch_size=32):
    global num_train_steps
    global num_validation_steps
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            left_images = []
            left_angles = []
            right_images = []
            right_angles = []
            for batch_sample in batch_samples:
                name = 'data/'+ batch_sample[0]
                #Append the image and angle
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                #Append the flipped image and inverse of angle
                images.append(np.fliplr(center_image))
                angles.append(center_angle * -1.0)
                
                name2 = 'data/'+ batch_sample[1]
                left_image = cv2.imread(name2)
                left_angle = float(batch_sample[3]) + 0.2
                #Append the left image and offsetted angle
                left_images.append(left_image)
                left_angles.append(left_angle)
                #Include flipped left images.
                left_images.append(np.fliplr(left_image))
                left_angles.append(left_angle * -1.0)
                
                name3 = 'data/'+ batch_sample[2]
                right_image = cv2.imread(name3)
                right_angle = float(batch_sample[3]) - 0.2
                #Append the right image and offsetted angle
                right_images.append(right_image)
                right_angles.append(right_angle)
                #Include flipped right images.
                right_images.append(np.fliplr(right_image))
                right_angles.append(right_angle * -1.0)

            if len(images) != 0:
                X_train = np.array(images)
                y_train = np.array(angles)

            if len(left_images) != 0:
                X_train_left = np.array(left_images)
                y_train_left = np.array(left_angles)

                X_train = np.concatenate((X_train, X_train_left), axis=0)
                y_train = np.concatenate((y_train, y_train_left), axis=0)

            if len(right_images) != 0:
                X_train_right = np.array(right_images)
                y_train_right = np.array(right_angles)

                X_train = np.concatenate((X_train, X_train_right), axis=0)
                y_train = np.concatenate((y_train, y_train_right), axis=0)
                
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, train=True, batch_size=batch_size)
validation_generator = generator(validation_samples, train=False, batch_size=batch_size)
#Load the model from h5 file
model = load_model('model.h5')
#To save the model after every epoch
checkpoint = ModelCheckpoint('model.h5')
#Fit the model for 20 epochs.
model.fit_generator(train_generator, steps_per_epoch= num_train_steps, validation_data=validation_generator, validation_steps= num_validation_steps, epochs=20, verbose=1, callbacks=[checkpoint])
#Save the model finally.
model.save('model.h5')