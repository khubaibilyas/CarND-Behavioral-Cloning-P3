import os
import cv2
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, Lambda, Cropping2D, Dropout
from keras.callbacks import Callback, ModelCheckpoint
import sklearn
from sklearn.model_selection import train_test_split
import csv

class CustomSaver (Callback):
    def on_epoch_end(self, epoch, log={}):
        self.model.save('model.h5')
samples = []

#Open driving log file and read all the lines
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile, skipinitialspace=True)
    for line in reader:
        samples.append(line)
#Delete the header.
del samples[0]

#Split the samples into train and validation set with the ratio of 80:20
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import math
from math import ceil
import random
from random import shuffle


# Set our batch size
batch_size=8
# Set our train and validations steps per epoch (Multiplied by 3 as we also include right and left images).
num_train_steps = ceil(4822.2/(batch_size))
num_validation_steps = ceil(3214.8/(batch_size))

#Generator function to yield the train and validation samples for each batch.
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

#Normaliser function to normalise the inpit data
def normalizer(x):
    x /= 255
    x -= 0.5
    return x

model = Sequential()
#Normaliser
model.add(Lambda(normalizer, input_shape=(160, 320, 3)))
#Crop the input image to only include the relevant section
model.add(Cropping2D(cropping=((70,20), (0,0))))
#Follow the architecture from nvidia
model.add(Conv2D(24, (5, 5), padding="same", activation="relu"))
model.add(Conv2D(36, (5, 5), padding="same", activation="relu"))
model.add(Conv2D(48, (5, 5), padding="same", activation="relu"))
model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.50))
model.add(Dense(50))
model.add(Dropout(0.50))
model.add(Dense(10))
model.add(Dropout(0.50))
model.add(Dense(1))
#Compile the model using adam optimiser using mean squared loss error
model.compile(loss='mse', optimizer='adam')
#To save model after every epoch
checkpoint = ModelCheckpoint('model.h5')
#Fit the model for 10 epochs
model.fit_generator(train_generator, steps_per_epoch= num_train_steps, validation_data=validation_generator, validation_steps= num_validation_steps, epochs=10, verbose=1, callbacks=[checkpoint])
#Save the model
model.save('model.h5')