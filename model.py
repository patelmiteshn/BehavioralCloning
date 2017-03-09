
# coding: utf-8

# In[37]:

import os
import csv
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn
import math
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import pandas as pd
import itertools

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, SpatialDropout2D, ELU
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D
from keras.layers import Lambda, Input
from keras.regularizers import l2
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import glob
import os
from pathlib import Path
import json




## read udacity dataset and recovery data
df = pd.read_csv('../dataFinal/data_udacity/driving_log.csv', header=0)

df.columns = ['center_image', 'left_image', 'right_image', 'steering_angle', 'throttle', 'break', 'speed']

df['center_image'] = df['center_image'] = '../dataFinal/data_udacity/IMG/' + df['center_image'].map(lambda x: x.rsplit('/')[-1])
df['left_image'] = df['left_image'] = '../dataFinal/data_udacity/IMG/' + df['left_image'].map(lambda x: x.rsplit('/')[-1])
df['right_image'] = df['right_image'] = '../dataFinal/data_udacity/IMG/' + df['right_image'].map(lambda x: x.rsplit('/')[-1])


df_recovery = pd.read_csv('../dataFinal/track1_recovery/driving_log.csv', header=0)
df_recovery.columns = ['center_image', 'left_image', 'right_image', 'steering_angle', 'throttle', 'break', 'speed']
df_recovery['center_image'] = df_recovery['center_image'] = '../dataFinal/track1_recovery/IMG/' + df_recovery['center_image'].map(lambda x: x.rsplit('/')[-1])        
df_recovery['left_image'] = df_recovery['left_image'] = '../dataFinal/track1_recovery/IMG/' + df_recovery['left_image'].map(lambda x: x.rsplit('/')[-1])        
df_recovery['right_image'] = df_recovery['center_image'] = '../dataFinal/track1_recovery/IMG/' + df_recovery['center_image'].map(lambda x: x.rsplit('/')[-1])        
print('length of udacity dataset: %d' %len(df_recovery))
print('length of udacity dataset: %d' %len(df))


################################################################################################################
# Data balancing setup
# Take a look at https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
################################################################################################################

def sampleData(df):
    df_right = []
    df_left = []
    df_center = []
    for i in range(len(df)):
        center_img = df["center_image"][i]
        left_img = df["left_image"][i]
        right_img = df["right_image"][i]
        angle = df["steering_angle"][i]
#         print (angle)

        if (angle > 0.15):
            df_right.append([center_img, left_img, right_img, angle])

            # I'm adding a small deviation of the angle 
            # This is to create more right turning samples for the same image
            for i in range(10):
                new_angle = angle * (1.0 + np.random.uniform(-1, 1)/30.0)
                df_right.append([center_img, left_img, right_img, new_angle])

        elif (angle < -0.15):
            df_left.append([center_img, left_img, right_img, angle])

            # I'm adding a small deviation of the angle
            # This is to create more left turning samples for the same image
            for i in range(15):
                new_angle = angle * (1.0 + np.random.uniform(-1, 1)/30.0)
                df_left.append([center_img, left_img, right_img, new_angle])

        else:
            if (angle != 0.0):
                # Include all near 0 angle data
                df_center.append([center_img, left_img, right_img, angle])
    return df_left, df_center, df_right

df_left_u, df_center_u, df_right_u = sampleData(df)
df_left_r, df_center_r, df_right_r = sampleData(df_recovery)

df_left = df_left_u + df_left_r 
df_center = df_center_u + df_center_r
df_right = df_right_u + df_right_r


print('data after udacity: left %d, center %d, right %d' %(len(df_left_u), len(df_center_u), len(df_right_u) ))
print('data after recovery: left %d, center %d, right %d' %(len(df_left_r), len(df_center_r), len(df_right_r) ))
print('data after concatenation: left %d, center %d, right %d' %(len(df_left), len(df_center), len(df_right) ))


# shuffle and name the columns in dataframe
np.random.shuffle(df_center)
np.random.shuffle(df_left)
np.random.shuffle(df_right)

df_center = pd.DataFrame(df_center, columns=["center_image", "left_image", "right_image", "steering_angle"])
df_left = pd.DataFrame(df_left, columns=["center_image", "left_image", "right_image", "steering_angle"])
df_right = pd.DataFrame(df_right, columns=["center_image", "left_image", "right_image", "steering_angle"])

## concate data and split into training testing 
data_list = [df_left, df_center, df_right]
data_df = pd.concat(data_list, ignore_index=True)
print('length of dataset: %d' %len(data_df) )

################################################################################################################
# Split data in training and testing

################################################################################################################

X_data = data_df[['center_image', 'left_image', 'right_image', 'steering_angle']]
y_data = data_df['steering_angle']
# train test split
X_train, X_valid, y_train, y_valid = train_test_split(X_data, y_data, test_size=0.2)
# reset index
X_train = X_train.reset_index(drop=True)
X_valid = X_valid.reset_index(drop=True)

print('print length of final dataset training: %d, validation: %d ' %(len(X_train), len(X_valid) ) )


################################################################################################################
# Process data 
# change brightness of images at random to add more variety in training data 
# gett sufficient data for turning in opposite direcion
# Remove unwanted region from the image

################################################################################################################
## processing data 

def change_brightness(image):
    # Randomly select a percent change
    change_pct = np.random.uniform(0.4, 1.2)
    # Change to HSV to change the brightness V
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = hsv[:,:,2] * change_pct
    #Convert back to RGB 
    img_brightness = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img_brightness

def flip_image(image, angle):
    img_flip = cv2.flip(image,1)
    angle = -angle
    return img_flip, angle



def preprocessImage(image):
    # Proportionally get lower half portion of the image
    nrow, ncol, nchannel = image.shape    
    start_row = int(nrow * 0.35)
    end_row = int(nrow * 0.875)   
    # This removes most of the sky and small amount below including the hood
    new_image = image[start_row:end_row, :]

    return new_image


def process_image(data_df, TRAIN = False):

    path_filename = data_df["center_image"][0]
    image = cv2.imread(path_filename)
    angle = data_df['steering_angle'][0]
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    
    image = change_brightness(image)    
    # adding fliped image only for training data
    if (TRAIN):
        if np.random.randint(2) == 1:
            image, angle = flip_image(image, angle)
    
    image = preprocessImage(image)
    image = np.array(image)

    return image, angle


            
def process_image_batch(data_df, TRAIN = False):
    images = []
    angles = []
    for i in range(len(data_df)):
        path_filename = data_df["center_image"][i]
        image = cv2.imread(path_filename)
        angle = data_df['steering_angle'][i]
    
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    
        image = change_brightness(image)
        # Coin flip to see to flip image and create a new sample of -angle
        if (TRAIN):
            if np.random.randint(2) == 1:
                image, angle = flip_image(image, angle)
        image = preprocessImage(image)
        images.append(image)
        angles.append(angle)
    images = np.array(images)
    angles = np.array(angles)
    return images, angles


def generator(data_df, batch_size=32, TRAIN = False):
    num_samples = len(data_df)
    
    while True: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            data_row = data_df[offset:offset+batch_size].reset_index()
            X_train, y_train = process_image_batch(data_row, TRAIN)
#             return X_train, y_train
            yield sklearn.utils.shuffle(X_train, y_train)



################################################################################################################
# Parameters Model Training
# 
# Set Hyper parameters of the model
# Train the model
# gett sufficient data for turning in opposite direcion
# Remove unwanted region from the image

################################################################################################################

INPUT_IMG_HEIGHT = 84
INPUT_IMG_WIDTH = 320
IMG_CHANNEL = 3


l2_reg = 0.001
keep_prob = 0.5
batch_size = 64
# Training settings
min_delta=1e-4
patience=4
nb_epoch = 100
beta2 = 0.999
beta1 = 0.9
epsilon = 1e-08
decay = 0.01
lr = 0.001

# Clear TensorBoard Logs
for file in os.listdir('./logs/'):
    os.remove('./logs/' + file)

# Clear any previously saved model files 
filelist = glob.glob("Models/*.*")
for f in filelist:
    os.remove(f)






print("img_height:", INPUT_IMG_HEIGHT)
print("img_width:", INPUT_IMG_WIDTH)
print("img_channel:", IMG_CHANNEL)
print("len(X_train_data:)", len(X_train))
print("len(X_valid_data:)", len(X_valid))






###################
# Nvidia model for driving behavior prediciton

def get_model_nvidia():
    model = Sequential()
    
    input_shape = (INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH, 3)
    model.add(Lambda(lambda x: x/127.5 - 1.,input_shape = input_shape))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid", init='he_normal', name='conv1'))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid", init='he_normal', name='conv2'))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid", init='he_normal', name='conv3'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init='he_normal', name='conv4'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init='he_normal', name='conv5'))
    model.add(Flatten(name='flatten1'))
    model.add(ELU())
    model.add(Dense(1164, init='he_normal', name='dense1'))
    model.add(ELU())
    model.add(Dense(100, init='he_normal', name='dense2'))
    model.add(ELU())
    model.add(Dense(50, init='he_normal', name='dense3'))
    model.add(ELU())
    model.add(Dense(10, init='he_normal', name='dense4'))
    model.add(ELU())
    model.add(Dense(1, init='he_normal', name='dense5'))

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
    model.compile(optimizer=adam, loss='mse')
    
    return model


###################
# Modified version of vgg16_net for driving behavior predicion 


def vgg16_net():
    
    model = Sequential()
    ## 2. Data Preprocessing functions
    input_shape = (INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH, 3)
    model.add(Lambda(lambda x: x/127.5 - 1.,input_shape = input_shape))
    
    # Block 1
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, border_mode='same') )
    model.add(ELU())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 2
    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(ELU())
    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(ELU())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 3
    model.add(Convolution2D(256, 3, 3, border_mode='same'))
    model.add(ELU())
    model.add(Convolution2D(256, 3, 3, border_mode='same'))
    model.add(ELU())
    model.add(Convolution2D(256, 3, 3, border_mode='same'))
    model.add(ELU())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 4
    model.add(Convolution2D(512, 3, 3, border_mode='same'))
    model.add(ELU())
    model.add(Convolution2D(512, 3, 3, border_mode='same'))
    model.add(ELU())
    model.add(Convolution2D(512, 3, 3, border_mode='same'))
    model.add(ELU())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 5
    model.add(Convolution2D(512, 3, 3, border_mode='same'))
    model.add(ELU())
    model.add(Convolution2D(512, 3, 3, border_mode='same'))
    model.add(ELU())
    model.add(Convolution2D(512, 3, 3, border_mode='same'))
    model.add(ELU())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten(name='flatten'))
    model.add(Dense(4096) )
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(2048) )
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1000) )
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    return model   


###################
# CNN model that I came up with for driving behavior predicion 

def model_cnn():

    model = Sequential()
    input_shape = (INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH, 3)
    model.add(Lambda(lambda x: x/127.5 - 1.,input_shape = input_shape))


    # CNN Model
    model.add(Convolution2D(32, 3, 3, init='he_normal', W_regularizer=l2(l2_reg)) )
    model.add(ELU())
    model.add(Convolution2D(32, 3, 3, init='he_normal', W_regularizer=l2(l2_reg)) )
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid') )

    model.add(Convolution2D(64, 3, 3, init='he_normal', W_regularizer=l2(l2_reg)) )
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid') )

    model.add(Convolution2D(128, 3, 3, init='he_normal', W_regularizer=l2(l2_reg)) )
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid') )

    model.add(Convolution2D(256, 3, 3, init='he_normal', W_regularizer=l2(l2_reg)) )
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid') )

    model.add(Flatten())
    model.add(Dense(512, init='he_normal' ) )
    model.add(ELU())
    model.add(Dropout(keep_prob) )
    model.add(Dense(256, init='he_normal') )
    model.add(ELU())
    model.add(Dropout(keep_prob) )
    model.add(Dense(128, init='he_normal') )
    model.add(ELU())
    model.add(Dropout(keep_prob) )
    model.add(Dense(64, init='he_normal', W_regularizer=l2(l2_reg)) )
    model.add(ELU())
    model.add(Dense(16, init='he_normal', W_regularizer=l2(l2_reg)) )
    model.add(ELU())
    model.add(Dense(1))

    return model

###################
# Save trained model
def save_model(fileModelJSON, fileWeights):
    prefix = "./Models/"
    
    filenameJSON = prefix + fileModelJSON
    if Path(filenameJSON).is_file():
        os.remove(filenameJSON)    
    with open (filenameJSON, 'w') as outfile:
        json.dump(model.to_json(), outfile)
        
    filenameWeights = prefix + fileWeights
    if Path(filenameWeights).is_file():
        os.remove(filenameWeights)
    model.save_weights(filenameWeights, True)


###################
# Select model and configure hyper parameters 

model = model_cnn() #vgg16_net() #get_model_nvidia()
adam = Adam(lr=lr, beta_1=beta1, beta_2=beta2, epsilon=epsilon, decay= decay)
print("Model summary:\n", model.summary())
model.compile(optimizer=adam, loss="mse", metrics=['mean_absolute_error'])


# compile and train the model using the generator function
train_generator = generator(X_train, batch_size, True)
validation_generator = generator(X_train, batch_size, False)

# Model training
filepath = 'Models/model.h5'

callbacks = [
    EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience, mode='min'),
    ModelCheckpoint(filepath=filepath, monitor='val_loss', save_best_only=True, save_weights_only=False, mode='min'),
    TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True)
  ]

model.fit_generator(train_generator, samples_per_epoch=len(X_train), validation_data=validation_generator,
                   nb_val_samples=len(X_valid),nb_epoch=nb_epoch, verbose=1, callbacks=callbacks)


fileModelJSON = 'model_cnn' + '.json'
fileWeights = 'model_cnn' + '.h5'
save_model(fileModelJSON, fileWeights)
print('model saved')


