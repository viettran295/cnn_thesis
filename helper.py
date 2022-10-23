import random
import pandas as pd 
import wandb
from wandb_config import config_defaults
import numpy as np 
import os
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import cv2
from sklearn.utils import shuffle
from keras import Sequential
from keras.layers import Conv2D, Flatten, Dense, Dropout
from keras.optimizers import Adam, SGD

# edit name of 'center' columns
def getName(name: str) -> str:
    return name.split("\\")[-1]

# load data to Dataframe
def load_data(path):
    cols = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
    df = pd.read_csv(path, names=cols)
    df['Center'] = df['Center'].apply(getName)
    return df 

# balance Steering data 
def balance_data(dataframe, cols_name: str, display=True, nbins=31):
    hist, bin = np.histogram(dataframe[cols_name], bins=nbins)
    samplePerBin = 1000
    center = (bin[:-1] + bin[1:]) * 0.5
    if display:
        plt.bar(center, hist, width=0.06)
        plt.title(f"Distribution of {cols_name} data")
        plt.plot((-1,1), (samplePerBin, samplePerBin))
        plt.show()

    # remove center angle to balance dataset 
    remove_list = [] 
    for i in range(nbins):
        bin_list = [] 
        for j in range(len(dataframe[cols_name])):
            if dataframe[cols_name][j] >= bin[i] and dataframe[cols_name][j] <= bin[i+1]:
                bin_list.append(j)
        bin_list = shuffle(bin_list)
        bin_list = bin_list[samplePerBin:]
        remove_list.extend(bin_list)
    
    dataframe.drop(dataframe.index[remove_list], inplace=True)
    print('removed imgs: ', len(remove_list))
    print('remain imgs: ', len(dataframe))

    if display:
        hist, bin = np.histogram(dataframe[cols_name], nbins)
        plt.bar(center, hist, width=0.06)
        plt.title(f"Distribution of {cols_name} data")
        plt.plot((-1,1), (samplePerBin, samplePerBin))
        plt.show()
    return dataframe

# load img path and steering from dataframe to np array
def load_data_toArray(path, dataframe):
    imgPath = []
    steering = [] 
    for i in range(len(dataframe)):
        tmp = dataframe.iloc[i]
        imgPath.append(os.path.join(path, 'IMG', tmp[0])) # first col of Dataframe
        steering.append(tmp[3]) # third col of Dataframe
    return np.asarray(imgPath), np.asarray(steering)

# image augmentation 
def augmentImg(imgPath, steering):
    img = mpimg.imread(imgPath)
    # shift image
    if np.random.rand() < 0.5: # random apply augmentation
        aff = iaa.Affine(translate_percent={'x':(-0.1, 0.1), 'y':(-0.1, 0.1)}) #shift image from -10% - 10%
        img = aff.augment_image(img)

    # zoom image
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1.2, 1.4)) #zoom image with x1.1 - 1.2
        img = zoom.augment_image(img)

    # brightness
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.5, 1.5)) # < 1: dark, > 1: bright
        img = brightness.augment_image(img)

    # flip 
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1) #1: flip around y-axis
        steering = -steering #steering have to be changed after flipping image

    return img, steering 

def img_preprocessing(img):
    # crop unnecessary region, keep road
    img = img[60:130,:,:] #[x, y, [R,G,B]]

    # change color space from RGB to YUV -> easy to define road
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    # Gaussian low-pass filter blur
    img = cv2.GaussianBlur(img, (5,5), 0)

    # resize and normalize img
    img = cv2.resize(img, (200,66))
    img = img/255

    return img 

# create and preprocessing imgs in batch 
def batch_generator(img_path_arr, steering_arr, batch_size, train_flag=True):
    while True:
        img_batch = []
        steering_batch = [] 
        for i in range(batch_size):
            idx = random.randint(0, len(img_path_arr)-1)
            if train_flag:
                img, steering = augmentImg(img_path_arr[idx], steering_arr[idx])
            else:
                img = mpimg.imread(img_path_arr[idx])
                steering = steering_arr[idx]
            img = img_preprocessing(img)
            img_batch.append(img)
            steering_batch.append(steering)
        yield(np.asarray(img_batch), np.asarray(steering_batch))

def build_network(activation, optimizer, dropout):
   
    model = Sequential()
    model.add(Conv2D(24, (5,5), (2,2), input_shape=(66, 200,3), activation=activation)) # (filter, kernel, stride, input shape)
    model.add(Conv2D(36, (5,5), (2,2), activation=activation)) 
    model.add(Conv2D(36, (5,5), (2,2), activation=activation)) 
    model.add(Conv2D(64, (3,3), activation=activation)) # size of img small -> stride = 1
    model.add(Conv2D(64, (3,3), activation=activation)) 

    model.add(Flatten())
    model.add(Dense(100, activation=activation))
    model.add(Dropout(dropout))
    model.add(Dense(50, activation=activation))
    model.add(Dropout(dropout))
    model.add(Dense(10, activation=activation))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

    return model 

def build_optimizer(optimizer, learning_rate):
    if optimizer == 'sgd':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer == 'adam':
        optimizer = SGD(learning_rate=learning_rate)
    return optimizer
