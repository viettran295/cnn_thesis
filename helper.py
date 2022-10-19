from re import S
import pandas as pd 
import numpy as np 
import os
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import cv2
from sklearn.utils import shuffle

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
def balance_data(data, cols_name: str, display=True, nbins=31):
    hist, bin = np.histogram(data[cols_name], bins=nbins)
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
        for j in range(len(data[cols_name])):
            if data[cols_name][j] >= bin[i] and data[cols_name][j] <= bin[i+1]:
                bin_list.append(j)
        bin_list = shuffle(bin_list)
        bin_list = bin_list[samplePerBin:]
        remove_list.extend(bin_list)
    
    data.drop(data.index[remove_list], inplace=True)
    print('removed imgs: ', len(remove_list))
    print('remain imgs: ', len(data))

    if display:
        hist, bin = np.histogram(data[cols_name], nbins)
        plt.bar(center, hist, width=0.06)
        plt.title(f"Distribution of {cols_name} data")
        plt.plot((-1,1), (samplePerBin, samplePerBin))
        plt.show()

# load img path and steering data to np array
def load_data_toArray(path, data):
    imgPath = []
    steering = [] 
    for i in range(len(data)):
        tmp = data.iloc[i]
        imgPath.append(os.path.join(path, 'IMG', tmp[0]))
        steering.append(tmp[3])
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
    img = cv2.GaussianBlur(img, (5,5))

    # resize and normalize img
    img = cv2.resize(img, (200,66))
    img = img/255

    return img 