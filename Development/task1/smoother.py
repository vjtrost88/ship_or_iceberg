#cnn2.py
#Author: Vince Trost
#Let's ramp up the complexity a bit now that we have the power

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from scipy import signal
import matplotlib.pyplot as plt
import json
import os

np.random.seed(1234)

#read in the data
DATA_DIR = "../../Data/"
train = pd.read_json(DATA_DIR + "train.json")
target_train=train['is_iceberg']
test = pd.read_json(DATA_DIR + "test.json")


#define SMOOTHER
smooth = np.array([[1, 1, 1], [1, 5, 1], [1, 1, 1]])

#get the images' pixel arrays out of the lists
X_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
X_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
#take the average of the two bands to make a third band
X_band3 = (X_band1+X_band2)/2

#SMOOTH THE BANDS
X_band1 = np.array([signal.convolve2d(band, smooth, mode='valid') for band in X_band1])
X_band2 = np.array([signal.convolve2d(band, smooth, mode='valid') for band in X_band2])
X_band3 = np.array([signal.convolve2d(band, smooth, mode='valid') for band in X_band3])

#concatenate to make training sample
X_train = np.concatenate([X_band1[:, :, :, np.newaxis]
                          , X_band2[:, :, :, np.newaxis]
                         , X_band3[:, :, :, np.newaxis]], axis=-1)

#do the same for the test set
X_band_test_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
X_band_test_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
X_band_test_3 = (X_band_test_1+X_band_test_2)/2

#SMOOTH THE BANDS
X_band_test_1 = np.array([signal.convolve2d(band, smooth, mode='valid') for band in X_band_test_1])
X_band_test_2 = np.array([signal.convolve2d(band, smooth, mode='valid') for band in X_band_test_2])
X_band_test_3 = np.array([signal.convolve2d(band, smooth, mode='valid') for band in X_band_test_3])

X_test = np.concatenate([X_band_test_1[:, :, :, np.newaxis]
                          , X_band_test_2[:, :, :, np.newaxis]
                         , X_band_test_3[:, :, :, np.newaxis]], axis=-1)

#sanity test
fig = plt.figure(1, figsize=(15, 15))
for i in range(9):
    ax = fig.add_subplot(3, 3, i+1)
    arr = X_band3[i]
    ax.imshow(arr, cmap='inferno')

plt.show()
