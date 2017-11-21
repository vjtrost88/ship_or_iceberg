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

X_train = X_train.reshape(X_train.shape[0], 73, 73, 1)

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

X_test = X_test.reshape(X_test.shape[0], 73, 73, 1)

#fix the labels
target_train = np_utils.to_categorical(target_train, 2)


#build the model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(73, 73, 1)))
model.add(Dense(2, activation='sigmoid'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, target_train, batch_size=802, epochs=10, verbose=1)

preds = model.predict(X_test)
scores = [preds[x][1] for x in range(len(preds))]

submission = pd.DataFrame()
submission['id'] = test['id']
submission['is_iceberg'] = scores
submission.to_csv('preds_smooth.csv', index=False)
