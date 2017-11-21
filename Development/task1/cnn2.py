#Author: Vince Trost
#Date: 11/20/2017
#cnn2 - beefy CNN

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import json
import os

np.random.seed(1234)

#read in the data
DATA_DIR = "../../Data/"
train = pd.read_json(DATA_DIR + "train.json")
target_train=train['is_iceberg']
test = pd.read_json(DATA_DIR + "test.json")

#get the images' pixel arrays out of the lists
X_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
X_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
#take the average of the two bands to make a third band
X_band3 = (X_band1+X_band2)/2

#concatenate to make training sample
X_train = np.concatenate([X_band1[:, :, :, np.newaxis]
                          , X_band2[:, :, :, np.newaxis]
                         , X_band3[:, :, :, np.newaxis]], axis=-1)

#do the same for the test set
X_band_test_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
X_band_test_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
X_band_test_3 = (X_band_test_1+X_band_test_2)/2

X_test = np.concatenate([X_band_test_1[:, :, :, np.newaxis]
                          , X_band_test_2[:, :, :, np.newaxis]
                         , X_band_test_3[:, :, :, np.newaxis]], axis=-1)

#split into train, test, and validation sets
X_Train, X_Test, y_train, y_test = train_test_split(X_train, target_train, test_size = 0.33)

#fix dimensions of label
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

#build the model
model = Sequential()
model.add(Convolution2D(32, (3, 3), padding='valid', input_shape=X_Train.shape[1:], activation='relu'))
model.add(Convolution2D(32, (3, 3), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.33))
model.add(Flatten())
model.add(Dense(152, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(74, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))

opt = Adam(lr=0.0001)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_Train, y_train, batch_size=100, epochs=50, verbose=1, validation_data=(X_Test, y_test))



