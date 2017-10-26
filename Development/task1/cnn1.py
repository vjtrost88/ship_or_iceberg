#Author: Vince Trost
#Implement simple Keras Conv NN

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import json
import os

np.random.seed(1234)

#read in the data
DATA_DIR = "Data/"

train_raw = open(DATA_DIR + "train.json", "r").read()
train = json.loads(train_raw)

test_raw = open(DATA_DIR + 'test.json', 'r').read()
test = json.loads(test_raw)

#get the images' pixel arrays out of the lists
train_band_2s = np.array([train[x]['band_2'] for x in range(len(train))])
train_band_1s = np.array([train[x]['band_1'] for x in range(len(train))])

train_labels = np.array([train[x]['is_iceberg'] for x in range(len(train))])

test_band_2s = np.array([test[x]['band_2'] for x in range(len(test))])
test_band_1s = np.array([test[x]['band_1'] for x in range(len(test))])


#preprocess the data so it's model-ready
#reshape the train and test sets
train_band_1s = train_band_1s.reshape(train_band_1s.shape[0], 75, 75, 1)
train_band_2s = train_band_2s.reshape(train_band_2s.shape[0], 75, 75, 1)

test_band_1s = test_band_1s.reshape(test_band_1s.shape[0], 75, 75, 1)
test_band_2s = test_band_2s.reshape(test_band_2s.shape[0], 75, 75, 1)

#change the number types
train_band_1s = train_band_1s.astype('float32')
train_band_2s = train_band_2s.astype('float32')
test_band_1s = test_band_1s.astype('float32')
test_band_2s = test_band_2s.astype('float32')

#put the pixel values on a spectrum of [0, 1]
train_band_1s /= 255
train_band_2s /= 255
test_band_1s /= 255
test_band_2s /= 255

#fix the dimensions of the labels
train_labels = np_utils.to_categorical(train_labels, 2)



#BUILD THE MODEL

model = Sequential()
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(75, 75, 1)))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics=['accuracy'])
model.fit(train_band_1s, train_labels, batch_size=32, epochs=10, verbose=1)

scores = model.predict(test_band_1s)
scores = [scores[x][1] for x in range(len(scores))]
ids = [test[x]['id'] for x in range(len(test))]

scores = pd.DataFrame(scores)
ids = pd.DataFrame(ids)
frames = [ids, scores]
preds = pd.concat(frames, axis=1)
preds.columns = ['id', 'is_iceberg']



preds.to_csv("preds.csv", sep=',', index=False)
model.save("cnn1.h5")
