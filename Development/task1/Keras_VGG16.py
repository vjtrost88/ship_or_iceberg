#Mandatory imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from os.path import join as opj
from mpl_toolkits.mplot3d import Axes3D
import pylab
import cv2

#Import Keras.
from keras import initializers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.optimizers import RMSprop, Adam, SGD, Adamax
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
from keras.utils import np_utils, multi_gpu_model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import load_model

def get_scaled_imgs(df):
    imgs = []

    for i, row in df.iterrows():
        #make 75x75 image
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 + band_2 # plus since log(x*y) = log(x) + log(y)

        # Rescale
        a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        b = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
        c = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())

        imgs.append(np.dstack((a, b, c)))

    return np.array(imgs)

def get_more_images(imgs):

    more_images = []
    vert_flip_imgs = []
    hori_flip_imgs = []

    for i in range(0,imgs.shape[0]):
        a=imgs[i,:,:,0]
        b=imgs[i,:,:,1]
        c=imgs[i,:,:,2]

        av=cv2.flip(a,1)
        ah=cv2.flip(a,0)
        bv=cv2.flip(b,1)
        bh=cv2.flip(b,0)
        cv=cv2.flip(c,1)
        ch=cv2.flip(c,0)

        vert_flip_imgs.append(np.dstack((av, bv, cv)))
        hori_flip_imgs.append(np.dstack((ah, bh, ch)))

    v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)

    more_images = np.concatenate((imgs,v,h))

    return more_images

#VGG19 Model
def getVggModel():
    base_model = VGG16(weights = 'imagenet', include_top=False,
                        input_shape = (75, 75, 3), classes=1)

    x = base_model.get_layer('block5_pool').output
    x = GlobalMaxPooling2D()(x)
    x = Dense(512, activation = 'relu', name = 'fc1')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation = 'relu', name = 'fc2')(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation = 'relu', name = 'fc3')(x)
    x = Dropout(0.2)(x)

    predictions = Dense(1, activation = 'sigmoid')(x)

    model = Model(input = base_model.input, output = predictions)
    parallel_model = multi_gpu_model(model, gpus=3)
    parallel_model.compile(loss ='binary_crossentropy',
                    optimizer = 'Adam',
                    metrics = ['accuracy'])
    return parallel_model

#Base CV Structure
def get_callbacks():
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint(filepath, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
    return [earlyStopping, mcp_save, reduce_lr_loss]

#Using K-fold Cross Validation.
# def myBaseCrossTrain(X_train, target_train):
#     folds = list(StratifiedKFold(n_splits = 4, shuffle = True, random_state = 16).split(X_train, target_train))
#     y_test_pred_log = 0
#     y_valid_pred_log = 0.0*target_train
#     for j, (train_idx, test_idx) in enumerate(folds):
#         print('\nFOLD = ',j)
#
#         X_train_cv = X_train[train_idx]
#         y_train_cv = target_train[train_idx]
#         X_holdout = X_train[test_idx]
#         Y_holdout= target_train[test_idx]
#         file_path = "%s_model_weights.hdf5"%j
#
#         callbacks = get_callbacks(filepath = file_path, patience = 5)
#         galaxyModel = getVggModel()
#
#         galaxyModel.fit(X_train_cv, y_train_cv,
#                         batch_size = 270,
#                         epochs = 100,
#                         verbose = 1,
#                         validation_data = (X_holdout, Y_holdout),
#                         callbacks = callbacks)
#
#         #Getting the Best Model
#         galaxyModel.load_weights(filepath = file_path)
#
#         #Getting Training Score
#         print('\n')
#         score = galaxyModel.evaluate(X_train_cv, y_train_cv)
#         print('Train loss:', score[0])
#         print('Train accuracy:', score[1])
#
#         #Getting Test Score
#         print('\n')
#         score = galaxyModel.evaluate(X_holdout, Y_holdout)
#         print('Test loss:', score[0])
#         print('Test accuracy:', score[1])
#
#         #Getting validation Score.
#         pred_valid = galaxyModel.predict(X_holdout)
#         y_valid_pred_log[test_idx] = pred_valid.reshape(pred_valid.shape[0])
#
#         #Getting Test Scores
#         temp_test = galaxyModel.predict(X_test)
#         y_test_pred_log += temp_test.reshape(temp_test.shape[0])
#
#     y_test_pred_log=y_test_pred_log/4
#     print('\nLog Loss Validation= ',log_loss(target_train, y_valid_pred_log))
#     return y_test_pred_log

def main():
    #read in data
    df_train = pd.read_json('../../Data/train.json')

    #get scaled images as one big array
    Xtrain = get_scaled_imgs(df_train)
    #get the labels
    Ytrain = np.array(df_train['is_iceberg'])

    #replace null values with 0
    df_train.inc_angle = df_train.inc_angle.replace('na',0)
    #filter out the images where there was no incidence angle values
    idx_tr = np.where(df_train.inc_angle>0)
    Ytrain = Ytrain[idx_tr[0]]
    Xtrain = Xtrain[idx_tr[0],...]

    #compute more images by flipping, rotating, etc. using openCV
    Xtr_more = get_more_images(Xtrain)
    #triple-replicate the labels to match the 3 new generated streams of rotated images
    Ytr_more = np.concatenate((Ytrain,Ytrain,Ytrain))

    #initialize and evaluate model
    model = getVggModel()
    model.summary()

    #get callbacks
    callbacks = get_callbacks()

    history = model.fit(Xtr_more, Ytr_more,
                             batch_size = 270,
                             epochs = 100,
                             verbose = 1,
                             validation_split = 0.25,
                             callbacks = callbacks)

    print(history.history.keys())

    score = model.evaluate(Xtrain, Ytrain, verbose=1)
    print('Train score:', score[0])
    print('Train accuracy:', score[1])


    df_test = pd.read_json('../../Data/test.json')
    df_test.inc_angle = df_test.inc_angle.replace('na',0)
    Xtest = (get_scaled_imgs(df_test))
    pred_test = model.predict(Xtest)

    submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': pred_test.reshape((pred_test.shape[0]))})
    print(submission.head(10))

    submission.to_csv('VGG16_submission.csv', index=False)






if __name__ == "__main__":
    main()
