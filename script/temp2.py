import numpy as np
import pandas as pd
import os
from skimage import transform
from skimage import io
from random import shuffle 

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from keras.preprocessing.image import ImageDataGenerator
from six.moves import range


nb_classes = 5
nb_epoch = 1
batch_size = 32

def split_index(row, batch_size):
    # split row to 6 processes for parallel processing
    res = []
    chunk = row/batch_size
    for i in range(1, chunk):
        res.append([(i-1)*batch_size, i*batch_size])
    res.append([batch_size*(chunk -1), row])
    return res

def data_gen(batch_size, fdir = '../input/processed/run-normal/train/'):
    labels = pd.read_csv('trainLabels.csv')
    folder = os.listdir(fdir)
    shuffle(folder)
    splits = split_index(labels.shape[0], batch_size)
    for idx in splits[:10]:
        X_train = np.zeros([idx[1] - idx[0], 3, 256, 256], dtype = 'float32')
        y_train = np.zeros(idx[1] - idx[0], dtype = 'uint8')
        i = 0
        for x in folder[idx[0]:idx[1]]:
            cur = io.imread(fdir + x)
            if len(cur.shape) != 3:
                continue
            cur = np.swapaxes(cur, 0, 2)
            X_train[i] = cur
            y_train[i] = int(labels.loc[labels.image == x[:-5],'level'])
            i += 1
        #Y_train = np_utils.to_categorical(y_train, nb_classes)
        X_train /= 255
        yield X_train, y_train

def gen_test(batch_size, fdir = '../input/processed/run-normal/test/'):
    folder = os.listdir(fdir)
    splits = split_index(len(folder), batch_size)
    for idx in splits:
        X_test = np.zeros([idx[1] - idx[0], 3, 256, 256], dtype = 'float32')
        i = 0
        for x in folder[idx[0]:idx[1]]:
            cur = io.imread(fdir + x)
            cur = np.swapaxes(cur, 0, 2)
            X_train[i] = cur
            i += 1
        X_train /= 255
        yield X_test


#y_train = y_train.reshape([1600,1])
#Y_train = np_utils.to_categorical(y_train, nb_classes)
if __name__ == "__main__":

    model = Sequential()

    model.add(Convolution2D(32, 3, 3, 3, border_mode='full')) 
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 32, 3, 3, border_mode='full')) 
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 64, 3, 3)) 
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 64, 3, 3, border_mode='full')) 
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 64, 3, 3)) 
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64*32*32, 512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512, 1))
    #model.add(Activation('linear'))

    # let's train the model using SGD + momentum (how original).
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer='rmsprop')

    print "finished model compilation"


    for e in range(nb_epoch):
            print('-'*40)
            print('Epoch', e)
            print('-'*40)
            print("Training...")
            # batch train with realtime data augmentation
            progbar = generic_utils.Progbar(61811)
            for X_batch, Y_batch in data_gen(batch_size):
                loss = model.train_on_batch(X_batch, Y_batch)
                progbar.add(X_batch.shape[0], values=[("train loss", loss)])