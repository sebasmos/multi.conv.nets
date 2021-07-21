
"""
@author: Sebastian Cajas
 
 Modified  u-net version for multi-class image segmentation 
"""
import tensorflow as tf
import sys
sys.path.insert(0,'..') # https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder 
from loss_functions.loss import jaccard_coefficient

from keras.models import *
from keras.layers import *
from keras.optimizers import *


def seg_66():

  model = Sequential()

  model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Conv2D(64, (3, 3), padding='same'))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(64, (3, 3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))

  model.add(Conv2D(64, (3, 3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(128, (3, 3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(256, (3, 3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(128))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.25))

  model.add(Dense(128))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.25))


  model.add(Dense(num_classes))
  model.add(Activation('softmax'))

  return model



def seg_55():

  model = Sequential()

  model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Conv2D(64, (3, 3), padding='same'))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(128, (3, 3), padding='same', input_shape=input_shape))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Conv2D(128, (3, 3), padding='same'))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
  model.add(Dropout(0.5))

  model.add(Conv2D(256, (3, 3), padding='same', input_shape=input_shape))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Conv2D(256, (3, 3), padding='same'))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.5))

  #model.add(Conv2D(64, (3, 3), padding='same'))
  #model.add(BatchNormalization())
  #model.add(Activation('relu'))
  '''
  model.add(Conv2D(64, (3, 3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(128, (3, 3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(256, (3, 3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  '''
  model.add(Flatten())
  model.add(Dense(128))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.25))

  model.add(Dense(128))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.25))


  model.add(Dense(num_classes))
  model.add(Activation('softmax'))

  return model


# Adapting contractive mode from previous tf versions 
def seg_():

  model = Sequential()

  model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Conv2D(64, (3, 3), padding='same'))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(128, (3, 3), padding='same'))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Conv2D(128, (3, 3), padding='same'))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
  model.add(Dropout(0.5))

  model.add(Conv2D(256, (3, 3), padding='same'))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Conv2D(256, (3, 3), padding='same'))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
  model.add(Dropout(0.5))

  #Decoder

  model.add(UpSampling2D(size=(2,2)))
  model.add(Conv2D(128, (3, 3), padding='same'))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Conv2D(128, (3, 3), padding='same'))
  model.add(Activation('relu'))
  model.add(BatchNormalization())

  model.add(Flatten())
  model.add(Dense(128))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.25))

  model.add(Dense(128))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.25))


  model.add(Dense(num_classes))
  model.add(Activation('softmax'))

  return model
