from tensorflow.python.keras.backend import dtype
from models.unet_1 import unet_truncated
import os
import tensorflow as tf
import numpy as np

from tqdm import tqdm 

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import random


img_height = 128
img_width = 128
ch = 3

path_train = "/home/sebasmos/Documentos/datasets/cells/stage1_train/"
path_test = "/home/sebasmos/Documentos/datasets/cells/stage1_test/"

train_ids = next(os.walk(path_train))[1]
test_ids = next(os.walk(path_test))[1]

X_train = np.zeros((len(train_ids),img_height,img_width,ch), dtype=np.uint8)
Y_train = np.zeros((len(train_ids),img_height,img_width,ch), dtype=np.bool_)

print("X_train.shape: ", X_train.shape)
print("X_train.shape: ", Y_train.shape)

print('Resizing training images and masks')

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
    path = path_train + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:ch]  
    img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
    X_train[n] = img  #Fill empty X_train with values from img
    mask = np.zeros((img_height, img_width, 1), dtype=np.bool_)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (img_height, img_width), mode='constant',  
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)  
            
    Y_train[n] = mask   

# test images
X_test = np.zeros((len(test_ids), img_height, img_width, ch), dtype=np.uint8)
sizes_test = []
print('Resizing test images') 
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = path_test + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:ch]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
    X_test[n] = img

print('Done!')

image_x = random.randint(0, len(train_ids))
imshow(X_train[image_x])
plt.show()
imshow(np.squeeze(Y_train[image_x]))
plt.show()

model = unet_truncated(img_height, img_width, ch)

#Modelcheckpoint
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_ch.h5', verbose=1, save_best_only=True)

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]

results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=25, callbacks=callbacks)
