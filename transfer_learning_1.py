from glob import glob

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Add pre-trained models 

from tensorflow.keras.applications.vgg16 import VGG16 as VGG
from tensorflow.keras.applications.densenet import DenseNet201 as DenseNet
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard


def models(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS,num_classes):

    use_vgg = True
    batch_size = 25

    # parameters for CNN
    if use_vgg:
        base_model = VGG(include_top=False,
                        weights=None,
                        input_shape = (img_rows, img_cols, img_bands),
                        #input_shape=(64, 64, 13)
                        )
    else:
        base_model = DenseNet(include_top=False,
                            weights=None,
                            input_shape = (img_rows, img_cols, img_bands))#input_shape=(64, 64, 13))
        
    # add a global spatial average pooling layer
    top_model = base_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    # or just flatten the layershttps://land.copernicus.eu/local/urban-atlas
    #    top_model = Flatten()(top_model)
    # let's add a fully-connected layer
    if use_vgg:
        # only in VGG19 a fully connected nn is added for classfication
        # DenseNet tends to overfitting if using additionally dense layers
        top_model = Dense(2048, activation='relu')(top_model)
        top_model = Dense(2048, activation='relu')(top_model)
    # and a logistic layer
    predictions = Dense(num_classes, activation='softmax')(top_model)
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    # print network structure
    model.summary()
    return model