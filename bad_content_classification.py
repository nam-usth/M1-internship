import cv2
import datetime
import efficientnet.tfkeras as efn
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from sklearn.metrics import accuracy_score
import sys
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, Input, Lambda, MaxPooling2D, ReLU, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
import zipfile

# %% Get the first digit in string

def first_digit(s):
    return re.search(r"\d", s).start()

# %% Preparation of the dataset

def get_data_shape(model_name):
    if model_name == 'AlexNet':
        img_shape = (227, 227)
    if model_name == 'EfficientNetB3':
        img_shape = (300, 300)
    if model_name == 'EfficientNetB4':
        img_shape = (380, 380)
    if model_name == 'InceptionV3':
        img_shape = (299, 299)
    if model_name == 'MobileNetV2':
        img_shape = (224, 224)
    if model_name == 'ResNet50':
        img_shape = (224, 224)
    return img_shape

def data_preparation(chosen_model):
    X_train, X_test, y_train, y_test = [], [], [], []

    # Hard-coded data_dir
    
    data_dir = r'D:/Frames/Data/'
    train_dir = data_dir + 'Train/'
    test_dir = data_dir + 'Test/'

    # NOTE: Class 0 = NonPorn, Class 1 = Porn
    # We can use the one-hot vector technique to represent the video content
    # For example: [1. 0.] means NonPorn, [0. 1.] means Porn
    
    for file in os.listdir(train_dir):
        if file.endswith('.jpg'):
            img = cv2.imread(train_dir + file)
            img = cv2.resize(img, get_data_shape(chosen_model))
        
            s = file[1:first_digit(file)]
        
            X_train.append(img)
        
            if (s == 'NonPorn'):
                y_train.append(0)
            else:
                y_train.append(1)
        
    for file in os.listdir(test_dir):
        if file.endswith('.jpg'):
            img = cv2.imread(test_dir + file)
            img = cv2.resize(img, get_data_shape(chosen_model))
        
            s = file[1:first_digit(file)]
        
            X_test.append(img) 
        
            if (s == 'NonPorn'):
                y_test.append(0)
            else:
                y_test.append(1)
        
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    target_shape = (-1,) + get_data_shape(chosen_model) + (3,)
    
    X_train = X_train.reshape(target_shape)
    X_test = X_test.reshape(target_shape)

    return X_train, X_test, y_train, y_test

# %% Define base model for transfer learning

def create_model(chosen_model):
    model_input = Input(shape=get_data_shape(chosen_model) + (3,))
    
    if chosen_model == 'AlexNet':
        x = Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), name="conv1", activation="relu")(model_input)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="pool1")(x)
        x = BatchNormalization()(x)

        x = ZeroPadding2D((2, 2))(x)
        con2_split1 = Lambda(lambda z: z[:,:,:,:48])(x)
        con2_split2 = Lambda(lambda z: z[:,:,:,48:])(x)
        x = Concatenate(axis=1)([con2_split1, con2_split2])
        x = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), name="conv2", activation="relu")(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="pool2")(x)
        x = BatchNormalization()(x)

        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), name="conv3", activation="relu")(x)
        
        x = ZeroPadding2D((1, 1))(x)
        con4_split1 = Lambda(lambda z: z[:,:,:,:192])(x)
        con4_split2 = Lambda(lambda z: z[:,:,:,192:])(x)
        x = Concatenate(axis=1)([con4_split1, con4_split2])
        x = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), name="conv4", activation="relu")(x)

        x = ZeroPadding2D((1, 1))(x)
        con5_split1 = Lambda(lambda z: z[:,:,:,:192])(x)
        con5_split2 = Lambda(lambda z: z[:,:,:,192:])(x)
        x = Concatenate(axis=1)([con5_split1, con5_split2])
        x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), name="conv5", activation="relu")(x)
        
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="pool5")(x)
        x = Flatten()(x)
        
        x = Dense(4096, activation='relu', name="fc6")(x)
        x = Dropout(0.5, name="droupout6")(x)
        x = Dense(4096, activation='relu', name="fc7")(x)
        x = Dropout(0.5, name="droupout7")(x)
        x = Dense(1000, activation='softmax', name="fc8")(x)

        base_model = Model(inputs=model_input, outputs=x)
        base_model.load_weights("alexnet_weights.h5", by_name=True)
        
    else:
        if chosen_model == 'EfficientNetB3':
            pretrained_model = efn.EfficientNetB3(weights='imagenet', include_top=False)
        
        if chosen_model == 'EfficientNetB4':
            pretrained_model = efn.EfficientNetB4(weights='imagenet', include_top=False)
            
        if chosen_model == 'InceptionV3':
            pretrained_model = InceptionV3(weights='imagenet', include_top=False)
            
        if chosen_model == 'MobileNetV2':
            pretrained_model = MobileNetV2(weights='imagenet', include_top=False)
        
        if chosen_model == 'ResNet50':
            pretrained_model = ResNet50(weights='imagenet', include_top=False)

        x = pretrained_model(model_input)
        if 'EfficientNet' not in chosen_model:
            x = GlobalAveragePooling2D()(x)
            x = Dense(512)(x) 

        base_model = Model(inputs=model_input, outputs=x)
        
    return base_model

# %% Main function

if __name__ == "__main__":
    # Choose a model's name
    # Available models: ['AlexNet', 'EfficientNetB3', 'EfficientNetB4', 'InceptionV3', 'MobileNetV2', 'ResNet50']
    
    avail_model = ['AlexNet', 'EfficientNetB3', 'EfficientNetB4', 'InceptionV3', 'MobileNetV2', 'ResNet50']
    
    for i in range(2, 3): # range(0, len(avail_model)):
        chosen_model = avail_model[i]
        
        # Getting data ready
        X_train, X_test, y_train, y_test = data_preparation(chosen_model)
        
        # Initialize a base_model
        base_model = create_model(chosen_model)
        
        # Freeze all layers of the base_model
        base_model.trainable = False
        
        # Do Transfer learning
        # 2 classes: NonPorn, Porn --> The last layer is Dense(2) to classify the input image
        
        if chosen_model == 'AlexNet':
            model = Sequential([base_model, Dense(2048), Dense(2, activation='softmax', batch_size=128)])
        elif 'EfficientNet' in chosen_model:
            model = Sequential([base_model, GlobalAveragePooling2D(), BatchNormalization(), Dropout(0.2), Dense(2, activation='softmax')])
        else:
            model = Sequential([base_model, BatchNormalization(), ReLU(), Dense(2, activation='softmax', batch_size=128)])
        
        model.summary()
        
        # Compile and train the model 
        if 'EfficientNet' not in chosen_model:
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            
        tensorboard_callback = TensorBoard(log_dir="logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
        model.fit(X_train, y_train, epochs=100, validation_split=0.1, callbacks=[tensorboard_callback])
    
        # Saving trained model 
        h5_file_name = 'bad_content_' + chosen_model + '.h5'
        model.save(h5_file_name)
    
        # Predict a cropped image/Calculate recall, precision and F1 score
        y_pred = model.predict(X_test)    