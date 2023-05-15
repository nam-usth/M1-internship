import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import sys
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# %% Get the first digit in string

def first_digit(s):
    return re.search(r"\d", s).start()

# %% Preparation of the dataset

def get_data_shape(model_name):
    if model_name == 'AlexNet':
        img_shape = (227, 227)
    if model_name == 'InceptionV3':
        img_shape = (299, 299)
    if model_name == 'MobileNetV2':
        img_shape = (224, 224)
    if model_name == 'ResNet50':
        img_shape = (224, 224)
    return img_shape

def data_preparation(chosen_model):
    X_test, y_test = [], []

    # Hard-coded data_dir
    
    data_dir = r'D:/Frames/Data/'
    test_dir = data_dir + 'Test/'

    # NOTE: Class 0 = NonPorn, Class 1 = Porn
    # We can use the one-hot vector technique to represent the video content
    # For example: [1. 0.] means NonPorn, [0. 1.] means Porn
          
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
        
    X_test = np.asarray(X_test)
    y_test = to_categorical(y_test)
    
    target_shape = (-1,) + get_data_shape(chosen_model) + (3,)
    
    X_test = X_test.reshape(target_shape)

    return X_test, y_test

# %% Main function

if __name__ == "__main__":
    # Choose a model's name
    # Available models: ['AlexNet', 'InceptionV3', 'MobileNetV2', 'ResNet50']
    
    avail_model = ['AlexNet', 'InceptionV3', 'MobileNetV2', 'ResNet50']
    # avail_model = ['AlexNet', 'EfficientNetB3', 'EfficientNetB4', 'InceptionV3', 'MobileNetV2', 'ResNet50']
    
    for i in range(0, len(avail_model)):
        chosen_model = avail_model[i]
        
        # Getting data ready
        X_test, y_test = data_preparation(chosen_model)
        
        # Loading a saved model 
        if chosen_model == 'InceptionV3':
            h5_file_name = 'bad_content_' + chosen_model + '-89-epochs' + '.h5'
        else:    
            h5_file_name = 'bad_content_' + chosen_model + '.h5'
            
        model = load_model(h5_file_name)
    
        # Predict a cropped image/Calculate recall, precision and F1 score
        y_pred = model.predict(X_test)
        
        # Calculate precision, recall
        ground_truth = np.argmax(y_test, axis=1)
        predicted = np.argmax(np.rint(y_pred), 1)
        
        # Choose an average_mode
        # Available modes: ['binary', 'macro', 'micro', 'weighted']
        mode = 'weighted'
        
        Acc = accuracy_score(ground_truth, predicted)
        P = precision_score(ground_truth, predicted, average=mode)
        R = recall_score(ground_truth, predicted, average=mode)
        F1 = f1_score(ground_truth, predicted, average=mode)
        
        print(chosen_model + ":\n Acc = " + str("{:.2f}".format(Acc*100)) + "% | P = " + str("{:.2f}".format(P)) + " | R = "  + str("{:.2f}".format(R)) + " | F1 = " + str("{:.2f}".format(F1)))