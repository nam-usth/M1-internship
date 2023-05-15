import cv2
from key_frame_extraction_batch import key_frame_extraction, key_frame_plot
import matplotlib.pyplot as plt
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
#from object_detection_image import preparation, detect_and_crop
from object_detection_image_merge_bounding_box import preparation, detect_and_crop
import os
from tensorflow.keras.models import load_model
import time
import csv

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

if __name__ == "__main__":
    # Load classification Neural Network weight first
    avail_model = ['AlexNet', 'InceptionV3', 'MobileNetV2', 'ResNet50']
      
    chosen_model = avail_model[3]
    
    if chosen_model == 'InceptionV3':
        h5_file_name = 'bad_content_' + chosen_model + '-89-epochs' + '.h5'
    else:
        h5_file_name = 'bad_content_' + chosen_model + '.h5'
            
    model = load_model(h5_file_name)
    
    target_shape = (-1,) + get_data_shape(chosen_model) + (3,)
    
    # Prepare Tensorflow model one time only
    MODEL_NAME = 'inference_graph'
    CWD_PATH = os.getcwd()
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph-faster_rcnn.pb')
    
    PATH_TO_LABELS = os.path.join(CWD_PATH,'training','classes.pbtxt')
    NUM_CLASSES = 9

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    sess, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections = preparation(PATH_TO_CKPT)
    
    data_dir = r'D:/Testing/Negative'
            
    # Extract keyframes
    #file_dir = r'D:/USTH_Master/BC-Application/vPorn2-cut-2.mp4'
    #file_dir = r'D:/USTH_Master/BC-Application/Test_10_second_360p.mp4'
    #file_dir = r'D:/USTH_Master/BC-Application/Test_10_second_480p.mp4'
    #file_dir = r'D:/USTH_Master/BC-Application/Test_15_second_360p.mp4'
    #file_dir = r'D:/USTH_Master/BC-Application/Test_15_second_480p.mp4'
    #file_dir = r'D:/USTH_Master/BC-Application/Test_20_second_360p.mp4'
    #file_dir = r'D:/USTH_Master/BC-Application/Test_20_second_480p.mp4'
    #file_dir = r'D:/USTH_Master/BC-Application/Test_25_second_360p.mp4'
    #file_dir = r'D:/USTH_Master/BC-Application/Test_25_second_480p.mp4'
    #file_dir = r'D:/USTH_Master/BC-Application/Test_30_second_360p.mp4'
    #file_dir = r'D:/USTH_Master/BC-Application/Test_30_second_480p.mp4'
    #file_dir = r'D:/USTH_Master/BC-Application/Test_35_second_360p.mp4'
    #file_dir = r'D:/USTH_Master/BC-Application/Test_35_second_480p.mp4'
    #file_dir = r'D:/USTH_Master/BC-Application/Test_40_second_360p.mp4'
    #file_dir = r'D:/USTH_Master/BC-Application/Test_40_second_480p.mp4'
    #file_dir = r'D:/USTH_Master/BC-Application/Test_45_second_360p.mp4'
    #file_dir = r'D:/USTH_Master/BC-Application/Test_45_second_480p.mp4'
    #file_dir = r'D:/USTH_Master/BC-Application/Test_1_minute_360p.mp4'
    
    N = 5

    dc = 3.000001 
    
    total_time, total_det_time, y_pred_overall = [], [], []

    for file in os.listdir(data_dir):
        if file.endswith('.avi'):
            file_dir = os.path.join(data_dir, file)

            start_time = time.time()
        
            key_frame = key_frame_extraction(file_dir, N, dc)
            
            key_frame_plot(file_dir, key_frame)
        
            end_time = time.time()
            
            KFE_time = end_time - start_time
            #print("KFE time: ", end_time - start_time)

            # Voter
            voteNonPorn = 0
            votePorn = 0
            threshold = 0.3
               
            # Start the working pipeline! Set timer
            start_time = time.time()
        
            # Detect and Crop human object [Tensorflow API]
            filelist = [ f for f in os.listdir(os.getcwd()) if f.endswith(".jpg") ]
            for f in filelist:
                img = cv2.imread(f)
                f_name = os.path.splitext(f)[0]
                digit = [int(s) for s in f_name.split() if s.isdigit()]
                detect_and_crop(img, str(digit[0]), category_index, sess, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections)
                
            # Do classification on cropped images
            filelist = [ f for f in os.listdir(os.getcwd()) if f.startswith("crop") ]
            
            try:
                for f in filelist:
                    X_test = [] 
                    cropped_img = cv2.imread(f)
                    
                    # Resize cropped image
                    cropped_img = cv2.resize(cropped_img, get_data_shape(chosen_model))
                    
                    X_test.append(cropped_img)
                    X_test = np.asarray(X_test)
                    
                    X_test = X_test.reshape(target_shape)
                    
                    y_pred = model.predict(X_test)
                
                    predicted = np.argmax(np.rint(y_pred), 1)
                    
                    if predicted == 0:
                        voteNonPorn += 1
                        #print('Negative')
                    else:
                        votePorn += 1
                        #print('Positive')
                
                if (voteNonPorn + votePorn) == 0:
                    continue
                else:
                    if votePorn/(voteNonPorn + votePorn) >= threshold:
                        #print('Classified as Adult video')
                        y_pred_overall.append(1)
                    else:
                        #print('Classified as Normal video')
                        y_pred_overall.append(0)
                        
                    end_time = time.time()
                    
                    det_time = end_time - start_time
                    
                    total_det_time.append(det_time)
                    total_time.append(KFE_time + det_time)
                    #print("Processing time: ", end_time - start_time)
                
                # Clean the Frame folder
            except:
                pass
    '''
    rows = zip(total_time, total_det_time, y_pred_overall)
    with open(r'D:/benchmark-positive-faster_rcnn_resnet50_coco-resnet50.csv', "w") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
    '''