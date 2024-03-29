######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/15/18
# Description: 
# This program uses a TensorFlow-trained neural network to perform object detection.
# It loads the classifier and uses it to perform object detection on an image.
# It draws boxes, scores, and labels around the objects of interest in the image.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import time
import sys

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def preparation(PATH_TO_CKPT):    
    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    
        sess = tf.Session(graph=detection_graph)
    
    # Define input and output tensors (i.e. data) for the object detection classifier
    
    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    
    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    
    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    
    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    return sess, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections
    
def detect_and_crop(image, fname, category_index, sess, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections):   
    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0)
    
    # Get image height, width
    height = image.shape[0]
    width = image.shape[1]
    
    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})
    
    # Crop images (aka 'Region of Interest' - ROI) from the detection
    for i in range(0,int(num[0])-1):
        ymin = int(boxes[0][i][0]*height)
        xmin = int(boxes[0][i][1]*width)
        ymax = int(boxes[0][i][2]*height)
        xmax = int(boxes[0][i][3]*width)
        
        ROI = image[ymin:ymax, xmin:xmax]
        
        cv2.imwrite('crop ' + fname + ' %d.jpg'%(i,), ROI)
    
    # Re-predict to calculate the time (since the first run is slow due to the model loading)
    start_time = time.time()
    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})
    end_time = time.time()
    
    #print("Detection time: ", end_time - start_time)
    
    # Draw the results of the detection (aka 'visulaize the results')
    
    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=1,
        min_score_thresh=0.30)
    
    # All the results have been drawn on image. Now display the image.
    # cv2.imshow('Object detector', image)
    
    # Press any key to close the image
    cv2.waitKey(0)
    
    # Clean up
    cv2.destroyAllWindows()
