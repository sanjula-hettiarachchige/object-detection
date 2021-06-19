import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
 
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
 
import cv2
cap = cv2.VideoCapture(1)

import sys
sys.path.append(r'C:\Users\Ktm Sam\Desktop\python\tensorflow\object detection\models\research\object_detection') # point to your tensorflow dir
sys.path.append(r'C:\Users\Ktm Sam\Desktop\python\tensorflow\object detection\models\research\slim') # point ot your slim dir
sys.path.append("..")
 
from utils import label_map_util
 
from utils import visualization_utils as vis_util
 


#Defines the model that should be downloaded as a pre-requisite
model_name = 'ssd_mobilenet_v1_coco_11_06_2017'
model_file = model_name + '.tar.gz'
download_base = "http://download.tensorflow.org/models/object_detection/"
 
# Path to frozen detection graph. This is the actual model that is used for the object detection.
path_to_ckpt = model_name + '/frozen_inference_graph.pb'
 
# List of the strings that is used to add correct label for each box.

path_to_labels = (r'C:\Users\Ktm Sam\Desktop\python\tensorflow\object detection\models\research\object_detection\data\mscoco_complete_label_map.pbtxt')
 

num_classes = 90

#Imports the graph which will be used for the object detection
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
 
label_map = label_map_util.load_labelmap(path_to_labels)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Retrieves the name of the object detected and calculates the relative position 
def get_class_name(data,coordinate_array):
   output = open("detected_classes.txt","w") # Opens a text file to save the names of detected classes
   classes = []
   for cl in data:
      classes.append(cl['name'])
   pan_array = []
   for obj in coordinate_array:
      pos = ((obj[2]+obj[3])/2)/640 #works out relative position of object (640 is width of the video source)
      if pos >0.5:
        pan=(pos-0.5)*2
        pan=pan/(0.65) #calculates relative pan according to screen size
      else:
        pan=(0.5-pos)*-2
        pan=pan/(0.65)
      if pan>=1:
        pan=1
      elif pan<=-1:
        pan=-1
      pan_array.append(pan)
    #Saves the objects and their relative position in the text file
    if len(pan_array)==len(classes):
        output.write(str(classes)[1:-1]+"\n")
         
         output.write(str(pan_array)[1:-1]) 
   output.close()


frame = 0
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      ret, image_np = cap.read()
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      array_dict = ([category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.5])
      #getting coordinates of boxes
      coordinates = vis_util.return_coordinates(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8,
                        min_score_thresh=0.80)
      get_class_name(array_dict, coordinates) # Saves the detected object and calculates the position
      cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
      if cv2.waitKey(25) & 0xFF == ord('q'):
        break
        cv2.destroyAllWindows()
        
output = open("detected_classes.txt","w")
output.close()
