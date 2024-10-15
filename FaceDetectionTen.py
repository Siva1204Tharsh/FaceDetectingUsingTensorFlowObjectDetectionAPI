# importing the necessary libraries
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

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#download the pre-trained model from the TensorFlow Object Detection API
MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Download the model

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)

# Extract the model

tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())

# Load the model

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# Define the labels

PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

category_index = {}
with open(PATH_TO_LABELS, 'r') as f:
  for line in f:
    if 'id:' in line:
          id = int(line.split(":")[1])
    elif 'display_name' in line:
        name = line.split(":")[1].strip().replace("'", "")
        category_index[id] = {'name': name}

# Define the detect_objects function
def detect_objects(image_path, graph, category_index, min_score_thresh=.5):
    # Load the image
    image = cv2.imread(image_path)
    image_np = np.array(image)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # Detect faces using the Haar Cascade classifier
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Iterate over each face and draw a rectangle around it
    for (x,y,w,h) in faces:
        cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),2)

    # Display the image
    cv2.imshow('Face Detection', image_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Define the path to the image
IMAGE_PATH = 'test_image.jpg'

# Detect faces in the image
detect_objects(IMAGE_PATH, detection_graph, category_index, min_score_thresh=.5)