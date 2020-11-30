import tensorflow as tf
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import cv2

PATH_TO_SAVED_MODEL = "export/saved_model"
print('Loading model...', end='')

detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

print('model loaded!')
category_index = label_map_util.create_category_index_from_labelmap("label_map.pbtxt",use_display_name=True)
image_path = "test.jpg"

def load_image_into_numpy_array(path):
    return np.array(Image.open(path))



image_np = load_image_into_numpy_array(image_path)
print(image_np)
input_tensor = tf.convert_to_tensor(image_np)

input_tensor = input_tensor[tf.newaxis, ...]
detections = detect_fn(input_tensor)
num_detections = int(detections.pop('num_detections'))
detections = {key:value[0,:num_detections].numpy() for key,value in detections.items()}
print(detection)
detections['num_detections'] = num_detections
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
image_np_with_detections = image_np.copy()
viz_utils.visualize_boxes_and_labels_on_image_array( image_np_with_detections, detections['detection_boxes'], detections['detection_classes'],
                                                    detections['detection_scores'], category_index,
                                                    use_normalized_coordinates=True, max_boxes_to_draw=100, min_score_thresh=.5,  agnostic_mode=False)
print(image_np_with_detections)
cv2.imwrite('final.jpg', image_np_with_detections)
# %matplotlib inline
plt.figure()
plt.imshow(image_np_with_detections)
print('Done')
plt.show()
