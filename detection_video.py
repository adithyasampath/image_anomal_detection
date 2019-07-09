import os
import sys
import cv2
import time
import math

import keras

# import keras_retinanet
from keras_retinanet.models.resnet import ResNetBackbone
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet import losses
import tensorflow as tf
import numpy as np

input_file = sys.argv[1]
out_path = sys.argv[2]
class_to_detect = sys.argv[3]


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

keras.backend.tensorflow_backend.set_session(get_session())
print("Loading weights and building model")
num_classes = 80
model_path = os.path.join('weights','resnet50_csv_anomaly_6.h5') 
model = ResNetBackbone('resnet50').retinanet(num_classes)
model.load_weights(model_path, by_name=True, skip_mismatch=True)
model.compile(
    loss={
        'regression': losses.smooth_l1(),
        'classification': losses.focal()
    },
    optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001))

model = models.convert_model(model)
print("Model Loaded and compiled")

initial_image_dir = os.path.dirname(__file__)
initial_image_dir = os.path.join(initial_image_dir, "data")
path = os.path.join(initial_image_dir, "inputs")
crop_save_path = os.path.join(initial_image_dir, 'train','cogs')
crop_save_path_valid = os.path.join(initial_image_dir, 'valid','cogs')
det_save_path = os.path.join(initial_image_dir, "detections")
# model = keras.models.load_model(model_path, custom_objects=custom_objects)
# labels_to_names = {0: "dont_know", 1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "trafficlight", 11: "firehydrant", 12: "streetsign", 13: "stopsign", 14: "parkingmeter", 15: "bench", 16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe", 26: "hat", 27: "backpack", 28: "umbrella", 29: "shoe", 30: "eyeglasses", 31: "handbag", 32: "tie", 33: "suitcase", 34: "frisbee", 35: "skis", 36: "snowboard", 37: "sportsball", 38: "kite", 39: "baseballbat", 40: "baseballglove", 41: "skateboard", 42: "surfboard", 43: "tennisracket", 44: "bottle", 45: "plate", 46: "wineglass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl", 52: "banana", 53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot", 58: "hotdog", 59: "pizza", 60: "donut", 61: "cake", 62: "chair", 63: "couch", 64: "pottedplant", 65: "bed", 66: "mirror", 67: "diningtable", 68: "window", 69: "desk", 70: "toilet", 71: "door", 72: "tv", 73: "laptop", 74: "mouse", 75: "keyboard", 76: "cellphone", 77: "microwave", 78: "oven", 79: "toaster", 80: "sink", 81: "refrigerator", 82: "blender", 83: "book", 84: "clock", 85: "clock", 86: "vase", 87: "scissors", 88: "teddybear", 89: "hairdrier", 90: "toothbrush", 91: "hairbrush"}
labels_to_names = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair drier',
    79: 'toothbrush',
    90: 'screw'
}

frame_skip = 1
print("Frame skip = %s" %(frame_skip))
flag = 0
proc_frames = 0
print("Starting Detection")
print("Class to detect = %s" %class_to_detect)
cap = cv2.VideoCapture(input_file)
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc('H','2','6','4')
video_out = out_path
out = cv2.VideoWriter(video_out,fourcc, fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

if fps>10.0:
    frame_skip = int(math.ceil(fps/10))
    print("Skipping frames : ",frame_skip)

else:
    frame_skip = 1
start_time1 = time.time()
print("start time = %s" %(start_time1))
while(cap.isOpened()):
    if flag % frame_skip == 0:
        ret, image = cap.read()
        if ret == False:
            break
        # print("detection")    
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image)

        boxes, scores, labels = model.predict(np.expand_dims(image, axis=0))
        boxes /= scale
        car_boxes = []
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
            if score < 0.9:
                break
            color = label_color(label)

            b = box.astype(int)
            if labels_to_names[label] == class_to_detect:
                crop = draw[b[1]:b[3],b[0]:b[2]].copy()
                crop = cv2.resize(crop, (1050,1050), interpolation = cv2.INTER_AREA)
                draw_box(draw, b, color=color)
                caption = "{} {:.3f}".format(labels_to_names[label], score)
                draw_caption(draw, b, caption)
                detection_file = "processed_"+file
                crop_file = "cropped_"+file
                cv2.imwrite(os.path.join(det_save_path,detection_file), draw)
                cv2.imwrite(os.path.join(crop_save_path,crop_file), crop)
                cv2.imwrite(os.path.join(crop_save_path_valid,crop_file), crop)
                flag = flag + 1

time_taken = time.time() - start_time1
print("Time taken to process = %s " %(time_taken))
print("Total number of frames processed = %s" %(proc_frames))
cap.release()
out.release()
print("Output path = %s " %(video_out))


