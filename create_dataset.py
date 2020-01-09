import keras
import datetime
# import keras_retinanet
from keras_retinanet.models.resnet import ResNetBackbone
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet import losses
# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
# import anomaly detection module
from deep_conv_anomaly import AnomalyPredictor
# from tf_serving_client import DetectObject
import tensorflow as tf

# set tf backend to allow memory to grow, instead of claiming everything
def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


# keras.backend.tensorflow_backend.set_session(get_session())
# tf_server = 'localhost:8500'
# model_name = 'retinanet'
# object_detect = DetectObject(tf_server,model_name)


num_classes = 80
# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = os.path.join('weights','resnet50_csv_anomaly_6.h5')  # os.path.join('..', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')

# load retinanet modeli
model = ResNetBackbone('resnet50').retinanet(num_classes)
model.load_weights(model_path, by_name=True, skip_mismatch=True)

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
#model = models.convert_model(model)

model.compile(
    loss={
        'regression': losses.smooth_l1(),
        'classification': losses.focal()
    },
    optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001))

model = models.convert_model(model)


print("Detection Model Loaded and compiled")
#print(model.summary())
anomaly = AnomalyPredictor()
anomaly.create_model()
anomaly.load_weights()
anomaly.get_threshold()

print("Anomaly detection model loaded and compiled")
# load label to names mapping for visualization purposes
labels_to_names = {
    0: 'person',
    1: 'gear',
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
    90: 'gear'
}

initial_image_dir = os.path.dirname(__file__)
initial_image_dir = os.path.join(initial_image_dir, "data")
path = os.path.join(initial_image_dir, "inputs")
crop_save_path_train = os.path.join(initial_image_dir, 'train','cogs')
crop_save_path_valid = os.path.join(initial_image_dir, 'valid','cogs')
crop_save_path_test = os.path.join(initial_image_dir, "crop")
det_save_path = os.path.join(initial_image_dir, "detections")

for file in os.listdir(path):
    image = read_image_bgr(os.path.join(path,file))
    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    
    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    boxes, scores, labels=  model.predict(np.expand_dims(image, axis=0)) #object_detect.do_inference(image)
    print("processing time: ", time.time() - start)
    
    # correct for image scale
    boxes /= scale
    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.9:
            break

        color = label_color(label)
        b = box.astype(int)
        if labels_to_names[label] == "gear":
            crop = draw[b[1]:b[3],b[0]:b[2]].copy()
            crop_file = "cropped_"+file
            filepath_1 = os.path.join(crop_save_path_train,crop_file)
            filepath_2 = os.path.join(crop_save_path_valid,crop_file)
            crop = cv2.resize(crop,(256,256), interpolation = cv2.INTER_AREA)
            cv2.imwrite(filepath_1, crop)
            cv2.imwrite(filepath_2, crop)
           

