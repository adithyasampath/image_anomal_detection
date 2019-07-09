import numpy as np
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Flatten, Activation
from keras.models import Sequential, Model
from keras import applications
from vgg_anomaly import AnomalyPredictor
import os
import cv2
import csv

if __name__ == "__main__":
    anomaly = AnomalyPredictor()
    anomaly.create_model()
    anomaly.load_weights()
    anomaly.get_threshold()
    for img in anomaly.test_image_list:
        print(
            "Is %s an anomaly: " % img,
            anomaly.IsImageHasAnomaly(
                os.path.join(anomaly.test_data_dir, "cogs", img), img))
