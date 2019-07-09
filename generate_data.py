import numpy as np
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Flatten, Activation, BatchNormalization
from keras.models import Sequential, Model
from keras import applications
import multiprocessing

import os
import cv2


class Generator:
    def __init__(self):
        self.initial_image_dir = os.path.dirname(__file__)
        self.initial_image_dir = os.path.join(self.initial_image_dir, "data")
        self.train_data_dir = os.path.join(self.initial_image_dir, 'train')
        self.train_save_to = os.path.join(self.train_data_dir, "cogs")
        self.image_list = os.listdir(os.path.join(self.train_data_dir, "cogs"))
        self.valid_data_dir = os.path.join(self.initial_image_dir, 'valid')
        self.valid_save_to = os.path.join(self.valid_data_dir, "cogs")
        self.valid_image_list = os.listdir(
            os.path.join(self.valid_data_dir, "cogs"))
        self.inital_image_count = 0
        self.valid_inital_image_count = 0
        self.datagen_without_mod = ImageDataGenerator()
        self.datagen_with_mod = ImageDataGenerator(
            rotation_range=180,
            width_shift_range=0.05,
            height_shift_range=0.05,
            # rescale=1. / 255,
            shear_range=0.05,
            zoom_range=0.05,
            horizontal_flip=False,
            fill_mode='nearest')

    def generate_images(self, datagen, mod):
        for img in self.image_list:
            if not os.path.isfile(os.path.join(self.train_save_to, img)):
                continue
            self.inital_image_count += 1
            img = load_img(os.path.join(self.train_save_to,
                                        img))  # this is a PIL image
            x = img_to_array(
                img)  # this is a Numpy array with shape (3, 150, 150)
            x = x.reshape(
                (1, ) +
                x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
            if mod is False:
                print("Generate additional No-Mod images for train in: ",
                      self.train_save_to)
            else:
                print("Generate additional Mod images for train in: ",
                      self.train_save_to)
            print("Number of images: ", len(os.listdir(self.train_save_to)))
            i = 0
            get_batch_size = lambda batch_size: 5 if mod else 2
            for _ in datagen.flow(
                    x,
                    batch_size=get_batch_size(mod),
                    save_to_dir=self.train_save_to,
                    save_prefix='sample',
                    save_format='jpeg'):
                i += 1
                if i > 5:
                    break

    def generate_valid_images(self, datagen, mod):
        for img in self.valid_image_list:
            if not os.path.isfile(os.path.join(self.valid_save_to, img)):
                continue
            self.valid_inital_image_count += 1
            img = load_img(os.path.join(self.valid_save_to,
                                        img))  # this is a PIL image
            x = img_to_array(
                img)  # this is a Numpy array with shape (3, 150, 150)
            x = x.reshape(
                (1, ) +
                x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
            if mod is False:
                print("Generate additional No-Mod images for valid in: ",
                      self.valid_save_to)
            else:
                print("Generate additional Mod images for valid in: ",
                      self.valid_save_to)
            print("Number of images: ", len(os.listdir(self.valid_save_to)))
            i = 0
            get_batch_size = lambda batch_size: 1 if mod else 2
            for _ in datagen.flow(
                    x,
                    batch_size=get_batch_size(mod),
                    save_to_dir=self.valid_save_to,
                    save_prefix='sample',
                    save_format='jpeg'):
                i += 1
                if i > 2:
                    break

    def generate(self):
        # p1 = multiprocessing.Process(
        #     target=self.generate_images,
        #     args=(
        #         self.datagen_without_mod,
        #         False,
        #     ))
        # p2 = multiprocessing.Process(
        #     target=self.generate_valid_images,
        #     args=(
        #         self.datagen_without_mod,
        #         False,
        #     ))
        # p3 = multiprocessing.Process(
        #     target=self.generate_images, args=(
        #         self.datagen_with_mod,
        #         True,
        #     ))
        # p4 = multiprocessing.Process(
        #     target=self.generate_valid_images,
        #     args=(
        #         self.datagen_with_mod,
        #         True,
        #     ))
        self.generate_images(self.datagen_without_mod, False)
        # self.generate_valid_images(self.datagen_without_mod, False)
        # self.generate_images(self.datagen_with_mod, True)
        # self.generate_valid_images(self.datagen_with_mod, True)
        # p1.start()
        # p2.start()
        # p3.start()
        # p4.start()
        # p1.join()
        # p2.join()
        # p3.join()
        # p4.join()

    def get_train_size(self):
        train_size = 0
        for t in os.listdir(self.train_save_to):
            if os.path.isfile(os.path.join(self.train_save_to, t)):
                train_size_1 += 1
        return train_size


if __name__ == "__main__":
    generate_images = Generator()
    generate_images.generate()
    print("Training dataset size: %d" % generate_images.get_train_size())
