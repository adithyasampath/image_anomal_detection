import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Flatten, Activation, BatchNormalization, Conv2DTranspose, LeakyReLU, GlobalAveragePooling2D,concatenate
from keras.models import Sequential, Model
from keras import applications
import keras
import os
import cv2
import csv
from tensorflow.contrib import lite
from skimage.measure import compare_ssim
import imutils
import time
import losses

class AnomalyPredictor:
    def __init__(self):
        self.img_dim = 256
        self.z_dim = 512
        self.batch_size = 1
        self.nb_validation_samples = 0
        self.test_datagen = None
        self.validation_generator = None
        self.nb_train_samples = 0
        self.train_datagen = None
        self.train_generator = None
        self.autoencoder = None
        self.error_df = None
        self.threshold = None
        self.nb_epoch = 50
        self.weights_dir = os.path.join(
            os.path.dirname(__file__), "weights", "autoencoder-vgg.h5")
        self.tf_weights_dir = os.path.join(
            os.path.dirname(__file__), "weights","1")
        self.initial_image_dir = os.path.dirname(__file__)
        self.initial_image_dir = os.path.join(self.initial_image_dir, "data")
        self.median_file = os.path.join(self.initial_image_dir, 'median.csv')
        self.train_analysis_file = os.path.join(self.initial_image_dir,
                                                'train_analysis.csv')
        self.train_analysis_image = os.path.join(self.initial_image_dir,
                                                 "train_error.png")
        self.train_data_dir = os.path.join(self.initial_image_dir, 'train')
        self.validation_data_dir = os.path.join(self.initial_image_dir,
                                                'valid')
        self.test_data_dir = os.path.join(self.initial_image_dir, 'test')
        self.test_image_list = os.listdir(
            os.path.join(self.test_data_dir, "cogs"))
        self.test_save_to = os.path.join(self.initial_image_dir, 'anomaly')
        

    def init_data(self):
        # this is the augmentation configuration we will use for training
        # only rescaling
        self.train_datagen = ImageDataGenerator(rescale=1/255)

        # this is the augmentation configuration we will use for testing:
        # only rescaling
        self.test_datagen = ImageDataGenerator(rescale=1/255)

        # this is a generator that will read pictures
        self.train_generator = self.train_datagen.flow_from_directory(
            self.train_data_dir,  # this is the target directory
            target_size=(self.img_dim, self.img_dim),
            batch_size=self.batch_size,
            color_mode='rgb',
            class_mode=None)

        self.nb_train_samples = self.train_generator.samples
        # this is a similar generator, for validation data
        self.validation_generator = self.test_datagen.flow_from_directory(
            self.validation_data_dir,
            target_size=(self.img_dim, self.img_dim),
            batch_size=self.batch_size,
            color_mode='rgb',
            class_mode=None)

        self.nb_validation_samples = self.validation_generator.samples
        self.nb_train_samples = self.train_generator.samples
        

    def fixed_generator(self, generator):
        for batch in generator:
            # print("Batch shape: ",np.asarray(batch).shape)
            yield (batch, batch)

    def mse(self, imageA, imageB):
        err = np.sum((imageA.astype("float") - imageB.astype("float"))**2)
        err /= float(imageA.shape[0] * imageA.shape[1])
        return err

    def get_threshold(self):
        with open(self.median_file) as csv_file:
            reader = csv.reader(csv_file)
            median = dict(reader)
        self.threshold = float(median["median"])


    def save_image(self,path,filename,img):
        cv2.imwrite(os.path.join(path,filename), img)

    def IsImageHasAnomaly(self, filePath, filename):
        im = cv2.resize(
            cv2.imread(filePath), (self.img_dim, self.img_dim))
        im = im * 1. / 255
        # validation_image = np.expand_dims(im,axis=0)
        validation_image = np.zeros((1, self.img_dim, self.img_dim, 3))
        validation_image[0, :, :, :] = im
        predicted_image = self.autoencoder.predict(validation_image)
        grayA = cv2.cvtColor(predicted_image[0].astype(np.float32)* 255.0, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(validation_image[0].astype(np.float32)* 255.0, cv2.COLOR_BGR2GRAY)
        (_mse, diff) = compare_ssim(grayA, grayB, full=True) #self.mse(predicted_image[0], validation_image[0])
        self.save_image("data","predicted.jpg",predicted_image[0].astype(np.float32)* 255.0)
        self.save_image("data","original.jpg",validation_image[0].astype(np.float32)* 255.0)
        if (_mse < self.threshold):
            original, anomaly, diff, thresh = self.image_diff(validation_image[0].astype(np.float32)* 255.0,predicted_image[0].astype(np.float32)* 255.0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(original, 'Anomaly', (10, 10), font, 1, (255, 0, 0), 1,
                        cv2.LINE_AA)
            self.save_image(self.test_save_to,"anomaly_" + filename,original)
            self.save_image(self.test_save_to,"predicted_" + filename,anomaly)
            self.save_image(self.test_save_to,"diff_" + filename,diff)
            self.save_image(self.test_save_to,"thresh_" + filename,thresh)
        print('_mse: {}'.format(_mse))
        return _mse < self.threshold

    def image_diff(self,imageA, imageB):
        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
        (score, diff) = compare_ssim(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")
        # print("SIMILARITY score: {}".format(score))
        thresh = cv2.threshold(diff, 0, 255,
	    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	    cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        cv2.drawContours(imageA, contours,-1, (0, 255, 0), 2)
        max_contour = max(contours, key = cv2.contourArea)
        rect = cv2.minAreaRect(max_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        im = cv2.drawContours(imageB,[box],0,(0,0,255),2)
        return imageA,imageB,diff,thresh
        
    def create_tf_serving_model(self):
        tf.keras.backend.set_learning_phase(0)
        self.autoencoder.load_weights(self.weights_dir)
        print("Started conversion")
        start = time.time()
        with tf.keras.backend.get_session() as sess:
            tf.initialize_all_variables().run()
            tf.saved_model.simple_save(
                sess,
                self.tf_weights_dir,
                inputs={'input_image': self.autoencoder.input},
                outputs={t.name:t for t in self.autoencoder.outputs})
            print("Converted in: ",time.time()-start)

    def create_model(self):
        inputs = Input((self.img_dim, self.img_dim, 3))
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        # conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        # conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        # conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        # conv4 = BatchNormalization()(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
        # conv5 = BatchNormalization()(conv5)
        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
        # conv6 = BatchNormalization()(conv6)
        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
        # conv7 = BatchNormalization()(conv7)
        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
        # conv8 = BatchNormalization()(conv8)
        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
        # conv9 = BatchNormalization()(conv9)
        conv10 = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(conv9)
        self.autoencoder = Model(inputs=[inputs], outputs=[conv10])
        for layer in self.autoencoder.layers[:19]:
           layer.trainable = False
        #self.autoencoder.summary()

    def train_model(self):
        callbacks = []
        checkpoint = keras.callbacks.ModelCheckpoint(
            self.weights_dir,
            monitor='val_loss',
            verbose=1,
            save_best_only=False,
            save_weights_only=False,
            mode='auto',
            period=1)
        callbacks.append(checkpoint)
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='loss', min_delta=0, patience=3, verbose=1, mode='auto')
        callbacks.append(early_stopping)
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.01,
                patience=1,
                verbose=1,
                mode='auto',
                min_delta=0.001,
                cooldown=0,
                min_lr=0))

        self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        self.autoencoder.fit_generator(
            self.fixed_generator(self.train_generator),
            steps_per_epoch=self.nb_train_samples // self.batch_size,
            epochs=self.nb_epoch,
            callbacks=callbacks,
            validation_data=self.fixed_generator(self.validation_generator),
            validation_steps=(self.nb_validation_samples // 2))
        self.autoencoder.save_weights(self.weights_dir)
        # converter = lite.TFLiteConverter.from_keras_model_file(
        #     'autoencoder-vgg.h5')
        # tfmodel = converter.convert()
        # open("autoencoder-vgg.tflite", "wb").write(tfmodel)

    def set_threshold(self):
        self.autoencoder.load_weights(self.weights_dir)
        all_mses = []
        step = 1
        for valid_image in self.train_generator:
            if step > self.nb_train_samples:
                break
            #print(step, sep=' ', end='>', flush=True)
            predicted_image = self.autoencoder.predict(valid_image)
            grayA = cv2.cvtColor(predicted_image[0].astype(np.float32)* 255.0, cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(valid_image[0].astype(np.float32)* 255.0, cv2.COLOR_BGR2GRAY)
            (mse_value, _) = compare_ssim(grayA, grayB, full=True)   #self.mse(predicted_image[0], valid_image[0])
            all_mses.append(mse_value)
            step = step + 1

        self.error_df = pd.DataFrame({'train_error': all_mses})
        self.error_df.to_csv(self.train_analysis_file)
        self.error_df.describe()
        self.threshold = np.mean(all_mses)
        median = {}
        median["median"] = self.threshold
        with open(self.median_file, 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in median.items():
                writer.writerow([key, value])

    def load_weights(self):
        self.autoencoder.load_weights(self.weights_dir)
        self.autoencoder.compile(
            optimizer='adadelta', loss='binary_crossentropy')

    def show_plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        _ = ax.hist(self.error_df.train_error.values, bins=5)
        ax.figure.savefig(self.train_analysis_image)
