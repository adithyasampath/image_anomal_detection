import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Flatten, Activation, BatchNormalization
from keras.models import Sequential, Model
from keras import applications
import keras
import os
import cv2
import csv
from tensorflow.contrib import lite
from skimage.measure import compare_ssim
import imutils
from keras.applications.vgg16 import preprocess_input

class AnomalyPredictor:
    def __init__(self):
        self.img_width, self.img_height = 420, 420
        self.img_dim = 420
        self.batch_size = 8
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
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            color_mode='rgb',
            class_mode=None)

        self.nb_train_samples = self.train_generator.samples
        # this is a similar generator, for validation data
        self.validation_generator = self.test_datagen.flow_from_directory(
            self.validation_data_dir,
            target_size=(self.img_width, self.img_height),
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
        # for c in cnts:
        # 	(x, y, w, h) = cv2.boundingRect(c)
	    #     cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
	    #     cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
        return imageA,imageB,diff,thresh

    def save_image(self,path,filename,img):
        cv2.imwrite(os.path.join(path,filename), img)

    def IsImageHasAnomaly(self, filePath, filename):
        im = cv2.resize(
            cv2.imread(filePath), (self.img_width, self.img_height))
        im = im * 1. / 255
        # validation_image = np.expand_dims(im,axis=0)
        validation_image = np.zeros((1, self.img_width, self.img_height, 3))
        validation_image[0, :, :, :] = im
        predicted_image = self.autoencoder.predict(validation_image)
        _mse = self.mse(predicted_image[0], validation_image[0])
        self.save_image("data","predicted.jpg",predicted_image[0].astype(np.float32)* 255.0)
        self.save_image("data","original.jpg",validation_image[0].astype(np.float32)* 255.0)
        if (_mse > self.threshold):
            original, anomaly, diff, thresh = self.image_diff(validation_image[0].astype(np.float32)* 255.0,predicted_image[0].astype(np.float32)* 255.0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(original, 'Anomaly', (10, 200), font, 1, (255, 0, 0), 3,
                        cv2.LINE_AA)
            self.save_image(self.test_save_to,"anomaly_" + filename,original)
            self.save_image(self.test_save_to,"predicted_" + filename,anomaly)
            self.save_image(self.test_save_to,"diff_" + filename,diff)
            self.save_image(self.test_save_to,"thresh_" + filename,thresh)
        print('_mse: {}'.format(_mse))
        return _mse > self.threshold

    def load_photos(self,directory):
        images = []
        for name in os.listdir(directory):
            # load an image from file
            print(name)
            filename = directory + '/' + name
            image = load_img(filename, target_size=(420, 420))
            # convert the image pixels to a numpy array
            image = img_to_array(image)
            # reshape data for the model
            image = image.reshape((image.shape[0], image.shape[1], image.shape[2]))
            # prepare the image for the VGG model
            image = preprocess_input(image)
            images.append(image)
        return np.array(images)

    def anomaly_function(self):
        self.create_model()
        self.load_weights()
        self.get_threshold()
        directory_train = '/home/drone/adithya/anomaly_detection_vgg/data/test/cogs'
        test_img = self.load_photos(directory_train)
        test_img = test_img.astype(np.float32) / 255.0
        original = [False,False,True,False,False,True,False,False,True,False]
        predicted = []
        for img in range(len(test_img)):
            validation_image = np.zeros((1, self.img_dim, self.img_dim, 3))
            validation_image[0, :, :, :] = test_img[img]
            predicted_image = self.autoencoder.predict(validation_image)
            grayA = cv2.cvtColor(predicted_image[0].astype(np.float32)* 255.0, cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(validation_image[0].astype(np.float32)* 255.0, cv2.COLOR_BGR2GRAY)
            (_mse, diff) = compare_ssim(grayA, grayB, full=True)
            predicted.append(_mse < self.threshold)
        return np.array(original),np.array(predicted)

    def create_model(self):
        base_model = applications.vgg16.VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_width, self.img_height, 3))
        proxy_model = Sequential()
        proxy_model.add(
            Conv2D(
                3, (1, 1),
                activation='relu',
                padding='same',
                input_shape=base_model.output_shape[1:]))
        proxy_model.add(BatchNormalization())
        proxy_model.add(UpSampling2D((32, 32)))
        proxy_model.add(ZeroPadding2D((1, 1)))

        # ------------------------------------------------------------------------
        base_model_and_proxy = Model(
            inputs=base_model.input, outputs=proxy_model(base_model.output))

        # -------------------------------------------------------------------------
        top_model = Sequential()
        top_model.add(
            Conv2D(
                16, (3, 3),
                activation='relu',
                padding='same',
                input_shape=base_model_and_proxy.output_shape[1:]))
        top_model.add(BatchNormalization())
        top_model.add(MaxPooling2D((2, 2), padding='same'))
        top_model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        top_model.add(BatchNormalization())
        top_model.add(MaxPooling2D((2, 2), padding='same'))
        top_model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        top_model.add(BatchNormalization())
        encoded = MaxPooling2D((2, 2), padding='same')
        top_model.add(encoded)
        top_model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        top_model.add(BatchNormalization())
        top_model.add(UpSampling2D((2, 2)))
        top_model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        top_model.add(BatchNormalization())
        top_model.add(UpSampling2D((2, 2)))
        top_model.add(Conv2D(16, (3, 3), activation='relu'))
        top_model.add(BatchNormalization())
        top_model.add(UpSampling2D((2, 2)))
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')
        top_model.add(decoded)
        self.autoencoder = Model(
            inputs=base_model_and_proxy.input,
            outputs=top_model(base_model_and_proxy.output))
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
        for valid_image in self.validation_generator:
            if step > self.nb_validation_samples:
                break
            #print(step, sep=' ', end='>', flush=True)
            predicted_image = self.autoencoder.predict(valid_image)
            mse_value = self.mse(predicted_image[0], valid_image[0])
            all_mses.append(mse_value)
            step = step + 1

        self.error_df = pd.DataFrame({'train_error': all_mses})
        self.error_df.to_csv(self.train_analysis_file)
        self.error_df.describe()
        self.threshold = np.percentile(all_mses, 85)
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
