from keras.layers import Conv2D, LeakyReLU, BatchNormalization, Dense, AveragePooling2D, GaussianNoise, Input
from keras.layers import Reshape, UpSampling2D, Activation, Dropout, Flatten, Conv2DTranspose
from keras.models import model_from_json, Sequential, Model
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from os import listdir
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from skimage.measure import compare_ssim
import cv2

def generator_model():
    G = Sequential()
        
    G.add(Reshape(target_shape = [1, 1, 4096], input_shape = [4096]))
    
    #1x1x4096 
    G.add(Conv2DTranspose(filters = 256, kernel_size = 4))
    G.add(Activation('relu'))
    
    #4x4x256 - kernel sized increased by 1
    G.add(Conv2D(filters = 256, kernel_size = 4, padding = 'same'))
    G.add(BatchNormalization(momentum = 0.7))
    G.add(Activation('relu'))
    G.add(UpSampling2D())
    
    #8x8x256 - kernel sized increased by 1
    G.add(Conv2D(filters = 128, kernel_size = 4, padding = 'same'))
    G.add(BatchNormalization(momentum = 0.7))
    G.add(Activation('relu'))
    G.add(UpSampling2D())
    
    #16x16x128
    G.add(Conv2D(filters = 64, kernel_size = 3, padding = 'same'))
    G.add(BatchNormalization(momentum = 0.7))
    G.add(Activation('relu'))
    G.add(UpSampling2D())
    
    #32x32x64
    G.add(Conv2D(filters = 32, kernel_size = 3, padding = 'same'))
    G.add(BatchNormalization(momentum = 0.7))
    G.add(Activation('relu'))
    G.add(UpSampling2D())
    
    #64x64x32
    G.add(Conv2D(filters = 16, kernel_size = 3, padding = 'same'))
    G.add(BatchNormalization(momentum = 0.7))
    G.add(Activation('relu'))
    G.add(UpSampling2D())
    
    #128x128x16
    G.add(Conv2D(filters = 8, kernel_size = 3, padding = 'same'))
    G.add(Activation('relu'))
    G.add(UpSampling2D())
    
    #256x256x8
    G.add(Conv2D(filters = 3, kernel_size = 3, padding = 'same'))
    G.add(Activation('sigmoid'))
    G.compile(loss='binary_crossentropy', optimizer='adam')
    return G


def discriminator_model():
    D = Sequential()
        
    #add Gaussian noise to prevent Discriminator overfitting
    D.add(GaussianNoise(0.2, input_shape = [256, 256, 3]))
    
    #256x256x3 Image
    D.add(Conv2D(filters = 8, kernel_size = 3, padding = 'same'))
    D.add(LeakyReLU(0.2))
    D.add(Dropout(0.25))
    D.add(AveragePooling2D())
    
    #128x128x8
    D.add(Conv2D(filters = 16, kernel_size = 3, padding = 'same'))
    D.add(BatchNormalization(momentum = 0.7))
    D.add(LeakyReLU(0.2))
    D.add(Dropout(0.25))
    D.add(AveragePooling2D())
    
    #64x64x16
    D.add(Conv2D(filters = 32, kernel_size = 3, padding = 'same'))
    D.add(BatchNormalization(momentum = 0.7))
    D.add(LeakyReLU(0.2))
    D.add(Dropout(0.25))
    D.add(AveragePooling2D())
    
    #32x32x32
    D.add(Conv2D(filters = 64, kernel_size = 3, padding = 'same'))
    D.add(BatchNormalization(momentum = 0.7))
    D.add(LeakyReLU(0.2))
    D.add(Dropout(0.25))
    D.add(AveragePooling2D())
    
    #16x16x64
    D.add(Conv2D(filters = 128, kernel_size = 3, padding = 'same'))
    D.add(BatchNormalization(momentum = 0.7))
    D.add(LeakyReLU(0.2))
    D.add(Dropout(0.25))
    D.add(AveragePooling2D())
    
    #8x8x128
    D.add(Conv2D(filters = 256, kernel_size = 3, padding = 'same'))
    D.add(BatchNormalization(momentum = 0.7))
    D.add(LeakyReLU(0.2))
    D.add(Dropout(0.25))
    D.add(AveragePooling2D())
    
    #4x4x256
    D.add(Flatten())
    
    #256
    D.add(Dense(128))
    D.add(LeakyReLU(0.2))
    
    D.add(Dense(1, activation = 'sigmoid'))
    D.compile(loss='binary_crossentropy', optimizer='adam')
    return D


def generator_containing_discriminator(g, d):
    d.trainable = False
    ganInput = Input(shape=(4096,))
    x = g(ganInput)
    ganOutput = d(x)
    gan = Model(inputs=ganInput, outputs=ganOutput)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan

def train(BATCH_SIZE, X_train):
    d = discriminator_model()
    print("#### discriminator ######")
    d.summary()
    g = generator_model()
    print("#### generator ######")
    g.summary()
    d_on_g = generator_containing_discriminator(g, d)
    d.trainable = True
    for epoch in tqdm(range(50)):
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            noise = np.random.uniform(0, 1, size=(BATCH_SIZE, 4096))
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = g.predict(noise, verbose=0)
            X = np.concatenate((image_batch, generated_images))
            y = np.array([1] * BATCH_SIZE + [0] * BATCH_SIZE)
            d_loss = d.train_on_batch(X, y)
            noise = np.random.uniform(0, 1, (BATCH_SIZE, 4096))
            d.trainable = False
            d_on_g_loss = d_on_g.train_on_batch(noise, np.array([1] * BATCH_SIZE))
            d.trainable = True
        g.save_weights('assets/generator.h5', True)
        d.save_weights('assets/discriminator.h5', True)
    return d, g


def generate(BATCH_SIZE):
    g = generator_model()
    g.load_weights('assets/generator.h5')
    noise = np.random.uniform(0, 1, (BATCH_SIZE, 4096))
    generated_images = g.predict(noise)
    return generated_images

def sum_of_residual(y_true, y_pred):
    return tf.reduce_sum(abs(y_true - y_pred))

def feature_extractor():
    d = discriminator_model()
    d.load_weights('assets/discriminator.h5') 
    intermidiate_model = Model(inputs=d.layers[0].input, outputs=d.layers[-5].output)
    intermidiate_model.compile(loss='binary_crossentropy', optimizer='adam')
    return intermidiate_model

def anomaly_detector():
    g = generator_model()
    g.load_weights('assets/generator.h5') # change back to generator if doesn't work
    g.trainable = False
    intermidiate_model = feature_extractor()
    intermidiate_model.trainable = False
    
    aInput = Input(shape=(4096,))
    gInput = Dense((4096))(aInput)
    G_out = g(gInput)
    D_out= intermidiate_model(G_out)    
    model = Model(inputs=aInput, outputs=[G_out, D_out])
    model.compile(loss=sum_of_residual, loss_weights= [0.9, 0.1], optimizer='adam')
    return model

def compute_anomaly_score(model, x):    
    z = np.random.uniform(0, 1, size=(1, 4096))
    intermidiate_model = feature_extractor()
    d_x = intermidiate_model.predict(x)
    loss = model.fit(z, [x, d_x], epochs=100, verbose=0)
    similar_data, _ = model.predict(z)
    return loss.history['loss'][-1], similar_data


def augment_data_leaf(dataset, augementation_factor=1, use_random_rotation=True, use_random_shear=True, use_random_shift=True, use_random_zoom=True):
	augmented_image = []

	for num in range (0, dataset.shape[0]):

		for i in range(0, augementation_factor):
			# original image:
			augmented_image.append(dataset[num])

			if use_random_rotation:
				augmented_image.append(tf.contrib.keras.preprocessing.image.random_rotation(dataset[num], 20, row_axis=0, col_axis=1, channel_axis=2))

			if use_random_shear:
				augmented_image.append(tf.contrib.keras.preprocessing.image.random_shear(dataset[num], 0.2, row_axis=0, col_axis=1, channel_axis=2))

			if use_random_shift:
				augmented_image.append(tf.contrib.keras.preprocessing.image.random_shift(dataset[num], 0.2, 0.2, row_axis=0, col_axis=1, channel_axis=2))

			if use_random_zoom:
				augmented_image.append(tf.contrib.keras.preprocessing.image.random_zoom(dataset[num], [0.9,1.1], row_axis=0, col_axis=1, channel_axis=2))

	return np.array(augmented_image)

def load_photos(directory):
    images = []
    for name in listdir(directory):
        filename = directory + '/' + name
        print(name)
        image = load_img(filename, target_size=(256, 256))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        images.append(image)
    return np.array(images)

def generator(features, batch_size):
 # Create empty arrays to contain batch of features and labels#
  batch_features = np.zeros((batch_size, 256, 256, 3))
  yield batch_features

class AnomalyPredictor:
    def anomaly_function(self):
        directory_train = '/home/drone/adithya/anomaly_detection_vgg/data/test/cogs'
        test_img = load_photos(directory_train)
        test_img = test_img.astype(np.float32) / 255.0
        threshold = 0.1329451485328614
        original = [False,False,True,False,False,True,False,False,True,False]
        predicted = []
        model = load_model(
        'assets/anoGAN_gear_v1.h5',
        custom_objects={'sum_of_residual':sum_of_residual})
        for img in range(len(test_img)):
            _, similar_img = compute_anomaly_score(
                model, test_img[img].reshape(1, 256, 256, 3))
            (_mse, diff) = compare_ssim(test_img[img].astype(np.float32) * 255.0, similar_img.reshape(256, 256, 3).astype(np.float32) * 255.0, full=True,multichannel=True)
            predicted.append(_mse < threshold)
            print(_mse)
        return np.array(original),np.array(predicted)