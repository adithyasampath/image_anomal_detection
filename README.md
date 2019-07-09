# image_anomal_detection
Detection of anomalous machine parts using deep convolution auto-encoders

# Generating train data
1) Add images to data/inputs folder.
2) Train keras-retinanet for the above images.
3) Run detection_image.py to generate cropped images for training anomaly detector in data/train
4) Run generate_data.py to generate augmented data for train and valid set.

# Training Anomaly detector
1) Deep Convolutional Model: (better results)
-> Train model by running train.py
-> test on images in data/test by running test.py. Anomalous images are saved in data/anomaly.

2) VGG based Convolutional Model:
-> Train model by running vgg_train.py
-> test on images in data/test by running vgg_test.py. Anomalous images are saved in data/anomaly.