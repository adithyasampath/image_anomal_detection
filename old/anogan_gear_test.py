import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import anogan_gear
import keras
from keras.models import load_model
import cv2
keras.losses.sum_of_residual = anogan_gear.sum_of_residual
model = anogan_gear.AnomalyPredictor()
y_test,y_pred=model.anomaly_function()
print("Test: ",y_test)
print("Pred: ",y_pred)
# model = load_model(
#     'assets/anoGAN_gear_v1.h5',
#     custom_objects={'sum_of_residual': anogan_gear.sum_of_residual})

# directory_train = '/home/drone/adithya/anomaly_detection_vgg/data/test/cogs'
# X_train = anogan_gear.load_photos(directory_train)
# X_train = X_train.astype(np.float32) / 255.0
# test_img_nos = np.random.randint(low=1, high=10, size=10)
# test_img_nos = test_img_nos.tolist()
# test_img = X_train[test_img_nos]

# # directory_test = '/home/drone/adithya/anomaly_detection_vgg/data/train/cogs'
# # X_test = anogan_gear.load_photos(directory_test)
# # X_test = X_test.astype(np.float32) / .astype(np.float32)* 255.0
# # X_test = X_test.reshape(-1, 256, 256, 3)
# # test_img = X_test

# for img in range(len(test_img)):
#     ano_score, similar_img = anogan_gear.compute_anomaly_score(
#         model, test_img[img].reshape(1, 256, 256, 3))
#     print("anomaly score test : " + str(ano_score))
#     cv2.imwrite('test/test_%d.jpg' % img,
#                 test_img[img].astype(np.float32) * 255.0)
#     cv2.imwrite('test/generated_%d.jpg' % img,
#                 similar_img.reshape(256, 256, 3).astype(np.float32) * 255.0)
#     residual = test_img[img] - similar_img.reshape(256, 256, 3)
#     cv2.imwrite('test/residual_%d.jpg' % img,
#                 residual.astype(np.float32) * 255.0)



