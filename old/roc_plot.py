from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import keras
from keras.models import load_model
import cv2
from anogan_gear import AnomalyPredictor as an4
import anogan_gear
keras.losses.sum_of_residual = anogan_gear.sum_of_residual
from deep_conv_anomaly import AnomalyPredictor as an1
from deep_conv_anomaly_1 import AnomalyPredictor as an2
from vgg_anomaly import AnomalyPredictor as an3

plt.figure()

# Add the models to the list that you want to view on the ROC plot
models = [{
    'label': 'Auto-Encoder',
    'model': an3(),
}, {
    'label': 'AnoGAN',
    'model': an4(),
}, {
    'label': 'My model with BCE',
    'model': an1(),
}, {
    'label': 'My model with SSIM',
    'model': an2(),
}]

# Below for loop iterates through your models list
for m in models:
    print("Model")
    model = m['model']  # select the model
    # model.fit(x_train, y_train) # train the model
    # y_pred=model.predict(x_test) # predict the test data
    y_test, y_pred = model.anomaly_function()
    # Compute False postive rate, and True positive rate
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    # Calculate Area under the curve to display on the plot
    auc = metrics.roc_auc_score(y_test, y_pred)
    # Now, plot the computed values
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (m['label'], auc))
# Custom settings for the plot
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("ROC.png")  # Display
