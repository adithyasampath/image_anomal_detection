import tensorflow as tf
import os
import time
import keras
from keras_retinanet.models.resnet import ResNetBackbone

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


keras.backend.tensorflow_backend.set_session(get_session())
tf.keras.backend.set_learning_phase(0) # Ignore dropout at inference
num_classes = 80
model_path = os.path.join('weights','resnet50_csv_anomaly_6.h5') 
# load retinanet model
model = ResNetBackbone('resnet50').retinanet(num_classes)
model.load_weights(model_path, by_name=True, skip_mismatch=True)

# model = tf.keras.models.load_model(os.path.join(".","weights","autoencoder-vgg.h5"))
tf_export_path = os.path.join(".","weights","2")
print("model loaded")
start = time.time()
# Fetch the Keras session and save the model
# The signature definition is defined by the input and output tensors
# And stored with the default serving key
with tf.keras.backend.get_session() as sess:
    tf.initialize_all_variables().run()
    tf.saved_model.simple_save(
        sess,
        tf_export_path,
        inputs={'input_image': model.input},
        outputs={t.name:t for t in model.outputs})
    print("Converted in: ",time.time()-start)
