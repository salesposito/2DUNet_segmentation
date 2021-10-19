from pathlib import Path
import os
import math
import numpy as np
import pathlib
from scipy.sparse import csc_matrix
import tensorflow as tf
from matplotlib import pyplot as plt
import warnings
import pickle
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.keras import layers
from sklearn.metrics import jaccard_score
warnings.filterwarnings("ignore", category=DeprecationWarning)
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Input
from tensorflow.python.keras.utils import conv_utils
from  SA_UNet import *
import unet_multiclass

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session =tf.compat.v1.InteractiveSession(config=config)
# gpus= tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

with open('train.pickle', 'rb') as f:
    X, y = pickle.load(f)




#Convert to np.array
X = np.array(X)
y = np.array(y)

print(y.shape)
X=X[:1500, :,:]
y=y[:1500, :,:]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# y_train = tf.keras.utils.to_categorical(y_train, 5)
# y_test = tf.keras.utils.to_categorical(y_test, 5)
# y_train = tf.cast(y_train, tf.float32)
# y_test = tf.cast(y_test, tf.float32)
y_train = tf.keras.utils.to_categorical(y_train, 5)
y_test = tf.keras.utils.to_categorical(y_test, 5)
# def dice_coefficient(y_true, y_pred):
#     numerator = 2 * tensorflow.reduce_sum(y_true * y_pred)
#     denominator = tensorflow.reduce_sum(y_true + y_pred)

#     return numerator / (denominator + tensorflow.keras.backend.epsilon())
smooth=1

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



# def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = -alpha*((1-p)^gamma)*log(p)
    Parameters:
        alpha -- the same as wighting factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    """
    def focal_loss(y_true, y_pred):
        # Define epsilon so that the backpropagation will not result in NaN
        # for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        #y_pred = y_pred + epsilon
        # Clip the prediction value
        y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
        # Calculate cross entropy
        cross_entropy = -y_true*K.log(y_pred)
        # Calculate weight that consists of  modulating factor and weighting factor
        weight = alpha * y_true * K.pow((1-y_pred), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.sum(loss, axis=1)
        return loss
    
    return focal_loss



# def loss(y_true, y_pred):
#     def dice_loss(y_true, y_pred):
#       y_pred = tf.math.sigmoid(y_pred)
#       numerator = 2 * tf.reduce_sum(y_true * y_pred)
#       denominator = tf.reduce_sum(y_true + y_pred)

#       return 1 - numerator / denominator

#     y_true = tf.cast(y_true, tf.float32)
#     o = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred) + dice_loss(y_true, y_pred)

#     return tf.reduce_mean(o)

img_size = (512,512,1) # 256 * 256 grayscale img with 1 channel
dr_rate = 0.6 # never mind
leakyrelu_alpha = 0.3

model=unet_multiclass.unet(pretrained_weights = None,input_size = img_size)
model.summary()



model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=['accuracy', dice_coef])
model.fit(X_train, y_train, validation_split=0.5,epochs=2, batch_size=1, verbose=1)
model.evaluate(X_test, y_test, batch_size=1, verbose=1,)
model.save("Unet")

result=model.predict(X_test[0:2])
fig=plt.figure()
for i in range(2):
  plt.subplot(2,2,i+1)
  plt.imshow(X_test[i].reshape(512,512),cmap='gray')
  plt.subplot(2,2,i+3)
  plt.imshow(result[i].reshape(512,512),cmap='gray')
  plt.savefig('unet_plot')
