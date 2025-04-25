import os
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
import matplotlib.pyplot as plt

import tensorflow as tf
Sequential = tf.keras.models.Sequential
load_model = tf.keras.models.load_model
Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
to_categorical = tf.keras.utils.to_categorical
import joblib
from sklearn.preprocessing import LabelEncoder

tf.keras.utils.set_random_seed(42)


trainingFolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Cards')
testFolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CardTest')
IMG_SHAPE = (40, 50)  
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'captcha_cnn_model.h5')
LABEL_ENCODER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'label_encoder.pkl')

dataTrain = []
dataLabels = []