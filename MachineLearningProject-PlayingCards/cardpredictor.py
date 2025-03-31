import os
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from sklearn.tree import DecisionTreeClassifier


trainingFolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Cards')
testFolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CardTest')
IMG_SHAPE = (32, 32)

dataTrain = []
dataLabels = []