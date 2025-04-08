import os
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline


trainingFolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Cards')    
testFolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CardTest')
IMG_SHAPE = (40, 50)

dataTrain = []
dataLabels = []