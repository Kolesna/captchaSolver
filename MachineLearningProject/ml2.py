import os   
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from sklearn.tree import DecisionTreeClassifier

trainingFolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TrainingImages')
testFolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TestImages')

dataTrain = []
dataLabels = []

