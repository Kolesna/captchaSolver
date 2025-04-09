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

for filename in sorted(os.listdir(trainingFolder)):
    if filename.endswith('.png'):
        img = imread(os.path.join(trainingFolder, filename))
        img = img[:, :, :3]
        img_gray = resize(rgb2gray(img), IMG_SHAPE)
        img_flat = img_gray.flatten()
        dataTrain.append(img_flat)
        dataLabels.append(os.path.splitext(filename)[0])

model = DecisionTreeClassifier(random_state=42)
model.fit(dataTrain, dataLabels)

for filename in sorted(os.listdir(testFolder)):
    if filename.endswith('.png'):
        img = imread(os.path.join(testFolder, filename))
        img = img[:, :, :3]
        img_gray = resize(rgb2gray(img), IMG_SHAPE)
        img_flat = img_gray.flatten()
        prediction = model.predict([img_flat])
        print(prediction)