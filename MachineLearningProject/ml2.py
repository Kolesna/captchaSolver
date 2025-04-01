import os   
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from sklearn.tree import DecisionTreeClassifier

trainingFolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TrainingImages')
testFolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TestImages')

dataTrain = []
dataLabels = []
for filename in sorted(os.listdir(trainingFolder)) :
    
    if filename.endswith('.png'):
        img = imread(os.path.join(trainingFolder,filename))
        
        img = img[:,:,:3]
        img_gray = rgb2gray(img)
        img_flat = img_gray.flatten()
        dataTrain.append(img_flat)
        filename = os.path.splitext(filename)[0]
        filename = filename.rstrip('0123456789')
   
        dataLabels.append(filename)
        

        
model = DecisionTreeClassifier(random_state=42) 
model.fit(dataTrain, dataLabels)

for filename in sorted(os.listdir(testFolder)):
    if filename.endswith('.png'):
        img = imread(os.path.join(testFolder, filename))
        img = img[:, :, :3]  
        img_gray = rgb2gray(img)
        img_flat = img_gray.flatten()
        prediction = model.predict([img_flat])
        print(prediction)

