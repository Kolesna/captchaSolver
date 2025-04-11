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


def maxcontrast(image):
    new_image = []

    inum = 0
    img = image
    for i in img:
        gnum = 0
        new_row = []
        if inum+1 == len(img):
                pass
        else:          
            for g in i:
                sorroundingAverage = 0
                if gnum+1 == len(i):
                    sorroundingAverage = 1

                else:
                    for h in range(-1,2):
                        for k in range(-1,2):
                            sorroundingAverage += int((img[inum+h][gnum+k][0]))/2295
                            
                
                if sorroundingAverage > 0.5:
                    new_row.append(1)
                else:
                    new_row.append(0)
                gnum += 1
                
                
            new_row2 = np.array(new_row)
            new_image.append(new_row2)

            
            inum += 1
    new_image = np.array(new_image)

    return new_image
                
def showArrayAsImage(input):
    plt.imshow(input, cmap='gray')
    plt.axis('off')
    plt.show()

def preProcessImage(input):


    img = np.array(input)
    img = maxcontrast(img)
    return img

trainingFolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Cards')    
testFolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CardTest')
IMG_SHAPE = (40, 50)

dataTrain = []
dataLabels = []