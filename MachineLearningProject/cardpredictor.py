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
                
def cropWhite(input):
    img = np.array(input)
    if len(img.shape) == 3:
        if img.shape[2] >= 3:
            img = np.mean(img[:,:,:3], axis=2)
        else:
            img = img[:,:,0]
    img = np.rot90(img)
    new_img = []

    for i in img:
        avgDark = 0
        lenI = len(i)
        for g in i:
            avgDark += (g)/lenI
            
        
        if avgDark < 0.75:
            new_img.append(i)

    if len(new_img) == 0:
        raise ValueError('No text found in image')

    new_img = np.rot90(new_img, 3)
    return new_img

def showArrayAsImage(input):
    plt.imshow(input, cmap='gray')
    plt.axis('off')
    plt.show()

def preProcessImage(input):


    img = np.array(input)
    img = maxcontrast(img)
    img = cropWhite(img)
    return img

def processTestImage(input):
    img = np.array(input)
    img = img[:, :, :3]
    imgProcessed = preProcessImage(img)
    
    fiveSplit = len(imgProcessed[0]) // 5
    spotlist = []
    processed_segments = []
    
    for i in range(0, 5):
        tempimage = []
        spotlist.append((fiveSplit*i) + (fiveSplit//2))
        hcount = 0
        for h in imgProcessed:
            temprow = []
            gcount = 0
            for g in h:
                for k in range(spotlist[i]-(fiveSplit//3 *2), spotlist[i]+(fiveSplit//3 *2)):
                    if gcount == k:
                        temprow.append(g)
                gcount += 1
            tempimage.append(temprow)
            hcount += 1    
        
        img_array = np.array(tempimage)
        img_resize = resize(img_array, IMG_SHAPE, anti_aliasing=True)
        img_flat = img_resize.flatten()
        processed_segments.append(img_flat)
    
    return processed_segments
trainingFolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Cards')    
testFolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CardTest')
IMG_SHAPE = (40, 50)

dataTrain = []
dataLabels = []

for filename in sorted(os.listdir(trainingFolder)):
    if filename.endswith('.png'):
        img = imread(os.path.join(trainingFolder, filename))
        img = img[:, :, :3]
        imgProcessed = preProcessImage(img)
        fiveSplit = len(imgProcessed[0]) // 5
        spotlist = []

        filename = filename[:-4]
        namelist = list(filename)
            
        for i in range(0,5):
            tempimage = []
            spotlist.append((fiveSplit*i) + (fiveSplit//2))
            hcount = 0
            for h in imgProcessed:
                temprow = []
                gcount = 0
                for g in h:
                    for k in range(spotlist[i]-(fiveSplit//3 *2),spotlist[i]+(fiveSplit//3 *2)):
                        if gcount == k:

                            temprow.append(g)
                    gcount += 1
                tempimage.append(temprow)
                hcount += 1    
        
            
            
            img_array = np.array(tempimage)
            img_resize = resize(img_array, IMG_SHAPE, anti_aliasing=True)
            
            img_flat = img_resize.flatten()
            
            dataTrain.append(img_flat)
            dataLabels.append(namelist[i])
        print("Processing " + filename + " done")
            

dataTrain = np.array(dataTrain)
dataLabels = np.array(dataLabels)

model = DecisionTreeClassifier(random_state=42)
model.fit(dataTrain, dataLabels)



correct = 0
wrong = 0
for filename in sorted(os.listdir(testFolder)):
    if filename.endswith('.png') and not os.path.isfile(os.path.join(trainingFolder, filename)):
        img = imread(os.path.join(testFolder, filename))
        
        img_segments = processTestImage(img)
        
        print(f"\nProcessing {os.path.splitext(filename)[0]}:")
        predictions = []
        
        for i, segment in enumerate(img_segments):
            prediction = model.predict([segment])
            predictions.append(prediction[0])

        filename = filename[:-4]
        namelist = list(filename)
        for i in range(0,len(predictions)):
            if predictions[i] == namelist[i]:
                print("yes")
                correct += 1
            else:
                wrong += 1
        print(f"  Full prediction: {''.join(predictions)}")
        
print(correct/(correct+wrong) if correct+wrong else 'No test images found')