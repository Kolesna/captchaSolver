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
        img_2d = img_resize.reshape(IMG_SHAPE[0], IMG_SHAPE[1], 1)
        processed_segments.append(img_2d)
    
    return processed_segments

trainingFolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Cards')
testFolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CardTest')
IMG_SHAPE = (40, 50)  
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'captcha_cnn_model.h5')
LABEL_ENCODER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'label_encoder.pkl')

def create_cnn_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SHAPE[0], IMG_SHAPE[1], 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model



label_encoder = LabelEncoder()

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
            img_2d = img_resize.reshape(IMG_SHAPE[0], IMG_SHAPE[1], 1)
            dataTrain.append(img_2d)
            dataLabels.append(namelist[i])
        print("Processing " + filename + " done")
            

dataTrain = np.array(dataTrain)
dataLabels = np.array(dataLabels)

dataLabels_encoded = label_encoder.fit_transform(dataLabels)
dataLabels_categorical = to_categorical(dataLabels_encoded)

num_classes = len(label_encoder.classes_)
model = create_cnn_model(num_classes)

model.fit(dataTrain, dataLabels_categorical, epochs=int(os.environ.get('CAPTCHA_EPOCHS', '10')), batch_size=64, validation_split=0.2)