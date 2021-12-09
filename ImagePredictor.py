# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 12:37:52 2021

@author: patse
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from ImagePreprocessing import preprocess_image

def decode_predictions(predictionArray):
    decoded = {}
    temp = []
    
    for value in predictionArray[0]:
        temp.append(value)
    
    decoded["alien"] = temp[0]
    decoded["predator"] = temp[1]
    
    return decoded

if __name__ == "__main__":
    H5FileName = r"G:\Python Projects\AlienVsPredator CNN\Run 2_97 Percent 40 Epochs\After Training CNN.h5"
    imageToPredict = r"G:\Python Projects\AlienVsPredator CNN\AlienVsPredatorDataset\test\predator\3.jpg"
    
    cnn = load_model(H5FileName)
    img = preprocess_image(imageToPredict)
    img = np.expand_dims(img, axis=0)
    prediction = cnn.predict(img)
    decodedPrediction = decode_predictions(prediction)
    print("DECIMAL REPRESENTATION\nAlien:", decodedPrediction["alien"], "\nPredator:", decodedPrediction["predator"])
