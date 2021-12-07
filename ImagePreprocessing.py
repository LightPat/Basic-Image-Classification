# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 15:58:15 2021

Loops through a base directory like the one in the github repo and preprocesses all images
After that it pickles (serializes) the image array and the label for that image

@author: patse
"""

import os
import cv2
import pickle

def preprocess_image(filepath):
    img_array = cv2.imread(filepath)
    img_array = cv2.resize(img_array, (224,224))
    img_array = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array)


if __name__ == "__main__":
    H5FileName = "CNN.h5"
    DatasetPath = r"G:\Python Projects\Cifar10 CNN\AlienVsPredatorDataset"
    PickledDatasetSavePath = r"G:\Python Projects\Cifar10 CNN\Pickled Datasets"
    
    preprocessedImageArrays = []

    for folder in os.listdir(DatasetPath):
        classPath = os.path.join(DatasetPath, folder)
        for category in os.listdir(classPath):
            categoryPath = os.path.join(classPath, category)
            for image in os.listdir(categoryPath):
                i = preprocess_image(os.path.join(categoryPath, image))
                preprocessedImageArrays.append([i, category])
            with open(os.path.join(PickledDatasetSavePath, category + " " + folder), "wb") as f:
                # Pickle means serialize
                pickle.dump(preprocessedImageArrays, f)
                preprocessedImageArrays = []
                