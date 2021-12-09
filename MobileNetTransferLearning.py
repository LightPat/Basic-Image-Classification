# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 16:48:29 2021

@author: patse
"""

import os
import pickle
import random
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import imagenet_utils

# Load pickled datasets from a path
def loadPickledDatasets(path):
    dsDict = {}
    
    for ds in os.listdir(path):
        dsPath = os.path.join(pickledDatasetLoc, ds)
        with open(dsPath, "rb") as f:
            loadedList = pickle.load(f)
            random.shuffle(loadedList)
            dsDict[ds] = loadedList

    return dsDict

# Separates image data and labels, one hot encodes labels
# Index 0 is alien, Index 1 is predator
def separateDataAndLabels(dsDict, dsName):
    data = []
    indexes = []
    
    for labeledImage in dsDict[dsName]:
        if labeledImage[1] == "alien":
            indexes.append(0)
        elif labeledImage[1] == "predator":
            indexes.append(1)
            
        data.append(labeledImage[0])
    
    labels = tf.keras.utils.to_categorical(indexes, 2)
    
    data = np.asarray(data)
    data = np.squeeze(data)
    
    print("Separated", len(data), "images from", len(labels), "labels")
    return data, labels
    

if __name__ == "__main__":
    # Location of pickled datasets
    pickledDatasetLoc = r"G:\Python Projects\AlienVsPredator CNN\Pickled Datasets"
    # Model file name during keras callback
    H5FileName = "checkpoint.h5"
    
    dsDict = loadPickledDatasets(pickledDatasetLoc)
    
    # Separate data and labels
    train_data, train_labels = separateDataAndLabels(dsDict, "train")
    test_data, test_labels = separateDataAndLabels(dsDict, "test")
    
    # Import mobilenet model and then put new output layer on it
    print("Prepping mobilenet model for training")
    mobileNetModel = tf.keras.applications.mobilenet.MobileNet(weights='imagenet', include_top=True)
    mobileNetLayer = mobileNetModel.layers[-6].output
    poolingLayer = tf.keras.layers.GlobalAveragePooling2D()(mobileNetLayer)
    outputLayer = tf.keras.layers.Dense(2, activation="softmax")(poolingLayer)
    model = Model(inputs=mobileNetModel.input, outputs=outputLayer)
    
    # Set all layers except the last 5 to untrainable
    for layer in model.layers[:-5]:
        layer.trainable = False
    print("Done.")
    
    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Keras callback to save model whenever validation loss decreases
    checkpoint = ModelCheckpoint(H5FileName, monitor='val_loss',
                                 verbose=1, save_best_only=True,
                                 mode='auto', save_freq="epoch")
    
    # Train Model
    history = model.fit(train_data, train_labels, epochs=40, verbose=1, validation_split=0.1, callbacks=[checkpoint])
    # Save Model after training
    model.save("After Training CNN.h5")
    
    # Loss Visuals    
    plt.figure()
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='val')
    plt.legend()
    plt.savefig("Loss")
    
    # Accuracy visuals
    plt.figure()
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='val')
    plt.legend()
    plt.savefig("Accuracy")
    
    # Evalutes model's performance on test set (data we've never seen before)
    # Batch size of 1 so that we go over the WHOLE testing dataset
    # I don't know if that is necessary or not,
    # I usually check both on default batch size and on batch size = 1 when testing my models
    loadedModel = tf.keras.models.load_model("After Training CNN.h5")
    testMetrics = loadedModel.evaluate(test_data, test_labels, batch_size=1)
