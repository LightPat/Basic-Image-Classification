# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 13:18:40 2021

@author: patse
"""

import tensorflow as tf
import tensorflow.keras.datasets.cifar10
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint

if __name__ == "__main__":
    
    
    H5FileName = "CNN.h5"
    
    # Import data and one hot encode labels
    (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.cifar10.load_data()
    train_labels_cat = tf.keras.utils.to_categorical(train_labels, 10)
    test_labels_cat = tf.keras.utils.to_categorical(test_labels, 10)
    
    # Normalize data
    train_data = train_data / 255
    test_data = test_data / 255
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                                     input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    
    model.summary()
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics='accuracy')
    model.save(H5FileName)
    
    checkpoint = ModelCheckpoint(H5FileName, monitor='val_loss',
                                 verbose=1, save_best_only=True, mode='auto', save_freq="epoch")
    
    history = model.fit(train_data, train_labels_cat, batch_size=32, epochs=75, validation_split=0.1, callbacks=[checkpoint])
    model.save("After Training.h5")
    
    plt.figure()
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='val')
    plt.legend()
    plt.savefig("Loss")
    
    plt.figure()
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='val')
    plt.legend()
    plt.savefig("Accuracy")
    
    testMetrics = model.evaluate(test_data, test_labels_cat)
    print(testMetrics)
    
    cnn = tf.keras.models.load_model("After Training.h5")
    testMetrics = cnn.evaluate(test_data, test_labels_cat, batch_size=1)