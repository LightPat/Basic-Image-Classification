# Basic-Image-Classification

This is a short project showcasing the power of transfer learning and CNNs for image recognition.
Our goal was to create a model capable of classifying the following image dataset: https://www.kaggle.com/pmigdal/alien-vs-predator-images

We are using a technique called transfer learning, which is taking a pretrained neural network and changing the output layer to suit our needs.

If you just want the completed model, download the Best CNN.h5 file. This is the file that contains the neural network's structure and weights for classification.

Cifar10 CNN.py is an example of how to build a CNN on the Cifar10 dataset, it's not important to the project, just an example.
ImagePreprocessing.py needs to be run in order to create the datasets for training later.
MobileNetTransferLearning.py This script loads the dataset and trains and tests the model.
