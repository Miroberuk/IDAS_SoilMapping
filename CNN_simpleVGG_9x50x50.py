#!/usr/bin/env python
"""Simple Convolutional Neural Network based on VGG16

Anaconda Environment Installations - Linux
python = 3.6
tensorflow = 2.1.0
numpy = 1.19
scipy = 1.5
scikit-learn = 0.23
keras = 1.0.8
openCV = 3.4.2
matplotlib = 3.3.2
pillow = 8.0.1
imutils = 0.5.3

Last Modified: Feb 4, 2021
"""


import tensorflow as tf
from tensorflow import keras
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
import numpy as np

import cv2
import pickle
import random
from imutils import paths

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

##

# Hyperparameters
Run_Name = "vgg_Feb04_01_Sentinel"
Epochs = 75
Learning_Rate = 1e-3
Batch_Size = 32

# Assume running execution from the following directory
# cd C:/Users/xxxx##/Documents/00_IDaS/01_SoilSample/

# Archaeology Informatics Lab Windows Directory
#base_dir = 'D:/SoilSampleProject/00_IDaS/2020_July_Sentinel/'

# Linux Directory
base_dir = "/home/mirober/Documents/2020_July_Sentinel/"

#dataset_dir = os.path.join(base_dir, 'Dataset/')
dataset_dir = "./Dataset_Combined/"
imagePlots_dir = os.path.join(base_dir, 'ImagePlots/')
modelCheckpoint_dir = os.path.join(base_dir, 'Checkpoint/')
modelCheckpointFiles = os.path.join(modelCheckpoint_dir, Run_Name)
Label_Binarizer_Save = modelCheckpointFiles + '_lb.pickle'
Model_Save = modelCheckpointFiles + '.h5'
#base_dir = '../kaggleCatsDogs/PetImages/'

data = []
labels = []

# Grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(dataset_dir)))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:

	image = cv2.imread(imagePath)
	try:
		#image = cv2.resize(image, (50,50))
		image.flatten()
		data.append(image)
		# Extract the class label from the image path and update the labels list
		label = imagePath.split(os.path.sep)[-2]
		labels.append(label)
	except:
		print(imagePath)


# Scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)


# Partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)

# Convert the labels from integers to vectors (for 2-class, binary
# classification you should use Keras' to_categorical function
# instead as the scikit-learn's LabelBinarizer will not return a
# vector)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
Number_Of_Classes = len(lb.classes_)

model = Sequential()
model.add(Conv2D(input_shape=(9,50,50),filters=64,kernel_size=(3,3),padding="same", activation="relu"))

#  Convolutional layers 1 (Conv 1-1, 1-2, Pooling)
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
# model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

#Convolutional Layers 2 (Conv 2-1, 2-2, Pooling)
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

#Convolutional Layers 3 (Conv 3-1, 3-2, 3-3, Pooling)
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

#Convolutional Layers 4 (Conv 4-1, 4-2, 4-3, Pooling)
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

#Convolutional Layers 5 (Conv 5-1, 5-2, 5-3, Pooling)
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=Number_Of_Classes, activation="softmax"))

##

from tensorflow.keras.optimizers import Adam
opt = Adam(lr=Learning_Rate)

# sparse_categorical_crossentropy for datasets consisting of two classes (so binary classification)
# categorical_crossentropy for greater than two classes

#model.compile(optimizer=opt, loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

#Save Labels
f = open(Label_Binarizer_Save, "wb")
f.write(pickle.dumps(lb))
f.close()

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
Trained_Model_Checkpoint = os.path.join(modelCheckpoint_dir, Model_Save)
checkpoint = ModelCheckpoint(Trained_Model_Checkpoint, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

#hist = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=5)
history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=Epochs, batch_size=Batch_Size, callbacks=[checkpoint,early])

##
# Evaluation utilities

# Metrics
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

# Accuracy and Loss Graphs
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig(imagePlots_dir + Run_Name + '_TrainValAcc.png')

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig(imagePlots_dir + Run_Name + '_TrainValLoss.png')

plt.show()

##
