#!/usr/bin/env python
# coding: utf-8

# alexnet_batchnorm.py

"""
The script is the realization of commandline-oriented style of AlexNet with Batch Normalization. It is 
similar to the original AlexNet by Michael Guerzhoy on 2017 but has a much consolidated structure in 
the purely Tensorflow 2.x. We set the same 1000 class numbers accordingly. 

According to the formula of Stanford cs231, W_output=(W-F+2P)S+1. W,F,P,S are input width, filter width, 
padding size and stride respectively. It is the apparent result of H_output = W_output since we requires 
square size of filters.

Stanford c231n 
https://cs231n.github.io/convolutional-networks/#conv

AlexNet, Krizhevsky, Alex, Ilya Sutskever and Geoffrey E. Hinton, 2012
https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

Michael Guerzhoy and Davi Frossard, 2017, AlexNet implementation in TensorFlow, with weights Details: 
http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
"""


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, BatchNormalization


# Set up the sequential model
model = Sequential()


# 1st Convolutional Layer: (227-11+2*0)/4 + 1 = 55
model.add(Conv2D(kernel_size=(11,11), strides=(4,4), padding="valid", filters=96, activation='relu', 
                 input_shape=(227,227,3)))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid"))
model.add(BatchNormalization())

# 2nd Convolutional Layer: (27-5+2*2)/1 + 1 = 27
model.add(Conv2D(kernel_size=(5,5), strides=(1,1), padding="same", filters=256, activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid"))
model.add(BatchNormalization())

# 3rd Convolutional Layer: (13-3+ 2*1)/1 + 1 = 13
model.add(Conv2D(kernel_size=(3,3), strides=(1,1), padding="same", filters=384, activation='relu'))
model.add(BatchNormalization())

# 4th Convolutional Layer: (13-3+2*1) + 1 = 13
model.add(Conv2D(kernel_size=(3,3), strides=(1,1), padding="same",filters=384, activation='relu'))
model.add(BatchNormalization())

# 5th Convolutional Layer: (13-3+2*1) + 1 = 13
model.add(Conv2D(kernel_size=(3,3), strides=(1,1), padding="same", filters=256, activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid"))
model.add(BatchNormalization())

# Flatten 256 x 6 x 6 = 9216 as one dimention vector and pass it to the 6th FC layer
model.add(Flatten())

# 6th layer: fully connected layer with 4096 neurons with 50% dropout and batch normalization.
model.add(Dense(units=4096, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(BatchNormalization())

# 7th layer: fully connected layer with 4096 neurons with 50% dropout and batch normalization.
model.add(Dense(units=4096, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(BatchNormalization())


# 8th Layer: a fully connected layer with 1000 neurons with batch normalization.
model.add(Dense(units=1000, activation= 'relu'))
model.add(Dense(units=1000, activation='softmax'))
model.add(BatchNormalization())


# Give the model structure summary 
model.summary()
