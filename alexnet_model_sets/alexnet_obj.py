#!/usr/bin/env python
# coding: utf-8

# alexnet_obj.py 

"""
The script is the realization of object-oriented style of the AlexNet model with he_normal. It has a 
consolidated structure in the purely Tensorflow 2.x. We set the same 1000 class numbers accordingly. 

According to the formula of Stanford cs231, W_output = (W-F+2P)/S + 1. W,F,P,S are input width, filter 
width, padding size and stride respectively. It is the apparent result of H_output = W_output since we 
requires the square size of filters.

Stanford c231n 
https://cs231n.github.io/convolutional-networks/#conv

AlexNet, Krizhevsky, Alex, Ilya Sutskever and Geoffrey E. Hinton, 2012
https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

Michael Guerzhoy and Davi Frossard, 2017, AlexNet implementation in TensorFlow, with weights Details: 
http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout


# Define the AlexNet Model 
class AlexNet(Sequential):

    def __init__(self, input_shape, num_classes):
        super().__init__()

        # No.1 Convolutional Layer: (227- 11 + 2 * 0) / 4 + 1 = 55
        self.add(Conv2D(kernel_size=(11,11), strides=(4,4), padding="valid", filters=96, activation='relu', kernel_initializer='he_normal',
                        input_shape=input_shape))
        # Max Pooling: (55- 3 + 2 * 0) / 2 + 1 = 27
        self.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
        
        # No.2 Conv Layer: (27- 5 + 2 * 2) / 1 + 1 = 27
        self.add(Conv2D(kernel_size=(5,5), strides=(1,1), padding="same", filters=256, activation='relu', kernel_initializer='he_normal'))
        # Max Pooling: (27-  3 + 2 * 0) / 2 + 1 = 13
        self.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
        
        # No.3 Conv Layer: (13 - 3 + 2 * 1) / 1 + 1 = 13
        self.add(Conv2D(kernel_size=(3,3), strides=(1,1), padding="same", filters=384, activation='relu', kernel_initializer='he_normal'))

        # No.4 Conv Layer: (13 - 3 + 2 * 1) / 1 + 1 = 13
        self.add(Conv2D(kernel_size=(3,3), strides=(1,1), padding="same",filters=384, activation='relu', kernel_initializer='he_normal'))

        # No.5 Conv Layer: (13 - 3 + 2 * 1) / 1 + 1 = 13
        self.add(Conv2D(kernel_size=(3,3), strides=(1,1), padding="same", filters=256, activation='relu', kernel_initializer='he_normal')) 
        # Max Pooling: (13 - 3 + 2 * 0) / 2 + 1 =  6
        self.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

        # Flatten the three dimensions of 256 x 6 x 6 into one dimension of 9216.
        self.add(Flatten())

        # 6th layer: fully connected layer with 4096 neurons with 50% dropout.
        self.add(Dense(units=4096, activation='relu'))
        # Add Dropout
        self.add(Dropout(0.5))
        
        # 7th layer: a fully connected layer with 4096 neurons with 50% dropout.        
        self.add(Dense(units=4096, activation='relu'))
        # Add Dropout 
        self.add(Dropout(0.5))

        # 8th Layer: a fully connected layer with 1000 neurons.
        self.add(Dense(units=1000, activation='relu'))
        self.add(Dense(units=num_classes, activation='softmax'))


# Provide the constants for the function. 
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227
CHANNELS = 3
NUM_CLASSES = 1000

# Assign the vlaues 
INPUT_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)


# Call the AlexNet model 
model = AlexNet(INPUT_SHAPE, NUM_CLASSES)

# Show the AlexNet Model 
model.summary()
