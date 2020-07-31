# AlexNet Model 

"""
AlexNet, Krizhevsky, Alex, Ilya Sutskever and Geoffrey E. Hinton, 2012
https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

Michael Guerzhoy and Davi Frossard, 2016
AlexNet implementation in TensorFlow, with weights Details: 
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/

The script is the realizatio of object oriented style of AlexNet. The construction method of AlexNet 
includes three parameters, including self, input_shape, num_classes, of which, input_shape works as a 
placeholder. 
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout


# Define the AlexNet Model 
class AlexNet(Sequential):
    def __init__(self, input_shape, num_classes):
        super().__init__()

        # No.1 Convolutional Layer: (227- 11 + 2 * 0) / 4 + 1 = 55
        self.add(Conv2D(filters=96, kernel_size=(11,11), strides=4,
                        padding='valid', activation='relu',
                        input_shape=input_shape, kernel_initializer='GlorotNormal'))
        # Max Pooling: (55- 3 + 2 * 0) / 2 + 1 = 27
        self.add(MaxPooling2D(pool_size=(3,3), strides=(2,2),
                              padding='valid', data_format=None))
        
        # No.2 Conv Layer: (27- 5 + 2 * 2) / 1 + 1 = 27
        self.add(Conv2D(filters=256, kernel_size=(5,5), strides=1,
                        padding='same', activation='relu',
                        kernel_initializer='GlorotNormal'))
        # Max Pooling: (27-  3 + 2 * 0) / 2 + 1 = 13
        self.add(MaxPooling2D(pool_size=(3,3), strides=(2,2),
                              padding='valid', data_format=None))
        
        # No.3 Conv Layer: (13 - 3 + 2 * 1) / 1 + 1 = 13
        self.add(Conv2D(filters=384, kernel_size=(3,3), strides=1,
                        padding='same', activation='relu',
                        kernel_initializer='GlorotNormal'))

        # No.4 Conv Layer: (13 - 3 + 2 * 1) / 1 + 1 = 13
        self.add(Conv2D(filters=384, kernel_size=(3,3), strides=1,
                 padding='same', activation='relu',
                 kernel_initializer='GlorotNormal'))

        # No.5 Conv Layer: (13 - 3 + 2 * 1) / 1 + 1 = 13
        self.add(Conv2D(filters=256, kernel_size=(3,3), strides=1,
                 padding='same', activation='relu',
                 kernel_initializer='GlorotNormal')) 
        # Max Pooling: (13 - 3 + 2 * 0) / 2 + 1 =  6
        self.add(MaxPooling2D(pool_size=(3,3), strides=(2,2),
                              padding='valid', data_format=None))

        # # Flatten the three dimensions of 256 x 6 x 6 into one dimension of 9216.
        self.add(Flatten())

        # No.6 FC Layer 
        self.add(Dense(4096, activation='relu'))
        # Add Dropout
        self.add(Dropout(0.5))
        
        # No.7 FC Layer         
        self.add(Dense(4096, activation='relu'))
        # Add Dropout 
        self.add(Dropout(0.5))

        # No.8 FC Layer
        self.add(Dense(1000, activation='relu'))
        self.add(Dense(num_classes, activation='softmax'))
        
        
