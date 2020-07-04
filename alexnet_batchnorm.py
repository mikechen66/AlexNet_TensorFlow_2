# alexnet_batchnorm.py

"""

AlexNet, Krizhevsky, Alex, Ilya Sutskever and Geoffrey E. Hinton, 2012
https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
Michael Guerzhoy and Davi Frossard, 2017
AlexNet implementation in TensorFlow, with weights Details: 
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
Stanford c231n 
https://cs231n.github.io/convolutional-networks/#conv

The script is the realization of command oriented style of AlexNet. It is similar to the original style 
of AlexNet realization by Michael Guerzhoy on 2017. In consideration of the actual inputs with 1000 
classes, we set the same class numbers. The model is distinguished with Batch Normalization. 

According to the formula of Stanford cs231, W_output=(W-F+2P)S+1. W,F,P,S are input width, filter width, 
padding size and stride respectively. It is the apparent result of H_output=W_output since we requires 
square size of filters. 
"""


import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D


# Set the class number as 1000
num_classes = 1000
# Set up the sequential model
model = keras.Sequential()


# 1st Convolutional Layer: (227-11+2*0)/4 + 1 = 55
model.add(keras.layers.Conv2D(input_shape=(227, 227, 3),
                              kernel_size=(11, 11),
                              strides=(4, 4),
                              padding="valid",
                              filters=96,
                              activation=tf.nn.relu))
model.add(keras.layers.MaxPooling2D(pool_size=(3, 3),
                                    strides=(2, 2),
                                    padding="valid"))
model.add(keras.layers.BatchNormalization())


# 2nd Convolutional Layer: (27-5+2*2)/1 + 1 = 27
model.add(keras.layers.Conv2D(kernel_size=(5, 5),
                              strides=(1, 1),
                              padding="same",
                              filters=256,
                              activation=tf.nn.relu))
model.add(keras.layers.MaxPooling2D(pool_size=(3, 3),
                                    strides=(2, 2),
                                    padding="valid"))
model.add(keras.layers.BatchNormalization())


# 3rd Convolutional Layer: (13-3+ 2*1)/1 + 1 = 13
model.add(keras.layers.Conv2D(kernel_size=(3,3),
                              strides=(1,1),
                              padding="same",
                              filters=384,
                              activation=tf.nn.relu))
model.add(keras.layers.BatchNormalization())


# 4th Convolutional Layer: (13-3+2*1) + 1 = 13
model.add(keras.layers.Conv2D(kernel_size=(3,3),
                              strides=(1,1),
                              padding="same",
                              filters=384,
                              activation=tf.nn.relu))
model.add(keras.layers.BatchNormalization())


# 5th Convolutional Layer: (13-3+2*1) + 1 = 13
model.add(keras.layers.Conv2D(kernel_size=(3,3),
                              strides=(1,1),
                              padding="same",
                              filters=256,
                              activation=tf.nn.relu))
model.add(keras.layers.MaxPooling2D(pool_size=(3,3),
                                    strides=(2,2),
                                    padding="valid"))
model.add(keras.layers.BatchNormalization())


# Flatten 256 x 6 x 6 = 9216 as one dimention vector and pass it to the 6th Fully
# Connected layer
model.add(keras.layers.Flatten())


# 6th layer: fully connected layer with 4096 neurons with 50% dropout and batch normalization.
model.add(keras.layers.Dense(units=4096,
                             activation=tf.nn.relu))
model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.BatchNormalization())


# 7th layer: fully connected layer with 4096 neurons with 50% dropout and batch normalization.
model.add(keras.layers.Dense(units=4096,
                             activation=tf.nn.relu))
model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.BatchNormalization())


# 8th Layer: fully connected layer with 1000 neurons with 50% dropout and batch normalization.
model.add(keras.layers.Dense(units=1000,
                             activation=tf.nn.relu))
model.add(Dense(num_classes, activation='softmax'))
model.add(keras.layers.BatchNormalization())


# Compile the model using Adam optimizer and categorical_crossentropy loss function.
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Give the model structure summary 
model.summary()
