#!/usr/bin/env python
# coding: utf-8

# alexnet_sequential.py

"""
The script is the realization of commanline-oriented programming of AlexNet. It is used as snippet 
of code in an application. We set the same 1000 class numbers accordingly. 

According to the formula of Stanford cs231, W_output=(W-F+2P)S+1. W,F,P,S are input width, filter 
width, padding size and stride respectively. It is the apparent result of H_output = W_output since 
we requires square size of filters. 

Stanford c231n 
https://cs231n.github.io/convolutional-networks/#conv

AlexNet, Krizhevsky, Alex, Ilya Sutskever and Geoffrey E. Hinton, 2012
https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

Michael Guerzhoy and Davi Frossard, 2017, AlexNet implementation in TensorFlow, with weights Details: 
http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/ 
"""


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout


# Instantiate an empty model
model = Sequential()

# 1st Convolutional Layer: (227-11+2*0)/4 + 1 = 55
model.add(Conv2D(input_shape=(227,227,3), kernel_size=(11,11), strides=(4,4), padding="valid", filters=96, activation='relu'))
# Max Pooling: (55-3+2*0)/2 + 1 = 27
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

# 2nd Convolutional Layer: (27-5+2*2)/1 + 1 = 27
model.add(Conv2D(kernel_size=(5,5), strides=(1,1), padding="same", filters=256, activation='relu'))
# Max Pooling: (27-3+2*0)/2 +1 = 13
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

# 3rd Convolutional Layer: (13-3+ 2*1)/1 + 1 = 13
model.add(Conv2D(kernel_size=(3,3), strides=(1,1), padding="same", filters=384, activation='relu'))

# 4th Convolutional Layer: (13-3+2*1) + 1 = 13
model.add(Conv2D(kernel_size=(3,3), strides=(1,1), padding="same", filters=384, activation='relu'))

# 5th Convolutional Layer: (13-3+2*1) + 1 = 13
model.add(Conv2D(kernel_size=(3,3), strides=(1,1), padding="same", filters=256, activation='relu'))
# Max Pooling: (13-3+2*0)/2 + 1 =  6
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

# Flatten 256x6x6=9216 as one dimention vector and pass it to the 6th fully connected layer
model.add(Flatten())

# 6th layer: a fully connected layer with 4096 neurons with 50% dropout.
model.add(Dense(units=4096, activation='relu'))
# Add Dropout
model.add(Dropout(0.5))

 # 7th layer: a fully connected layer with 4096 neurons with 50% dropout.
model.add(Dense(units=4096, activation='relu'))
# Add Dropout 
model.add(Dropout(0.5))

# 8th Layer: a fully connected layer with 1000 neurons. 
model.add(Dense(units=1000, activation='relu'))
model.add(Dense(units=1000, activation='softmax'))


# Show the AlexNet Model 
model.summary()


