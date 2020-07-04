# alexnet_keras.py

"""
AlexNet, Krizhevsky, Alex, Ilya Sutskever and Geoffrey E. Hinton, 2012
https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

Michael Guerzhoy and Davi Frossard, 2017
AlexNet implementation in TensorFlow, with weights Details: 
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/

Stanford c231n 
https://cs231n.github.io/convolutional-networks/#conv

The script is the realization of command oriented style of AlexNet. It is similar to the original style of
AlexNet realization on 2017. In consideration of the actual inputs has at least 2 classes, we set the class 
nunber from 2 to 1000. 

According to the formula of Stanford cs231, W_output=(W-F+2P)S+1. W,F,P,S are input width, filter width, 
padding size and stride respectively. It is apparent result of H_output=W_output since the square size of 
filters is necessary. 
"""

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
# -from keras.layers.normalization import BatchNormalization
import numpy as np

# Generate the random integer from 2 to 1000
num_classes = np.random.randint(2, 1000)
# Instantiate an empty model
model = Sequential() 


# 1st Convolutional Layer: (227-11+2*0)/4 + 1 = 55
model.add(Conv2D(input_shape=(227,227,3), filters=96, kernel_size=(11,11), 
               strides=4, padding='valid', activation='relu'))
# Max Pooling: (55-3+2*0)/2 + 1 = 27
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

# 2nd Convolutional Layer: (27-5+2*2)/1 + 1 = 27
model.add(Conv2D(filters=256, kernel_size=(5,5), strides=1, 
               padding='same', activation='relu'))
# Max Pooling: (27-3+2*0)/2 +1 = 13
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

# 3rd Convolutional Layer: (13-3+ 2*1)/1 + 1 = 13
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=1, 
               padding='same', activation='relu'))

# 4th Convolutional Layer: (13-3+2*1) + 1 = 13
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=1, 
               padding='same', activation='relu'))

# 5th Convolutional Layer: (13-3+2*1) + 1 = 13
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=1, 
               padding='same', activation='relu'))
# Max Pooling: (13-3+2*0)/2 + 1 =  6
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

# Flatten 256 x 6 x 6 = 9216 as one dimention vector and pass it to the 6th Fully
# Connected layer
model.add(Flatten())

# 6th Fully Connected Layer
model.add(Dense(4096, activation='relu'))
# Add Dropout
model.add(Dropout(0.5))

# 7th Fully Connected Layer
model.add(Dense(4096, activation='relu'))
# Add Dropout 
model.add(Dropout(0.5))

# No.8 FC Layer
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Show the AlexNet Model 
model.summary()


