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
from tensorflow.keras import optimizers, metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Input


# Define the AlexNet Model 
# class AlexNet(tf.Module):
class AlexNet: 

    # def __init__(self, input_shape, num_classes):
    def __init__(self):
        self.conv_base(input_shape)
        self.fc_base(num_classes)

    def conv_base(self, input_shape): 

        model = Sequential()

        # No.1 Convolutional Layer: (227- 11 + 2 * 0) / 4 + 1 = 55
        model.add(Conv2D(filters=96, kernel_size=11, strides=4, padding='valid', activation='relu',
               input_shape=input_shape, kernel_initializer='GlorotNormal'))
        # Max Pooling: (55- 3 + 2 * 0) / 2 + 1 = 27
        model.add(MaxPooling2D(pool_size=3, strides=2, padding='valid', data_format=None))
    
        # No.2 Conv Layer: (27- 5 + 2 * 2) / 1 + 1 = 27
        model.add(Conv2D(filters=256, kernel_size=5, strides=1, padding='same', activation='relu',
               kernel_initializer='GlorotNormal'))
        # Max Pooling: (27 - 3 + 2 * 0) / 2 + 1 = 13
        model.add(MaxPooling2D(pool_size=3, strides=2, padding='valid', data_format=None))
    
        # No.3 Conv Layer: (13 - 3 + 2 * 1) / 1 + 1 = 13
        model.add(Conv2D(filters=384, kernel_size=3, strides=1, padding='same', activation='relu',
               kernel_initializer='GlorotNormal'))

        # No.4 Conv Layer: (13 - 3 + 2 * 1) / 1 + 1 = 13
        model.add(Conv2D(filters=384, kernel_size=(3,3), strides=1, padding='same', activation='relu',
               kernel_initializer='GlorotNormal'))

        # No.5 Conv Layer: (13 - 3 + 2 * 1) / 1 + 1 = 13
        model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu',
               kernel_initializer='GlorotNormal'))
        # Max Pooling: (13 - 3 + 2 * 0) / 2 + 1 =  6
        model.add(MaxPooling2D(pool_size=3, strides=2, padding='valid', data_format=None))
        # Flatten the three dimensions of 256 x 6 x 6 into one dimension of 9216.
        # Flatten()

        return model 

    def fc_base(self, num_classes): 

        model = self.conv_base(input_shape)

        # Flatten the three dimensions of 256 x 6 x 6 into one dimension of 9216.
        model.add(Flatten())
        # No.6 FC Layer 
        model.add(Dense(4096, activation='relu'))
        # Add the Dropout
        model.add(Dropout(0.5))
    
        # No.7 FC Layer         
        model.add(Dense(4096, activation='relu'))
        # Add the Dropout 
        model.add(Dropout(0.5))

        # No.8 FC Layer
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        return model


if __name__ == "__main__":  


    input_shape = (227, 227, 3)
    num_classes = 1000


    # Call the AlexNet model 
    model = AlexNet().fc_base(num_classes)
 
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Show the full model structure of AlexNet 
    model.summary()