"""
# AlexNet Model Functional 

# AlexNet is the milestone in the image recognition benchmark competition. However, most of the textbook and 
# referencebooks take the LeNet as AI code example for granted. I think that the LeNet code far less addresses 
# the real value of the Deep Learning and stops the deeper AI learning of the general users. So it is necessary 
# to focus on AlexNet - the really important AI model with practical code examples in the truely large-scale 
# (1 million images) datasets such as ILSVRC2012 ~ ILSVRC2012 not the toy datasets most of the code examples 
# refers to. Please note the two critical points as follows. 
# 
# 1.Functional API
# 
# The model complies with the Keras Funcational API. It means input is feedforwarded each time until the output 
# stage. Users may use the Sequential API for a simple usage. But the Funcational API is more flexible for the 
# transition to the more complex models such as GoogLeNet, Inception variations and ResNet. 

# AlexNet Paper: 
# https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
#
# 2.BatchNormalization
# 
# BatchNormalization is introduced by Google in Batch Normalization paper. It set mean and variance as 0 and 1 
# respectively in order to reduce Internal Covariate Shift within the hidden layers during the training. But
# users need to cautiously make use of them becuase it greatly affects the training time. Here set only two 
# times of BatchNormalization for speed up the training time. 
# 
# Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
# https://arxiv.org/pdf/1502.03167.pdf
#
# Rethinking “Batch” in BatchNorm
# https://arxiv.org/pdf/2105.07576.pdf
"""


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPooling2D, Dense, Flatten, Dropout

import os 
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


def alexnet(input_shape):
    
    # inputs = Input(shape=(227,227,3), name="alexnet_input")
    inputs = Input(input_shape, name="alexnet_input")

    # Layer 1    
    x = Conv2D(96, (11,11), strides=4, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((3,3), strides=2)(x)

    # Layer 2
    x = Conv2D(256, (5,5), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((3,3), strides=2)(x)

    # Layer 3
    x = Conv2D(384, (3,3) , strides=1, padding='same')(x)
    x = ReLU()(x)

    # Layer 4
    x = Conv2D(384, (3,3) , strides=1, padding='same')(x)
    x = ReLU()(x)

    # Layer 5
    x = Conv2D(256, (3,3) , strides=1, padding='same')(x)
    x = ReLU()(x)
    x = MaxPooling2D((3,3), strides=2)(x)

    # Layer 6
    x = Flatten()(x)

    x = Dense(4096)(x)
    x = ReLU()(x)
    x = Dropout(rate=0.5)(x)

    # Layer 7
    x = Dense(4096)(x)
    x = ReLU()(x)
    x = Dropout(rate=0.5)(x)

    # Layer 8
    outputs = Dense(1000, activation='softmax', dtype=tf.float32, name="alexnet_output")(x)

    alexnet = Model(inputs=inputs, outputs=outputs)

    return alexnet


alexnet(input_shape=(227,227,3)).summary()

# Show the dtructure diagram only for Jupyter Notebook 
plot_model(alexnet(input_shape=(227,227,3)), show_layer_names=False, show_shapes=True, show_dtype=True)