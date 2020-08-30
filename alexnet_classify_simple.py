#!/usr/bin/env python
# coding: utf-8

"""
After completion of downloading the data with alexnet_classify.py, users can use the lightweight
script to call alexnet.py. Therefore, the download and processing lines of code are deleted. To 
run the script on the environment of TensorFlow 2.2.0 and CUDA 11.0/cudNN 8.0.1, users need to 
add -cap-add=CAP_SYS_ADMIN

$ python3 alexnet_classify_simple.py --cap-add=CAP_SYS_ADMIN
"""

import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from alexnet import AlexNet

import os 
import cv2
import urllib
import requests
import PIL.Image
import numpy as np
from bs4 import BeautifulSoup


# Set up the GPU in the condition of allocation exceeds system memory.The following lines 
# of code can avoids the sudden stop of the runtime. 
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# Confirm the directories 
path1 = '/home/mike/Documents/Alexnet_Callback/content/train'
path2 = '/home/mike/Documents/Alexnet_Callback/content/train/ships'
path3 = '/home/mike/Documents/Alexnet_Callback/content/train/bikes'
path4 = '/home/mike/Documents/Alexnet_Callback/content/validation'
path5 = '/home/mike/Documents/Alexnet_Callback/content/validation/ships'
path6 = '/home/mike/Documents/Alexnet_Callback/content/validation/bikes'


# It has pre-defined two classes including bike and ship. 
EPOCHS = 100
BATCH_SIZE = 32
image_width = 227
image_height = 227
channels = 3
num_classes = 2


# Change the original images dimentions to 32 x 32. 
img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols, 3)

 
# Modify the original ipython code into the lines of code in Python          
array1 = os.listdir('/home/mike/Documents/Alexnet_Callback/content/train/ships')
array2 = os.listdir('/home/mike/Documents/Alexnet_Callback/content/train/bikes')
array3 = os.listdir('/home/mike/Documents/Alexnet_Callback/content/validation/ships')
array4 = os.listdir('/home/mike/Documents/Alexnet_Callback/content/validation/bikes')


# Add the print function to show the results of images ended with jpg
print(array1, array2, array3, array4) 


# Call the alexnet model in alexnet.py
model = AlexNet((227, 227, 3), num_classes)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# It will output the AlexNet model after executing the command 
model.summary()


# Designate the directories for training, validation and model saved. 
train_dir = '/home/mike/Documents/Alexnet_Callback/content/train'
valid_dir = '/home/mike/Documents/Alexnet_Callback/content/validation'
model_dir = '/home/mike/Documents/Alexnet_Callback/content/my_model.h5'


# Assign both the image and the diretory generators
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=10,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.1,
                                   zoom_range=0.1)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(image_height, image_width),
                                                    color_mode="rgb",
                                                    batch_size=BATCH_SIZE,
                                                    seed=1,
                                                    shuffle=True,
                                                    class_mode="categorical")

valid_datagen = ImageDataGenerator(rescale=1.0/255.0)
valid_generator = valid_datagen.flow_from_directory(valid_dir,
                                                    target_size=(image_height, image_width),
                                                    color_mode="rgb",
                                                    batch_size=BATCH_SIZE,
                                                    seed=7,
                                                    shuffle=True,
                                                    class_mode="categorical")

# The terminal shows the number of images belonging to 2 classes. 
train_num = train_generator.samples
valid_num = valid_generator.samples


# Need to start the following command in Ubuntu Terminal after executing the script. 
# tensorboard --logdir logs/fit
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
callback_list = [tensorboard_callback]


# Set up the verbose=1 (or verbose=1) for visible (or invisible) epochs. 
model.fit(train_generator,
          epochs=EPOCHS,
          steps_per_epoch=train_num//BATCH_SIZE,
          validation_data=valid_generator,
          validation_steps=valid_num//BATCH_SIZE,
          callbacks=callback_list,
          verbose=1)


# The system saves the whole model into the direcotry: /home/mike/Documents/AlexNet-tf2/content. The 
# model of my_model.h5 has the quite big size of 748.6 MB. 
model.save(model_dir)


# To view the diagrams, users need to upload the Python script into Jupyter Notebook and run the 
# the script or directly upload and run the original ipython script. 
class_names = ['bike', 'ship']
x_valid, label_batch  = next(iter(valid_generator))
prediction_values = model.predict_classes(x_valid)


# The plot will be realized in the Jupyter Notebook after running the script in either Python or 
# ipython. 
fig = plt.figure(figsize=(10, 6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# Plot the images: each image is 227x227 pixels
for i in range(8):
    ax = fig.add_subplot(2, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(x_valid[i,:], cmap=plt.cm.gray_r, interpolation='nearest')
    if prediction_values[i] == np.argmax(label_batch[i]):
        # Label the image with the blue text
        ax.text(3, 17, class_names[prediction_values[i]], color='blue', fontsize=14)
    else:
        # Label the image with the red text
        ax.text(3, 17, class_names[prediction_values[i]], color='red', fontsize=14)
