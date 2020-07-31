#!/usr/bin/env python
# coding: utf-8

"""
# Set up the GPU in the condition of allocation exceeds system memory with the reminding message: Could not 
# create cuDNN handle... The following lines of code can avoids the sudden stop of the runtime. 
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
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


# Get the ship synset
page = requests.get("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n04194289")
soup = BeautifulSoup(page.content, 'html.parser')

# Get the bicycle synset
bikes_page = requests.get("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n02834778")
bikes_soup = BeautifulSoup(bikes_page.content, 'html.parser')

str_soup=str(soup)
split_urls=str_soup.split('\r\n')

bikes_str_soup=str(bikes_soup)
bikes_split_urls=bikes_str_soup.split('\r\n')

# Create the respective directories
path1 = os.makedirs('/home/john/Documents/Alexnet_Callback/content/train', mode=0o777)
path2 = os.makedirs('/home/john/Documents/Alexnet_Callback/content/train/ships', mode=0o777)
path3 = os.makedirs('/home/john/Documents/Alexnet_Callback/content/train/bikes', mode=0o777)
path4 = os.makedirs('/home/john/Documents/Alexnet_Callback/content/validation', mode=0o777)
path5 = os.makedirs('/home/john/Documents/Alexnet_Callback/content/validation/ships', mode=0o777)
path6 = os.makedirs('/home/john/Documents/Alexnet_Callback/content/validation/bikes', mode=0o777)

# Dimentions of input images
img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols, 3)


def url_to_image(url):
    resp = urllib.request.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)

	return image

num_of_training_images=100
for progress in range(num_of_training_images):
    if not split_urls[progress] == None:
        try:
            img = url_to_image(split_urls[progress])
            if (len(img.shape))==3:
                save_path = '/home/john/Documents/Alexnet_Callback/content/train/ships/img'+str(progress)+'.jpg'
                cv2.imwrite(save_path,img)
        except:
            None

for progress in range(num_of_training_images):
    if not bikes_split_urls[progress] == None:
        try:
            img = url_to_image(bikes_split_urls[progress])
            if (len(img.shape))==3:
                save_path = '/home/john/Documents/Alexnet_Callback/content/train/bikes/img'+str(progress)+'.jpg'
                cv2.imwrite(save_path,img)
        except:
            None

num_of_validation_images=50
for progress in range(num_of_validation_images):
    if not split_urls[progress] == None:
        try:
            img = url_to_image(split_urls[num_of_training_images+progress])
            if (len(img.shape))==3:
                save_path = '/home/john/Documents/Alexnet_Callback/content/validation/ships/img'+str(progress)+'.jpg'
                cv2.imwrite(save_path,img)
        except:
            None

for progress in range(num_of_validation_images):
    if not bikes_split_urls[progress] == None:
        try:
            img = url_to_image(bikes_split_urls[num_of_training_images+progress])
            if (len(img.shape))==3:
                save_path = '/home/john/Documents/Alexnet_Callback/content/validation/bikes/img'+str(progress)+'.jpg'
                cv2.imwrite(save_path,img)
        except:
            None
 
# Modify the original ipython code into the lines of code in Python          
array1 = os.listdir('/home/john/Documents/Alexnet_Callback/content/train/ships')
array2 = os.listdir('/home/john/Documents/Alexnet_Callback/content/train/bikes')
array3 = os.listdir('/home/john/Documents/Alexnet_Callback/content/validation/ships')
array4 = os.listdir('/home/john/Documents/Alexnet_Callback/content/validation/bikes')

# Add the print function to show the results of images ended with jpg
print(array1, array2, array3, array4) 


# Set up the GPU in the condition of allocation exceeds system memory. 
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# It has pre-defined two classes including bike and ship. 
num_classes = 2

# It calls the alexnet model in alexnet.py
model = AlexNet((227, 227, 3), num_classes)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# It will output the AlexNet model after executing the command 
model.summary()


# Give some training parameters
EPOCHS = 100
BATCH_SIZE = 32
image_height = 227
image_width = 227
train_dir = '/home/john/Documents/Alexnet_Callback/content/train'
valid_dir = '/home/john/Documents/Alexnet_Callback/content/validation'
model_dir = '/home/john/Documents/Alexnet_Callback/content/my_model.h5'


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


# Start Tensorboard --logdir logs/fit
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
callback_list = [tensorboard_callback]

# The system starts the training. We change the verbose=0 to verbose=1. So users can see the dynamic 
# precudure of training and validation execution. 
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
