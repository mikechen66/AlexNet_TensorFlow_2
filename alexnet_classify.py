#!/usr/bin/env python
# coding: utf-8

# It is originally written with ipython by datahacker and re-written with python and adding some 
# lines of code in order to adapt the Python practice by Mike Chen. 
# http://media5.datahacker.rs/2018/06/logo-crno.png)

"""
The User Guide: 

Part One

To use the script in Python, users need to create the folder such as AlexNet-tf2. The application 
automatically downloads the pictures into the created folders. 

Part Two. Script running procedure

1. Enter into current directory

   $ cd /home/user/Documents/Alexnet_Callback

Anaconda defaults the pre-installed Python3 and the Ubuntu 18.04 has both Python2 and Python3. Therefore, 
users need to follow the procedures. 

2. Script running command

  In the Conda Environment, please execute the following command in the Ubuntu terminal at the current 
  directory.  
  
  $ python alexnet_classify.py  
  
  or 

  In the native Ubuntu 18.04 env, please execute the following command. 

  $ python3 alexnet_classifying.py

  While executing the above-mentioned command, the Linux Terminal shows the arrays of image name ended 
  with jpg. 

  Moreover, the Terminal show the complete Model: alex_net". Furthermore, it show "Found 117 images 
  belonging to 2 classes". 

  In the meantime, it also address the following warning. However, users can ignore the warning becuase it
  does not influence the script running. 

  WARNING:tensorflow:sample_weight modes were coerced from
  ...
    to  
  ['...']
  

3. Start the TensorBoard

   After completing the script excution, users can start the TensorBoard command in the Linux Terminal 
   at the current directory. 

  $ tensorboard --logdir logs/fit

  After the above-mentioned command is given, the ｔerminal shows the reminding message as follows. 
  Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
  TensorBoard 2.2.1 at http://localhost:6006/ (Press CTRL+C to quit)

4. Enter the weblink in a browser

   http://localhost:6006/

   After entering the weblink into either Chrome or Firefox browser, the TensorBoard will show the diagrams
   that the scrip defines. 

5. Images showing 
   The browser could not show the images. If users want to plot the images, please upload the Python script 
   into the Jupyter Notebook or just directly adopts the original ipython script. 

Part Three Trouble shooting 

Issue: 
AttributeError: module 'tensorflow' has no attribute 'compat'

Solution: 
It is the conflict between Conda and TensorFlow 2.x if users adopt the Anaconda/Miniconda env. I recommend 
the users to install tensorflow 2.1 and then install tensorflw-estimator as follows. 

$ conda install tensorflow-estimator==2.1.0
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

# Change the original ipython code into the Python code
path1 = os.makedirs('/home/mic/Documents/Alexnet_Callback/content/train', mode=0o777)
path2 = os.makedirs('/home/mic/Documents/Alexnet_Callback/content/train/ships', mode=0o777)
path3 = os.makedirs('/home/mic/Documents/Alexnet_Callback/content/train/bikes', mode=0o777)

path4 = os.makedirs('/home/mic/Documents/Alexnet_Callback/content/validation', mode=0o777)
path5 = os.makedirs('/home/mic/Documents/Alexnet_Callback/content/validation/ships', mode=0o777)
path6 = os.makedirs('/home/mic/Documents/Alexnet_Callback/content/validation/bikes', mode=0o777)

# Change the dimentions of input images to (32, 32)
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
                save_path = '/home/mic/Documents/Alexnet_Callback/content/train/ships/img'+str(progress)+'.jpg'
                cv2.imwrite(save_path,img)
        except:
            None

for progress in range(num_of_training_images):
    if not bikes_split_urls[progress] == None:
        try:
            img = url_to_image(bikes_split_urls[progress])
            if (len(img.shape))==3:
                save_path = '/home/mic/Documents/Alexnet_Callback/content/train/bikes/img'+str(progress)+'.jpg'
                cv2.imwrite(save_path,img)
        except:
            None

num_of_validation_images=50
for progress in range(num_of_validation_images):
    if not split_urls[progress] == None:
        try:
            img = url_to_image(split_urls[num_of_training_images+progress])
            if (len(img.shape))==3:
                save_path = '/home/mic/Documents/Alexnet_Callback/content/validation/ships/img'+str(progress)+'.jpg'
                cv2.imwrite(save_path,img)
        except:
            None

for progress in range(num_of_validation_images):
    if not bikes_split_urls[progress] == None:
        try:
            img = url_to_image(bikes_split_urls[num_of_training_images+progress])
            if (len(img.shape))==3:
                save_path = '/home/mic/Documents/Alexnet_Callback/content/validation/bikes/img'+str(progress)+'.jpg'
                cv2.imwrite(save_path,img)
        except:
            None
 
# Modify the original ipython code into the lines of code in Python          
array1 = os.listdir('/home/mic/Documents/Alexnet_Callback/content/train/ships')
array2 = os.listdir('/home/mic/Documents/Alexnet_Callback/content/train/bikes')
array3 = os.listdir('/home/mic/Documents/Alexnet_Callback/content/validation/ships')
array4 = os.listdir('/home/mic/Documents/Alexnet_Callback/content/validation/bikes')

# Add the print function to show the results of images ended with jpg
print(array1, array2, array3, array4) 


# Set up the GPU in the condition of allocation exceeds system memory with the reminding message: Could not 
# create cuDNN handle... The following lines of code can avoids the sudden stop of the runtime. 
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
train_dir = '/home/mic/Documents/Alexnet_Callback/content/train'
valid_dir = '/home/mic/Documents/Alexnet_Callback/content/validation'
model_dir = '/home/mic/Documents/Alexnet_Callback/content/my_model.h5'


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

# Show the diagram once upon entering the weblink in a browser
# http://localhost:6006/

# The system starts the training. We change the verbose=0 to verbose=1. So users can see the dynamic 
# precudure of training and validation execution. 
model.fit(train_generator,
          epochs=EPOCHS,
          steps_per_epoch=train_num//BATCH_SIZE,
          validation_data=valid_generator,
          validation_steps=valid_num//BATCH_SIZE,
          callbacks=callback_list,
          verbose=1)

# The system saves the whole model into the direcotry: /home/mic/Documents/AlexNet/content. The model 
# of my_model.h5 has the quite big size of 748.6 MB. 
model.save(model_dir)


# To view the diagrams, users need to upload the Python script into Jupyter Notebook and run the the 
# script or directly upload and run the original ipython script. 
class_names = ['bike', 'ship']
x_valid, label_batch = next(iter(valid_generator))
prediction_values = model.predict_classes(x_valid)

# The plot will be realized in the Jupyter Notebook after running the script in either Python or ipython. 
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