#!/usr/bin/env python
# coding: utf-8
"""

# ##AlexNet Model Functional 

# AlexNet is the milestone in the ILSVRC benchmark image recognition competition. However, most of the textbook 
# and reference books take the LeNet as AI code example for granted. I think the LeNet code far less addresses 
# the real value of the Deep Learning and prevent the users from enjoying the real interesting large-scale image
# classification on big data. So it is necessary to focus on AlexNet - the really important AI model with practical 
# code examples in the truely large-scale(1.2 million images) datasets such as ILSVRC2012 rather than the toy 
# datasets with 10GB provided whithin most of the image deep learning code examples. Please note the two critical 
# points as follows. 
# 
# 1.Functional API
# 
# The model complies with the Keras Funcational API. It means input is feedforwarded each time until the output 
# stage. Users may use the Sequential API for a simple usage. But the Funcational API is more flexible for the 
# transition to the more complex models such as GoogLeNet, Inception variations and ResNet. 

# AlexNet - ImageNet Classification with Deep Convolutional Neural Networks: 
# https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
#
# 2.BatchNormalization
# 
# BatchNormalization is introduced by Google in Batch Normalization paper. It set mean and variance as 0 and 1 
# respectively in order to reduce Internal Covariate Shift within the hidden layers during the training. But users 
# need to cautiously make use of them becuase it greatly affects the training time. Here set only two times of 
# BatchNormalization for speed up the training time. 
# 
# Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
# https://arxiv.org/pdf/1502.03167.pdf
#
# Rethinking “Batch” in BatchNorm
# https://arxiv.org/pdf/2105.07576.pdf
#
#
# ## Data Loading Notes: 
#
# Please download the ImageNet dataset from the above link. There are two methods to get the dataset. One is the 
# official image-net website but users may need an college student indntity and complex application procedure. 
# And another (both simple and practical) method is to download from the public resources. 
# 
# 1.ImageNet Official Website for College Users: 
#
# https://image-net.org/download-images
# 
# 2.Academictorrents Website for Other Users: 
# 
# https://academictorrents.com/
# 
# The original name of ImageNet dataset is ILSVRC2012. Users can change the name to imagenet2012 or keep it no 
# changed. Principlly, users can use the dataset up to ILVVRC2017 that is the last iamge recognition benchmark 
# competition [https://image-net.org/challenges/LSVRC/2017/] in the history. 
# 
# ```
# imagenet2012/
#     ├── ILSVRC2012_img_test.tar
#     ├── ILSVRC2012_img_train.tar
#     └── ILSVRC2012_img_val.tar
# ```
# 
# Create another folder and create the folders `data`, `download` & `extracted` like shown:
# ```
# datasets/
#     └──imagenet/
#            ├── data/
#            ├── downloaded/
#            └── extracted/  
# ```
#
# 
# Data Processing 
# 
# It only includes image preprocessing and augment batch. If users have large-capability computers and GPU, users 
# can add more data processing code to enjoy the essence of the  big data of ILSVRC2012 ~ ILSVRC2017.
# 
# ## Train
# 
# Conduct both the train and the callback
# 
# Users can run the train srcipt with the following command in the Linux Terminal. 
# 
# $ python train.py 
# 
# If the dev environments are the updates such as TensorFlow 2.11.0 and Keras 2.11.0, the script can run but have 
# the ValueError issue as follows after one epoch. 
# 
# ValueError: Expected scalar shape, saw shape: (1000,).
# 
# It incurs due to the TensorFlow updates. It is not user's problem but Google's probelm. Users need to correct the 
# error with commenting the following line of code in the script of summary_v2.py within the TensorBoard environment. 
# 
# 1.Find out the directory of summary_v2.py
#
# For example, please have a look at the following absolute path.
# 
# miniconda3/lib/Python3.9/site-packages/tensorboard/plugins/scalar/summary_v2.py
# 
# 2.Comment the line of code
# 
# change the original line of code
# 
# "tf.debugging.assert_scalar(data)"
# 
# to the following code:
#
# "# tf.debugging.assert_scalar(data)"
# 
# 3. Save the script of summary_v2.py
# 
# And then run the script again and it will be no problem.
# 
# Please remember save the original script in case any problems such as wrong-doing.
"""


import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds

from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint


plt.rcParams["figure.figsize"] = 30, 30


## ImageNet Data Loader

# Constants 

BATCH_SIZE = 512
# EPOCHS = 100 # for the formal training
EPOCHS = 5 # For running a review on the script. 
input_shape = (227,227,3)
image_dims = (227,227)
num_classes = 1000
NUM_CLASSES = num_classes
BATCH_SIZE = None
NUM_CHANNELS = 3
AUTOTUNE = tf.data.experimental.AUTOTUNE
HEIGHT, WIDTH = image_dims


## Data Processing 

def preprocess_image(image, label):
    """
    Process the image and label to perform the following operations:
    - Min-Max scale the image divided by 255
    - Convert the numerical values of the lables to one-hot encoded format
    - Resize the image to (227, 227)
    Args:
        image(image tensor): Raw image
        label(tensor): Numeric labels 1, 2, 3, ...
    Returns:
        tuple: Scaled image, one-hot encoded label
    """
    # Change the tf.unint8 into tf.float32 in the code of AlexNet model.  
    image = tf.cast(image, tf.uint8)
    image = tf.image.resize(image, [HEIGHT, WIDTH])
    image = image / tf.math.reduce_max(image)
    label = tf.one_hot(indices=label, depth=NUM_CLASSES)

    return image, label


@tf.function
def augment_batch(image, label):
    """
    Image augmentation for training:
    - Random Contrast
    - Random Brightness
    - Random Hue(Color)
    - Random Saturation
    - random Flip Left Right
    - Random Jpeg Quality
    Args:
        image(Tensor Image): Raw Image
        label(Tensor): Numeric Labels 1, 2, 3, ...
    Returns:
        tuple: Augmented image, numeric labels 1, 2, 3, ...
    """
    if tf.random.normal([1]) < 0:
        image = tf.image.random_contrast(image, 0.2, 0.9)
    if tf.random.normal([1]) < 0:
        image = tf.image.random_brightness(image, 0.2)
    if NUM_CHANNELS == 3 and tf.random.normal([1]) < 0:
        image = tf.image.random_hue(image, 0.3)
    if NUM_CHANNELS == 3 and tf.random.normal([1]) < 0:
        image = tf.image.random_saturation(image, 0, 15)
    
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_jpeg_quality(image, 10, 100)

    return image, label


# Load the ImageNet2012 dataset

class DataLoader:


    def __init__(self, source_dir, dest_dir, split="train"):
        """
        - Instance variable initialization
        - Download and set up the dataset with one-off operation
        - Use the TensorFlow tfds to Load and convert the ImageNet Dataset
        Args:
            source_dir(str): Path to downloaded tar files
            dest_dir(str): Path to the location where the dataset will be unpacked
            split(str): Split data for example with[80%:20%], defaulted as "train"
        """  
        # Download the Config(see tfds reference)
        download_config = tfds.download.DownloadConfig(
            extract_dir=os.path.join(dest_dir, 'extracted'), 
            manual_dir=source_dir)

        # download_and_prepare() is a method under tfds.core.DatasetBuilder
        download_and_prepare_kwargs = {
            'download_dir': os.path.join(dest_dir, 'downloaded'),
            'download_config': download_config}

        # TFDS Data Loader with performing dataset conversion to TFRecord
        self.ds, self.ds_info = tfds.load(
            'imagenet2012', 
            # for the following big TFRecord data
            data_dir=os.path.join(dest_dir, 'data'),  
            # Load the designated dataset such as "train", "test" of ["train", "test"]
            split=split, 
            shuffle_files=True,  
            # Convert the ImageNet data into TFRecord and save it into data directory. If set it 
            # to False, it does not execute the method: builder.download_and_prepare().
            download=True, # If False, data is expected to be in data_dir
            # Set it to True for saving each piece of 2-tuple data:(input,label); or set if to False 
            # for saving it as dict type such as {feature1:input, feature:label}. 
            as_supervised=True,
            # Set it to True for returning a tuple (tf.data.Dataset,tfds.core.DatasetInfo) or set it
            # to Flase for returning as a tf.data.Dataset object. 
            with_info=True,
            # Set it to True for passing kwargs to tfds.core.DatasetBuilder.download_and_prepare.
            download_and_prepare_kwargs=download_and_prepare_kwargs)


    def dataset_generator(self, batch_size=32, augment=False):
        """
        Create the data loader pipeline and return a generator to generate datsets
        Args:
            batch_size(int, optional): Batch size defaulted to 32.
            augment(bool, optional): Enable/Disable augmentation defaulted to False.
        Returns:
            Tf.Data Generator: Dataset generator
        """
        # self.BATCH_SIZE = batch_size
        BATCH_SIZE = batch_size

        dataset = self.ds.apply(tf.data.experimental.ignore_errors())
        dataset = dataset.shuffle(batch_size * 10)
        dataset = dataset.repeat()
          
        if augment:
            dataset = dataset.map(augment_batch, num_parallel_calls=AUTOTUNE)
        
        dataset = dataset.map(preprocess_image, num_parallel_calls=AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)

        return dataset


    def get_dataset_size(self):
        """
        get_dataset_size
        Get the dataset size(number of images)
        Returns:
            int: Total number of images in the dataset
        """

        # return len(self.dataset)
        return len(self.ds)
    

    def get_num_steps(self):
        """
        Get the number of steps required per batch for training
        Raises:
            AssertionError: Dataset generator needs to be initialized first
        Returns:
            int: Number of steps required for training per batch
        """
        if BATCH_SIZE is None:

            raise AssertionError(
                # f"Batch Size is not Initialized. Call this method only after calling: {dataset_generator}"
                f"Batch Size is not Initialized. Call this method only after calling: {self.dataset_generator}"
            )   
        num_steps = self.get_dataset_size() // BATCH_SIZE + 1

        return num_steps
    

    def visualize_batch(self, augment=True):
        """
        Dataset sample visualization
        - Support augmentation
        - Automatically adjust for grayscale images
        Args:
            augment(bool, optional): Enable/Disable augmentation defaulted to True.
        """
        if NUM_CHANNELS == 1:
            cmap = "gray"
        else:
            cmap = "viridis"

        dataset = self.dataset_generator(batch_size=36, augment=augment)
        image_batch, label_batch = next(iter(dataset))
        image_batch, label_batch = (image_batch.numpy(), label_batch.numpy(),)

        for n in range(len(image_batch)):
            ax = plt.subplot(6, 6, n + 1)
            plt.imshow(image_batch[n], cmap=cmap)
            plt.title(np.argmax(label_batch[n]))
            plt.axis("off")

        plt.show() 


# For the display of batch images in Jupyter Notebook 

data_loader = ImageNetDataLoader(
        source_dir = "/media/drive1/ImageNet2012",
        dest_dir = "/media/drive1/imdata/imagenet",
        split = "train",
)


data_loader.visualize_batch(augment=False)

data_loader.visualize_batch(augment=True)


# AlexNet Architecture - Single GPU 

inputs = Input(shape=(227,227,3), name="alexnet_input")

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
x = Conv2D(384, (3,3), strides=1, padding='same')(x)
x = ReLU()(x)

# Layer 4
x = Conv2D(384, (3,3), strides=1, padding='same')(x)
x = ReLU()(x)

# Layer 5
x = Conv2D(256, (3,3), strides=1, padding='same')(x)
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

alexnet.summary()

# tf.keras.utils.plot_model(alexnet, show_layer_names=False, show_shapes=True, show_dtype=True)
plot_model(alexnet, show_layer_names=False, show_shapes=True, show_dtype=True)


# Callbacks

# Need to create the folders of weights and tb_logs within the AlexNet main folder
my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.ModelCheckpoint(filepath='weights/model.{epoch:02d}-{val_categorical_accuracy:.2f}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='tb_logs./'),
]


# Metrics
metrics = [
    tf.keras.metrics.CategoricalAccuracy(),
    tf.keras.metrics.FalseNegatives(),
    tf.keras.metrics.FalsePositives(),
    tf.keras.metrics.Precision(),
    tf.keras.metrics.Recall(),
    tfa.metrics.F1Score(num_classes=1000)]


# Mixed Precision
tf.keras.mixed_precision.set_global_policy("mixed_float16")


# Initialize DataLoader for Training and Validation

# Init DataLoaders
train_data_loader = DataLoader(
        source_dir = "/media/drive1/ImageNet2012",
        dest_dir = "/media/drive1/datasets/imagenet",
        split = "train",
)

val_data_loader = DataLoader(
        source_dir = "/media/drive1/ImageNet2012",
        dest_dir = "/media/drive1/datasets/imagenet",
        split = "validation",
)

train_generator = train_data_loader.dataset_generator(batch_size=BATCH_SIZE, augment=False)
val_generator = val_data_loader.dataset_generator(batch_size=BATCH_SIZE, augment=False)

train_steps = train_data_loader.get_num_steps()
val_steps = val_data_loader.get_num_steps()


# Compile Model and Start Training

# Compile the model  

alexnet(input_shape).compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=False, name='SGD'),
    metrics=metrics,)


# Train the ImageNet with a quite longer time

history = alexnet(input_shape).fit(
    epochs=EPOCHS,
    x=train_generator,
    steps_per_epoch=train_steps,
    validation_data=val_generator,
    validation_steps=val_steps,
    callbacks=callbacks
)