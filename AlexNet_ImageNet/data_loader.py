"""
#
# Data Loading Notes: 
#
# Please download the ImageNet dataset from the above link. There are two methods to get the dataset. 
# One is the official image-net website but users may need an college student indntity and complex 
# application procedure. And another (both simple and practical) method is to download from the public 
# resources. 
# 
# 1.ImageNet Official Website for Collage Users: 
#
# https://image-net.org/download-images
# 
# 2.Academictorrents Website for Other Users: 
# 
# https://academictorrents.com/
# 
# 
# The original name of ImageNet dataset is ILSVRC2012. Users can change the name to imagenet2012 or keep 
# it no changed. Principlly, users can use the dataset up to ILVVRC2017 that is the last iamge recognition
# benchmark competition [https://image-net.org/challenges/LSVRC/2017/] in the history. 
# 
# imagenet2012/
#     ├── ILSVRC2012_img_test.tar
#     ├── ILSVRC2012_img_train.tar
#     └── ILSVRC2012_img_val.tar
# 
# 
# Please note TFRecord and related files will be saved in the data directory for the usage of the train 
# script. Right now, the fileholders of downloaded and extracted are empty and only used as placeholders 
# for future updates. Please manually create the folders as follows or use self-defined names which users 
# prefer to. 
#
# datasets/
#     └──imagenet/
#            ├── data/
#            ├── downloaded/
#            └── extracted/      
# 
# Training and Validation Metrics
# 
# In the F1 statistics, we will use the four benchmarks as follows. 
# 
# TP - True Positive
# FP - False Positive
# TN - True Negative
# FN - False Negative
# 
# Metrices:
#
# Categorical Accuracy = $\frac{TP+FP}{TP+FP+TN+FN}$
# Precision = $\frac{TP}{TP+FP}$
# Recall = $\frac{TP}{TP+FN}$
# F1 Score = $2*\frac{Precision*Recall}{Precision+Recall}$
"""


import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
from data_process import preprocess_image, augment_batch


plt.rcParams["figure.figsize"] = 30, 30


# Constants 

BATCH_SIZE = None
NUM_CHANNELS = 3
AUTOTUNE = tf.data.experimental.AUTOTUNE


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
          
        # if augment:
        if augment:
            # Call the augment_batch() method in data_process.py
            dataset = dataset.map(augment_batch, num_parallel_calls=AUTOTUNE)
        # # Call the preprocess_image() method in data_process.py
        dataset = dataset.map(preprocess_image, num_parallel_calls=AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)

        return dataset


    def get_dataset_size(self):
        """
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
                f"Batch Size is not Initialized, invoke this method only after calling: {self.dataset_generator}"
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

        # Only for the usage of Jupyter Notebook 
        plt.show() 