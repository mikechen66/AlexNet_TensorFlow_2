#/usr/bin/python3
# -*- coding: utf-8 -*-

# library modules
import os
import logging

from collections import namedtuple
from random import shuffle
from random import sample
from math import ceil

# External library modules
import numpy as np
from scipy.io import loadmat
from PIL import Image

# local modules
from utils import preprocess
# from tensorflow.keras.utils import preprocess
from utils import gen_mean_activity
# from tensorflow.keras.utils import gen_mean_activity
from utils import Store
# from tensorflow.keras.utils import Store 
from data_augment import augment


# Define the class of LSVRC2012 but has no parameter. 
class LSVRC2012:
    """
    Read the train data of ILSVRC2012.
    """

    def __init__(self, path, batch_size, augment=False):
        """
        Find which folder has the kind of images and which image belongs to a folder and a category.
        :param path: The directory path for the ILSVRC2012 training data
        """
        # self.logger = logging.getLogger('AlexNet2012.ILSVRC2012')
        self.logger = logging.getLogger('AlexNet.ILSVRC2012')
        self.batch_size = batch_size
        self.augment = augment
        self.image_size = (227, 227, 3)

        # Directory paths
        # -path = '/media/drive1/datasets/ILSVRC2012' 
        self.base_dir = path
        # Delete the prefix "ILSVRC2012_" of the original filenames 
        """
        self.train_dir = os.path.join(path, 'ILSVRC2012_img_train')
        self.val_dir = os.path.join(path, 'ILSVRC2012_img_val')
        self.test_dir = os.path.join(path, 'ILSVRC2012_img_test')
        """
        self.train_dir = os.path.join(path, 'img_train')
        self.val_dir = os.path.join(path, 'img_val')
        self.test_dir = os.path.join(path, 'img_test')

        # Store the folder name to label info
        self.wnid2label = {}
        self.gen_labels()

        self.lsvrcid2wnid = {}
        self.store_lsvrcid2wnid()

        self.image_names = {}
        self.find_image_names()

        self.image_names_val = {}
        self.find_image_names_val()

        self.image_names_test = {}
        self.find_image_names_test()

        if not os.path.exists('mean.pkl'):
            gen_mean_activity(self.train_dir)

    def gen_labels(self):
        """
        Store the folder to label map in a dict. This will be helpful while creating one-hot encodings.
        Example:
        >>> self.folders = ['hi', 'Alex', 'deep']
        >>> self.get_folder_indices()
        {'deep': 1, 'hi': 2, 'Alex': 0}
        """
        self.wnid2label = dict((folder,i) for i, folder in enumerate(sorted(os.listdir(self.train_dir))))
        self.logger.info("There are %d categories in total", len(self.wnid2label))

    def store_lsvrcid2wnid(self):
        """
        Store the mapping of ILSVRC2012_ID to WNID. For more information about both ILSVRC2012_ID and WNID, 
        please read the devkit_12 readme(or information). In short, WNID are the foldernames in the training 
        folder and ILSVRC2012_ID is an id that is assigned to each folder category to uniquely identify the 
        category for the folder

        After running this you should have
        >>> self.lsvrcid2wnid[330] == 'n01910747'
        """
        mat = loadmat(os.path.join(self.base_dir, 'devkit_t12',
                                   'data', 'meta.mat'))
        synsets = mat['synsets']

        for i in range(len(synsets)):
            # matlab datas are not coming nicely for python objects ;)
            self.lsvrcid2wnid[synsets[i][0][0][0][0]] = str(synsets[i][0][1][0])

    def find_image_names(self):
        """
        Find category information for all training images. With regard to  all images in the train directory
        please find which WNID(folder) and label that image belongs to and store it in self.image_names:

        If there are 1000 images in folder `f`, all images inside `f` are `f_0.JPEG`, `f_1.JPEG`, ..., `f_999.JPEG`.
        But not necessarily as `0, 1, 2, ...`(increasing order from 0). So it is better to read what files are 
        present in `f` rather than assuming that all files are present in the increasing order.
        """
        # Each folder belongs to a folder and corresponding label. This label represents the number in output softmax
        # in the AlexNet graph(model)
        category = namedtuple('Category', ['folder','label'])
        for folder in os.listdir(self.train_dir):
            for image in os.listdir(os.path.join(self.train_dir, folder)):
                self.image_names[image] = category(folder, self.wnid2label[folder])

        self.logger.info("There are %d total training images in the dataset",
                         len(self.image_names))

    def find_image_names_val(self):
        """
        Find the label of each validation image
        """
        with open(os.path.join(self.base_dir, 'devkit_t12', 'data',
                               'ILSVRC2012_validation_ground_truth.txt')) as f:
            for image, lsvrcid in zip(sorted(os.listdir(self.val_dir)), f):
                self.image_names_val[image] = \
                    self.wnid2label[self.lsvrcid2wnid[int(lsvrcid.strip())]]

    def find_image_names_test(self):
        """
        Find the label of each test image
        """
        with open(os.path.join(self.base_dir, 'devkit_t12', 'data',
                               'ILSVRC2012_test_ground_truth.txt')) as f:
            for image, lsvrcid in zip(sorted(os.listdir(self.test_dir)), f):
                self.image_names_test[image] = \
                    self.wnid2label[self.lsvrcid2wnid[int(lsvrcid.strip())]]

    def image_path(self, image_name, val=False, test=False):
        """
        Return full image path
        e.g. ~/datasets/ILSVRC2012/ILSVRC2012_img_train/n03854065/n03854065_297.JPEG
        or
        e.g. ~/datasets/ILSVRC2012/ILSVRC2012_img_val/ILSVRC2012_val_00000303.JPEG

        :param image_name: The name of the image. e.g. n03854065_297.JPEG
        """
        if val:
            return os.path.join(self.val_dir, image_name)
        if test:
            return os.path.join(self.test_dir, image_name)
        return os.path.join(self.train_dir,
                            self.image_names[image_name].folder,
                            image_name)

    def one_hot(self, labels):
        """
        Get the one hot encoding of `labels:` The size of the output encoding matrix has 
        to be (batch size x no of categories).

        :param labels: list of labels for current batch, the type of labels is `list`
        """
        batch_size = len(labels)
        y_hat = np.zeros((batch_size, len(self.wnid2label)))
        y_hat[np.arange(batch_size), labels] = 1

        return y_hat

    @preprocess
    def get_image(self, image_path):
        """
        Get the image in the path `image_path`
        """
        return Image.open(image_path)

    def cur_batch_images(self, images, val=False):
        """
        Convert all images in `images` to numpy array
        Return numpy size (`:py:self.batch_size:`, 227, 227, 3)
        """
        npimages = []
        for image in images:
            npimages.append(self.get_image(self.image_path(image, val)))

        return np.array(npimages)

    def cur_batch_labels(self, images, val=False):
        """
        Get the one hot encoding for all `images` in one array
        """
        labels = []
        for image in images:
            if val:
                labels.append(self.image_names_val[image])
            else:
                labels.append(self.image_names[image].label)
        return self.one_hot(labels)

    @property
    def gen_batch(self):
        """
        A generator which returns `:py:self.batch_size:` of images(in a numpy array) and corresponding 
        labels
        """
        images = list(self.image_names.keys())
        shuffle(images)
        def get_batch(idx):
            """
            Get current batch of data give batch index.
            :param idx: The batch index in the dataset
            """
            self.logger.debug("Reading batch for index: %d", idx)
            _images = images[idx * self.batch_size: (idx + 1) * self.batch_size]
            X = self.cur_batch_images(_images)
            Y = self.cur_batch_labels(_images)
            return X, Y

        source = (get_batch, len(self.image_names.keys()), self.batch_size)
        store = Store(source, 10)

        batch = store.read()
        for i in range(ceil(len(self.image_names.keys()) / self.batch_size)):
            yield next(batch)

        raise StopIteration

    @property
    def gen_batch_non_threaded(self):
        """
        A generator returns `:py:self.batch_size:` of images(in a numpy array) and corresponding labels
        """
        images = list(self.image_names.keys())
        shuffle(images)

        for idx in range(ceil(len(images) / self.batch_size)):
            _images = images[idx * self.batch_size: (idx + 1) * self.batch_size]
            X = self.cur_batch_images(_images)
            Y = self.cur_batch_labels(_images)
            yield X, Y

        raise StopIteration

    @property
    def get_batch_val(self):
        """
        A generator returns `:py:self.batch_size:` of images(in a numpy array) and corresponding labels
        for validation dataset
        """
        images = list(self.image_names_val.keys())
        shuffle(images)

        _images = sample(images, self.batch_size)
        X = self.cur_batch_images(_images, True)
        Y = self.cur_batch_labels(_images, True)

        return X, Y

    def get_5_patches(self, image_path):
        """
        Get 5 patches for an image. It returns a list of 5 patches(top left, top right, bottom left, bottom 
        right and center) of an image.

        :param image_path: the path of an image
        """
        img = Image.open(image_path)
        # Resize the shorter size to 256
        if img.width < 256:
            img = img.resize((256, img.height))
        if img.height < 256:
            img = img.resize((img.width, 256))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Take 5 patches(top left, top right, bottom left, bottom right, center)
        img_crop = [None] * 5
        img_crop[0] = img.crop((0, 0, self.image_size[0],
                                self.image_size[1]))
        img_crop[1] = img.crop((img.width - self.image_size[0], 0,
                                img.width - self.image_size[0] + self.image_size[1],
                                self.image_size[1]))
        img_crop[2] = img.crop((0, img.height - self.image_size[1],
                                self.image_size[0], img.height))
        img_crop[3] = img.crop((img.width - self.image_size[0],
                                img.height - self.image_size[1],
                                img.width, img.height))
        img_crop[4] = img.crop((img.width // 2 - self.image_size[0] // 2,
                                img.height // 2 - self.image_size[1] // 2,
                                img.width // 2 - self.image_size[0] // 2 + self.image_size[0],
                                img.height // 2 - self.image_size[1] // 2 + self.image_size[1]))

        patches = [None] * 5
        for i, img in enumerate(img_crop):
            patches[i] = preprocess(lambda self, img: img, False)(self, img_crop[i])

        return patches

    @property
    def gen_batch_test(self):
        """
        A generator gives test images one by one after doing preproessing. For each batch return X, Y
        Where X is a list of 5 patches: each patch will have batch no of images. Y is the labels which 
        size is batch size.
        """
        logger_test = logging.getLogger('LSVRC2012')
        batch_size = 128
        images = list(self.image_names_test.keys())
        def get_batch(idx):
            """
            Get current batch of data give batch index.
            :param idx: The batch index in the dataset
            """
            logger_test.debug("Reading batch for index: %d", idx)
            _images = images[idx * batch_size: (idx + 1) * batch_size]

            X = [[] for _ in range(5)]
            Y = []
            for image in _images:
                patches = self.get_5_patches(self.image_path(image, test=True))
                for i, patch in enumerate(patches):
                    X[i].append(patch)
                Y.append(self.image_names_test[image])

            for i in range(len(X)):
                X[i] = np.array(X[i])

            return X, np.array(Y)

        source = (get_batch, len(self.image_names_test), batch_size)
        store = Store(source, 10)

        batch = store.read()
        for i in range(ceil(len(self.image_names_test) / batch_size)):
            yield next(batch)

        raise StopIteration


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', metavar = 'image-path',
                        help = 'ImageNet dataset path')
    args = parser.parse_args()

    lsvrc2012 = LSVRC2012(args.image_path, 128)

    image_cur_batch = lsvrc2012.gen_batch
    first_batch = next(image_cur_batch)
    print("The first batch shape:", first_batch[0].shape)
    print("The first one hot vector shape:", first_batch[1].shape)

    first_batch = lsvrc2012.get_batch_val
    print("The first batch shape:", first_batch[0].shape)
    print("The first one hot vector shape:", first_batch[1].shape)