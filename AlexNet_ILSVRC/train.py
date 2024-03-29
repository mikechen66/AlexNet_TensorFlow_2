#!/usr/bin/env python
# coding: utf-8

# It is originally written with ipython by datahacker and re-written with python and adding some 
# lines of code in order to adapt the Python practice by Mike Chen. 
# http://media5.datahacker.rs/2018/06/logo-crno.png)

"""

The User Guide: 

To use the script in Python, users need to create the folder such as AlexNet_ILSVRC. Users need to 
do the following actions to run the script. 

1.Download ILSVRC2012 Dataset

# Download the dataset 

Please donwloand the ILSVRC dataset in the public dataset. Either get the dataset by registering in 
the the official website or get the dataset via the dataset search. Since it has 150GB, users need 
to download it in a very long time. It is very big but an exciting momment for users to access to 
the big data. 

Official Website 
https://www.image-net.org/challenges/LSVRC/2012/

Academictorrents (dowload by uTorrent): 

Download and install uTorrent in Linux 
https://www.utorrent.com/

Download ILSVRC2012 Dataset
https://academictorrents.com/collection/imagenet-lsvrc-2015

# Unzip

After unzipping the ILSVRC datasets, please do the following actions to adapt the scripts. 

Change the original filenames

ILSVRC2012_bbox_train_v2, ILSVRC2012_img_test, ILSVRC2012_img_train, ILSVRC2012_img_val

to the new filenames as follows 

bbox_train_v2, img_test, img_train, img_val

2.Enter into current directory

$ cd /home/user/Documents/Alexnet_ILSVRC

Anaconda defaults the pre-installed Python3 and the Ubuntu 18.04 has both Python2 and Python3. Therefore, 
users need to follow the procedures. 

3.Script running command

In the Conda Environment, please execute the following command in the Ubuntu terminal at the current 
directory.  

# Train the model from scrach

  $ python train.py /home/user/datasets/ILSVRC2012 --resume False --train true

#  Resume the training: 

   $ python model.py /home/user/datasets/ILSVRC2012 --resume True --train true 

# Test the model 

   $ python train.py /home/user/datasets/ILSVRC2012 --test true

4.Release GPU Memory

The training time for the 100GB data is quite  long, users need to wait the automatic release of GPU memory 
after completing the train period or manully kill the GPU process. Otherwise, the Linux Termianl would show
"CUDA runtime implicit initialization on GPU:0 failed. Status: out of memory". 

# Automatic Release 

I have already put the following code in the script of train.py. 

from numba import cuda
cuda.select_device(0)
cuda.close() 

# Manual Release with System Monitor 

Users may install System Monitor, select the most heavy-load process of GPU and click "kill" menu. 

Install System Monitor
$ sudo apt install system-monitor

Open System Monitor in Show Application, Ubuntu Linux 

Kill the selective Python Memory - GPU process


# Release with the command in Linux Terminal 

It is also an esay way to kill the GPU process with the command as follows. 

$ sudo kill -9 PID


# AlexNet Paper

https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
"""


from math import ceil
import json
import time
import os
import threading

import tensorflow as tf
import numpy as np

from data import LSVRC2012
import logs
from numba import cuda


# Set up the GPU in the condition of allocation exceeds system memory with the reminding message: Could not 
# create cuDNN handle... The following lines of code can avoids the sudden stop of the runtime. 
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class AlexNet:

    def __init__(self, path, batch_size, resume):
        """
        Build the AlexNet model
        """
        self.logger = logs.get_logger()

        self.resume = resume
        self.path = path
        self.batch_size = batch_size
        self.lsvrc2012 = LSVRC2012(self.path, batch_size)
        self.num_classes = len(self.lsvrc2012.wnid2label)

        self.lr = 0.001
        self.momentum = 0.9
        self.lambd = tf.constant(0.0005, name='lambda')
        self.input_shape = (None, 227, 227, 3)
        self.output_shape = (None, self.num_classes)

        self.logger.info("Creating placeholders for graph...")
        self.create_tf_placeholders()

        self.logger.info("Creating variables for graph...")
        self.create_tf_variables()

        self.logger.info("Initialize hyper parameters...")
        self.hyper_param = {}
        self.init_hyper_param()

    def create_tf_placeholders(self):
        """
        Create placeholders for the graph. The input for these will be given 
        # while training or testing.
        """
        self.input_image = tf.compat.v1.placeholder(tf.float32, shape=self.input_shape,
                                          name='input_image')
        self.labels = tf.compat.v1.placeholder(tf.float32, shape=self.output_shape,
                                     name='output')
        self.learning_rate = tf.compat.v1.placeholder(tf.float32, shape=(),
                                            name='learning_rate')
        self.dropout = tf.compat.v1.placeholder(tf.float32, shape=(),
                                      name='dropout')

    def create_tf_variables(self):
        """
        Create variables for epoch, batch and global step
        """
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.cur_epoch = tf.Variable(0, name='epoch', trainable=False)
        self.cur_batch = tf.Variable(0, name='batch', trainable=False)

        self.increment_epoch_op = tf.compat.v1.assign(self.cur_epoch, self.cur_epoch+1)
        self.increment_batch_op = tf.compat.v1.assign(self.cur_batch, self.cur_batch+1)
        self.init_batch_op = tf.compat.v1.assign(self.cur_batch, 0)

    def init_hyper_param(self):
        """
        Store the hyper parameters. For each layer store number of filters
        (kernels) and filter size. If it's a fully connected layer then store 
        the number of neurons.
        """
        with open('hparam.json') as f:
            self.hyper_param = json.load(f)

    def get_filter(self, layer_num, layer_name):
        """
        :param layer_num: Indicate the layer number in the graph
        :type layer_num: int
        :param layer_name: Name of the filter
        """
        layer = 'L' + str(layer_num)

        filter_height, filter_width, in_channels = self.hyper_param[layer]['filter_size']
        out_channels = self.hyper_param[layer]['filters']

        return tf.Variable(tf.random.truncated_normal(
            [filter_height, filter_width, in_channels, out_channels],
            dtype = tf.float32, stddev = 1e-2), name = layer_name)

    def get_strides(self, layer_num):
        """
        :param layer_num: Indicate the layer number in the graph
        :type layer_num: int
        """
        layer = 'L' + str(layer_num)
        stride = self.hyper_param[layer]['stride']
        strides = [1, stride, stride, 1]

        return strides

    def get_bias(self, layer_num, value=0.0):
        """
        Get the bias variable for current layer
        :param layer_num: Indicate the layer number in the graph
        :type layer_num: int
        """
        layer = 'L' + str(layer_num)
        initial = tf.constant(value,
                              shape=[self.hyper_param[layer]['filters']],
                              name='C' + str(layer_num))
        return tf.Variable(initial, name='B' + str(layer_num))

    @property
    def l2_loss(self):
        """
        Compute the l2 loss for all the weights
        """
        conv_bias_names = ['B' + str(i) for i in range(1, 6)]
        weights = []
        for v in tf.compat.v1.trainable_variables():
            if 'biases' in v.name: continue
            if v.name.split(':')[0] in conv_bias_names: continue
            weights.append(v)

        return self.lambd * tf.add_n([tf.nn.l2_loss(weight) for weight in weights])

    def build_graph(self):
        """
        Build the tensorflow graph for AlexNet.

        First 5 layers are Convolutional layers. Out of which
        first 2 and last layer will be followed by max pooling
        layers. Next 2 layers are fully connected layers.

        L1_conv -> L1_MP -> L2_conv -> L2_MP -> L3_conv
        -> L4_conv -> L5_conv -> L5_MP -> L6_FC -> L7_FC

        Where L1_conv -> Convolutional layer 1
              L5_MP -> Max pooling layer 5
              L7_FC -> Fully Connected layer 7

        Use tf.nn.conv2d to initialize the filters so as to reduce 
        training time and tf.layers.max_pooling2d as we don't need 
        to initialize in the pooling layer.
        """
        # Layer 1 Convolutional layer
        filter1 = self.get_filter(1, 'L1_filter')
        l1_conv = tf.nn.conv2d(input=self.input_image, filters=filter1,
                               strides=self.get_strides(1),
                               padding = self.hyper_param['L1']['padding'],
                               name='L1_conv')
        l1_conv = tf.add(l1_conv, self.get_bias(1))
        l1_conv = tf.nn.local_response_normalization(l1_conv,
                                                     depth_radius=5,
                                                     bias=2,
                                                     alpha=1e-4,
                                                     beta=.75)
        l1_conv = tf.nn.relu(l1_conv)

        # Layer 1 Max Pooling layer
        l1_MP = tf.compat.v1.layers.max_pooling2d(l1_conv,
                                        self.hyper_param['L1_MP']['filter_size'],
                                        self.hyper_param['L1_MP']['stride'],
                                        name='L1_MP')

        # Layer 2 Convolutional layer
        filter2 = self.get_filter(2, 'L2_filter')
        l2_conv = tf.nn.conv2d(input=l1_MP, filters=filter2,
                               strides=self.get_strides(2),
                               padding = self.hyper_param['L2']['padding'],
                               name='L2_conv')
        l2_conv = tf.add(l2_conv, self.get_bias(2, 1.0))
        l2_conv = tf.nn.local_response_normalization(l2_conv,
                                                     depth_radius=5,
                                                     bias=2,
                                                     alpha=1e-4,
                                                     beta=.75)
        l2_conv = tf.nn.relu(l2_conv)

        # Layer 2 Max Pooling layer
        l2_MP = tf.compat.v1.layers.max_pooling2d(l2_conv,
                                        self.hyper_param['L2_MP']['filter_size'],
                                        self.hyper_param['L2_MP']['stride'],
                                        name='L2_MP')

        # Layer 3 Convolutional layer
        filter3 = self.get_filter(3, 'L3_filter')
        l3_conv = tf.nn.conv2d(input=l2_MP, filters=filter3,
                               strides=self.get_strides(3),
                               padding = self.hyper_param['L3']['padding'],
                               name='L3_conv')
        l3_conv = tf.add(l3_conv, self.get_bias(3))
        l3_conv = tf.nn.relu(l3_conv)

        # Layer 4 Convolutional layer
        filter4 = self.get_filter(4, 'L4_filter')
        l4_conv = tf.nn.conv2d(input=l3_conv, filters=filter4,
                               strides=self.get_strides(4),
                               padding = self.hyper_param['L4']['padding'],
                               name='L4_conv')
        l4_conv = tf.add(l4_conv, self.get_bias(4, 1.0))
        l4_conv = tf.nn.relu(l4_conv)

        # Layer 5 Convolutional layer
        filter5 = self.get_filter(5, 'L5_filter')
        l5_conv = tf.nn.conv2d(input=l4_conv, filters=filter5,
                               strides=self.get_strides(5),
                               padding = self.hyper_param['L5']['padding'],
                               name='L5_conv')
        l5_conv = tf.add(l5_conv, self.get_bias(5, 1.0))
        l5_conv = tf.nn.relu(l5_conv)

        # Layer 5 Max Pooling layer
        l5_MP = tf.compat.v1.layers.max_pooling2d(l5_conv,
                                        self.hyper_param['L5_MP']['filter_size'],
                                        self.hyper_param['L5_MP']['stride'],
                                        name='L5_MP')

        flatten = tf.compat.v1.layers.flatten(l5_MP)

        # Layer 6 Fully connected layer
        # Adopt tf.compat.v1.layers.dense to replace the commented code 
        # l6_FC = tf.layers.fully_connected(flatten, self.hyper_param['FC6'])
        l6_FC = tf.compat.v1.layers.dense(flatten, self.hyper_param['FC6'])

        # Dropout layer
        l6_dropout = tf.nn.dropout(l6_FC, 1 - (self.dropout),
                                   name='l6_dropout')

        # Layer 7 Fully connected layer
        # self.l7_FC = tf.contrib.layers.fully_connected(l6_dropout, self.hyper_param['FC7'])
        self.l7_FC = tf.compat.v1.layers.dense(l6_dropout, self.hyper_param['FC7'])

        # Dropout layer
        l7_dropout = tf.nn.dropout(self.l7_FC, 1 - (self.dropout),
                                   name='l7_dropout')

        # final layer before softmax
        # self.logits = tf.contrib.layers.fully_connected(l7_dropout, self.num_classes, None)
        self.logits = tf.compat.v1.layers.dense(l7_dropout, self.num_classes, None)

        # loss function
        loss_function = tf.nn.softmax_cross_entropy_with_logits(
            logits = self.logits,
            labels = tf.stop_gradient( self.labels)
        )

        # total loss
        self.loss = tf.reduce_mean(input_tensor=loss_function) + self.l2_loss

        self.optimizer = tf.compat.v1.train.MomentumOptimizer(self.learning_rate, momentum=self.momentum)\
                                 .minimize(self.loss, global_step=self.global_step)

        correct = tf.equal(tf.argmax(input=self.logits, axis=1), tf.argmax(input=self.labels, axis=1))
        self.accuracy = tf.reduce_mean(input_tensor=tf.cast(correct, tf.float32))

        self.top5_correct = tf.nn.in_top_k(predictions=self.logits, targets=tf.argmax(input=self.labels, axis=1), k=5)
        self.top5_accuracy = tf.reduce_mean(input_tensor=tf.cast(self.top5_correct, tf.float32))

        self.add_summaries()

    def add_summaries(self):
        """
        Add summaries for loss, top1 and top5 accuracies
        Add loss, top1 and top5 accuracies to summary files
        in order to visualize in tensorboard
        """
        tf.compat.v1.summary.scalar('loss', self.loss)
        tf.compat.v1.summary.scalar('Top-1-Acc', self.accuracy)
        tf.compat.v1.summary.scalar('Top-5-Acc', self.top5_accuracy)

        self.merged = tf.compat.v1.summary.merge_all()

    def save_model(self, sess, saver):
        """
        Save the current model
        :param sess: Session object
        :param saver: Saver object responsible to store
        """
        model_base_path = os.path.join(os.getcwd(), 'model')
        if not os.path.exists(model_base_path):
            os.mkdir(model_base_path)
        model_save_path = os.path.join(os.getcwd(), 'model', 'model.ckpt')
        save_path = saver.save(sess, model_save_path)
        self.logger.info("Model saved in path: %s", save_path)

    def restore_model(self, sess, saver):
        """
        Restore previously saved model

        :param sess: Session object
        :param saver: Saver object responsible to store
        """
        model_base_path = os.path.join(os.getcwd(), 'model')
        model_restore_path = os.path.join(os.getcwd(), 'model', 'model.ckpt')
        saver.restore(sess, model_restore_path)
        self.logger.info("Model Restored from path: %s",
                         model_restore_path)

    def get_summary_writer(self, sess):
        """
        Get summary writer for training and validation
        Responsible for creating summary writer so it can
        write summaries to a file so it can be read by
        tensorboard later.
        """
        if not os.path.exists(os.path.join('summary', 'train')):
            os.makedirs(os.path.join('summary', 'train'))
        if not os.path.exists(os.path.join('summary', 'val')):
            os.makedirs(os.path.join('summary', 'val'))
        return (tf.compat.v1.summary.FileWriter(os.path.join(os.getcwd(),
                                                  'summary', 'train'),
                                      sess.graph),
                tf.compat.v1.summary.FileWriter(os.path.join(os.getcwd(),
                                                   'summary', 'val'),
                                      sess.graph))

    def train(self, epochs, thread='false'):
        """
        Train the AlexNet.
        """
        batch_step, val_step = 10, 500

        self.logger.info("Building the graph...")
        self.build_graph()

        init = tf.compat.v1.global_variables_initializer()

        saver = tf.compat.v1.train.Saver()
        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)) as sess:
            (summary_writer_train,
             summary_writer_val) = self.get_summary_writer(sess)
            if self.resume and os.path.exists(os.path.join(os.getcwd(),
                                                           'model')):
                self.restore_model(sess, saver)
            else:
                sess.run(init)

            resume_batch = True
            best_loss = float('inf')
            while sess.run(self.cur_epoch) < epochs:
                losses = []
                accuracies = []

                epoch = sess.run(self.cur_epoch)
                if not self.resume or (
                        self.resume and not resume_batch):
                    sess.run(self.init_batch_op)
                resume_batch = False
                start = time.time()
                # AttributeError: 'AlexNet' object has no attribute 'lsvrc2012'
                gen_batch = self.lsvrc2012.gen_batch
                for images, labels in gen_batch:
                    batch_i = sess.run(self.cur_batch)
                    # If it's resumed from stored model,
                    # this will save from messing up the batch number
                    # in subsequent epoch
                    if batch_i >= ceil(len(self.lsvrc2012.image_names) / self.batch_size):
                        break
                    (_, global_step,
                     _) = sess.run([self.optimizer,
                                    self.global_step, self.increment_batch_op],
                                   feed_dict = {
                                       self.input_image: images,
                                       self.labels: labels,
                                       self.learning_rate: self.lr,
                                       self.dropout: 0.5
                                   })

                    if global_step == 150000:
                        self.lr = 0.0001 # Halve the learning rate

                    if batch_i % batch_step == 0:
                        (summary, loss, acc, top5_acc, _top5,
                         logits, l7_FC) = sess.run([self.merged, self.loss,
                                                    self.accuracy, self.top5_accuracy,
                                                    self.top5_correct,
                                                    self.logits, self.l7_FC],
                                                   feed_dict = {
                                                       self.input_image: images,
                                                       self.labels: labels,
                                                       self.learning_rate: self.lr,
                                                       self.dropout: 1.0
                                                   })
                        losses.append(loss)
                        accuracies.append(acc)
                        summary_writer_train.add_summary(summary, global_step)
                        summary_writer_train.flush()
                        end = time.time()
                        try:
                            self.logger.debug("l7 no of non zeros: %d", np.count_nonzero(l7_FC))
                            true_idx = np.where(_top5[0]==True)[0][0]
                            self.logger.debug("logit at %d: %s", true_idx,
                                              str(logits[true_idx]))
                        except IndexError as ie:
                            self.logger.debug(ie)
                        self.logger.info("Time: %f Epoch: %d Batch: %d Loss: %f "
                                         "Avg loss: %f Accuracy: %f Avg Accuracy: %f "
                                         "Top 5 Accuracy: %f",
                                         end - start, epoch, batch_i,
                                         loss, sum(losses) / len(losses),
                                         acc, sum(accuracies) / len(accuracies),
                                         top5_acc)
                        start = time.time()

                    if batch_i % val_step == 0:
                        images_val, labels_val = self.lsvrc2012.get_batch_val
                        (summary, acc, top5_acc,
                         loss) = sess.run([self.merged,
                                           self.accuracy,
                                           self.top5_accuracy, self.loss],
                                          feed_dict = {
                                              self.input_image: images_val,
                                              self.labels: labels_val,
                                              self.learning_rate: self.lr,
                                              self.dropout: 1.0
                                          })
                        summary_writer_val.add_summary(summary, global_step)
                        summary_writer_val.flush()
                        self.logger.info("Validation - Accuracy: %f Top 5 Accuracy: %f Loss: %f",
                                         acc, top5_acc, loss)

                        cur_loss = sum(losses) / len(losses)
                        if cur_loss < best_loss:
                            best_loss = cur_loss
                            self.save_model(sess, saver)

                # Increase epoch number
                sess.run(self.increment_epoch_op)

    def test(self):
        step = 10

        self.logger_test = logs.get_logger('AlexNetTest', file_name='logs_test.log')
        self.logger_test.info("In Test: Building the graph...")
        self.build_graph()

        init = tf.compat.v1.global_variables_initializer()

        saver = tf.compat.v1.train.Saver()
        top1_count, top5_count, count = 0, 0, 0
        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)) as sess:
            self.restore_model(sess, saver)

            start = time.time()
            batch = self.lsvrc2012.gen_batch_test
            for i, (patches, labels) in enumerate(batch):
                count += patches[0].shape[0]
                avg_logits = np.zeros((patches[0].shape[0], self.num_classes))
                for patch in patches:
                    logits = sess.run(self.logits,
                                      feed_dict = {
                                          self.input_image: patch,
                                          self.dropout: 1.0
                                      })
                    avg_logits += logits
                avg_logits /= len(patches)
                top1_count += np.sum(np.argmax(avg_logits, 1) == labels)
                top5_count += np.sum(avg_logits.argsort()[:, -5:] == \
                                     np.repeat(labels, 5).reshape(patches[0].shape[0], 5))

                if i % step == 0:
                    end = time.time()
                    self.logger_test.info("Time: %f Step: %d "
                                          "Avg Accuracy: %f "
                                          "Avg Top 5 Accuracy: %f",
                                          end - start, i,
                                          top1_count / count,
                                          top5_count / count)
                    start = time.time()

            self.logger_test.info("Final - Avg Accuracy: %f "
                                  "Avg Top 5 Accuracy: %f",
                                  top1_count / count,
                                  top5_count / count)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', metavar = 'image-path',
                        help = 'ImageNet dataset path')
    parser.add_argument('--resume', metavar='resume',
                        type=lambda x: x != 'False', default=True,
                        required=False,
                        help='Resume training (True or False)')
    parser.add_argument('--train', help='Train AlexNet')
    parser.add_argument('--test', help='Test AlexNet')
    args = parser.parse_args()

    alexnet = AlexNet(args.image_path, batch_size=128, resume=args.resume)

    if args.train == 'true':
        alexnet.train(50)
    elif args.test == 'true':
        alexnet.test()


# Release the GPU memory elegantly. 
cuda.select_device(0)
cuda.close()