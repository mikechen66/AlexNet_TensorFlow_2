
"""
# Conduct both the train and the callback
# 
# Users can run the train srcipt with the following command in the Linux Terminal. 
# 
# $ python train.py 
# 
# If the dev environments are the updates such as TensorFlow 2.11.0 and Keras 2.11.0, 
# the script can run but have the ValueError issue as follows after one epoch. 
# 
# ValueError: Expected scalar shape, saw shape: (1000,).
# 
# It incurs due to the TensorFlow updates. It is not user's problem but Google's 
# probelm. Users need to correct the error with commenting the following line of code 
# in the script of summary_v2.py within the TensorBoard environment. 
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


import tensorflow as tf
from model import alexnet
from data_loader import DataLoader
from data_process import get_num_steps


# Constants
BATCH_SIZE = 512
# EPOCHS = 100 # for the formal training
EPOCHS = 5 # For running a review on the script. 
input_shape = (227,227,3)


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