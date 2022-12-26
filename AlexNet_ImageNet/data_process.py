"""
# Data Processing 

# It only includes image preprocessing and augment batch. If users have large-capability 
# computers and GPU, users can add more data processing code to enjoy the essence of the 
# big data of ILSVRC2012 ~ ILSVRC2017.
"""


import tensorflow as tf


# Constants 

image_dims = (227, 227)
num_classes = 1000

NUM_CHANNELS = 3
NUM_CLASSES = num_classes
HEIGHT, WIDTH  = image_dims


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