
# client.py

"""
The application of the client is used to call the alexnet model. 
"""

import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential, metrics
import os, glob
import random, csv
import matplotlib.pyplot as plt
from alexnet import AlexNet

# Set the GPU growth to avoid the cuDNN runtime error. 
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# Create image path and lables and write them into the .csv file; root: dataset root directory, 
# filename:csv name, name2label:class coding table. 
def load_csv(root, filename, name2label):
    # If there is no csv, create a csv file and save it into the directory of home/mike/datasets/pokemon
    if not os.path.exists(os.path.join(root, filename)): 
        # Initialize the array in which image paths are saved.  
        images = []
        # Iterate all sub-directories and obtain all the paths of images. 
        for name in name2label.keys():  
            # Adopt the glob filename to match and obtain all files with the formats of png, jpg and jpeg
            images += glob.glob(os.path.join(root, name, '*.png'))
            images += glob.glob(os.path.join(root, name, '*.jpg'))
            images += glob.glob(os.path.join(root, name, '*.jpeg'))
        # Print it if necessary     
        # -print(len(images), images) 
        random.shuffle(images)  
        # Create the csv file and write both the paths of images and the info of labels
        with open(os.path.join(root,filename), mode='w', newline='') as f:
            writer = csv.writer(f)
            for img in images:  
            	# Adopt the symbol of \\ to divide items and take all 2nd items as class names
                name = img.split(os.sep)[-2]  
                # Find the values(as labels) related to the keys of classes
                label = name2label[name]  
                # Write them into the csv and divide them with comma, i.e, pokemon\\mewtwo\\00001.png, 2
                writer.writerow([img, label])  #
            print('written into csv file:', filename)

    # Read the csv file and create two empty arrays of which y is used to save both paths and labels.
    images, labels = [], []  

    with open(os.path.join(root,filename)) as f:
        reader = csv.reader(f)
        for row in reader:  
            img, label = row 
            label = int(label) 
            images.append(img) 
            labels.append(label)
    # Determine whether the images have the same size as the labels
    assert len(images) == len(labels)  
    return images, labels


# Iterate all sub-directories under the pokemon; take classnames as keys and lenghs as class labels
def load_pokemon(root, mode='train'):
    # Create an empty dictionary{key:value} which holds both classnames and labels
    name2label = {}  
    # Iterate sub-directories under the roor dir and sort them. 
    for name in sorted(os.listdir(os.path.join(root))):  
        if not os.path.isdir(os.path.join(root, name)):  
            continue
        name2label[name] = len(name2label.keys())
    # Read the paths and the labels of the csv
    images, labels = load_csv(root, 'images.csv', name2label)  
    # Divide the dataset with the ratio of 6：2：2 for train、val and test sets. 
    if mode == 'train':  
        images = images[:int(0.6*len(images))]
        labels = labels[:int(0.6*len(labels))]
    elif mode == 'val': 
        images = images[int(0.6*len(images)) : int(0.8*len(images))]
        labels = labels[int(0.6*len(labels)) : int(0.8*len(labels))]
    else:
        images = images[int(0.8*len(images)):]
        labels = labels[int(0.8*len(labels)):]
    return images, labels, name2label

img_mean = tf.constant([0.485, 0.456, 0.406])
img_std  = tf.constant([0.229, 0.224, 0.225])

def normalize(x, mean=img_mean, std=img_std):
    x = (x - mean) / std
    return x

# Denormalize the images if necessary. 
# -def denormalize(x, mean=img_mean, std=img_std):
    # -x = x * std + mean
    # -return x

def preprocess(image_path, label):
    x = tf.io.read_file(image_path)
    x = tf.image.decode_jpeg(x, channels=3)  
    x = tf.image.resize(x, [244,244])
    # Conduct data augmentation if necessary
    # -x = tf.image.random_flip_up_down(x)    # Flip up and down
    # -x = tf.image.random_flip_left_right(x) # Mirror from left to right
    x = tf.image.random_crop(x, [227,227,3]) # Crop images 
    x = tf.cast(x, dtype=tf.float32) / 255.  # Normalization 
    x = normalize(x)
    y = tf.convert_to_tensor(label)
    return x, y

# Upload the self-defined datasets
images, labels, table = load_pokemon('/home/mike/datasets/pokemon', 'train')
# Print them if necessary
# -print('images', len(images), images)
# -print('labels', len(labels), labels)
# -print(table)
# images:string path，labels:number
db = tf.data.Dataset.from_tensor_slices((images, labels))  
db = db.shuffle(1000).map(preprocess).batch(32).repeat(20)


# Set the global constant as 150
num_classes = 150

model = AlexNet((227,227,3), num_classes) 

model.summary()


# Train the model: compute gradients and update network parameters 
optimizer = optimizers.SGD(lr=0.01)  
acc_meter = metrics.Accuracy()
x_step = []
y_accuracy = []
# Input the batch for the training
for step, (x,y) in enumerate(db):  
	# Buld the gradient records
    with tf.GradientTape() as tape:  
    	# Flatten the input such as [b,28,28]->[b,784]
        x = tf.reshape(x, (-1,227,227,3))  
        output = model(x)  
        y_onehot = tf.one_hot(y, depth=150)
        loss = tf.square(output - y_onehot)
        loss = tf.reduce_sum(loss) / 32
        # Compute the gradients of each parameter
        grads = tape.gradient(loss, model.trainable_variables)  
        # Update network parameters
        optimizer.apply_gradients(zip(grads, model.trainable_variables)) 
        # Compare the predicted value and labels and related precision
        acc_meter.update_state(tf.argmax(output,axis=1), y)  
     # Print the results per 200 steps
    if step % 10 == 0: 
        print('Step', step, ': Loss is: ', float(loss), ' Accuracy: ', acc_meter.result().numpy())
        x_step.append(step)
        y_accuracy.append(acc_meter.result().numpy())
        acc_meter.reset_states()

# Visulize the result with matplolib
plt.plot(x_step, y_accuracy, label="training")
plt.xlabel("step")
plt.ylabel("accuracy")
plt.title("accuracy of training")
plt.legend()
plt.show()