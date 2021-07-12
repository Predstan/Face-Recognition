from os import name
import os
os.environ["CUDA_VISIBLE_DEVICE"] = "1,2"
import random
import numpy as np
import cv2 
import tensorflow as tf
from inception_blocks_v2 import faceRecoModel
import fr_utils
tf.compat.v1.disable_eager_execution()
import tensorflow.keras.backend as K
sess = tf.compat.v1.Session()
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
print("HERE::::::::::::::::::::::::::::::::::::::::::::::::::::::::",tf.test.gpu_device_name())
#tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
tf.config.experimental.set_memory_growth(physical_devices[1], enable=True)



def generator(dictionary, dataframe, batch_size, augument=True, size=(160, 160)):
    num_samples = 0
    names = list(dictionary.keys())
    
    for keys in dictionary:num_samples+=len(dictionary[keys])
    while 1:
        anchor_= []
        positive_ = []
        negative_ = []
        for i in range(batch_size):
            anchor_name = random.choice(names)
            anchor_pix = random.choice(dictionary[anchor_name])
            positive = random.choice(dictionary[anchor_name]) 
            while positive == anchor_pix:positive = random.choice(dictionary[anchor_name])
            neg_name = random.choice(names)
            while neg_name ==anchor_name: neg_name = random.choice(names)
            negative = random.choice(dictionary[neg_name])
            anchor_.append(anchor_pix)
            positive_.append(positive)
            negative_.append(negative)
            
        anchor = []
        positive = []
        negative = []
        for i in range(batch_size):
            anchor.append(load_image(anchor_[i], dataframe))
            positive.append(load_image(positive_[i], dataframe))
            negative.append(load_image(negative_[i], dataframe))
    
        yield np.array(anchor, dtype = np.int), np.array(positive, dtype = np.int), np.array(negative, dtype = np.int)
                                                    
            
            

def load_image(index, dataframe, size = (160, 160)):
    global j
    x, y, h, w = [int(i) for i in dataframe.iloc[index, 1].split(",")]
    img =  cv2.imread(dataframe.iloc[index, 0])
    return cv2.resize(img[y:y+h,x:x+w], size, interpolation = cv2.INTER_AREA)


def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss as defined by formula (3)
    
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- anchor number, value of the loss
    """
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    # Step 1: Compute the (encoding) distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss =tf.reduce_sum(tf.maximum(0.0, basic_loss))
    
    
    return loss
        
    
model = faceRecoModel([3, 160, 160])
model.summary()



alpha = 0.2
anchor = tf.compat.v1.placeholder(tf.float32, shape=[None, 3, 160, 160], name="anchor")
positive = tf.compat.v1.placeholder(tf.float32, shape=[None, 3, 160, 160], name="positive")
negative = tf.compat.v1.placeholder(tf.float32, shape=[None, 3, 160, 160], name="negative")

alpha = tf.constant(alpha)


anchor_pred = model(anchor)
positive_pred = model(positive)
negative_pred = model(negative)

y_pred = (anchor_pred, positive_pred, negative_pred)

print(tf.shape(y_pred))

loss = triplet_loss(None, y_pred, alpha=alpha)


train_step = tf.compat.v1.train.AdagradOptimizer(0.001).minimize(loss)





