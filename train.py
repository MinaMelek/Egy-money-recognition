# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 15:17:00 2017

@author: MinaMelek
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import dataset
import random
#    
## Convolutional Layer 1.
filter_size1 = 3 
#num_filters1 = 32
#
## Convolutional Layer 2.
#filter_size2 = 3
#num_filters2 = 32
#
## Convolutional Layer 3.
#filter_size3 = 3
#num_filters3 = 64
#    
## Fully-connected layer.
#fc_size = 128             # Number of neurons in fully-connected layer.

# Number of color channels for the images: 1 channel for gray-scale.
num_channels = 3

# image dimensions (only squares for now)
height = 40
width = 80

# Size of image when flattened to a single dimension
img_size_flat = height * width * num_channels

# Tuple with height and width of images used to reshape arrays.
img_shape = (height, width)

# class info

classes = ['1', '2', '3', '4', '5', '6', '7']
num_classes = len(classes)

# batch size
batch_size = 32 # batch size

keep_prob1 = tf.placeholder(tf.float32)
keep_prob2 = tf.placeholder(tf.float32)
keep_prob3 = tf.placeholder(tf.float32)
keep_probf = tf.placeholder(tf.float32)

# validation split
validation_size = .12

# how long to wait after validation loss stops improving before terminating training
early_stopping = None  # use None if you don't want to implement early stoping


train_path='./training_data'
test_path='./testing_data'


data = dataset.read_train_sets(train_path, height, width, classes, validation_size=validation_size)
test_images, test_ids = dataset.read_test_set(test_path, height, width,classes)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(test_images)))
print("- Validation-set:\t{}".format(len(data.valid.labels)))



def new_weights(shape,name=None):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05),name=name)

def new_biases(length,name=None):
    return tf.Variable(tf.constant(0.05, shape=[length]),name=name)



def new_conv_layer(input,              # The previous layer.
               num_input_channels, # Num. channels in prev. layer.
               filter_size,        # Width and height of each filter.
               num_filters,        # Number of filters.
               use_pooling=True,name=None):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape,name="W_conve"+str(name))#if name!=None: name="W_conve"+name else: name=name

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters, name="b_conve"+str(name))

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
    	             filter=weights,
    	             strides=[1, 1, 1, 1],
    	             padding='SAME',name=name)

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights, biases

    

def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features


def new_fc_layer(input,          # The previous layer.
             num_inputs,     # Num. inputs from prev. layer.
             num_outputs,    # Num. outputs.
             use_relu=True,
             name=None): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs],name="W_fc"+str(name))
    biases = new_biases(length=num_outputs,name="b_fc"+str(name))

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer,weights,biases
""" My Functions """
def conv2d(x, W, name=None):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)

def max_pool_2x2(x, name=None):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME',name=name)

def extend_labels(labels,classes):
    sets_num = labels.shape[0]
    ex = np.reshape([False]* sets_num *classes,[sets_num,classes])
    for n in range(sets_num): ex[n,int(labels[n])] = True
    return ex
    
def ConvolutionalNeuralNetwork(x):#, keep_rate):
    weights = {
        # 3 x 3 convolution, 1 input image, 32 outputs
        'W_conv11': tf.Variable(tf.random_normal([3, 3, num_channels, 64],stddev=0.01),name='conv11_w'),
        # 3 x 3 conv, 32 inputs, 64 outputs
        'W_conv12': tf.Variable(tf.random_normal([3, 3, 64, 128],stddev=0.01),name='conv12_w'),
        # 3 x 3 convolution, 1 input image, 32 outputs
        'W_conv21': tf.Variable(tf.random_normal([3, 3, 128, 256],stddev=0.01),name='conv21_w'),
        # 3 x 3 conv, 32 inputs, 64 outputs
        'W_conv22': tf.Variable(tf.random_normal([3, 3, 256, 256],stddev=0.01),name='conv22_w'),
        # 3 x 3 convolution, 1 input image, 32 outputs
        'W_conv31': tf.Variable(tf.random_normal([3, 3, 256, 256],stddev=0.01),name='conv31_w'),
        # 3 x 3 conv, 32 inputs, 64 outputs
        'W_conv32': tf.Variable(tf.random_normal([3, 3, 256, 512],stddev=0.01),name='conv32_w'),
        # fully connected, 8*8*64 inputs, 1024 outputs
        'W_fc': tf.Variable(tf.random_normal([5*10*512, 1024],stddev=0.01),name='fc1_w'),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, num_classes],stddev=0.01),name='out_w')
    }
    
    biases = {
        'b_conv11': tf.Variable(tf.ones([64]),name='conv11_b'),
        'b_conv12': tf.Variable(tf.ones([128]),name='conv12_b'),
        'b_conv21': tf.Variable(tf.ones([256]),name='conv21_b'),
        'b_conv22': tf.Variable(tf.ones([256]),name='conv22_b'),
        'b_conv31': tf.Variable(tf.ones([256]),name='conv31_b'),
        'b_conv32': tf.Variable(tf.ones([512]),name='conv32_b'),
        'b_fc': tf.Variable(tf.ones([1024]),name='fc1_b'),
        'out': tf.Variable(tf.ones([num_classes]),name='out_b')
    }
    
    global saver_w
    global saver_b
    saver_w = tf.train.Saver(weights)
    saver_b = tf.train.Saver(biases)
    
    
    # Reshape input to a 4D tensor
    x = tf.reshape(x, shape=[-1, height , width , num_channels ])
    
    
    # First Convolution Layer, using our function
    conv1 = tf.nn.relu(conv2d(x, weights['W_conv11'],name='CONV11') + biases['b_conv11'])
    # Convolution Layer, using our function
    conv1 = tf.nn.relu(conv2d(conv1, weights['W_conv12'],name='CONV12') + biases['b_conv12'])
    # Max Pooling (down-sampling)
    conv1 = max_pool_2x2(conv1, name='MAX_POOL1')
    # Dropout
    conv1_drop = tf.nn.dropout(conv1, keep_prob1, name='DROPOUT1')
    # Second Convolution Layer, using our function
    conv2 = tf.nn.relu(conv2d(conv1_drop, weights['W_conv21'],name='CONV21') + biases['b_conv21'])
    # Convolution Layer, using our function
    conv2 = tf.nn.relu(conv2d(conv2, weights['W_conv22'],name='CONV22') + biases['b_conv22'])
    # Max Pooling (down-sampling)
    conv2 = max_pool_2x2(conv2, name='MAX_POOL2')
    # Dropout
    conv2_drop = tf.nn.dropout(conv2, keep_prob2, name='DROPOUT2')   
    # Third Convolution Layer, using our function
    conv3 = tf.nn.relu(conv2d(conv2_drop, weights['W_conv31'],name='CONV31') + biases['b_conv31'])
    # Convolution Layer, using our function
    conv3 = tf.nn.relu(conv2d(conv3, weights['W_conv32'],name='CONV32') + biases['b_conv32'])
    # Max Pooling (down-sampling)
    conv3 = max_pool_2x2(conv3, name='MAX_POOL3')
    # Dropout
    conv3_drop = tf.nn.dropout(conv3, keep_prob3, name='DROPOUT3')    
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer
    fc = tf.reshape(conv3_drop, [-1, 5*10*512])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'],name='FC1') + biases['b_fc'])
    # Dropout
    fc = tf.nn.dropout(fc, keep_probf, name='DROPOUTf')
    output = tf.matmul(fc, weights['out'], name='OUT') + biases['out']
    return output
    
#*********************************************

session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='IN')
x_image = tf.reshape(x, [-1, height, width, num_channels])

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)


########################################################1
layer_conv11, weights_conv11, biases_conv11 = \
new_conv_layer(input=x_image,
               num_input_channels=num_channels,
               filter_size=filter_size1,
               num_filters=64,
               use_pooling=False,
               name="11")
#print("now layer2 input")
#print(layer_conv1.get_shape())     
layer_conv12, weights_conv12, biases_conv12 = \
new_conv_layer(input=layer_conv11,
               num_input_channels=64,
               filter_size=filter_size1,
               num_filters=128,
               use_pooling=True,
               name="12")
# Dropout
layer_conv12 = tf.nn.dropout(layer_conv12, keep_prob1, name='DROPOUT1')
########################################################2
layer_conv21, weights_conv21, biases_conv21 = \
new_conv_layer(input=layer_conv12,
               num_input_channels=128,
               filter_size=filter_size1,
               num_filters=256,
               use_pooling=False,
               name="21")
#print("now layer2 input")
#print(layer_conv1.get_shape())     
layer_conv22, weights_conv22, biases_conv22 = \
new_conv_layer(input=layer_conv21,
               num_input_channels=256,
               filter_size=filter_size1,
               num_filters=256,
               use_pooling=True,
               name="22")
# Dropout
layer_conv22 = tf.nn.dropout(layer_conv22, keep_prob2, name='DROPOUT2')
########################################################3
layer_conv31, weights_conv31, biases_conv31 = \
new_conv_layer(input=layer_conv22,
               num_input_channels=256,
               filter_size=filter_size1,
               num_filters=256,
               use_pooling=False,
               name="31")
#print("now layer2 input")
#print(layer_conv1.get_shape())     
layer_conv32, weights_conv32, biases_conv32 = \
new_conv_layer(input=layer_conv31,
               num_input_channels=256,
               filter_size=filter_size1,
               num_filters=512,
               use_pooling=True,
               name="32")
# Dropout
layer_conv32 = tf.nn.dropout(layer_conv32, keep_prob3, name='DROPOUT3')
#print("now layer flatten input")
#print(layer_conv3.get_shape())     
########################################################FCN          
layer_flat, num_features = flatten_layer(layer_conv32)

layer_fc1, weights_fc1, biases_fc1 = \
new_fc_layer(input=layer_flat,
             num_inputs=num_features,
             num_outputs=1024,
             use_relu=True,
               name="1")
# Dropout
layer_fc1 = tf.nn.dropout(layer_fc1, keep_probf, name='DROPOUTf')
layer_fc2, weights_fc2, biases_fc2 = \
new_fc_layer(input=layer_fc1,
             num_inputs=1024,
             num_outputs=num_classes,
             use_relu=False,
               name="out")

Weights = {        
# 3 x 3 convolution, 1 input image, 32 outputs
        'W_conv11': weights_conv11,
        # 3 x 3 conv, 32 inputs, 64 outputs
        'W_conv12': weights_conv12,
        # 3 x 3 convolution, 1 input image, 32 outputs
        'W_conv21': weights_conv21,
        # 3 x 3 conv, 32 inputs, 64 outputs
        'W_conv22': weights_conv22,
        # 3 x 3 convolution, 1 input image, 32 outputs
        'W_conv31': weights_conv31,
        # 3 x 3 conv, 32 inputs, 64 outputs
        'W_conv32': weights_conv32,
        # fully connected, 8*8*64 inputs, 1024 outputs
        'W_fc': weights_fc1,
        # 1024 inputs, 10 outputs (class prediction)
        'out': weights_fc2
           }
biases = {
        'b_conv11': biases_conv11,
        'b_conv12': biases_conv12,
        'b_conv21': biases_conv21,
        'b_conv22': biases_conv22,
        'b_conv31': biases_conv31,
        'b_conv32': biases_conv32,
        'b_fc': biases_fc1,
        'out': biases_fc2
          }
saver_w = tf.train.Saver(Weights)
saver_b = tf.train.Saver(biases)

y_pred = tf.nn.softmax(layer_fc2,name='y_pred')

y_pred_cls = tf.argmax(y_pred, dimension=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                    labels=y_true)
cost = tf.reduce_mean(cross_entropy)

#optimizer = tf.train.AdamOptimizer(learning_rate=5e-5).minimize(cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#session.run(tf.global_variables_initializer()) # for newer versions
session.run(tf.initialize_all_variables()) # for older versions
train_batch_size = batch_size

### FOR TENSORBOARD TUTORIAL ONLY
#writer= tf.summary.FileWriter('/tmp/tensorboard_tut')
#writer.add_graph(session.graph)


def print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    # Calculate the accuracy on the training-set.
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))



total_iterations = 0

def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    best_val_loss = float("inf")

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(train_batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(train_batch_size)
       
        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, flattened image shape]
        x_batch = x_batch.reshape(train_batch_size, img_size_flat)
        x_valid_batch = x_valid_batch.reshape(train_batch_size, img_size_flat)
        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch,
                           keep_prob1: 0.75, keep_prob2: 0.75, keep_prob3: 0.75, keep_probf: 0.5}
        
        feed_dict_validate = {x: x_valid_batch,
                              y_true: y_valid_batch,
                              keep_prob1: 1, keep_prob2: 1, keep_prob3: 1, keep_probf: 1}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)
        saver = tf.train.Saver()
        saver.save(session, 'my_test_model') 

        # Print status at end of each epoch (defined as full pass through training dataset).
        if i % int(data.train.num_examples/batch_size) == 0: 
            val_loss = session.run(cost, feed_dict=feed_dict_validate)
            epoch = int(i / int(data.train.num_examples/batch_size))
            
            print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss)
            

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    
optimize(num_iterations=3000)
#print_validation_accuracy()