import tensorflow as tf
import numpy as np
from data import *
import scipy.io as sio
import cv2
import matplotlib.pyplot as plt


classes=['background', 'circular', 'rod', 'algea_1', 'algea_2', 'letter', 'ruler', 'black_rod', 'mixed']
num_classes = len(classes)
size_input = 100;
num_input_channels=3;

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

def create_convolution_layer(input, num_input_channels, conv_filter_size,num_filters):
    weights = create_weights(shape=[conv_filter_size,conv_filter_size,num_input_channels,num_filters])
    biases = create_biases(num_filters)
    layer = tf.nn.conv2d(input=input, filter=weights,strides=[1,1,1,1], padding ='SAME')
    layer += biases
    layer = tf.nn.max_pool(value=layer,ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding = 'SAME')
    layer = tf.nn.relu(layer)
    return layer

def create_flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer,[-1,num_features])
    return layer


def create_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)
    layer = tf.matmul(input,weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer

X = tf.placeholder(tf.float32, shape=[None, size_input, size_input, num_input_channels], name='X')
Y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='Y_true')
Y_true_class = tf.argmax(Y_true, axis=1)

layer_conv1 = create_convolution_layer(input=X, num_input_channels=num_input_channels,conv_filter_size=3,num_filters=32)
layer_conv2 = create_convolution_layer(input=layer_conv1, num_input_channels=32,conv_filter_size=3,num_filters=64)
layer_conv3 = create_convolution_layer(input=layer_conv2, num_input_channels=64,conv_filter_size=3,num_filters=128)
layer_conv4 = create_convolution_layer(input=layer_conv3, num_input_channels=128,conv_filter_size=3,num_filters=256)
layer_flat = create_flatten_layer(layer_conv4)
layer_fc1 = create_fc_layer(input=layer_flat, num_inputs = layer_flat.get_shape()[1:4].num_elements(), num_outputs=256, use_relu=True)
layer_fc2 = create_fc_layer(input=layer_fc1, num_inputs=256, num_outputs=9, use_relu=False)


y_pred = tf.nn.softmax(layer_fc2,name='y_pred')
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2, labels=Y_true)
cost = tf.reduce_mean(cross_entropy)


optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

def train(num_iteration): 
    saver = tf.train.Saver()
    for i in range(num_iteration):
        sess.run(optimizer,feed_dict={X:X_train/255, Y_true:Y_train})
        train_loss = sess.run(cost,feed_dict={X:X_train/255, Y_true:Y_train})
        if (i%100 == 0):
            print (train_loss)
    
    saver.save(sess,"./microorganism_model/microorganisms_model.ckpt")

X_train,Y_train = get_data(resize=size_input)
print(X_train.shape)
print(Y_train.shape)

train(4000)