#   ___ _         _
#  / __| |_  __ _(_)_ _  ___ _ _
# | (__| ' \/ _` | | ' \/ -_) '_|
#  \___|_||_\__,_|_|_||_\___|_|
#
# Construct neural networks using chaining function calls like so...
#
#  y_conv = (input(x_image, [28, 28, 1])
#              .to_conv(filter=[5,5,32]).pool(2).normalize([1, 2])
#              .to_conv(filter=[5,5,64]).pool(2)
#              .to_fc(1024)
#              .output_vector(10))

import tensorflow as tf
import numpy as np
import math

def last(arr):
    return arr[len(arr) - 1]


def input(tensor, shape):
    return LayerNode(shape, input_tensor=tensor)


class LayerNode:
    def __init__(self, shape, prev_layer=None, input_tensor=None):
        self.shape = shape
        self.prev_layer = prev_layer
        self.tensor = None

        # propogate input tensor, or set the one specifically indicated
        if input_tensor is not None:
            self.tensor = input_tensor
        elif prev_layer is not None:
            self.tensor = prev_layer.tensor

        # at this point, the input tensor should have been selected
        assert self.tensor is not None, "No input tensor could be determined."

        if prev_layer:
            # a reshape is needed
            if self.is_rank(1) and prev_layer.is_gt_rank(1):
                self.reshape([np.prod(prev_layer.shape)]) # flatten

    def is_rank(self, n):
        return len(self.shape) == n

    def is_gt_rank(self, n):
        return len(self.shape) > n

    def is_lt_rank(self, n):
        return len(self.shape) < n

    def normalize(self, across_dimensions):
        counter = 0
        for dim_idx in across_dimensions:
            assert dim_idx < len(self.shape) and dim_idx > 0, "Cannot normalize across dimension %d, out of bounds" % dim_idx
            across_dimensions[counter] += 1
            counter += 1

        self.tensor = tf.nn.l2_normalize(self.tensor, across_dimensions);
        return self

    def reshape(self, shape):
        # ensure that the sizes of each layer are compatible
        assert np.prod(self.shape) == np.prod(shape), "Layer shape '%r' is mismatched with new shape '%r'" % (self.shape, shape)

        self.shape = shape.copy()
        shape.insert(0, -1)

        self.tensor = tf.reshape(self.tensor, shape)
        return self

    def out(self, size):
        return self.output_vector(size)

    def output_vector(self, size):
        if self.is_gt_rank(2):
            self.reshape([np.prod(self.shape)]) # flatten

        weights = tf.Variable(tf.truncated_normal([last(self.shape), size], stddev=0.1))
        biases  = tf.Variable(tf.truncated_normal([size], stddev=0.1))

        self.tensor = tf.matmul(self.tensor, weights) + biases

        return self.tensor

    def to_convolution(self, filter, stride=[1, 1, 1, 1], activation_function=tf.nn.relu):
        conv_layer = LayerNode(self.shape, prev_layer=self)

        _filter = filter.copy()
        W = [_filter[0], _filter[1], last(self.shape), last(_filter)]

        weights = tf.Variable(tf.truncated_normal(W, stddev=0.1))
        biases  = tf.Variable(tf.truncated_normal([last(_filter)], stddev=0.1))

        if len(stride) < 4:
            stride.insert(0, 1)

        Z = tf.nn.conv2d(conv_layer.tensor, weights, strides=stride, padding='VALID') + biases

        conv_layer.tensor = activation_function(Z)
        conv_layer.shape = [
            math.ceil(float(self.shape[0] - filter[0] + 1) / float(stride[2])),
            math.ceil(float(self.shape[1] - filter[1] + 1) / float(stride[1])),
            last(filter)
        ]

        return conv_layer

    def to_conv(self, filter, stride=[1, 1, 1, 1], activation_function=tf.nn.relu):
        return self.to_convolution(filter=filter, stride=stride, activation_function=activation_function)

    def pool(self, reduction):
        assert len(self.shape) > 2, "Can only perform pooling ops on convolution layers"

        n = reduction
        dim = [1, n, n, 1]
        self.tensor = tf.nn.max_pool(self.tensor, ksize=dim, strides=dim, padding='SAME')

        for i in range(len(self.shape) - 1):
            self.shape[i] = self.shape[i] // n

        return self

    def to_fully_connected(self, size, activation_function=tf.nn.relu):
        fc_layer = LayerNode([last(self.shape), size], prev_layer=self)

        if self.is_gt_rank(2):
            self.reshape([np.prod(self.shape)]) # flatten
            fc_layer.shape = [last(self.shape), size]

        weights = tf.Variable(tf.truncated_normal(fc_layer.shape, stddev=0.1))
        biases  = tf.Variable(tf.truncated_normal([size], stddev=0.1))

        Z = tf.matmul(self.tensor, weights) + biases
        fc_layer.tensor = activation_function(Z)

        return fc_layer

    def to_fc(self, size, activation_function=tf.nn.relu):
        return self.to_fully_connected(size=size, activation_function=activation_function)
