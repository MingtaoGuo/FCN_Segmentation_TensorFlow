import tensorflow as tf
import numpy as np

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

def conv(name, inputs, W, b, is_trainable=True):
    with tf.variable_scope(name):
        W = tf.Variable(W, name="W", trainable=is_trainable)
        b = tf.Variable(b, name="b", trainable=is_trainable)
        inputs = tf.nn.conv2d(inputs, W, [1, 1, 1, 1], "SAME")
        inputs = tf.nn.bias_add(inputs, b)
    return inputs

def max_pooling(inputs):
    return tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

def relu(inputs):
    return tf.nn.relu(inputs)

def preprocess(inputs):
    inputs_r = inputs[:, :, :, 0]
    inputs_g = inputs[:, :, :, 1]
    inputs_b = inputs[:, :, :, 2]
    inputs = tf.stack([inputs_b - _B_MEAN, inputs_g - _G_MEAN, inputs_r - _R_MEAN], axis=-1)
    return inputs / 255.0

def fully_conv(name, inputs, nums_out, k_size, strides, padding):
    nums_in = inputs.shape[-1]
    with tf.variable_scope(name):
        W = tf.get_variable("W", [k_size, k_size, nums_in, nums_out], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b", [nums_out], initializer=tf.constant_initializer([0.]))
        inputs = tf.nn.conv2d(inputs, W, [1, strides, strides, 1], padding)
        inputs = tf.nn.bias_add(inputs, b)
    return inputs

def upsample(name, inputs, scale, nums_out):
    h, w = inputs.shape[1], inputs.shape[2]
    inputs = tf.image.resize_bilinear(inputs, [h*scale, w*scale])
    inputs = fully_conv(name, inputs, nums_out, 1, 1, "SAME")
    return inputs


def vgg16(inputs):
    weights = np.load("./vgg_para/vgg16.npy", encoding="latin1").item()
    inputs = preprocess(inputs)
    inputs = relu(conv("conv1_1", inputs, weights["conv1_1"][0], weights["conv1_1"][1]))
    inputs = relu(conv("conv1_2", inputs, weights["conv1_2"][0], weights["conv1_2"][1]))
    inputs = max_pooling(inputs)
    inputs = relu(conv("conv2_1", inputs, weights["conv2_1"][0], weights["conv2_1"][1]))
    inputs = relu(conv("conv2_2", inputs, weights["conv2_2"][0], weights["conv2_2"][1]))
    inputs = max_pooling(inputs)
    inputs = relu(conv("conv3_1", inputs, weights["conv3_1"][0], weights["conv3_1"][1]))
    inputs = relu(conv("conv3_2", inputs, weights["conv3_2"][0], weights["conv3_2"][1]))
    inputs = relu(conv("conv3_3", inputs, weights["conv3_3"][0], weights["conv3_3"][1]))
    inputs = max_pooling(inputs)
    pool3 = tf.identity(inputs)
    inputs = relu(conv("conv4_1", inputs, weights["conv4_1"][0], weights["conv4_1"][1]))
    inputs = relu(conv("conv4_2", inputs, weights["conv4_2"][0], weights["conv4_2"][1]))
    inputs = relu(conv("conv4_3", inputs, weights["conv4_3"][0], weights["conv4_3"][1]))
    inputs = max_pooling(inputs)
    pool4 = tf.identity(inputs)
    inputs = relu(conv("conv5_1", inputs, weights["conv5_1"][0], weights["conv5_1"][1]))
    inputs = relu(conv("conv5_2", inputs, weights["conv5_2"][0], weights["conv5_2"][1]))
    inputs = relu(conv("conv5_3", inputs, weights["conv5_3"][0], weights["conv5_3"][1]))
    inputs = max_pooling(inputs)
    pool5 = tf.identity(inputs)
    fc1 = fully_conv("fc1", pool5, 4096, 1, 1, "SAME")
    fc2 = fully_conv("fc2", fc1, 4094, 1, 1, "SAME")
    up1 = relu(upsample("up1", fc2, 2, pool4.shape[-1]) + pool4)
    up2 = relu(upsample("up2", up1, 2, pool3.shape[-1]) + pool3)
    logits = upsample("up3", up2, 8, 21)#B x H x W x 21
    pred = tf.nn.softmax(logits)
    return pred









