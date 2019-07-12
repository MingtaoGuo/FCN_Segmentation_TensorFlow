import tensorflow as tf
from utils import read_batch
from vgg16 import vgg16
import matplotlib.pyplot as plt
from config import *


LABEL_PATH = "./VOCdevkit/VOC2007/SegmentationClass/"
IMG_PATH = "./VOCdevkit/VOC2007/JPEGImages/"

def train():
    inputs = tf.placeholder(tf.float32, [None, IMG_H, IMG_W, 3])
    labels = tf.placeholder(tf.int32, [None, IMG_H, IMG_W])#448 x 448
    labels_onehot = tf.one_hot(labels, CLASS_NUMS)
    pred = vgg16(inputs)#448 x 448 x 21
    pred_heat_map = tf.argmax(pred, axis=-1)
    pred_ = tf.reshape(pred, [-1, CLASS_NUMS])
    labels_ = tf.reshape(labels_onehot, [-1, CLASS_NUMS])
    loss = tf.reduce_mean(tf.reduce_sum(-tf.log(pred_ + EPSILON) * labels_, axis=-1))
    Opt = tf.train.AdamOptimizer(1e-4).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # saver.restore(sess, "./save_para/model.ckpt")
    for i in range(10000):
        BATCH_IMG, BATCH_LABEL = read_batch(IMG_PATH, LABEL_PATH, BATCH_SIZE)
        sess.run(Opt, feed_dict={inputs: BATCH_IMG, labels: BATCH_LABEL})
        LOSS = sess.run(loss, feed_dict={inputs: BATCH_IMG, labels: BATCH_LABEL})
        print("Iteration: %d, Loss: %f"%(i, LOSS))
        if i % 100 == 0:
            HEAT_MAT = sess.run(pred_heat_map, feed_dict={inputs: BATCH_IMG, labels: BATCH_LABEL})
            plt.imshow(HEAT_MAT[0])
            plt.savefig("./save_img/"+str(i)+".jpg")
        if i % 1000 == 0:
            saver.save(sess, "./save_para/model.ckpt")


if __name__ == "__main__":
    train()

