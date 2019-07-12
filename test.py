import tensorflow as tf
from PIL import Image
import numpy as np
from utils import img_mask_blend
from vgg16 import vgg16
import matplotlib.pyplot as plt
from config import *


class FCN:
    def __init__(self):
        self.inputs = tf.placeholder(tf.float32, [None, IMG_H, IMG_W, 3])
        pred = vgg16(self.inputs)  # 448 x 448 x 21
        self.pred_heat_map = tf.argmax(pred, axis=-1)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.sess, "./save_para/model.ckpt")

    def predict(self, img_path):
        img = np.array(Image.open(img_path).resize([IMG_W, IMG_H]))[np.newaxis, :, :, :]
        HEAT_MAT = self.sess.run(self.pred_heat_map, feed_dict={self.inputs: img})
        plt.imshow(HEAT_MAT[0])
        return img[0], HEAT_MAT[0]

if __name__ == "__main__":
    img_path = "C:/Users/gmt/Desktop/cats/9.jpg"
    fcn = FCN()
    img, mask = fcn.predict(img_path)
    result = img_mask_blend(img, mask)
    Image.fromarray(np.uint8(result)).show()






