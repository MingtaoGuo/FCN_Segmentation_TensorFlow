from PIL import Image
import numpy as np
import scipy.io as sio
import os
from config import *

def data_augment(img, label):
    p = np.random.uniform(0, 1)
    if p < 1/3:
        return img, label
    elif p >= 1/3 and p < 2/3:
        img = np.flip(img, axis=0)
        label = np.flip(label, axis=0)
        return img, label
    else:
        img = np.flip(img, axis=1)
        label = np.flip(label, axis=1)
        return img, label

def data_crop_and_resize(img, label):
    img_h, img_w = img.shape[0], img.shape[1]
    if img_h < img_w:
        start_w = np.random.randint(0, img_w - img_h + 1)
        img = img[:, start_w:start_w + img_h]
        label = label[:, start_w:start_w + img_h]
        img = np.array(Image.fromarray(np.uint8(img)).resize([IMG_W, IMG_H]))
        label = np.array(Image.fromarray(np.uint8(label)).resize([IMG_W, IMG_H]))
    else:
        start_h = np.random.randint(0, img_h - img_w + 1)
        img = img[start_h:start_h + img_w, :]
        label = label[start_h:start_h + img_w, :]
        img = np.array(Image.fromarray(np.uint8(img)).resize([IMG_W, IMG_H]))
        label = np.array(Image.fromarray(np.uint8(label)).resize([IMG_W, IMG_H]))
    return img, label

def read_batch(img_path, label_path, batch_size):
    file_names = os.listdir(label_path)
    nums = file_names.__len__()
    batch_set = np.random.randint(0, nums, [batch_size])
    batch = np.zeros([batch_size, IMG_H, IMG_W, 3])
    labels = np.zeros([batch_size, IMG_H, IMG_W])
    for idx, n in enumerate(batch_set):
        img = np.array(Image.open(img_path + file_names[n][:-3] + "jpg"))
        label = np.array(Image.open(label_path + file_names[n]))
        label[label > 20] = 0
        img, label = data_augment(img, label)
        img, label = data_crop_and_resize(img, label)
        batch[idx, :, :, :] = img
        labels[idx, :, :] = label
    return batch, labels

def read_colab_batch(img_dict, batch_size):
    nums = 422
    batch_set = np.random.randint(0, nums, [batch_size])
    batch = np.zeros([batch_size, IMG_H, IMG_W, 3])
    labels = np.zeros([batch_size, IMG_H, IMG_W])
    for idx, n in enumerate(batch_set):
        img = img_dict[str(n)][:, :, :3]
        label = img_dict[str(n)][:, :, 3]
        label[label > 20] = 0
        img, label = data_augment(img, label)
        img, label = data_crop_and_resize(img, label)
        batch[idx, :, :, :] = img
        labels[idx, :, :] = label
    return batch, labels

def img2mat(img_path, label_path):
    label_name = os.listdir(label_path)
    data = {}
    for idx, name in enumerate(label_name):
        img = np.array(Image.open(img_path + name[:-3] + "jpg"), dtype=np.uint8)
        label = np.array(Image.open(label_path + name), dtype=np.uint8)
        concat = np.dstack((img, label))
        data[str(idx)] = concat
        print(idx)
    sio.savemat("data.mat", data)

def img_mask_blend(img, mask):
    for i in range(1, 21):
        temp_mask = mask * 1
        temp_mask[mask != i] = 0
        temp_mask[mask == i] = 1
        if np.sum(temp_mask) < 1:
            continue
        img = Image.fromarray(np.uint8(img))
        temp = np.dstack((temp_mask * color[i][0], temp_mask * color[i][1], temp_mask * color[i][2]))
        temp = Image.fromarray(np.uint8(temp))
        img = np.array(Image.blend(img, temp, 0.5)) * temp_mask[:, :, np.newaxis] + img * (1 - temp_mask[:, :, np.newaxis])
    return img

