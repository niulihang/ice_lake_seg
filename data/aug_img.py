# 数据增广
import glob

import numpy as np
import os
from PIL import Image

import global_var

s2_path = global_var.TRAIN_S2_PATH
img_path = global_var.TRAIN_IMGS_PATH
label_path = global_var.TRAIN_MASKS_PATH

s2_save_path = global_var.TRAIN_S2_PATH
img_save_path = global_var.TRAIN_IMGS_PATH
label_save_path = global_var.TRAIN_MASKS_PATH

s2_list = os.listdir(s2_path)
img_list = os.listdir(img_path)
label_list = os.listdir(label_path)

tran_num = 5000

bin_colormap = [0, 0, 0] + [255, 255, 255] * 254


def s2_save(ar, save_path):
    np.save(save_path, ar)


def masks_save(ar, save_path):
    mask_save = Image.fromarray(ar.squeeze(), 'P')
    mask_save.putpalette(bin_colormap)
    mask_save.save(save_path)


def img_save(ar, save_path):
    Image.fromarray(ar, 'RGBA').save(save_path)


for s2_file in glob.glob(s2_path + r'\s2_*.npy'):
    print(s2_file)
    index = os.path.basename(s2_file).split('_')[1].split('.')[0]
    label_file = label_path + r'\label_' + index + '.png'
    img_file = img_path + r'\img_' + index + '.png'  # HWC模式

    s2 = np.load(s2_file)

    label_tif = Image.open(label_file)
    label = np.array(label_tif)
    label = np.expand_dims(label, axis=0)

    img_tif = Image.open(img_file)
    img = np.array(img_tif)

    # 水平翻转
    s2_hor = np.flip(s2, axis=2)
    s2_save(s2_hor, s2_save_path + r'\s2_' + str(tran_num) + '.npy')
    label_hor = np.flip(label, axis=2)
    masks_save(label_hor, label_save_path + r'\label_' + str(tran_num) + '.png')
    img_hor = np.flip(img, axis=1)
    img_save(img_hor, img_save_path + r'\img_' + str(tran_num) + '.png')
    tran_num += 1

    # 垂直翻转
    s2_vec = np.flip(s2, axis=1)
    s2_save(s2_vec, s2_save_path + r'\s2_' + str(tran_num) + '.npy')
    label_vec = np.flip(label, axis=1)
    masks_save(label_vec, label_save_path + r'\label_' + str(tran_num) + '.png')
    img_vec = np.flip(img, axis=0)
    img_save(img_vec, img_save_path + r'\img_' + str(tran_num) + '.png')
    tran_num += 1

    # 对角镜像
    s2_dia = np.flip(s2_vec, axis=2)
    s2_save(s2_dia, s2_save_path + r'\s2_' + str(tran_num) + '.npy')
    label_dia = np.flip(label_vec, axis=2)
    masks_save(label_dia, label_save_path + r'\label_' + str(tran_num) + '.png')
    img_dia = np.flip(img_vec, axis=1)
    img_save(img_dia, img_save_path + r'\img_' + str(tran_num) + '.png')
    tran_num += 1

    label_tif.close()
    img_tif.close()
