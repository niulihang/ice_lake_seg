# 分割训练集和测试集

import glob
import random
import os
import shutil
import global_var


TEST_SIZE = 0.25

mask_files = glob.glob(global_var.PREP_PATH + '/masks/label_*.png')
random.seed(42)
mask_chos = random.sample(mask_files, int(len(mask_files) * TEST_SIZE))

for mask_file in mask_files:
    mask_name = os.path.basename(mask_file)
    index = mask_name.split('_')[1].split('.')[0]
    test_mask_path = global_var.TEST_PATH + r'\masks' + '\\' + mask_name
    train_mask_path = global_var.TRAIN_PATH + r'\masks' + '\\' + mask_name
    img_name = f'img_{index}.png'
    prep_img_path = global_var.PREP_PATH + r'\imgs' + '\\' + img_name
    test_img_path = global_var.TEST_PATH + r'\imgs' + '\\' + img_name
    train_img_path = global_var.TRAIN_PATH + r'\imgs' + '\\' + img_name
    s2_name = f's2_{index}.npy'
    prep_s2_path = global_var.PREP_PATH + r'\s2' + '\\' + s2_name
    test_s2_path = global_var.TEST_PATH + r'\s2' + '\\' + s2_name
    train_s2_path = global_var.TRAIN_PATH + r'\s2' + '\\' + s2_name
    if mask_file in mask_chos:
        print(f'{mask_name} put into test')
        shutil.copy(mask_file, test_mask_path)
        shutil.copy(prep_img_path, test_img_path)
        shutil.copy(prep_s2_path, test_s2_path)
    else:
        print(f'{mask_name} put into train')
        shutil.copy(mask_file, train_mask_path)
        shutil.copy(prep_img_path, train_img_path)
        shutil.copy(prep_s2_path, train_s2_path)
