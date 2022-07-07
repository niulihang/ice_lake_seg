import glob
import shutil

import gdal
import rasterio
import scipy.ndimage
import numpy as np
import os
import torch
from PIL import Image

from model.unet import UNet
from predict import predict_img
from model.mod_att_unet import ModAttUnet


def full_to_crop(s2_path):
    all_bands = None
    ct = 1
    crop_pos = []

    new_dir = os.path.dirname(s2_path) + '/predict_dir' + \
              '/Pred_' + os.path.basename(s2_path)[:26] + \
              os.path.basename(s2_path)[37:44]
    s2_dir = new_dir + '/s2/'
    mask_dir = new_dir + '/masks/'
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    if not os.path.exists(s2_dir):
        os.makedirs(s2_dir)
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    for band_path in glob.glob(s2_path + '/GRANULE/L1C_*/IMG_DATA/*_B*.jp2'):
        with rasterio.open(band_path) as band_obj:
            band_np = band_obj.read()
            if band_np.shape[1] != SET_X_SHAPE or band_np.shape[2] != SET_Y_SHAPE:
                zoom_x_size = SET_X_SHAPE / band_np.shape[1]
                zoom_y_size = SET_Y_SHAPE / band_np.shape[2]
                band_np = scipy.ndimage.zoom(band_np, (1, zoom_x_size, zoom_y_size), order=0)
            if all_bands is None:
                all_bands = band_np
            else:
                all_bands = np.concatenate((all_bands, band_np), axis=0)
    for i in range(SET_X_SHAPE // CROP_SIZE + 1):
        x_start = i * CROP_SIZE
        x_end = (i + 1) * CROP_SIZE
        for j in range(SET_Y_SHAPE // CROP_SIZE + 1):
            y_start = j * CROP_SIZE
            y_end = (j + 1) * CROP_SIZE
            bands_crop = all_bands[:, x_start:x_end, y_start:y_end]
            x_len = bands_crop.shape[1]
            y_len = bands_crop.shape[2]
            crop_pos.append([x_start, x_end, y_start, y_end])
            if x_len != CROP_SIZE or y_len != CROP_SIZE:  # 填充
                bands_crop = np.pad(bands_crop, ((0, 0), (0, CROP_SIZE - x_len), (0, CROP_SIZE - y_len)),
                                    'constant', constant_values=0)
            np.save(f'{s2_dir}s2_{ct}.npy', bands_crop)
            ct += 1
    crop_pos = np.array(crop_pos)
    np.save(new_dir + '/crop_pos.npy', crop_pos)
    return new_dir, s2_dir, mask_dir


CROP_SIZE = 128
SET_X_SHAPE = 10980
SET_Y_SHAPE = 10980

S2_DIR = r'D:\data\s-2\2022'


net = ModAttUnet(n_channels=13, n_classes=2)
net_path = 'checkpoints/modattunet_epoch102_0519.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device=device)
net.load_state_dict(torch.load(net_path, map_location=device))

net.eval()

for s2_img_path in glob.glob(S2_DIR + r'\*.SAFE'):
    print(s2_img_path)
    # 切割原始图像
    dir, s2_dir, mask_dir = full_to_crop(s2_img_path)  # 裁剪s2
    print('Crop s2 data completed.')
    # dir = 'data/original/predict/Pred_S2B_MSIL1C_20210131T041729_T41CPV'
    # s2_dir = 'data/original/predict/Pred_S2B_MSIL1C_20210131T041729_T41CPV/s2/'
    # mask_dir = 'data/original/predict/Pred_S2B_MSIL1C_20210131T041729_T41CPV/masks/'
    # crop_pos = np.array(crop_pos)
    crop_pos = np.load(dir + '/crop_pos.npy')
    s2_list = os.listdir(s2_dir)
    s2_num = len(s2_list)
    # 预测
    for i in range(s2_num):
        s2 = np.load(s2_dir + s2_list[i])
        index = s2_list[i].split('_')[1].split('.')[0]
        out_mask_name = mask_dir + 'label_' + index + '.npy'
        mask_pred = predict_img(net=net,
                                full_img=s2,
                                scale_factor=1.0,
                                out_threshold=0.5,
                                device=device)
        mask_pred = np.argmax(mask_pred, axis=0)
        # mask_pred = mask_pred[np.newaxis, ...]
        # with rasterio.open(out_mask_name, 'w', driver='PNG',
        #                    width=mask_pred.shape[1], height=mask_pred.shape[2], count=mask_pred.shape[0],
        #                    dtype='uint8', nodata=0) as dst:
        #     dst.write(mask_pred.astype(np.uint8))
        # mask_img = Image.new('RGB', (mask_pred.shape[0], mask_pred.shape[1]), (255, 255, 255))
        # for x in range(mask_pred.shape[0]):
        #     for y in range(mask_pred.shape[1]):
        #         if mask_pred[x, y] == 1:
        #             mask_img.putpixel((x, y), (0, 0, 255))
        # mask_img.save(out_mask_name)
        np.save(out_mask_name, mask_pred)

    print('Predict completed.')
    # 拼接mask
    mask_full = np.zeros((SET_X_SHAPE, SET_Y_SHAPE))
    out_mask_full_name = dir + r'/' + os.path.basename(s2_img_path).split('.')[0] + '.tif'
    ct_l = 0
    for i in range(s2_num):
        mask_np = np.load(mask_dir + 'label_' + str(i + 1) + '.npy')
        x_start, x_end, y_start, y_end = crop_pos[i]
        if mask_np.sum():
            ct_l += mask_np.sum()
            if x_end > SET_X_SHAPE or y_end > SET_Y_SHAPE:
                if x_end > SET_X_SHAPE:
                    x_end = SET_X_SHAPE
                if y_end > SET_Y_SHAPE:
                    y_end = SET_Y_SHAPE
                mask_full[x_start:x_end, y_start:y_end] = mask_np[0:(x_end-x_start), 0:(y_end-y_start)]
            else:
                mask_full[x_start:x_end, y_start:y_end] = mask_np
    print(f'ct_l:{ct_l}, mask_full:{mask_full.sum()}')
    # with rasterio.open(out_mask_full_name, 'w', driver='PNG',
    #                    width=mask_full.shape[1], height=mask_full.shape[2], count=mask_full.shape[0],
    #                    dtype='uint8', nodata=0) as dst:
    #     dst.write(mask_full.astype(np.uint8))
    # mask_full_img = Image.new('RGB', (mask_full.shape[0], mask_full.shape[1]), (255, 255, 255))
    # for x in range(mask_full.shape[0]):
    #     for y in range(mask_full.shape[1]):
    #         if mask_full[x, y] == 1:
    #             mask_full_img.putpixel((y, x), (0, 0, 255))
    # mask_full_img.save(out_mask_full_name)
    s2_ds = gdal.Open(s2_img_path + r'\MTD_MSIL1C.xml')
    s2_ds_list = s2_ds.GetSubDatasets()
    vis_ds = gdal.Open(s2_ds_list[0][0])
    driver = gdal.GetDriverByName('GTIFF')
    out_tif = driver.Create(out_mask_full_name, mask_full.shape[0], mask_full.shape[1],
                            1, gdal.GDT_Float32)
    out_tif.SetProjection(vis_ds.GetProjection())
    out_tif.SetGeoTransform(vis_ds.GetGeoTransform())
    out_tif.GetRasterBand(1).WriteArray(mask_full)
    out_tif.FlushCache()
    out_tif = None
    vis_ds = None
    s2_ds = None

    print('Merge masks completed.')









