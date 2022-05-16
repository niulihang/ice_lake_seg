from skimage import io
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import glob
import scipy.ndimage
from PIL import Image

# LABEL_PATH = '../data/original/Lake_S2B_MSIL1C_20210131T041729.png'
# PNG_PATH = '../data/original/S2B_MSIL1C_20210131T041729_N0209_R061_T42CVD_20210131T072137_RGB.png'
# S2_PATH = '../data/original/S2B_MSIL1C_20210131T041729_N0209_R061_T42CVD_20210131T072137.SAFE'

CROP_SIZE = 128
# CROP_AREA = CROP_SIZE * CROP_SIZE
SET_X_SHAPE = 10980
SET_Y_SHAPE = 10980

count = count_start = 0  # label切割计数器

Image.MAX_IMAGE_PIXELS = 120560400
bin_colormap = [0, 0, 0] + [255, 255, 255] * 254

# 裁剪label
for s2_img_path in glob.glob('../data/original/S2*_MSIL1C_***.SAFE'):
# for s2_img_path in glob.glob('../data/original/S2B_MSIL1C_20200119T131859_N0208_R095_T19DEA_20200119T142732.SAFE'):
    s2_name = s2_img_path[17:]
    label_name = 'Lake_' + s2_name[:26] + s2_name[37:44] + '.png'
    png_name = s2_name[:-5] + '_RGB.png'

    label_path = '../data/original/' + label_name
    s2_path = '../data/original/' + s2_name
    png_path = '../data/original/' + png_name

    print(f'{s2_name} start')
    crop_pos = []  # [[x_start, x_end, y_start, y_end]]
    percents = []  # water像素占比

    with Image.open(label_path) as label_tif:
        # label_data = label_tif.read()
        label_data = np.array(label_tif)
        x_size = label_data.shape[0]
        y_size = label_data.shape[1]
        # 裁剪label，记录裁剪坐标到crop_pos
        for i in range(x_size // CROP_SIZE + 1):
            x_start = i * CROP_SIZE
            x_end = (i + 1) * CROP_SIZE
            for j in range(y_size // CROP_SIZE + 1):
                y_start = j * CROP_SIZE
                y_end = (j + 1) * CROP_SIZE
                label_crop = label_data[x_start:x_end, y_start:y_end]
                x_len = label_crop.shape[0]
                y_len = label_crop.shape[1]
                percent = label_crop.sum() / (x_len * y_len)
                if not percent >= 0.1:  # 设置过滤比
                    continue
                else:
                    count += 1
                    percents.append(percent)
                    crop_pos.append([x_start, x_end, y_start, y_end])
                # 填充尺寸不足的图像
                if x_len != CROP_SIZE or y_len != CROP_SIZE:
                    print(f'第{count}个label需要填充')
                    label_crop = np.pad(label_crop, ((0, CROP_SIZE - x_len), (0, CROP_SIZE - y_len)),
                                        'constant', constant_values=0)
                # print(f'{count}: {label_crop.shape}\t'
                #       f'x_start: {x_start}\tx_end: {x_end}\ty_start: {y_start}\ty_end: {y_end}')
                # with rasterio.open(f'../data/preprocess/masks/label_{count}.png', 'w', driver='PNG',
                #                    width=CROP_SIZE, height=CROP_SIZE, count=label_tif.count,
                #                    dtype=label_tif.dtypes[0], nodata=0, transform=label_tif.transform,
                #                    crs=label_tif.crs) as dst:
                #     dst.write(label_crop)
                label_crop = label_crop.astype(np.uint8)
                label_save = Image.fromarray(label_crop, 'P')
                label_save.putpalette(bin_colormap)
                label_save.save(f'../data/preprocess/masks/label_{count}.png')
                label_save.close()
    print(f'label crop done.')

    print(f'{count - count_start}张图片')

    # 制作S2原始图像数据矩阵
    if count - count_start:
        all_bands = None
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
        print(f'original s2 done.')

        # 裁剪rgb图像和S2原始波段
        with Image.open(png_path) as png_img:
            png_data = np.array(png_img)
            for ct in range(len(crop_pos)):
                [x_start, x_end, y_start, y_end] = crop_pos[ct]
                new_img = png_data[x_start:x_end, y_start:y_end, :]
                # with rasterio.open(f'../data/preprocess/imgs/img_{ct + count_start + 1}.png', 'w', driver='PNG',
                #                    width=CROP_SIZE, height=CROP_SIZE, count=png_img.count,
                #                    dtype=png_img.dtypes[0], nodata=0, transform=png_img.transform,
                #                    crs=png_img.crs) as dst:
                #     dst.write(new_img)
                if new_img.shape[0] != CROP_SIZE or new_img.shape[1] != CROP_SIZE:
                    new_img = np.pad(new_img, ((0, CROP_SIZE - new_img.shape[0]),
                                               (0, CROP_SIZE - new_img.shape[1]), (0, 0)),
                                     'constant', constant_values=0)
                new_img = new_img.astype(np.uint8)
                new_img_save = Image.fromarray(new_img, 'RGBA')
                new_img_save.save(f'../data/preprocess/imgs/img_{ct + count_start + 1}.png')
                new_img_save.close()
                bands_crop = all_bands[:, x_start:x_end, y_start:y_end]
                x_len = bands_crop.shape[1]
                y_len = bands_crop.shape[2]
                if x_len != CROP_SIZE or y_len != CROP_SIZE:
                    print(f'第{ct + count_start + 1}个s2需要填充')
                    bands_crop = np.pad(bands_crop, ((0, 0), (0, CROP_SIZE - x_len), (0, CROP_SIZE - y_len)),
                                        'constant', constant_values=0)
                np.save(f'../data/preprocess/s2/s2_{ct + count_start + 1}.npy', bands_crop)
        count_start = count

    print(f'png and s2 data crop done.')

    print(f'{s2_name} end')



