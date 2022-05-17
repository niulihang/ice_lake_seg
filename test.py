# 测试文件
import glob

import global_var
import numpy as np
from PIL import Image


# def translate(band, number=2):
#     """
#     Convert the value of a band to the range 0-255
#     2% linear conversion
#     """
#     # band_data = band.ReadAsArray(0, 0, cols, rows)
#     # min = np.min(band)
#     max_val = np.max(band)
#     min_val = np.min(band)
#     # nodata = band.GetNoDataValue()
#     band = band.astype(np.float64)
#     band[band == -9999] = np.nan
#     band[band == 0] = np.nan
#
#     band_data = band / max_val * 255
#     # Convert nan in the data to a specific value, for example
#     band_data[np.isnan(band_data)] = 0
#     d2 = np.percentile(band_data, number)
#     u98 = np.percentile(band_data, 100 - number)
#
#     maxout = 255
#     minout = 0
#
#     data_8bit_new = minout + ((band_data - d2) / (u98 - d2)) * (maxout - minout)
#     data_8bit_new[data_8bit_new < minout] = minout
#     data_8bit_new[data_8bit_new > maxout] = maxout
#     return data_8bit_new
#
#
# s2_file = prep_s2_file = global_var.PREP_PATH + r'\s2\s2_1.npy'
# s2_data = np.load(s2_file)
# rgb = s2_data[1:4].copy()



# ar = np.arange(1, 17).reshape((4, 1, 4))
# print(ar)
# rgb = ar[1:4].copy()
# print(rgb)
# rgb[[0, 1, 2]] = rgb[[2, 1, 0]]
# print(rgb)
# print(ar)

s2_list = glob.glob(r'D:\data\s-2\model_datasets\original\S2*_MSIL1C_***.SAFE')
for s2 in s2_list:
    jp2_list = glob.glob(s2 + r'\GRANULE\L1C_*\IMG_DATA\*_B*.jp2')
    print()

