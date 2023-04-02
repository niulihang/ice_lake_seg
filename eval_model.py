# 评价模型

import numpy as np
import glob
import os
from PIL import Image
from predict import predict_img
import global_var
from skimage import morphology
import torch
from model.mod_att_unet import ModAttUnet
from model.unet import UNet
from model.fcn import VGG_fcn32s as FCN
from model.segnet import SegNet


def conf_matrix(mask_true, mask_pred, num_class):
    mask = (mask_true >= 0) & (mask_true < num_class)
    hist = np.bincount(
        num_class * mask_true[mask].astype(int) + mask_pred[mask],
        minlength=num_class * num_class
    ).reshape((num_class, num_class))
    return hist


def cal_pix_acc(conf_mat):
    return round(
        np.diagonal(conf_mat).sum() / conf_mat.sum(),
        4)


def cal_water_acc(conf_mat):
    res = round(conf_mat[1, 1] / (conf_mat[1, 0] + conf_mat[1, 1]), 4)
    return res if not np.isnan(res) else 0.0


def cal_non_water_acc(conf_mat):
    res = round(conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1]), 4)
    return res if not np.isnan(res) else 0.0


def cal_water_iou(conf_mat):
    res = round(conf_mat[1, 1] / (conf_mat[0, 1] + conf_mat[1, 0] + conf_mat[1, 1]), 4)
    return res if not np.isnan(res) else 0.0


def cal_non_water_iou(conf_mat):
    res = round(conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1] + conf_mat[1, 0]), 4)
    return res if not np.isnan(res) else 0.0


def cal_miou(conf_mat):
    return round(
        (cal_water_iou(conf_mat) + cal_non_water_iou(conf_mat)) / 2,
        4
    )


def cal_kappa(conf_mat):
    p0 = cal_pix_acc(conf_mat)
    pe = (conf_mat[0, :].sum() * conf_mat[:, 0].sum() +
          conf_mat[1, :].sum() * conf_mat[:, 1].sum()) / (conf_mat.sum() * conf_mat.sum())
    res = round((p0 - pe) / (1 - pe), 4)
    return res if not np.isnan(res) else 0.0


def dndwi(s2_data):
    band2 = blue = np.array([s2_data[1]])
    band3 = green = np.array([s2_data[2]])
    band4 = red = np.array([s2_data[3]])
    band8 = nir = np.array([s2_data[7]])
    band10 = swirCirrus = np.array([s2_data[9]])
    band11 = swir = np.array([s2_data[10]])

    np.seterr(divide='ignore', invalid='ignore')
    ndwi = np.empty(band2.shape, dtype=band2.dtype)
    ndwi = np.where(
        (green + nir) == 0,
        0,
        (green - nir) / (green + nir))

    np.seterr(divide='ignore', invalid='ignore')
    ndsi = np.empty(band2.shape, dtype=band2.dtype)
    ndsi = np.where(
        (green + swir) == 0,
        0,
        (green - swir) / (green + swir))

    rock = np.empty(band2.shape, dtype=band2.dtype)
    rock = np.where((ndsi < 0.9) & (blue > 0) & (blue < 4000) & (green < 4000), 1, 0)
    rock_mask = rock

    cloud = np.empty(band2.shape, dtype=band2.dtype)
    cloud = np.where(((swirCirrus > 30) & (swir > 1100)
                      & (blue > 6000) & (blue < 9700)),
                     1, 0)
    cloud_mask = np.where(rock_mask == 1, 0, cloud)

    ndwiL = 0.16
    lakeL = np.empty(band2.shape, dtype=band2.dtype)
    lakeL = np.where((ndwi > ndwiL) & ((blue - red) / (blue + red) > 0.18) & ((green - red) > 800)
                     & ((green - red) < 4000) & ((blue - green) > 400),
                     1, 0)
    lakeL_mask = np.where((cloud_mask == 1), 0, lakeL)
    lakeL_mask = np.where((rock_mask == 1), 0, lakeL_mask)

    CC_L = morphology.label(lakeL_mask, connectivity=3)
    processingL = morphology.remove_small_objects(CC_L.astype(bool), min_size=2,
                                                  connectivity=3).astype('float32')
    lakeL_mask_cc = np.where(processingL == 0, 0, 1)

    return lakeL_mask_cc


def evaluate_dndwi(dir_img, dir_mask):
    img_path_list = glob.glob(dir_img + r'\*.npy')
    img_num = len(img_path_list)
    pix_acc = 0
    wt_acc = 0
    n_wt_acc = 0
    wt_iou = 0
    n_wt_iou = 0
    miou = 0
    kp = 0
    for img_path in img_path_list:
        image = np.load(img_path)
        index = os.path.basename(img_path).split('_')[1].split('.')[0]
        mask_file = dir_mask + r'\label_' + index + '.png'
        mask_true_img = Image.open(mask_file)
        mask_true = np.asarray(mask_true_img)
        mask_dndwi = dndwi(image)

        mask_dndwi = np.squeeze(mask_dndwi, axis=0)
        conf_mat = conf_matrix(mask_true, mask_dndwi, 2)
        pix_acc += cal_pix_acc(conf_mat)
        wt_acc += cal_water_acc(conf_mat)
        n_wt_acc += cal_non_water_acc(conf_mat)
        wt_iou += cal_water_iou(conf_mat)
        n_wt_iou += cal_non_water_iou(conf_mat)
        miou += cal_miou(conf_mat)
        kp += cal_kappa(conf_mat)

    pix_acc /= img_num
    wt_acc /= img_num
    n_wt_acc /= img_num
    wt_iou /= img_num
    n_wt_iou /= img_num
    miou /= img_num
    kp /= img_num
    print(f'像素准确率：{round(pix_acc, 4)}\n'
          f'水准确率：{round(wt_acc, 4)}\n'
          f'非水准确率：{round(n_wt_acc, 4)}\n'
          f'水IOU：{round(wt_iou, 4)}\n'
          f'非水IOU：{round(n_wt_iou, 4)}\n'
          f'MIOU：{round(miou, 4)}\n'
          f'kappa：{round(kp, 4)}')



def evaluate_models(net, dir_img, dir_mask, device):
    net.eval()
    img_path_list = glob.glob(dir_img + r'\*.npy')
    img_num = len(img_path_list)
    pix_acc = 0
    wt_acc = 0
    n_wt_acc = 0
    wt_iou = 0
    n_wt_iou = 0
    miou = 0
    kp = 0
    for img_path in img_path_list:
        image = np.load(img_path)
        index = os.path.basename(img_path).split('_')[1].split('.')[0]
        mask_file = dir_mask + r'\label_' + index + '.png'
        mask_true_img = Image.open(mask_file)
        mask_true = np.asarray(mask_true_img)
        mask_pred = predict_img(net=net,
                                full_img=image,
                                scale_factor=1.0,
                                out_threshold=0.5,
                                device=device)
        mask_pred = np.argmax(mask_pred, axis=0)
        mask_pred = mask_pred[np.newaxis, ...]

        # 评价过程
        # mask_true = np.squeeze(mask_true, axis=0)
        mask_pred = np.squeeze(mask_pred, axis=0)
        conf_mat = conf_matrix(mask_true, mask_pred, 2)
        pix_acc += cal_pix_acc(conf_mat)
        wt_acc += cal_water_acc(conf_mat)
        n_wt_acc += cal_non_water_acc(conf_mat)
        wt_iou += cal_water_iou(conf_mat)
        n_wt_iou += cal_non_water_iou(conf_mat)
        miou += cal_miou(conf_mat)
        kp += cal_kappa(conf_mat)

    pix_acc /= img_num
    wt_acc /= img_num
    n_wt_acc /= img_num
    wt_iou /= img_num
    n_wt_iou /= img_num
    miou /= img_num
    kp /= img_num
    print(f'像素准确率：{round(pix_acc, 4)}\n'
          f'水准确率：{round(wt_acc, 4)}\n'
          f'非水准确率：{round(n_wt_acc, 4)}\n'
          f'水IOU：{round(wt_iou, 4)}\n'
          f'非水IOU：{round(n_wt_iou, 4)}\n'
          f'MIOU：{round(miou, 4)}\n'
          f'kappa：{round(kp, 4)}')


if __name__ == '__main__':
    dir_img = global_var.TEST_S2_PATH
    dir_mask = global_var.TEST_MASKS_PATH

    for net_path in glob.glob(r'checkpoints\modattunet_*_0519.pth'):
        print(f'load model path: {net_path}')
        net_name = os.path.basename(net_path).split('_')[0]
        if net_name == 'modattunet':
            net = ModAttUnet(n_channels=13, n_classes=2)
        elif net_name == 'unet':
            net = UNet(n_channels=13, n_classes=2)
        elif net_name == 'fcn':
            net = FCN()
        elif net_name == 'segnet':
            net = SegNet()
        else:
            raise Exception('未识别的net_name {}'.format(net_name))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net.to(device=device)
        net.load_state_dict(torch.load(net_path, map_location=device))
        evaluate_models(net, dir_img, dir_mask, device)

    # evaluate_dndwi(dir_img, dir_mask)

