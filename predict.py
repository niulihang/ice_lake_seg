import argparse
import logging
import os

import gdal
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import rasterio

from utils.data_loading import BasicDataset
from model.unet import UNet
from utils.utils import plot_img_and_mask
from model.mod_att_unet import ModAttUnet
import global_var

import glob


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        # tf = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize((full_img.size[1], full_img.size[0])),
        #     transforms.ToTensor()
        # ])
        #
        # full_mask = tf(probs.cpu()).squeeze()
        full_mask = probs.cpu().squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()
 

if __name__ == '__main__':
    net = ModAttUnet(n_channels=13, n_classes=2)
    net_path = 'checkpoints/modattunet_epoch102_0519.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    net.load_state_dict(torch.load(net_path, map_location=device))

    input_imgs = glob.glob(global_var.TEST_S2_PATH + r'\*.npy')
    print(f'{len(input_imgs)} s2 imgs')
    output_imgs = global_var.TEST_PATH + r'\predict_masks'
    if not os.path.exists(output_imgs):
        os.makedirs(output_imgs)

    for s2_path in input_imgs:
        print(f'predict {s2_path}')
        s2_np = np.load(s2_path)
        mask_index = os.path.basename(s2_path).split('.')[0].split('_')[1]
        save_path = output_imgs + rf'\label_{mask_index}.tif'
        print(f'save to {save_path}')

        mask_pred = predict_img(net=net,
                                full_img=s2_np,
                                scale_factor=1.0,
                                out_threshold=0.5,
                                device=device)
        mask_pred = np.argmax(mask_pred, axis=0)
        print(f'{mask_pred.shape}')

        driver = gdal.GetDriverByName('GTIFF')
        img_out = driver.Create(save_path, mask_pred.shape[0], mask_pred.shape[1],
                                1, gdal.GDT_Float32)
        img_out.GetRasterBand(1).WriteArray(mask_pred)
        img_out.FlushCache()
        img_out = None
