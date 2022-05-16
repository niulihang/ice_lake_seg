import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import rasterio

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

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


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))
 

if __name__ == '__main__':
    # args = get_args()
    model = 'checkpoints/checkpoint_epoch200.pth'
    # input_img = ['data/train/s2/s2_13.npy']
    # output_img = ['label_pred_13.png']
    input_img = glob.glob('data/test/s2/s2_*.npy')
    output_img = ['data/test/predict']
    no_save = False
    scale = 1.0
    n_channels = 13
    n_classes = 2
    mask_threshold = 0.5
    viz = False

    in_files = input_img
    out_files = output_img

    net = UNet(n_channels=n_channels, n_classes=n_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(model, map_location=device))

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'\nPredicting image {filename} ...')
        # img = Image.open(filename)
        index = os.path.basename(filename).split('_')[1].split('.')[0]
        img = np.load(filename)
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=scale,
                           out_threshold=mask_threshold,
                           device=device)

        if not no_save:
            out_filename = out_files[0] + f'/predict_{index}.png'
            # result = mask_to_image(mask)
            # result.save(out_filename)
            result = np.argmax(mask, axis=0)
            result = result[np.newaxis, ...]
            with rasterio.open(out_filename, 'w', driver='PNG',
                               width=result.shape[1], height=result.shape[2], count=result.shape[0],
                               dtype='uint8', nodata=0) as dst:
                dst.write(result.astype(np.uint8))
            logging.info(f'Mask saved to {out_filename}')

        if viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
