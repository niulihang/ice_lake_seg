import logging

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import glob
import os

import global_var


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [self.get_index(f_path) for f_path in glob.glob(f'{self.images_dir}/s2_*.npy')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def get_index(cls, f_path):
        f_name = os.path.basename(f_path).split('_')[1].split('.')[0]
        return f_name

    @classmethod
    def preprocess(cls, img, scale, is_mask):
        if not is_mask:
            img = img / 10000

        return img

    @classmethod
    def load(cls, filename):
        file_type = os.path.basename(filename).split('_')[0]
        if file_type == 's2':
            return np.load(filename).astype(np.float32)
        elif file_type == 'label':
            return np.expand_dims(np.array(Image.open(filename)), axis=0)
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = rf'{self.masks_dir}\label_{name}.png'
        img_file = rf'{self.images_dir}\s2_{name}.npy'

        mask = self.load(mask_file)
        img = self.load(img_file)

        if len(mask.shape) == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(axis=0)

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'name': name,
        }


class S2Dataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')


if __name__ == '__main__':
    img_dir = global_var.TRAIN_S2_PATH
    masks_dir = global_var.TRAIN_MASKS_PATH
    datasets = S2Dataset(img_dir, masks_dir)
    read_0 = datasets.__getitem__(0)
    print()
