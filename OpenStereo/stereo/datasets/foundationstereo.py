# @Time    : 2025/4/30 11:28
# @Author  : zhangchenming
import os
import numpy as np
import cv2
import glob
import imageio
import torch.utils.data as torch_data
from pathlib import Path
from .dataset_template import build_transform_by_cfg


def depth_uint8_decoding(depth_uint8, scale=1000):
    depth_uint8 = depth_uint8.astype(float)
    out = depth_uint8[...,0]*255*255 + depth_uint8[..., 1]*255 + depth_uint8[..., 2]
    return out/float(scale)


class FoundationStereoDataset(torch_data.Dataset):
    def __init__(self, data_info, data_cfg, mode):
        super().__init__()
        self.data_info = data_info
        self.data_cfg = data_cfg
        self.mode = mode
        self.root = self.data_info.DATA_PATH

        self.data_list = []
        if self.mode.upper() in self.data_info.DATA_SPLIT:
            transform_config = self.data_cfg.DATA_TRANSFORM[self.mode.upper()]
            self.transform = build_transform_by_cfg(transform_config)
            data_dirs = glob.glob(os.path.join(self.root, '*/dataset/data/'))
            for each_data_dir in data_dirs:
                left_images = glob.glob(os.path.join(each_data_dir, 'left/rgb/*.jpg'))
                self._append_sample(left_images)
        else:
            self.transform = None

        if hasattr(self.data_info, 'RETURN_SUPER_PIXEL'):
            self.retrun_super_pixel = self.data_info.RETURN_SUPER_PIXEL
        else:
            self.retrun_super_pixel = False

    def _append_sample(self, left_images):
        for each in left_images:
            data = {'left': each,
                    'right': each.replace('left', 'right'),
                    'disp': each.replace('rgb', 'disparity').replace('.jpg', '.png')}
            self.data_list.append(data)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        left_img = imageio.imread(item['left'])
        right_img = imageio.imread(item['right'])
        disp = depth_uint8_decoding(imageio.imread(item['disp']))
        occ_mask = np.zeros_like(disp, dtype=bool)

        sample = {
            'left': left_img.astype(np.float32),
            'right': right_img.astype(np.float32),
            'disp': disp.astype(np.float32),
            'occ_mask': occ_mask
        }

        if self.retrun_super_pixel:
            save_name = Path(item['left']).relative_to(self.root)
            super_pixel_label = Path(self.root).parent.joinpath('SuperPixelLabel/FoundationStereoDataset', save_name)
            super_pixel_label = str(super_pixel_label)[:-len('.png')] + "_lsc_lbl.png"
            if not os.path.exists(os.path.dirname(super_pixel_label)):
                os.makedirs(os.path.dirname(super_pixel_label), exist_ok=True)
            if not os.path.exists(super_pixel_label):
                img = cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR)
                lsc = cv2.ximgproc.createSuperpixelLSC(img, region_size=10, ratio=0.075)
                lsc.iterate(20)
                label = lsc.getLabels()
                cv2.imwrite(super_pixel_label, label.astype(np.uint16))
            super_pixel_label = cv2.imread(super_pixel_label, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if super_pixel_label is None:
                img = cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR)
                lsc = cv2.ximgproc.createSuperpixelLSC(img, region_size=10, ratio=0.075)
                lsc.iterate(20)
                label = lsc.getLabels()
                super_pixel_label = label.astype(np.int32)
            else:
                super_pixel_label = super_pixel_label.astype(np.int32)
            sample['super_pixel_label'] = super_pixel_label

        if self.transform is not None:
            sample = self.transform(sample)

        sample['valid'] = sample['disp'] < 512
        sample['index'] = idx
        sample['name'] = item['left']

        return sample

    def __len__(self):
        return len(self.data_list)
