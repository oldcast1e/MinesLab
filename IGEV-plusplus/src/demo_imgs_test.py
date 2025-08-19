# src/demo_imgs_test.py
"""
cd /home/mines/Documents/oldcast1e/MinesLab/IGEV-plusplus
PYTHONPATH=src python src/demo_imgs_test.py --save_numpy
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "core"))

import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from core.igev_stereo import IGEVStereo
from core.utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import cv2

DEVICE = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)[..., :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def demo(args):
    # 모델 로드
    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))
    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            disp = model(image1, image2, iters=args.valid_iters, test_mode=True)
            disp = padder.unpad(disp)

            # 파일명 추출 (000.png → 000)
            file_stem = Path(imfile1).stem

            # disparity map 저장
            disp_np = disp.cpu().numpy().squeeze()
            plt.imsave(output_directory / f"{file_stem}_disp.png", disp_np, cmap='jet')

            # numpy 저장 옵션
            if args.save_numpy:
                np.save(output_directory / f"{file_stem}.npy", disp_np)

            # depth(mm) 저장 (baseline/focal 값은 calib_out에서 가져올 수 있음)
            # 여기서는 단순 disparity visualization 예시
            depth_vis = (disp_np - disp_np.min()) / (disp_np.max() - disp_np.min() + 1e-6)
            cv2.imwrite(str(output_directory / f"{file_stem}_depth_vis.png"),
                        (depth_vis * 255).astype(np.uint8))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint",
                        default='/home/mines/Documents/oldcast1e/MinesLab/IGEV-plusplus/asset/calib/pretrained_models/igev_plusplus/sceneflow.pth')
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames",
                        default="./asset/test_imgs/rect/left/*.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames",
                        default="./asset/test_imgs/rect/right/*.png")
    parser.add_argument('--output_directory', help="directory to save output",
                        default="./asset/test_imgs/output")
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                        help='use mixed precision')
    parser.add_argument('--precision_dtype', default='float32',
                        choices=['float16', 'bfloat16', 'float32'],
                        help='Choose precision type: float16 or bfloat16 or float32')
    parser.add_argument('--valid_iters', type=int, default=16,
                        help='number of flow-field updates during forward pass')

    # Architecture parameters
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3)
    parser.add_argument('--corr_levels', type=int, default=2)
    parser.add_argument('--corr_radius', type=int, default=4)
    parser.add_argument('--n_downsample', type=int, default=2)
    parser.add_argument('--n_gru_layers', type=int, default=3)
    parser.add_argument('--max_disp', type=int, default=768)
    parser.add_argument('--s_disp_range', type=int, default=48)
    parser.add_argument('--m_disp_range', type=int, default=96)
    parser.add_argument('--l_disp_range', type=int, default=192)
    parser.add_argument('--s_disp_interval', type=int, default=1)
    parser.add_argument('--m_disp_interval', type=int, default=2)
    parser.add_argument('--l_disp_interval', type=int, default=4)
    
    args = parser.parse_args()
    demo(args)
