"""
cd /home/mines/Documents/oldcast1e/MinesLab/IGEV-plusplus

PYTHONPATH=. python src/demo_imgs_test.py \
  --input_dir ./asset/img \
  --output_dir ./asset/output \
  --restore_ckpt ./asset/calib/pretrained_models/igev_plusplus/sceneflow.pth

"""

import sys
sys.path.append('core')

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
import os


DEVICE = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)[..., :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def demo(args):
    # 모델 불러오기
    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))
    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_dir)
    output_directory.mkdir(exist_ok=True, parents=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(os.path.join(args.input_dir, "*/im0.png")))
        right_images = sorted(glob.glob(os.path.join(args.input_dir, "*/im1.png")))
        print(f"Found {len(left_images)} image pairs. Saving files to {output_directory}/")

        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            disp = model(image1, image2, iters=args.valid_iters, test_mode=True)
            disp = padder.unpad(disp)

            # 출력 파일명은 폴더 이름 기준
            file_stem = Path(imfile1).parent.name
            filename = os.path.join(output_directory, f'{file_stem}.png')

            disp = disp.cpu().numpy().squeeze()
            plt.imsave(filename, disp, cmap='jet')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help="path to input stereo pairs", default="./asset/img")
    parser.add_argument('--output_dir', help="directory to save output", default="./asset/output")
    parser.add_argument('--restore_ckpt', help="restore checkpoint", 
                        default="./asset/calib/pretrained_models/igev_plusplus/sceneflow.pth")

    # 모델 설정
    parser.add_argument('--mixed_precision', action='store_true', default=True, help='use mixed precision')
    parser.add_argument('--precision_dtype', default='float32',
                        choices=['float16', 'bfloat16', 'float32'], help='precision type')
    parser.add_argument('--valid_iters', type=int, default=16, help='number of iterations')

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
