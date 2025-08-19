"""
# asset/real_imgs/rect 폴더의 모든 이미지를 처리
PYTHONPATH=. python src/demo_imgs_test_baseline.py

# 특정 폴더만 지정하여 처리하고 싶을 경우
PYTHONPATH=. python src/demo_imgs_test_baseline.py --input_dir ./asset/real_imgs/rect/real_world_1
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
import cv2

DEVICE = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)[..., :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def disp_to_depth(disp, baseline_m, fx_pixels):
    """
    Disparity map을 Depth map으로 변환합니다.
    :param disp: (H, W) 형태의 disparity map (numpy array)
    :param baseline_m: 미터 단위의 카메라 베이스라인
    :param fx_pixels: 픽셀 단위의 카메라 초점 거리
    :return: (H, W) 형태의 depth map (numpy array, 미터 단위)
    """
    # 0 또는 음수 disparity 값으로 인한 나누기 오류 방지
    disp[disp <= 0] = 0.1
    
    depth = (baseline_m * fx_pixels) / disp
    return depth

def visualize_depth(depth, min_depth=0.1, max_depth=10):
    """
    Depth map을 시각화용 컬러 이미지로 변환합니다.
    """
    # 지정된 범위 밖의 값들을 클리핑
    depth_clipped = np.clip(depth, min_depth, max_depth)
    # 0-255 범위로 정규화 (가까울수록 작은 값, 멀수록 큰 값)
    depth_normalized = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # 가까운 곳(값이 작음)이 빨간색, 먼 곳(값이 큼)이 파란색이 되도록 값의 범위를 반전시킵니다.
    depth_inverted = 255 - depth_normalized
    
    # 컬러맵 적용
    depth_colored = cv2.applyColorMap(depth_inverted, cv2.COLORMAP_JET)
    return depth_colored


def demo(args):
    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    print(f"Loading checkpoint from {args.restore_ckpt}")
    model.load_state_dict(torch.load(args.restore_ckpt))
    
    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_dir)
    output_directory.mkdir(exist_ok=True, parents=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(os.path.join(args.input_dir, "**/im0.png"), recursive=True))
        right_images = sorted(glob.glob(os.path.join(args.input_dir, "**/im1.png"), recursive=True))
        
        if not left_images:
            print(f"Error: No image pairs found in '{args.input_dir}'.")
            return
            
        print(f"Found {len(left_images)} image pairs in '{args.input_dir}'. Saving files to '{output_directory}/'")

        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            disp_pred = model(image1, image2, iters=args.valid_iters, test_mode=True)
            disp_pred = padder.unpad(disp_pred).cpu().numpy().squeeze()

            file_stem = Path(imfile1).parent.name
            
            # # 1. Disparity map 저장 (시각화용) -> 주석 처리
            # disp_filename = output_directory / f"{file_stem}_disp.png"
            # plt.imsave(disp_filename, disp_pred, cmap='jet')

            # 2. Depth map 계산
            depth_map = disp_to_depth(disp_pred.copy(), args.baseline, args.focal_length)
            
            # # 2a. Depth map 원본 데이터 저장 (.npy) -> 주석 처리
            # depth_npy_filename = output_directory / f"{file_stem}_depth.npy"
            # np.save(depth_npy_filename, depth_map)
            
            # 2b. Depth map 시각화 이미지 저장 (.png)
            depth_viz = visualize_depth(depth_map, args.min_depth, args.max_depth)
            depth_viz_filename = output_directory / f"{file_stem}_depth.png"
            cv2.imwrite(str(depth_viz_filename), depth_viz)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # 경로 설정
    parser.add_argument('--input_dir', help="입력 스테레오 쌍 경로", default="./asset/tests")
    parser.add_argument('--output_dir', help="출력 저장 경로", default="./asset/output")
    parser.add_argument('--restore_ckpt', help="체크포인트 복원", default="./pretrained_models/igev_plusplus/sceneflow.pth")

    # 깊이 계산 파라미터 (Middlebury 2021 데이터셋 기준)
    parser.add_argument('--baseline', type=float, help="카메라 베이스라인 (미터 단위)", default=0.11153) # 111.53mm
    parser.add_argument('--focal_length', type=float, help="카메라 초점거리 (픽셀 단위)", default=1758.23)
    
    # 깊이 시각화 파라미터
    parser.add_argument('--min_depth', type=float, help="시각화 최소 깊이 (미터)", default=0.2)
    parser.add_argument('--max_depth', type=float, help="시각화 최대 깊이 (미터)", default=10.0)

    # 모델 설정 (기존과 동일)
    parser.add_argument('--mixed_precision', action='store_true', default=True)
    parser.add_argument('--precision_dtype', default='float32', choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--valid_iters', type=int, default=16)
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
