"""
cd /home/mines/Documents/oldcast1e/MinesLab/IGEV-plusplus
PYTHONPATH=. python src/demo_imgs_test.py

# 특정 폴더(예: tests)만 테스트하고 싶을 경우
PYTHONPATH=. python src/demo_imgs_test.py --input_dir ./asset/imgs/tests
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
    img_torch = torch.from_numpy(img).permute(2, 0, 1).float()
    return img_torch[None].to(DEVICE), img

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

def visualize_depth(depth, original_image, min_depth=0.2, max_depth=10):
    """
    Depth map을 시각화용 컬러 이미지로 변환하고, 원본 이미지의 검은 배경을 마스킹합니다.
    """
    # 1. 마스크 생성: 원본 이미지에서 검은색 영역을 찾습니다.
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    mask = (gray == 0)

    # 2. 동적 범위 결정: 통계적으로 강건한 최소/최대 깊이 값을 찾습니다.
    valid_depth = depth[~mask & (depth > 0.1)]
    if len(valid_depth) == 0:
        # 유효한 깊이 값이 없으면 검은 이미지 반환
        return np.zeros_like(original_image)

    # 퍼센타일을 사용하여 이상치(outlier)에 덜 민감하게 만듭니다.
    min_val = np.percentile(valid_depth, 5)
    max_val = np.percentile(valid_depth, 95)

    # 3. 정규화 및 색상 변환
    depth_clipped = np.clip(depth, min_val, max_val)
    depth_normalized = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_inverted = 255 - depth_normalized
    depth_colored = cv2.applyColorMap(depth_inverted, cv2.COLORMAP_JET)

    # 4. 마스킹 적용: 원본의 검은 배경을 최종 이미지에도 적용합니다.
    depth_colored[mask] = 0

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
            image1_torch, image1_cv = load_image(imfile1)
            image2_torch, _ = load_image(imfile2)

            padder = InputPadder(image1_torch.shape, divis_by=32)
            image1_torch, image2_torch = padder.pad(image1_torch, image2_torch)

            disp_pred = model(image1_torch, image2_torch, iters=args.valid_iters, test_mode=True)
            disp_pred = padder.unpad(disp_pred).cpu().numpy().squeeze()
            
            # 패딩된 크기에 맞춰 원본 이미지도 리사이즈
            # --- 수정된 부분 ---
            # 'pad_dims'를 '_pad'로 변경하여 오류를 해결합니다.
            h_pad, w_pad = padder._pad[2], padder._pad[0]
            # --- 수정 끝 ---
            image1_cv_padded = cv2.copyMakeBorder(image1_cv, h_pad, 0, w_pad, 0, cv2.BORDER_CONSTANT, value=[0,0,0])

            file_stem = Path(imfile1).parent.name
            
            # 2. Depth map 계산
            depth_map = disp_to_depth(disp_pred.copy(), args.baseline, args.focal_length)
            
            # 2b. Depth map 시각화 이미지 저장 (.png)
            depth_viz = visualize_depth(depth_map, image1_cv_padded, args.min_depth, args.max_depth)
            
            # 패딩 제거
            depth_viz_unpadded = padder.unpad(torch.from_numpy(depth_viz).permute(2,0,1).unsqueeze(0)).squeeze().permute(1,2,0).numpy()

            depth_viz_filename = output_directory / f"{file_stem}_depth.png"
            cv2.imwrite(str(depth_viz_filename), depth_viz_unpadded)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # 경로 설정
    parser.add_argument('--input_dir', help="입력 스테레오 쌍 경로", default="./asset/real_imgs/rect")
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
