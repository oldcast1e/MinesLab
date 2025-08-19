"""
# asset/real_imgs/rect 폴더의 모든 이미지를 처리
PYTHONPATH=. python src/demo_imgs_test_baseline_and_Distance.py

# 특정 폴더만 지정하여 처리하고 싶을 경우
PYTHONPATH=. python src/demo_imgs_test_baseline_and_Distance.py --input_dir ./asset/real_imgs/rect/real_world_1
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
    # OpenCV는 BGR 순서를 사용하므로 변환
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_torch = torch.from_numpy(img).permute(2, 0, 1).float()
    return img_torch[None].to(DEVICE), img_bgr

def disp_to_depth(disp, baseline_m, fx_pixels):
    """
    Disparity map을 Depth map으로 변환합니다.
    """
    disp[disp <= 0] = 0.1
    depth = (baseline_m * fx_pixels) / disp
    return depth

def analyze_and_visualize_depth(depth, original_image):
    """
    Depth map을 분석하고 시각화합니다.
    """
    # 1. 마스크 생성
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    mask = (gray < 10)

    # 2. 유효 깊이 데이터 추출
    valid_depth = depth[~mask & (depth > 0.1)]
    if len(valid_depth) == 0:
        return np.zeros_like(original_image)

    # 3. 최소/최대 깊이 값 및 위치 찾기
    min_depth_val = np.min(valid_depth)
    max_depth_val = np.max(valid_depth)

    # 임시 깊이 맵을 만들어 최소/최대 값의 좌표를 찾음
    temp_depth_min = depth.copy()
    temp_depth_min[mask] = np.inf
    min_loc_yx = np.unravel_index(np.argmin(temp_depth_min), depth.shape)
    
    temp_depth_max = depth.copy()
    temp_depth_max[mask] = -np.inf
    max_loc_yx = np.unravel_index(np.argmax(temp_depth_max), depth.shape)

    # (y, x) -> (x, y) 순서로 변환
    min_loc_xy = (min_loc_yx[1], min_loc_yx[0])
    max_loc_xy = (max_loc_yx[1], max_loc_yx[0])

    # 4. 동적 범위 결정 및 시각화
    min_val_norm = np.percentile(valid_depth, 2)
    max_val_norm = np.percentile(valid_depth, 95)
    
    depth_clipped = np.clip(depth, min_val_norm, max_val_norm)
    depth_normalized = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_inverted = 255 - depth_normalized
    depth_colored = cv2.applyColorMap(depth_inverted, cv2.COLORMAP_JET)

    # 5. 마스킹 적용
    depth_colored[mask] = 0

    # 6. 결과 이미지에 포인트 및 텍스트 추가
    # 최소 지점 (가장 가까움) - 빨간색 원
    cv2.circle(depth_colored, min_loc_xy, 12, (0, 0, 255), -1)
    cv2.circle(depth_colored, min_loc_xy, 12, (255, 255, 255), 2)

    # 최대 지점 (가장 멈) - 파란색 원
    cv2.circle(depth_colored, max_loc_xy, 12, (255, 0, 0), -1)
    cv2.circle(depth_colored, max_loc_xy, 12, (255, 255, 255), 2)

    # 거리 정보 텍스트
    text = f"Min: {min_depth_val:.2f}m, Max: {max_depth_val:.2f}m"
    cv2.putText(depth_colored, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

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
            image1_torch_padded, image2_torch_padded = padder.pad(image1_torch, image2_torch)

            disp_pred_padded = model(image1_torch_padded, image2_torch_padded, iters=args.valid_iters, test_mode=True)
            disp_pred_unpadded = padder.unpad(disp_pred_padded).cpu().numpy().squeeze()
            
            depth_map = disp_to_depth(disp_pred_unpadded.copy(), args.baseline, args.focal_length)
            depth_viz = analyze_and_visualize_depth(depth_map, image1_cv)
            
            file_stem = Path(imfile1).parent.name
            depth_viz_filename = output_directory / f"{file_stem}_depth.png"
            cv2.imwrite(str(depth_viz_filename), depth_viz)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # 경로 설정
    parser.add_argument('--input_dir', help="입력 스테레오 쌍 경로", default="./asset/tests")
    parser.add_argument('--output_dir', help="출력 저장 경로", default="./asset/output")
    parser.add_argument('--restore_ckpt', help="체크포인트 복원", default="./pretrained_models/igev_plusplus/sceneflow.pth")

    # 깊이 계산 파라미터
    parser.add_argument('--baseline', type=float, help="카메라 베이스라인 (미터 단위)", default=0.11153) # 111.53mm
    parser.add_argument('--focal_length', type=float, help="카메라 초점거리 (픽셀 단위)", default=1758.23)
    
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
