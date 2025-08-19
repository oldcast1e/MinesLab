"""
실행 방법:
conda activate IGEV_plusplus
cd ~/Documents/oldcast1e/MinesLab/IGEV-plusplus
python src/demo_imgs_test.py
"""

import sys
import os
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from core.igev_stereo import IGEVStereo
from core.utils.utils import InputPadder
from PIL import Image
import matplotlib.pyplot as plt

DEVICE = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ==========================
# 경로 설정 (asset 하위)
# ==========================
PROJECT_ROOT = "/home/mines/Documents/oldcast1e/MinesLab/IGEV-plusplus"
ASSET_DIR   = os.path.join(PROJECT_ROOT, "asset")

CALIB_DIR   = os.path.join(ASSET_DIR, "calib_out")
RAW_DIR     = os.path.join(ASSET_DIR, "test_imgs", "raw")
OUTPUT_DIR  = os.path.join(ASSET_DIR, "test_imgs", "output")
CKPT_PATH   = os.path.join(PROJECT_ROOT, "pretrained_models", "igev_plusplus", "sceneflow.pth")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================
# 이미지 로드 함수
# ==========================
def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)[..., :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


# ==========================
# 모델 실행
# ==========================
def run_demo():
    model = torch.nn.DataParallel(IGEVStereo({}), device_ids=[0])
    model.load_state_dict(torch.load(CKPT_PATH))

    model = model.module
    model.to(DEVICE)
    model.eval()

    left_images = sorted(glob.glob(os.path.join(RAW_DIR, "left", "*.png")))
    right_images = sorted(glob.glob(os.path.join(RAW_DIR, "right", "*.png")))

    print(f"Found {len(left_images)} test images. Saving results to {OUTPUT_DIR}/")

    with torch.no_grad():
        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            disp = model(image1, image2, iters=16, test_mode=True)
            disp = padder.unpad(disp)
            disp = disp.cpu().numpy().squeeze()

            # 파일명: 원래 이미지 이름 기반으로 저장
            stem = os.path.splitext(os.path.basename(imfile1))[0]  # ex) 002
            out_path = os.path.join(OUTPUT_DIR, f"{stem}.png")
            npy_path = os.path.join(OUTPUT_DIR, f"{stem}.npy")

            plt.imsave(out_path, disp, cmap='jet')
            np.save(npy_path, disp)

            print(f"[SAVED] {out_path}, {npy_path}")


if __name__ == "__main__":
    run_demo()
