"""
실행 방법:
conda activate IGEV_plusplus
cd ~/Documents/oldcast1e/MinesLab/IGEV-plusplus
python src/run_test_imgs.py
"""
import os, json, glob, subprocess
import numpy as np
import cv2

PROJECT_ROOT = "/home/mines/Documents/oldcast1e/MinesLab/IGEV-plusplus"

ASSET_DIR   = os.path.join(PROJECT_ROOT, "asset")

CALIB_DIR   = os.path.join(ASSET_DIR, "calib_out")
RAW_DIR     = os.path.join(ASSET_DIR, "test_imgs", "raw")
OUTPUT_DIR  = os.path.join(ASSET_DIR, "test_imgs", "output")

RECT_DIR    = os.path.join(PROJECT_ROOT, "test_imgs", "rect")
CKPT_PATH   = os.path.join(PROJECT_ROOT, "pretrained_models", "igev_plusplus", "sceneflow.pth")

IMG_ID = "001"

LEFT_IMG  = os.path.join(RAW_DIR, "left",  f"{IMG_ID}.png")
RIGHT_IMG = os.path.join(RAW_DIR, "right", f"{IMG_ID}.png")
RECT_L    = os.path.join(RECT_DIR, "left",  f"{IMG_ID}.png")
RECT_R    = os.path.join(RECT_DIR, "right", f"{IMG_ID}.png")

CALIB_INFO = os.path.join(CALIB_DIR, "info.json")

EPS = 1e-6
BASELINE_M = 0.11153

os.makedirs(os.path.dirname(RECT_L), exist_ok=True)
os.makedirs(os.path.dirname(RECT_R), exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1) Rectify
mapLx = np.load(os.path.join(CALIB_DIR, "mapLx.npy"))
mapLy = np.load(os.path.join(CALIB_DIR, "mapLy.npy"))
mapRx = np.load(os.path.join(CALIB_DIR, "mapRx.npy"))
mapRy = np.load(os.path.join(CALIB_DIR, "mapRy.npy"))

L = cv2.imread(LEFT_IMG)
R = cv2.imread(RIGHT_IMG)
rectL = cv2.remap(L, mapLx, mapLy, cv2.INTER_LINEAR)
rectR = cv2.remap(R, mapRx, mapRy, cv2.INTER_LINEAR)
cv2.imwrite(RECT_L, rectL)
cv2.imwrite(RECT_R, rectR)
print(f"[OK] Rectified → {RECT_L}, {RECT_R}")

# 2) Run demo_imgs.py
subprocess.run([
    "python", "demo_imgs.py",
    "--restore_ckpt", CKPT_PATH,
    "--left_imgs", RECT_L,
    "--right_imgs", RECT_R,
    "--output_directory", OUTPUT_DIR,
    "--save_numpy"
])
print(f"[OK] Disparity estimation complete → {OUTPUT_DIR}")

# 3) Rename demo_imgs.py 결과물을 IMG_ID 기반으로 통일
#    (PNG = jet colormap, NPY = raw disparity)
for ext in [".png", ".npy"]:
    candidates = glob.glob(os.path.join(OUTPUT_DIR, f"*{ext}"))
    for old_file in candidates:
        if not old_file.endswith(f"{IMG_ID}{ext}"):
            new_file = os.path.join(OUTPUT_DIR, f"{IMG_ID}{ext}")
            os.replace(old_file, new_file)
            print(f"[RENAMED] {old_file} → {new_file}")

# 4) 깊이맵 변환 (from disparity)
with open(CALIB_INFO, "r") as f:
    info = json.load(f)
fx_calib = float(info["fx_pixel"])
fx_runtime = fx_calib

DISP_NPY = os.path.join(OUTPUT_DIR, f"{IMG_ID}.npy")
disp = np.load(DISP_NPY).astype(np.float32)
disp = np.maximum(disp, EPS)

depth_m = fx_runtime * BASELINE_M / disp

# Depth(mm) 저장
depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
cv2.imwrite(os.path.join(OUTPUT_DIR, f"{IMG_ID}_depth_mm.png"), depth_mm)

# Depth 시각화 저장
vis = np.clip(depth_m, 0, np.percentile(depth_m, 99))
vis = (vis / (vis.max() + EPS) * 255.0).astype(np.uint8)
if vis.ndim == 3:
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
vis_color = cv2.applyColorMap(255 - vis, cv2.COLORMAP_MAGMA)
cv2.imwrite(os.path.join(OUTPUT_DIR, f"{IMG_ID}_depth_vis.png"), vis_color)

print(f"[OK] Depth maps saved → {OUTPUT_DIR}")
