import cv2
import numpy as np
import os
import glob
from tqdm import tqdm

def rectify_image_pairs():
    """
    'asset/real_imgs/raw' 내의 각 장면 폴더에서 원본 이미지들을 찾아
    'asset/calib_out'의 캘리브레이션 데이터를 이용해 정렬하고,
    'asset/real_imgs/rect'에 동일한 구조로 저장합니다.
    """
    # --- 경로 설정 ---
    PROJECT_ROOT = os.getcwd()
    CALIB_DIR = os.path.join(PROJECT_ROOT, "asset", "calib_out")
    RAW_DIR = os.path.join(PROJECT_ROOT, "asset", "real_imgs", "raw")
    RECT_DIR = os.path.join(PROJECT_ROOT, "asset", "real_imgs", "rect")

    # --- 출력 폴더 생성 ---
    os.makedirs(RECT_DIR, exist_ok=True)

    # --- 캘리브레이션 맵 로드 ---
    try:
        print(f"캘리브레이션 데이터를 '{CALIB_DIR}'에서 로드합니다.")
        mapLx = np.load(os.path.join(CALIB_DIR, "mapLx.npy"))
        mapLy = np.load(os.path.join(CALIB_DIR, "mapLy.npy"))
        mapRx = np.load(os.path.join(CALIB_DIR, "mapRx.npy"))
        mapRy = np.load(os.path.join(CALIB_DIR, "mapRy.npy"))
    except FileNotFoundError:
        print(f"오류: '{CALIB_DIR}'에서 캘리브레이션 파일(.npy)을 찾을 수 없습니다.")
        print("먼저 'stereo_calibrate.py'를 실행하여 캘리브레이션을 완료해야 합니다.")
        return

    # --- 원본 이미지가 있는 장면 폴더 목록 가져오기 ---
    scene_dirs = sorted([d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))])

    if not scene_dirs:
        print(f"오류: '{RAW_DIR}'에서 처리할 장면 폴더를 찾을 수 없습니다.")
        return

    print(f"총 {len(scene_dirs)}개의 장면 폴더를 정렬합니다...")

    # --- 이미지 정렬 및 저장 ---
    for scene_name in tqdm(scene_dirs, desc="Rectifying Scenes"):
        scene_path = os.path.join(RAW_DIR, scene_name)
        left_path = os.path.join(scene_path, "im0.png")
        right_path = os.path.join(scene_path, "im1.png")

        # 각 장면 폴더에 im0.png와 im1.png가 모두 있는지 확인
        if not (os.path.exists(left_path) and os.path.exists(right_path)):
            print(f"  - [{scene_name}] im0.png/im1.png 쌍을 찾을 수 없어 건너뜁니다.")
            continue
            
        # 이미지 로드
        img_l = cv2.imread(left_path)
        img_r = cv2.imread(right_path)

        # 이미지 정렬 (Rectification)
        rect_l = cv2.remap(img_l, mapLx, mapLy, cv2.INTER_LINEAR)
        rect_r = cv2.remap(img_r, mapRx, mapRy, cv2.INTER_LINEAR)

        # 정렬된 이미지를 저장할 출력 폴더 생성
        rect_scene_dir = os.path.join(RECT_DIR, scene_name)
        os.makedirs(rect_scene_dir, exist_ok=True)

        # 정렬된 이미지 저장
        cv2.imwrite(os.path.join(rect_scene_dir, "im0.png"), rect_l)
        cv2.imwrite(os.path.join(rect_scene_dir, "im1.png"), rect_r)

    print(f"\n이미지 정렬 완료. 결과가 '{RECT_DIR}' 폴더에 저장되었습니다.")


if __name__ == "__main__":
    rectify_image_pairs()
