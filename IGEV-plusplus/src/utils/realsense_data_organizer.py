import os
import shutil
import glob

# --- 설정 ---
# 프로젝트의 최상위 경로를 기준으로 스크립트를 실행한다고 가정합니다.
PROJECT_ROOT = os.getcwd() 
ASSET_DIR = os.path.join(PROJECT_ROOT, "asset")

# 원본 및 대상 폴더 경로 설정 (raw 폴더를 대상으로 변경)
SOURCE_DIR = os.path.join(ASSET_DIR, "real_imgs", "raw")
TARGET_DIR = os.path.join(ASSET_DIR, "imgs") # 'img'가 'imgs'로 변경되었다고 가정

# --- 스크립트 시작 ---

def organize_real_world_images():
    """
    real_imgs/raw 폴더의 이미지들을 imgs/real_world_X 형식으로 재구성합니다.
    """
    print(f"'{SOURCE_DIR}'의 이미지들을 '{TARGET_DIR}'로 재구성합니다.")

    left_source_dir = os.path.join(SOURCE_DIR, "left")
    right_source_dir = os.path.join(SOURCE_DIR, "right")

    if not os.path.isdir(left_source_dir) or not os.path.isdir(right_source_dir):
        print(f"오류: 원본 폴더 '{left_source_dir}' 또는 '{right_source_dir}'를 찾을 수 없습니다.")
        return

    # 정렬된 이미지 파일 목록 가져오기
    left_images = sorted(glob.glob(os.path.join(left_source_dir, "*.png")))
    right_images = sorted(glob.glob(os.path.join(right_source_dir, "*.png")))

    if len(left_images) != len(right_images):
        print("오류: 좌우 이미지의 개수가 일치하지 않습니다.")
        return
        
    if not left_images:
        print("처리할 이미지가 없습니다.")
        return

    # 대상 폴더가 없으면 생성
    os.makedirs(TARGET_DIR, exist_ok=True)

    # 이미지 재구성
    for i, (left_path, right_path) in enumerate(zip(left_images, right_images)):
        scene_name = f"real_world_{i+1}"
        scene_dir = os.path.join(TARGET_DIR, scene_name)
        
        # 새로운 씬 폴더 생성
        os.makedirs(scene_dir, exist_ok=True)
        
        # 대상 파일 경로 정의 (im0.png, im1.png)
        target_left_path = os.path.join(scene_dir, "im0.png")
        target_right_path = os.path.join(scene_dir, "im1.png")
        
        # 파일 복사
        shutil.copy2(left_path, target_left_path)
        shutil.copy2(right_path, target_right_path)
        
        print(f"  - [{scene_name}] 생성: '{os.path.basename(left_path)}' -> 'im0.png', '{os.path.basename(right_path)}' -> 'im1.png'")

    print(f"\n총 {len(left_images)}개의 이미지 쌍을 성공적으로 재구성했습니다.")
    print(f"이제 원본 '{os.path.join(ASSET_DIR, 'real_imgs')}' 폴더는 필요 시 수동으로 삭제할 수 있습니다.")


if __name__ == "__main__":
    organize_real_world_images()
