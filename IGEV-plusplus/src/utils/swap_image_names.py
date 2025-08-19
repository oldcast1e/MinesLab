import os
import glob

def swap_image_filenames_in_subdirs(target_parent_dir):
    """
    지정된 부모 디렉토리의 모든 하위 디렉토리를 순회하며
    'im0.png'와 'im1.png' 파일의 이름을 서로 바꿉니다.
    """
    
    # target_parent_dir이 존재하는지 확인
    if not os.path.isdir(target_parent_dir):
        print(f"오류: '{target_parent_dir}' 디렉토리를 찾을 수 없습니다.")
        return

    # 하위 디렉토리 목록을 가져옴
    subdirectories = [d for d in os.listdir(target_parent_dir) if os.path.isdir(os.path.join(target_parent_dir, d))]

    if not subdirectories:
        print(f"'{target_parent_dir}' 내에 처리할 하위 디렉토리가 없습니다.")
        return

    print(f"'{target_parent_dir}' 경로의 하위 폴더들을 스캔합니다...")
    
    swapped_count = 0
    for subdir_name in subdirectories:
        current_dir = os.path.join(target_parent_dir, subdir_name)
        
        im0_path = os.path.join(current_dir, "im0.png")
        im1_path = os.path.join(current_dir, "im1.png")
        
        # 두 파일이 모두 존재하는지 확인
        if os.path.exists(im0_path) and os.path.exists(im1_path):
            # 임시 파일 이름을 사용하여 이름 충돌 방지
            temp_path = os.path.join(current_dir, "temp_swap_image.png")
            
            # 이름 변경 프로세스
            os.rename(im0_path, temp_path)      # im0.png -> temp
            os.rename(im1_path, im0_path)      # im1.png -> im0.png
            os.rename(temp_path, im1_path)      # temp    -> im1.png
            
            print(f"  - [{subdir_name}]: im0.png <-> im1.png 이름 교체 완료.")
            swapped_count += 1
        else:
            print(f"  - [{subdir_name}]: im0.png와 im1.png 쌍이 모두 존재하지 않아 건너뜁니다.")

    print(f"\n총 {swapped_count}개 폴더의 파일 이름을 성공적으로 교체했습니다.")


if __name__ == "__main__":
    # 프로젝트 최상위 경로를 기준으로 실행한다고 가정
    TARGET_DIRECTORY = os.path.join(os.getcwd(), "asset", "real_imgs", "raw")
    swap_image_filenames_in_subdirs(TARGET_DIRECTORY)
