"""
cd /Users/apple/Desktop/Python/MinesLab/IGEV-plusplus
python src/utils/capture_pair_linux_mac.py
"""

import cv2
import os
import platform
import re

# 사용자 설정

# 카메라 장치 인덱스 (macOS: 'ffplay -f avfoundation -list_devices true -i ""' / Linux: 'v4l2-ctl --list-devices' 명령어로 확인)
LEFT_DEV_ID = 0
RIGHT_DEV_ID = 1

# 해상도
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# 카메라가 거꾸로 설치된 경우 True로 설정하여 이미지를 뒤집습니다.
APPLY_FLIP = True
# ====================================================================

def get_next_scene_number(path):
    """
    지정된 경로에서 'real_world_X' 형식의 폴더를 찾아
    다음 저장될 번호를 반환합니다.
    """
    if not os.path.exists(path):
        return 1
        
    dir_list = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    
    max_num = 0
    for dir_name in dir_list:
        match = re.match(r'real_world_(\d+)', dir_name)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
                
    return max_num + 1

def main():
    """
    스테레오 카메라에서 이미지 쌍을 캡처하고 지정된 형식으로 저장합니다.
    """
    # --- 경로 설정 ---
    # 수정: 스크립트 위치 대신 현재 작업 디렉토리를 프로젝트 루트로 사용합니다.
    # 이는 사용자가 항상 프로젝트 최상위 폴더에서 스크립트를 실행하기 때문입니다.
    project_root = os.getcwd()

    save_root = os.path.join(project_root, "asset", "real_imgs", "raw")
    os.makedirs(save_root, exist_ok=True)
    
    print(f"이미지 저장 경로: {save_root}")

    # --- 카메라 설정 ---
    # 운영체제에 따라 적절한 백엔드(API)를 선택
    if platform.system() == 'Darwin':  # macOS
        backend_api = cv2.CAP_AVFOUNDATION
    else:  # Linux, Windows 등
        backend_api = cv2.CAP_V4L2

    capL = cv2.VideoCapture(LEFT_DEV_ID, backend_api)
    capR = cv2.VideoCapture(RIGHT_DEV_ID, backend_api)

    if not capL.isOpened() or not capR.isOpened():
        print("오류: 카메라를 열 수 없습니다. 장치 인덱스 또는 권한을 확인하세요.")
        return

    # 해상도 설정
    capL.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    capL.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    capR.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    capR.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("\n미리보기 창이 활성화됩니다.")
    print("  - 's' 키: 현재 프레임 저장")
    print("  - 'ESC' 키 또는 창의 'X' 버튼: 촬영 종료")

    scene_counter = get_next_scene_number(save_root)

    while True:
        okL, frameL = capL.read()
        okR, frameR = capR.read()

        if not (okL and okR):
            print("프레임 캡처 실패. 카메라 연결을 확인하세요.")
            break

        if APPLY_FLIP:
            frameL = cv2.flip(frameL, -1)
            frameR = cv2.flip(frameR, -1)

        # 화면에 미리보기 표시
        preview_combined = cv2.hconcat([frameL, frameR])
        preview_combined_resized = cv2.resize(preview_combined, (FRAME_WIDTH, FRAME_HEIGHT // 2))
        cv2.imshow("Stereo Camera Preview (Left | Right) - Press 's' to save, 'ESC' to exit", preview_combined_resized)

        key = cv2.waitKey(1) & 0xFF

        # 's' 키를 눌러 저장
        if key == ord('s'):
            scene_dir = os.path.join(save_root, f"real_world_{scene_counter}")
            os.makedirs(scene_dir, exist_ok=True)
            
            left_path = os.path.join(scene_dir, "im0.png")
            right_path = os.path.join(scene_dir, "im1.png")
            
            cv2.imwrite(left_path, frameL)
            cv2.imwrite(right_path, frameR)
            
            print(f"-> 저장 완료: {scene_dir}")
            scene_counter += 1

        # ESC 키로 종료
        if key == 27:
            print("ESC 키 입력으로 프로그램을 종료합니다.")
            break
            
        # 창의 X 버튼으로 종료
        try:
            if cv2.getWindowProperty("Stereo Camera Preview (Left | Right) - Press 's' to save, 'ESC' to exit", cv2.WND_PROP_VISIBLE) < 1:
                print("미리보기 창이 닫혀 프로그램을 종료합니다.")
                break
        except cv2.error:
            # 창이 이미 닫힌 경우 발생할 수 있는 오류 무시
            break


    print("촬영을 종료합니다.")
    capL.release()
    capR.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
