# 맥부 기준 소스 코드
# IGEV-plusplus 프로젝트의 캡처 및 보정 스크립트

import cv2, os

# 사용자 설정
PROJECT_ROOT = "/Users/apple/Desktop/Python/MinesLab/IGEV-plusplus"

# 저장 경로 (프로젝트 루트 아래 calib/left, calib/right)
LEFT_DIR = os.path.join(PROJECT_ROOT, "calib/left")
RIGHT_DIR = os.path.join(PROJECT_ROOT, "calib/right")

# 카메라 인덱스 (맥북 AVFoundation 기준, 네 환경에 맞게 조정)
L_ID, R_ID = 0, 1   # 예시: 왼쪽 카메라=0, 오른쪽 카메라=2
W, H = 1920, 1080   # 1080p
MAX_PAIRS = 80      # 저장할 쌍 수

# 출력 폴더 생성
os.makedirs(LEFT_DIR, exist_ok=True)
os.makedirs(RIGHT_DIR, exist_ok=True)

# 카메라 열기
capL = cv2.VideoCapture(L_ID, cv2.CAP_AVFOUNDATION)
capR = cv2.VideoCapture(R_ID, cv2.CAP_AVFOUNDATION)

capL.set(cv2.CAP_PROP_FRAME_WIDTH, W)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, W)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

i = 0
while i < MAX_PAIRS:
    okL, frameL = capL.read()
    okR, frameR = capR.read()
    if not (okL and okR):
        print("캡처 실패. 카메라 연결을 확인하세요.")
        break

    # 카메라가 거꾸로 설치 → flip(-1): 상하좌우 반전
    frameL = cv2.flip(frameL, -1)
    frameR = cv2.flip(frameR, -1)

    # 화면에 미리보기 표시
    cv2.imshow("Left", frameL)
    cv2.imshow("Right", frameR)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # 's'를 눌러 저장
        cv2.imwrite(os.path.join(LEFT_DIR, f"{i:03d}.png"), frameL)
        cv2.imwrite(os.path.join(RIGHT_DIR, f"{i:03d}.png"), frameR)
        print(f"[{i+1}/{MAX_PAIRS}] Saved pair {i:03d}")
        i += 1

    if key == 27:  # ESC 조기 종료
        break

print("촬영 종료.")
capL.release()
capR.release()
cv2.destroyAllWindows()
