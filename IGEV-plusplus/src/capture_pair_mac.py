import cv2, os, time

# =======================
# 사용자 설정
# =======================
LEFT_DEV  = 0   # ffplay -f avfoundation -list_devices true -i "" 로 확인
RIGHT_DEV = 1  # 예: USB 카메라 0번, 2번
W, H = 1920, 1080
APPLY_FLIP = True   # 카메라가 거꾸로 달려 있다면 flip(-1)
# =======================

# 현재 스크립트(src/) 기준 → 상위 폴더(IGEV-plusplus/) → test_imgs/raw
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SAVE_ROOT = os.path.join(PROJECT_ROOT, "test_imgs", "raw")

LEFT_DIR  = os.path.join(SAVE_ROOT, "left")
RIGHT_DIR = os.path.join(SAVE_ROOT, "right")
os.makedirs(LEFT_DIR, exist_ok=True)
os.makedirs(RIGHT_DIR, exist_ok=True)

# AVFoundation 백엔드 사용
capL = cv2.VideoCapture(LEFT_DEV, cv2.CAP_AVFOUNDATION)
capR = cv2.VideoCapture(RIGHT_DEV, cv2.CAP_AVFOUNDATION)

capL.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
capR.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

i = 0
print("미리보기 창이 뜹니다. 's' 키 = 한 쌍 저장, ESC = 종료")
while True:
    okL, frameL = capL.read()
    okR, frameR = capR.read()
    if not (okL and okR):
        print("캡처 실패. 카메라 인덱스 확인 필요.")
        break

    if APPLY_FLIP:
        frameL = cv2.flip(frameL, -1)
        frameR = cv2.flip(frameR, -1)

    # 미리보기
    cv2.imshow("Left", frameL)
    cv2.imshow("Right", frameR)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite(os.path.join(LEFT_DIR,  f"{i:03d}.png"), frameL)
        cv2.imwrite(os.path.join(RIGHT_DIR, f"{i:03d}.png"), frameR)
        print(f"[{i}] Saved pair")
        i += 1
        time.sleep(0.2)  # 키 중복 방지
    if key == 27:  # ESC
        break

capL.release()
capR.release()
cv2.destroyAllWindows()
