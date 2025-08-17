import cv2
import torch

# YOLOv5 모델 로드 (로컬 .pt 파일 사용)
model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path='/Users/apple/Desktop/Python/Labs/StereoCam/src/yolov5s.pt',
    force_reload=False  # 필요시 True로 변경하여 캐시 무시
)

# 모델을 평가 모드로 설정
model.eval()

# USB 카메라 열기 (AVFoundation 기준 index 1)
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# 카메라 정상 연결 확인
if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다.")
    exit()

print("✅ YOLOv5 객체 인식 시작 (q 키로 종료)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임을 수신할 수 없습니다.")
        break

    # BGR → RGB 변환
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 객체 탐지
    results = model(img_rgb)

    # 결과 이미지 렌더링
    results.render()

    # 출력: results.ims[0]는 numpy 배열(RGB), 다시 BGR로 변환해 OpenCV에 표시
    output_frame = results.ims[0][:, :, ::-1]
    cv2.imshow('YOLOv5 Object Detection', output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
