# ==============================================================================
# 단일 카메라 실시간 객체 탐지 시스템
#
# 주요 기능:
# 1. 단일 카메라(ESP32-CAM 또는 웹캠)에서 실시간 영상 스트림 수신
# 2. YOLOv5 모델을 이용한 객체 탐지
# 3. 탐지된 객체를 영상에 표시하여 실시간으로 화면에 출력
# ==============================================================================

# --- 섹션 1: 라이브러리 임포트 ---
import cv2
import torch
import numpy as np
import requests
import warnings
from typing import Optional, Tuple

# ==============================================================================
# --- 섹션 2: 설정 (CONFIGURATION) ---
# 사용자가 직접 수정할 수 있는 파라미터들
# ==============================================================================

# --- 카메라 선택 ---
# ESP32 카메라를 사용하려면 'ESP32', PC에 연결된 웹캠을 사용하려면 'WEBCAM'으로 설정
CAMERA_MODE = 'ESP32'  # 'ESP32' 또는 'WEBCAM'

# ESP32 카메라 스트리밍 URL (CAMERA_MODE가 'ESP32'일 때만 사용)
# ESP32_CAM_URL = "http://192.168.0.13/capture" #RIGHT
ESP32_CAM_URL = "http://192.168.0.14/capture" #LEFT

# 웹캠 장치 번호 (CAMERA_MODE가 'WEBCAM'일 때만 사용, 보통 0 또는 1)
WEBCAM_ID = 0

# YOLOv5 모델 가중치 파일 경로
YOLO_WEIGHTS_PATH = '/Users/apple/Desktop/Python/Labs/StereoCam/src/yolov5s.pt'

# 객체 탐지 신뢰도 임계값 (0.0 ~ 1.0)
CONFIDENCE_THRESHOLD = 0.45

# ==============================================================================
# --- 섹션 3: 클래스 이름 정의 ---
# ==============================================================================

# YOLOv5 모델의 클래스 인덱스와 실제 이름 매핑
CLASS_MAP = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus", 6: "train",
    7: "truck", 8: "boat", 9: "traffic light", 10: "fire hydrant", 11: "stop sign", 12: "parking meter",
    13: "bench", 14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella",
    26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee", 30: "skis", 31: "snowboard",
    32: "sports ball", 33: "kite", 34: "baseball bat", 35: "baseball glove", 36: "skateboard",
    37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass", 41: "cup", 42: "fork",
    43: "knife", 44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
    50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake", 56: "chair",
    57: "couch", 58: "potted plant", 59: "bed", 60: "dining table", 61: "toilet", 62: "TV",
    63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book", 74: "clock",
    75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush"
}

# ==============================================================================
# --- 섹션 4: 핵심 기능 함수 ---
# ==============================================================================

def load_yolo_model(weights_path: str, confidence: float) -> torch.nn.Module:
    """
    YOLOv5 모델을 로드하고 신뢰도 임계값을 설정합니다.
    """
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
        model.conf = confidence
        model.eval()
        print("YOLOv5 모델 로딩 성공.")
        return model
    except Exception as e:
        print(f"모델 로딩 중 오류 발생: {e}")
        raise

def capture_esp32_frame(cam_url: str) -> Optional[np.ndarray]:
    """
    ESP32 카메라에서 이미지를 캡처합니다.
    """
    try:
        response = requests.get(cam_url, timeout=3)
        response.raise_for_status()
        img_array = np.array(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, -1)
        return img
    except requests.exceptions.RequestException as e:
        print(f"ESP32 카메라({cam_url}) 연결 오류: {e}")
        return None

def detect_objects(model: torch.nn.Module, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    이미지에서 객체를 탐지하고 라벨, 신뢰도, 바운딩 박스를 반환합니다.
    """
    results = model(img)
    # xyxyn 포맷: [x_min, y_min, x_max, y_max, confidence, class]
    detections = results.xyxyn[0].cpu().numpy()
    labels = detections[:, -1]
    confidences = detections[:, -2]
    boxes = detections[:, :-2]
    return labels, confidences, boxes

def annotate_frame(img: np.ndarray, labels: np.ndarray, confidences: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    이미지에 탐지된 객체의 바운딩 박스와 클래스 이름, 신뢰도를 그립니다.
    """
    annotated_img = img.copy()
    height, width, _ = annotated_img.shape
    
    for i, label_int in enumerate(labels):
        label = int(label_int)
        box = boxes[i]
        confidence = confidences[i]
        
        x1, y1, x2, y2 = int(box[0] * width), int(box[1] * height), int(box[2] * width), int(box[3] * height)
        
        category = CLASS_MAP.get(label, "Unknown")
        text = f'{category}: {confidence:.2f}'
        
        # 바운딩 박스 그리기 (초록색)
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 텍스트 배경 사각형 그리기
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated_img, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
        
        # 텍스트 그리기 (검정색)
        cv2.putText(annotated_img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
    return annotated_img

# ==============================================================================
# --- 섹션 5: 메인 실행 로직 ---
# ==============================================================================

def main():
    """
    단일 카메라 객체 탐지 시스템을 초기화하고 실행합니다.
    """
    warnings.filterwarnings("ignore")

    # YOLOv5 모델 로드
    model = load_yolo_model(YOLO_WEIGHTS_PATH, CONFIDENCE_THRESHOLD)

    # 카메라 초기화
    cap = None
    if CAMERA_MODE == 'WEBCAM':
        cap = cv2.VideoCapture(WEBCAM_ID)
        if not cap.isOpened():
            print(f"웹캠(ID: {WEBCAM_ID})을 열 수 없습니다.")
            return
        print(f"웹캠(ID: {WEBCAM_ID}) 사용을 시작합니다.")
    elif CAMERA_MODE == 'ESP32':
        print(f"ESP32 카메라({ESP32_CAM_URL}) 사용을 시작합니다.")
    else:
        print(f"오류: 잘못된 CAMERA_MODE ('{CAMERA_MODE}') 입니다. 'ESP32' 또는 'WEBCAM'을 사용하세요.")
        return

    print("\n실시간 객체 탐지를 시작합니다. 종료하려면 'q'를 누르세요.")
    
    while True:
        frame = None
        # 프레임 캡처
        if CAMERA_MODE == 'WEBCAM':
            ret, frame = cap.read()
            if not ret:
                print("웹캠에서 프레임을 읽을 수 없습니다. 종료합니다.")
                break
        elif CAMERA_MODE == 'ESP32':
            frame = capture_esp32_frame(ESP32_CAM_URL)
            if frame is None:
                print("ESP32에서 프레임을 가져오지 못했습니다. 1초 후 재시도합니다.")
                cv2.waitKey(1000)
                continue

        # 객체 탐지
        labels, confidences, boxes = detect_objects(model, frame)
        
        # 터미널에 탐지 결과 출력
        if len(labels) > 0:
            print(f"탐지된 객체 수: {len(labels)}")
            for i in range(len(labels)):
                category = CLASS_MAP.get(int(labels[i]), "Unknown")
                print(f"  - {category} (신뢰도: {confidences[i]:.2f})")
        
        # 프레임에 결과 시각화
        annotated_frame = annotate_frame(frame, labels, confidences, boxes)

        # 화면에 결과 출력
        cv2.imshow("Single Camera Object Detection", annotated_frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제
    if CAMERA_MODE == 'WEBCAM' and cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("프로그램을 종료합니다.")

if __name__ == '__main__':
    main()