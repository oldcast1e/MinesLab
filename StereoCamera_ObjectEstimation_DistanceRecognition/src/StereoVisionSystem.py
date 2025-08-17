# ==============================================================================
# 통합 실시간 스테레오 비전 시스템 (개선 버전)
#
# 원본 파일 목록:
# capture.py, distance_calculation.py, image_processing.py, class_map.py,
# detection.py, main.py, model.py, processing.py, RealTime.py, visualization.py
#
# 주요 기능:
# 1. 두 대의 ESP32 카메라에서 실시간으로 영상 스트림 수신
# 2. YOLOv5 모델을 이용한 객체 탐지
# 3. 스테레오 비전 원리를 이용한 객체까지의 거리 측정
# 4. 탐지된 객체와 거리를 영상에 표시하여 실시간 출력
# ==============================================================================

# --- 섹션 1: 라이브러리 임포트 ---
import cv2
import torch
import numpy as np
import requests
import warnings
from datetime import datetime
import concurrent.futures
from typing import Dict, List, Tuple, Optional

# ==============================================================================
# --- 섹션 2: 설정 (CONFIGURATION) ---
# 사용자가 직접 수정할 수 있는 파라미터들
# ==============================================================================

# ESP32 카메라 스트리밍 URL
LEFT_CAM_URL = "http://192.168.0.14/capture"
RIGHT_CAM_URL = "http://192.168.0.13/capture"

# YOLOv5 모델 가중치 파일 경로
# 사용자가 제공한 경로를 정확히 반영합니다.
YOLO_WEIGHTS_PATH = '/Users/apple/Desktop/Python/Labs/StereoCam/src/yolov5s.pt'

# 객체 탐지 신뢰도 임계값 (0.0 ~ 1.0)
# 이 값보다 낮은 신뢰도를 가진 객체는 무시됩니다.
CONFIDENCE_THRESHOLD = 0.45

# 스테레오 비전 파라미터
FOCAL_LENGTH = 2.043636363636363  # 카메라 초점 거리 (mm)
TAN_THETA = 0.5443642625           # 카메라 화각(FOV)의 절반에 대한 탄젠트 값
IMAGE_WIDTH_PX = 480               # 카메라 이미지의 너비 (픽셀 단위, HVGA 해상도 기준)
BASELINE = 7.05                    # 두 카메라 렌즈 간의 거리 (mm)

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
    
    Args:
        weights_path (str): .pt 파일의 경로.
        confidence (float): 객체 탐지를 위한 신뢰도 임계값.

    Returns:
        torch.nn.Module: 로드된 YOLOv5 모델 객체.
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

def capture_frame(cam_url: str) -> Optional[np.ndarray]:
    """
    지정된 URL에서 이미지를 캡처하여 OpenCV 이미지 객체로 반환합니다.
    
    Args:
        cam_url (str): ESP32 카메라의 캡처 URL.

    Returns:
        Optional[np.ndarray]: 성공 시 이미지 배열, 실패 시 None.
    """
    try:
        response = requests.get(cam_url, timeout=5)
        response.raise_for_status()  # HTTP 오류가 있으면 예외 발생
        img_array = np.array(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, -1)
        return img
    except requests.exceptions.RequestException as e:
        print(f"카메라({cam_url}) 연결 오류: {e}")
        return None

def detect_and_get_boxes(model: torch.nn.Module, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    이미지에서 객체를 탐지하고, 라벨과 정규화된 바운딩 박스를 반환합니다.
    
    Args:
        model: YOLOv5 모델 객체.
        img: 분석할 이미지.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (라벨 배열, 바운딩 박스 배열).
    """
    results = model(img)
    # xyxyn 포맷: [x_min, y_min, x_max, y_max, confidence, class] (정규화된 좌표)
    detections = results.xyxyn[0].cpu().numpy()
    labels = detections[:, -1]
    boxes = detections[:, :-1]
    return labels, boxes

def calculate_distance_from_disparity(disparity: float) -> float:
    """
    픽셀 단위의 시차(disparity)를 사용하여 실제 거리를 계산합니다.

    Args:
        disparity (float): 두 이미지 간의 객체 중심점의 픽셀 차이.

    Returns:
        float: 계산된 거리 (미터 단위).
    """
    if disparity <= 0:
        return float('inf')  # 시차가 0 이하면 거리 계산 불가
    
    # 거리 계산 공식 (단위: mm)
    distance_mm = (BASELINE / 2) * IMAGE_WIDTH_PX * (1 / TAN_THETA) / disparity + FOCAL_LENGTH
    return distance_mm / 1000  # 미터 단위로 변환하여 반환

def compute_stereo_distances(labels1: np.ndarray, boxes1: np.ndarray, labels2: np.ndarray, boxes2: np.ndarray) -> Dict[int, float]:
    """
    두 이미지의 탐지 결과를 비교하여 동일 객체의 거리를 계산합니다.

    참고: 이 함수는 각 클래스별로 첫 번째로 탐지된 객체만 매칭합니다.
          동일 클래스의 객체가 여러 개 있을 경우, 가장 유사한 객체를 찾는
          고급 매칭 알고리즘(예: IoU 기반 매칭)이 필요할 수 있습니다.
          
    Args:
        labels1, boxes1: 왼쪽 이미지의 라벨과 바운딩 박스.
        labels2, boxes2: 오른쪽 이미지의 라벨과 바운딩 박스.

    Returns:
        Dict[int, float]: {클래스 라벨: 계산된 거리(m)} 형태의 딕셔너리.
    """
    distances = {}
    unique_labels = np.unique(labels1)
    
    for label in unique_labels:
        if label in labels2:
            idx1 = np.where(labels1 == label)[0][0]
            idx2 = np.where(labels2 == label)[0][0]
            
            box1 = boxes1[idx1]
            box2 = boxes2[idx2]

            # 바운딩 박스 중심의 x좌표 계산 (픽셀 단위)
            center1_px = ((box1[0] + box1[2]) / 2) * IMAGE_WIDTH_PX
            center2_px = ((box2[0] + box2[2]) / 2) * IMAGE_WIDTH_PX
            
            disparity = abs(center1_px - center2_px)
            distance = calculate_distance_from_disparity(disparity)
            
            distances[int(label)] = distance
            
    return distances

def annotate_image(img: np.ndarray, labels: np.ndarray, boxes: np.ndarray, distances: Dict[int, float]) -> np.ndarray:
    """
    이미지에 탐지된 객체의 바운딩 박스와 계산된 거리를 그립니다.
    
    Args:
        img: 원본 이미지.
        labels: 탐지된 객체들의 라벨.
        boxes: 탐지된 객체들의 바운딩 박스.
        distances: 계산된 거리 딕셔너리.

    Returns:
        np.ndarray: 정보가 그려진 이미지.
    """
    annotated_img = img.copy()
    height, width, _ = annotated_img.shape
    
    for i, label_int in enumerate(labels):
        label = int(label_int)
        distance = distances.get(label)
        
        if distance is not None:
            box = boxes[i]
            x1, y1, x2, y2 = int(box[0] * width), int(box[1] * height), int(box[2] * width), int(box[3] * height)
            
            category = CLASS_MAP.get(label, "Unknown")
            text = f'{category}: {distance:.2f}m'
            
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
    return annotated_img

def process_frame_pair(model: torch.nn.Module) -> Optional[Tuple]:
    """
    한 쌍의 프레임을 캡처하고 처리하는 전체 파이프라인.
    """
    img_left = capture_frame(LEFT_CAM_URL)
    img_right = capture_frame(RIGHT_CAM_URL)

    if img_left is None or img_right is None:
        print("프레임 캡처 실패, 다음 시도까지 대기합니다.")
        return None

    labels_left, boxes_left = detect_and_get_boxes(model, img_left)
    labels_right, boxes_right = detect_and_get_boxes(model, img_right)

    distances = compute_stereo_distances(labels_left, boxes_left, labels_right, boxes_right)
    
    return (img_left, labels_left, boxes_left, distances)

# ==============================================================================
# --- 섹션 5: 메인 실행 로직 ---
# ==============================================================================

def main():
    """
    실시간 스테레오 비전 시스템을 초기화하고 실행합니다.
    """
    warnings.filterwarnings("ignore") # 경고 메시지 숨기기

    # 모델 로드
    model = load_yolo_model(YOLO_WEIGHTS_PATH, CONFIDENCE_THRESHOLD)

    # 병렬 처리를 위한 스레드 풀 실행기
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        print("\n실시간 처리를 시작합니다. 종료하려면 'q'를 누르세요.")
        
        while True:
            try:
                # 스레드에서 프레임 처리 작업을 비동기적으로 실행
                future = executor.submit(process_frame_pair, model)
                result = future.result()

                if result is None:
                    cv2.waitKey(1000) # 실패 시 1초 대기
                    continue

                img_left, labels_left, boxes_left, distances = result

                # 터미널에 결과 출력
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n--- {current_time} ---")
                if not distances:
                    print("매칭되는 객체를 찾지 못했습니다.")
                else:
                    for label, dist in distances.items():
                        category = CLASS_MAP.get(label, "Unknown")
                        print(f"  - 카테고리: {category:<15} | 거리: {dist:.2f} m")
                
                # 이미지에 결과 시각화
                annotated_image = annotate_image(img_left, labels_left, boxes_left, distances)

                # 화면에 출력
                cv2.imshow("Real-time Stereo Vision", annotated_image)

                # 'q' 키를 누르면 종료 (약 30 FPS)
                if cv2.waitKey(33) & 0xFF == ord('q'):
                    break

            except Exception as e:
                print(f"메인 루프에서 예외 발생: {e}")
                break

    cv2.destroyAllWindows()
    print("프로그램을 종료합니다.")

if __name__ == '__main__':
    main()