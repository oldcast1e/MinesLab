# ==============================================================================
# 통합 실시간 스테레오 비전 시스템 (디버깅 버전)
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
# ==============================================================================

LEFT_CAM_URL = "http://192.168.0.14/capture"
RIGHT_CAM_URL = "http://192.168.0.13/capture"
YOLO_WEIGHTS_PATH = '/Users/apple/Desktop/Python/Labs/StereoCam/src/yolov5s.pt'
# [디버깅] 신뢰도를 약간 낮춰서 화질이 안 좋은 카메라에서도 탐지될 가능성을 높여봅니다.
CONFIDENCE_THRESHOLD = 0.35 
FOCAL_LENGTH = 2.043636363636363
TAN_THETA = 0.5443642625
IMAGE_WIDTH_PX = 480
BASELINE = 7.05

# ==============================================================================
# --- 섹션 3: 클래스 이름 정의 ---
# ==============================================================================

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
    try:
        response = requests.get(cam_url, timeout=5)
        response.raise_for_status()
        img_array = np.array(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, -1)
        return img
    except requests.exceptions.RequestException as e:
        print(f"카메라({cam_url}) 연결 오류: {e}")
        return None

def detect_and_get_boxes(model: torch.nn.Module, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    results = model(img)
    detections = results.xyxyn[0].cpu().numpy()
    labels = detections[:, -1]
    boxes = detections[:, :-1]
    return labels, boxes

def calculate_distance_from_disparity(disparity: float) -> float:
    if disparity <= 0:
        return float('inf')
    distance_mm = (BASELINE / 2) * IMAGE_WIDTH_PX * (1 / TAN_THETA) / disparity + FOCAL_LENGTH
    return distance_mm / 1000

def compute_stereo_distances(labels1: np.ndarray, boxes1: np.ndarray, labels2: np.ndarray, boxes2: np.ndarray) -> Dict[int, float]:
    distances = {}
    unique_labels = np.unique(labels1)
    for label in unique_labels:
        if label in labels2:
            idx1 = np.where(labels1 == label)[0][0]
            idx2 = np.where(labels2 == label)[0][0]
            box1 = boxes1[idx1]
            box2 = boxes2[idx2]
            center1_px = ((box1[0] + box1[2]) / 2) * IMAGE_WIDTH_PX
            center2_px = ((box2[0] + box2[2]) / 2) * IMAGE_WIDTH_PX
            disparity = abs(center1_px - center2_px)
            distance = calculate_distance_from_disparity(disparity)
            distances[int(label)] = distance
    return distances

# [디버깅] 새로운 시각화 함수: 탐지된 모든 객체를 표시
def annotate_all_detections(img: np.ndarray, labels: np.ndarray, boxes: np.ndarray, title: str) -> np.ndarray:
    """단일 이미지에 탐지된 모든 객체를 바운딩 박스로 표시"""
    annotated_img = img.copy()
    height, width, _ = annotated_img.shape
    cv2.putText(annotated_img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    for i, label_int in enumerate(labels):
        label = int(label_int)
        box = boxes[i]
        x1, y1, x2, y2 = int(box[0] * width), int(box[1] * height), int(box[2] * width), int(box[3] * height)
        category = CLASS_MAP.get(label, "Unknown")
        
        # 순수 탐지 결과는 파란색으로 표시
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(annotated_img, category, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
    return annotated_img

# [디버깅] 수정된 주 이미지 시각화 함수
def annotate_left_image(img: np.ndarray, labels: np.ndarray, boxes: np.ndarray, distances: Dict[int, float]) -> np.ndarray:
    """왼쪽 이미지에 매칭 성공/실패 객체를 다른 색으로 표시"""
    annotated_img = img.copy()
    height, width, _ = annotated_img.shape
    cv2.putText(annotated_img, "Main View (Left Cam)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    for i, label_int in enumerate(labels):
        label = int(label_int)
        distance = distances.get(label)
        box = boxes[i]
        x1, y1, x2, y2 = int(box[0] * width), int(box[1] * height), int(box[2] * width), int(box[3] * height)
        category = CLASS_MAP.get(label, "Unknown")
        
        if distance is not None:
            # 매칭 성공: 초록색, 거리 표시
            text = f'{category}: {distance:.2f}m'
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # 매칭 실패: 빨간색, 카테고리만 표시
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated_img, category, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
    return annotated_img

def process_frame_pair(model: torch.nn.Module) -> Optional[Tuple]:
    img_left = capture_frame(LEFT_CAM_URL)
    img_right = capture_frame(RIGHT_CAM_URL)

    if img_left is None or img_right is None:
        print("프레임 캡처 실패, 다음 시도까지 대기합니다.")
        return None

    labels_left, boxes_left = detect_and_get_boxes(model, img_left)
    labels_right, boxes_right = detect_and_get_boxes(model, img_right)

    distances = compute_stereo_distances(labels_left, boxes_left, labels_right, boxes_right)
    
    # [디버깅] 오른쪽 이미지와 탐지 결과도 반환
    return (img_left, labels_left, boxes_left, img_right, labels_right, boxes_right, distances)

# ==============================================================================
# --- 섹션 5: 메인 실행 로직 ---
# ==============================================================================

def main():
    warnings.filterwarnings("ignore")
    model = load_yolo_model(YOLO_WEIGHTS_PATH, CONFIDENCE_THRESHOLD)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        print("\n[디버그 모드] 실시간 처리를 시작합니다. 종료하려면 'q'를 누르세요.")
        
        while True:
            try:
                future = executor.submit(process_frame_pair, model)
                result = future.result()

                if result is None:
                    cv2.waitKey(1000)
                    continue

                # [디버깅] 반환값 확장
                img_left, labels_left, boxes_left, img_right, labels_right, boxes_right, distances = result

                # [디버깅] 터미널에 상세 정보 출력
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n--- {current_time} ---")
                print(f"왼쪽 탐지 객체 수: {len(labels_left)}")
                print(f"오른쪽 탐지 객체 수: {len(labels_right)}")
                print(f"  - 왼쪽 라벨: {[CLASS_MAP.get(int(l), 'N/A') for l in labels_left]}")
                print(f"  - 오른쪽 라벨: {[CLASS_MAP.get(int(l), 'N/A') for l in labels_right]}")

                if not distances:
                    print("결과: 매칭되는 객체를 찾지 못했습니다.")
                else:
                    print("결과: 매칭 성공!")
                    for label, dist in distances.items():
                        category = CLASS_MAP.get(label, "Unknown")
                        print(f"  - 카테고리: {category:<15} | 거리: {dist:.2f} m")
                
                # [디버깅] 시각화
                annotated_left = annotate_left_image(img_left, labels_left, boxes_left, distances)
                annotated_right = annotate_all_detections(img_right, labels_right, boxes_right, "Debug View (Right Cam)")

                # [디버깅] 두 이미지를 가로로 합쳐서 한 창에 표시
                # 해상도가 다를 경우를 대비해 높이를 맞춰줌
                h1, w1, _ = annotated_left.shape
                h2, w2, _ = annotated_right.shape
                if h1 != h2:
                    # 왼쪽 이미지 높이에 맞춤
                    new_w2 = int(w2 * h1 / h2)
                    annotated_right = cv2.resize(annotated_right, (new_w2, h1))
                
                combined_view = np.hstack((annotated_left, annotated_right))
                cv2.imshow("Stereo Vision Debug View", combined_view)

                if cv2.waitKey(33) & 0xFF == ord('q'):
                    break

            except Exception as e:
                print(f"메인 루프에서 예외 발생: {e}")
                break

    cv2.destroyAllWindows()
    print("프로그램을 종료합니다.")

if __name__ == '__main__':
    main()