import detection as det
import distance_calculation as dist_calc
from capture import capture_frame_from_esp32

def process_frames(left_cam_url, right_cam_url, model, fl, tantheta, img_width):
    """
    두 개의 ESP32 카메라에서 프레임을 가져와 처리하는 함수
    """
    # ESP32에서 이미지 캡처
    img1 = capture_frame_from_esp32(left_cam_url)
    img2 = capture_frame_from_esp32(right_cam_url)

    if img1 is None or img2 is None:
        print("Failed to capture images from ESP32 cameras.")
        return None, None

    # 객체 탐지
    results1 = det.detect_objects(model, img1)
    results2 = det.detect_objects(model, img2)

    # 바운더리 박스와 라벨 추출
    labels1, boxes1 = det.get_bounding_boxes(results1)
    labels2, boxes2 = det.get_bounding_boxes(results2)

    # 거리 및 시차 계산
    distances, disparities = dist_calc.compute_distances_and_disparity(labels1, boxes1, labels2, boxes2, fl, tantheta, img_width)

    return (img1, labels1, boxes1, distances, disparities)
