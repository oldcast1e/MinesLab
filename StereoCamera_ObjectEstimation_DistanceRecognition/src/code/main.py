import model as yolo_model  # 'model' 모듈을 'yolo_model'로 임포트
import image_processing as ip
import detection as det
import distance_calculation as dist_calc
import visualization as vis
from class_map import CLASS_MAP
import cv2
import warnings
import sys

def main():
    # 경고 메시지 무시 (선택사항)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # 이미지 경로
    img1_path = "/Users/apple/Desktop/Python/Smarcle/MakersDay/사진자료/SVGA/clock_SVGA_20_left.jpg"
    img2_path = "/Users/apple/Desktop/Python/Smarcle/MakersDay/사진자료/SVGA/clock_SVGA_20_right.jpg"

    # 스테레오 비전 설정
    fl = 2.043636363636363
    tantheta = 0.5443642625
    img_width = 240

    # YOLOv5 모델 로드
    weights_path = '/Users/apple/Desktop/Python/Labs/StereoCam/src/yolov5s.pt'

    # YOLOv5 루트 경로를 최상단에 추가
    sys.path.insert(0, '/Users/apple/Desktop/Python/yolov5')

    from models.experimental import attempt_load
    model = attempt_load(weights_path)

    # 이미지 로드
    img1 = ip.load_image(img1_path)
    img2 = ip.load_image(img2_path)

    # 객체 탐지
    results1 = det.detect_objects(model, img1)
    results2 = det.detect_objects(model, img2)

    # 바운더리 박스와 라벨 추출
    labels1, boxes1 = det.get_bounding_boxes(results1)
    labels2, boxes2 = det.get_bounding_boxes(results2)

    # 거리 및 시차 계산
    distances, disparities = dist_calc.compute_distances_and_disparity(labels1, boxes1, labels2, boxes2, fl, tantheta, img_width)

    # 시차 및 거리 출력
    for label in disparities:
        category = CLASS_MAP[int(label)]
        disparity = disparities[label]
        distance = distances.get(label, "Unknown")
        print(f"Category: {category}, Disparity: {disparity:.2f} pixels")
        print(f"Category: {category}, Distance: {distance:.2f} meters")

    # 이미지에 주석 추가
    img1_annotated = vis.annotate_image_with_distances(img1, labels1, boxes1, distances, CLASS_MAP)

    # 최종 이미지 파이썬 창에 띄우기
    cv2.imshow("Annotated Image", img1_annotated)
    cv2.waitKey(0)  # 키 입력이 있을 때까지 창을 유지합니다.
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
