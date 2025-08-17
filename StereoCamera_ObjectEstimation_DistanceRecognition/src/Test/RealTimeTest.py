import cv2
import multiprocessing as mp
from datetime import datetime
import warnings
import sys
import os

# 모든 경고 무시
warnings.filterwarnings("ignore")

def process_and_display(left_cam_url, right_cam_url, weights_path, fl, tantheta, img_width):
    import model as yolo_model  # 모델을 프로세스 내에서 임포트
    from capture import capture_frame_from_esp32
    from processing import process_frames
    from visualization import annotate_image_with_distances
    from class_map import CLASS_MAP
    
    # YOLOv5 모델 로드
    model = yolo_model.load_model(weights_path)
    
    while True:
        try:
            # 이미지 캡처
            img1 = capture_frame_from_esp32(left_cam_url)
            img2 = capture_frame_from_esp32(right_cam_url)

            if img1 is None or img2 is None:
                print("Failed to capture images from ESP32 cameras.")
                continue

            # 프레임 처리
            result = process_frames(img1, img2, model, fl, tantheta, img_width)
            if result is not None:
                img1, labels1, boxes1, distances, disparities = result

                # 현재 시간 출력
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"Time: {current_time}")

                # 시차 및 거리 출력
                for label in labels1:
                    if label in distances:
                        category = CLASS_MAP.get(int(label), "Unknown")
                        distance = distances.get(label, "Unknown")
                        print(f"Category: {category}, Distance: {distance:.2f} meters")

                print("-" * 50)

                # 이미지에 주석 추가
                img1_annotated = annotate_image_with_distances(img1, labels1, boxes1, distances, CLASS_MAP)

                # 최종 이미지 파이썬 창에 띄우기
                cv2.imshow("Annotated Image", img1_annotated)

                # 초당 30프레임 설정 (33ms 대기)
                if cv2.waitKey(33) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"An error occurred: {e}")
            break

    cv2.destroyAllWindows()

def main():
    # ESP32 카메라 주소 (해상도 QVGA)
    left_cam_url = "http://192.168.0.14/capture"
    right_cam_url = "http://192.168.0.13/capture"

    # YOLOv5 모델 경로
    weights_path = '/Users/apple/Desktop/Python/yolov5s.pt'

    # 스테레오 비전 설정
    fl = 2.043636363636363
    tantheta = 0.5443642625
    img_width = 480  # HVGA 해상도 기준

    # 프로세스를 실행하여 이미지 처리 및 표시 작업 수행
    process = mp.Process(target=process_and_display, args=(left_cam_url, right_cam_url, weights_path, fl, tantheta, img_width))
    process.start()

    try:
        process.join()
    except KeyboardInterrupt:
        print("Interrupted by user")
        process.terminate()
        process.join()

if __name__ == '__main__':
    main()
