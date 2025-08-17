import cv2
import concurrent.futures
from datetime import datetime
import warnings
from capture import capture_frame_from_esp32
from processing import process_frames
from visualization import annotate_image_with_distances
from class_map import CLASS_MAP
import model as yolo_model

# 모든 경고 무시
warnings.filterwarnings("ignore")

def main():
    # ESP32 카메라 주소 (해상도 QVGA)
    left_cam_url = "http://192.168.0.14/capture"
    right_cam_url = "http://192.168.0.13/capture"

    # YOLOv5 모델 로드
    weights_path = '/Users/apple/Desktop/Python/Labs/StereoCam/src/yolov5s.pt'
    model = yolo_model.load_model(weights_path)

    # 스테레오 비전 설정
    fl = 2.043636363636363
    tantheta = 0.5443642625

    # img_width = 640  # VGA 해상도 기준
    img_width = 480  # HVGA 해상도 기준

    with concurrent.futures.ThreadPoolExecutor() as executor:
        while True:
            try:
                # 병렬 처리로 프레임을 처리
                future = executor.submit(process_frames, left_cam_url, right_cam_url, model, fl, tantheta, img_width)
                result = future.result()

                if result is None:
                    continue

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

                print("-"*50)

                # 이미지에 주석 추가
                img1_annotated = annotate_image_with_distances(img1, labels1, boxes1, distances, CLASS_MAP)

                # 최종 이미지 파이썬 창에 띄우기
                cv2.imshow("Annotated Image", img1_annotated)

                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

                # # # 초당 10프레임 설정 
                # if cv2.waitKey(100) & 0xFF == ord('q'):
                #     break

                # 초당 30프레임 설정 (33ms 대기)
                if cv2.waitKey(33) & 0xFF == ord('q'):
                    break

                # # 초당 60프레임 설정 
                # if cv2.waitKey(16) & 0xFF == ord('q'):
                #     break

            except KeyError as e:
                print(f"KeyError: {e} - Skipping this frame.")
                continue
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
