import cv2  # OpenCV 라이브러리를 임포트하여 이미지 처리와 관련된 함수들을 사용
import serial  # pySerial 라이브러리를 임포트하여 시리얼 통신을 사용
import time  # 시간 관련 함수들을 사용하기 위해 time 모듈을 임포트
import concurrent.futures  # 병렬 처리를 위해 concurrent.futures 모듈을 임포트
from datetime import datetime  # 날짜와 시간 관련 기능을 사용하기 위해 datetime 모듈을 임포트
import warnings  # 경고 메시지를 제어하기 위해 warnings 모듈을 임포트
from capture import capture_frame_from_esp32  # ESP32로부터 프레임을 캡처하는 함수를 임포트
from processing import process_frames  # 프레임을 처리하는 함수를 임포트
from visualization import annotate_image_with_distances  # 객체 감지 결과를 이미지에 주석으로 추가하는 함수를 임포트
from class_map import CLASS_MAP  # 객체 분류를 위한 클래스 맵을 임포트
import model as yolo_model  # YOLO 모델을 사용하기 위해 모델 모듈을 임포트

# 모든 경고를 무시합니다.
warnings.filterwarnings("ignore")

# 아두이노와의 시리얼 통신 설정
arduino_port = '/dev/cu.usbserial-120'  # 아두이노가 연결된 시리얼 포트
baud_rate = 115200  # 시리얼 통신의 보드레이트 설정 (아두이노와 일치해야 함)
ser = serial.Serial(arduino_port, baud_rate, timeout=1)  # 시리얼 통신을 위한 객체 생성

# 서보 모터 제어 함수
def control_servo(angle):
    command = f"{angle}\n"  # 아두이노로 보낼 각도를 문자열로 변환하고 줄바꿈 문자 추가
    ser.write(command.encode())  # 문자열을 바이트로 변환하여 시리얼로 전송

def main():
    # ESP32 카메라 주소 (해상도 QVGA)
    left_cam_url = "http://192.168.0.14/capture"  # 좌측 카메라의 주소
    right_cam_url = "http://192.168.0.13/capture"  # 우측 카메라의 주소

    # YOLOv5 모델 로드
    weights_path = '/Users/apple/Desktop/Python/yolov5s.pt'  # YOLO 모델 가중치 파일 경로
    model = yolo_model.load_model(weights_path)  # YOLO 모델을 로드

    # 스테레오 비전 설정
    fl = 2.043636363636363  # 초점 거리 설정
    tantheta = 0.5443642625  # 삼각측량에 사용할 파라미터 설정

    img_width = 480  # HVGA 해상도 기준의 이미지 너비 설정
    center_x = img_width // 2  # 화면의 중점 x 좌표 계산

    with concurrent.futures.ThreadPoolExecutor() as executor:
        while True:
            try:
                # 병렬 처리로 프레임을 처리
                future = executor.submit(process_frames, left_cam_url, right_cam_url, model, fl, tantheta, img_width)
                result = future.result()  # 프레임 처리 결과를 가져옴

                if result is not None:
                    left_frame, right_frame, distances = result  # 좌측 프레임, 우측 프레임, 거리 정보를 분리

                    # 객체 감지 결과에서 중점 좌표 추출
                    for obj in distances:
                        x, y, w, h, cls_id, distance = obj  # 객체의 위치와 크기, 클래스 ID, 거리를 분리
                        obj_center_x = x + (w // 2)  # 객체의 중심 x 좌표 계산

                        # 화면 중점과 객체 중점의 차이 계산
                        diff = center_x - obj_center_x  # 화면 중점과 객체 중점 간의 거리 계산

                        # 차이에 비례하여 서보 모터 회전
                        if diff != 0:
                            # 각도는 20도(오른쪽 끝)에서 80도(왼쪽 끝) 사이
                            servo_angle = 50 - (diff / center_x) * 30  # 중점 차이에 비례한 각도 계산
                            servo_angle = max(20, min(80, servo_angle))  # 각도를 20도에서 80도 사이로 제한
                            control_servo(int(servo_angle))  # 서보 모터 제어 함수 호출

                    # 결과 영상에 거리 정보 추가
                    annotated_image = annotate_image_with_distances(left_frame, distances)  # 이미지에 주석 추가

                    # 영상 출력
                    cv2.imshow("Object Detection", annotated_image)  # 주석이 추가된 이미지를 출력

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break  # 'q' 키를 누르면 루프 종료

            except KeyboardInterrupt:
                break  # 키보드 인터럽트(Ctrl+C) 시 루프 종료

    ser.close()  # 시리얼 통신 종료
    cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기

if __name__ == "__main__":
    main()  # 메인 함수 호출
