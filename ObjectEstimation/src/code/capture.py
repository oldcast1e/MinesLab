import requests
import numpy as np
import cv2

def capture_frame_from_esp32(cam_url):
    """
    ESP32 카메라 서버에서 프레임을 캡처하여 반환합니다.
    """
    img_resp = requests.get(cam_url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    return img
