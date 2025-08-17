import cv2

# 이미지 로드 함수
def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 로드할 수 없습니다: {image_path}")
    return img
