import numpy as np

# 이미지에서 객체 탐지
def detect_objects(model, img):
    results = model(img)
    return results

# 바운더리 박스와 라벨을 가져오기
def get_bounding_boxes(results):
    labels = results.xyxyn[0][:,-1].numpy()
    boxes = results.xyxyn[0][:,:-1].numpy()
    return labels, boxes
