import numpy as np

# 시차를 계산하는 함수
def calculate_disparity(box1, box2, img_width):
    center1 = (box1[0] + box1[2]) / 2 * img_width
    center2 = (box2[0] + box2[2]) / 2 * img_width
    disparity = abs(center1 - center2)
    return disparity

# 카메라와 객체 사이의 거리를 계산하는 함수
def calculate_distance(disparity, fl, tantheta, img_width):
    distance = (7.05 / 2) * img_width * (1 / tantheta) / disparity + fl
    return distance

# 같은 카테고리 객체 사이의 거리 및 시차 계산
def compute_distances_and_disparity(labels1, boxes1, labels2, boxes2, fl, tantheta, img_width):
    distances = {}
    disparities = {}
    for i, label in enumerate(labels1):
        if label in labels2:
            j = np.where(labels2 == label)[0][0]
            disparity = calculate_disparity(boxes1[i], boxes2[j], img_width)
            distance = calculate_distance(disparity, fl, tantheta, img_width)
            distances[label] = distance
            disparities[label] = disparity
    return distances, disparities
