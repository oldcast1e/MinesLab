import cv2

# 이미지에 객체 카테고리와 거리를 표시
def annotate_image_with_distances(img, labels, boxes, distances, class_map):
    for i, (label, box) in enumerate(zip(labels, boxes)):
        category = class_map[int(label)]
        distance = distances[label]
        x1, y1, x2, y2 = int(box[0] * img.shape[1]), int(box[1] * img.shape[0]), int(box[2] * img.shape[1]), int(box[3] * img.shape[0])

        # 경계 상자 그리기
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

        # 객체의 카테고리와 거리를 표시하는 텍스트
        text = f'{category}: {distance:.2f}m'
        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img
