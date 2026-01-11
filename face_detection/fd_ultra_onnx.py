#!/usr/bin/env python3
import cv2
import argparse

def detect_faces_yunet(image_path: str,
                       conf_threshold: float = 0.8,
                       model_path: str = "./model/face_detection_yunet_2023mar.onnx") -> None:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    h, w = img.shape[:2]

    # 初始化 YuNet
    detector = cv2.FaceDetectorYN_create(
        model=model_path,
        config="",
        input_size=(w, h),
        score_threshold=conf_threshold,
        nms_threshold=0.4,
        top_k=5000
    )
    detector.setInputSize((w, h))

    # detect 返回 (status, faces)  取第 1 个元素
    faces = detector.detect(img)[1]
    if faces is None:
        faces = []

    for face in faces:
        x, y, w_box, h_box, *_ = map(int, face[:4])
        score = face[-1]
        cv2.rectangle(img, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
        label = f"{score:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x, y - label_size[1] - 4),
                      (x + label_size[0], y), (0, 255, 0), -1)
        cv2.putText(img, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # 树莓派小屏：允许鼠标拖拽缩放
    cv2.namedWindow("YuNet Face Detection", cv2.WINDOW_NORMAL)
    cv2.imshow("YuNet Face Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
