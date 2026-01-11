#!/usr/bin/env python3
import cv2
import argparse
import os
import numpy as np
from pathlib import Path

# ------------------- Face Detection ------------------
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

    cv2.namedWindow("YuNet Face Detection", cv2.WINDOW_NORMAL)
    cv2.imshow("YuNet Face Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -------------------- Face Recognition -------------------
def recognize_faces(img_path: str,
                    face_dir: str = "./face",
                    model_path: str = "./model/face_detection_yunet_2023mar.onnx",
                    rec_model: str = "./model/face_recognition_sface_2021dec.onnx") -> None:
    """
    1. 读取 img_path 并检测人脸
    2. 对 face_dir 下的每张注册照提取特征
    3. 将目标人脸与注册照逐一比对，取最高余弦相似度
    4. 弹窗画出框+姓名（或 Unknown）
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    h, w = img.shape[:2]

    # 检测器
    detector = cv2.FaceDetectorYN_create(
        model=model_path, config="", input_size=(w, h),
        score_threshold=0.8, nms_threshold=0.4, top_k=5000)
    detector.setInputSize((w, h))
    faces = detector.detect(img)[1]
    if faces is None:
        print("未检测到人脸")
        return

    # 识别器
    recognizer = cv2.FaceRecognizerSF_create(rec_model, "")

    # 注册照特征库
    regist = {}  # name -> feature
    for fp in Path(face_dir).glob("*.*"):
        name = fp.stem
        reg_img = cv2.imread(str(fp))
        if reg_img is None:
            continue
        rh, rw = reg_img.shape[:2]
        detector.setInputSize((rw, rh))
        reg_faces = detector.detect(reg_img)[1]
        if reg_faces is not None:
            # 只取第一张脸
            aligned = recognizer.alignCrop(reg_img, reg_faces[0])
            feat = recognizer.feature(aligned)
            regist[name] = feat
    detector.setInputSize((w, h))

    if not regist:
        print("注册库为空")
        return

    # 逐一比对
    for face in faces:
        aligned = recognizer.alignCrop(img, face)
        feat = recognizer.feature(aligned)
        best_score, best_name = -1, "Unknown"
        for name, reg_feat in regist.items():
            score = recognizer.match(feat, reg_feat, cv2.FaceRecognizerSF_FR_COSINE)
            if score > best_score:
                best_score, best_name = score, name
        # 画框+名字
        x, y, w_box, h_box = map(int, face[:4])

        SIM_TH = 0.3                         # 可调，OpenCV 推荐 0.3~0.4
        if best_score < SIM_TH:
            best_name = "Unknown"

        print(f"[{best_name}]  score={best_score:.3f}  box=({x},{y},{w_box},{h_box})")
        color = (0, 255, 0) if best_name != "Unknown" else (0, 0, 255)
        cv2.rectangle(img, (x, y), (x + w_box, y + h_box), color, 2)
        cv2.putText(img, f"{best_name}:{best_score:.2f}",
                    (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
    cv2.imshow("Face Recognition", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ---------- 命令行入口 ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="目标图片路径")
    parser.add_argument("-m", "--mode", choices=["detect", "recognize"],
                        default="recognize", help="detect：仅检测；recognize：识别")
    args = parser.parse_args()

    if args.mode == "detect":
        detect_faces_yunet(args.image)
    else:
        recognize_faces(args.image)
