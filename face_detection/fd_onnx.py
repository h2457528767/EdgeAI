import cv2
import numpy as np

def detect_faces_yunet(image_path: str,
                       model_path: str = "model/face_detection_yunet_2023mar.onnx",
                       conf_threshold: float = 0.8) -> None:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    h0, w0 = img.shape[:2]

    # 1. 对齐到 32 倍数
    def align32(x): return (x + 31) // 32 * 32
    w_align, h_align = align32(w0), align32(h0)

    # 2. letterbox 缩放（保持比例，边缘灰条）
    scale = min(w_align / w0, h_align / h0)
    new_w, new_h = int(w0 * scale), int(h0 * scale)
    pad_x, pad_y = (w_align - new_w) // 2, (h_align - new_h) // 2
    letter = 128 * np.ones((h_align, w_align, 3), dtype=np.uint8)
    letter[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = \
        cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 3. 初始化 & 检测
    detector = cv2.FaceDetectorYN_create(
        model=model_path,
        config="",
        input_size=(w_align, h_align),
        score_threshold=conf_threshold,
        nms_threshold=0.4,
        top_k=5000
    )
    faces = detector.detect(letter)[1]
    if faces is None:
        faces = []

    # 4. 把框映射回原图坐标
    for face in faces:
        x, y, w, h, *_ = map(int, face[:4])
        score = face[-1]
        # 去掉 letterbox 偏移并反缩放
        x = int((x - pad_x) / scale)
        y = int((y - pad_y) / scale)
        w = int(w / scale)
        h = int(h / scale)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{score:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x, y - label_size[1] - 4),
                      (x + label_size[0], y), (0, 255, 0), -1)
        cv2.putText(img, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # 5. 一次性显示
    max_h = 500
    show_img = cv2.resize(img, (int(w0 * max_h / h0), max_h)) if h0 > max_h else img
    print("[INFO] press 'q' or ESC in the window to quit")
    while True:
        cv2.imshow("YuNet", show_img)
        k = cv2.waitKey(200) & 0xFF
        if k == ord('q') or k == 27:          # 27 = ESC
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_faces_yunet("img/friends.jpg")