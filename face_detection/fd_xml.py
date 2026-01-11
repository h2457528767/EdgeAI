import cv2
from pathlib import Path

def detect_faces(image_path: str,
                 max_side: int = 1280,
                 padding: float = 0.05) -> None:
    """
    零切割人脸检测
    :param image_path:  原图路径
    :param max_side:    检测前最长边上限（越大越慢，越小越可能漏）
    :param padding:     矩形向外扩的边距比例（0.05 = 5 %）
    """
    # --------- 1. 读图 ------------
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    h0, w0 = img.shape[:2]

    # ------------ 2. 等比例缩放 ------------
    scale = min(1.0, max_side / max(h0, w0))
    if scale < 1.0:
        img_small = cv2.resize(img, (int(w0 * scale), int(h0 * scale)),
                               interpolation=cv2.INTER_LINEAR)
    else:
        img_small = img
    h1, w1 = img_small.shape[:2]

    # -------- 3. 灰度 + 检测 -----------------
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    cascade_path = "model/haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(str(cascade_path))
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(60, 60)
    )

    # ----------- 4. 映射回原图 + 边缘修正 --------------
    for (x, y, w, h) in faces:
        # 映射回原图坐标
        x = int(x / scale)
        y = int(y / scale)
        w = int(w / scale)
        h = int(h / scale)

        # 外扩边距
        dw = int(w * padding)
        dh = int(h * padding)
        x = max(0, x - dw)
        y = max(0, y - dh)
        x2 = min(w0, x + w + 2 * dw)
        y2 = min(h0, y + h + 2 * dh)

        cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)

    # -------------- 5. 显示 ----------------
    window_name = "Face Detection"
    max_h = 500                        # 高度不超过 500 px
    if h0 > max_h:
        scale_show = max_h / h0
        new_w = int(w0 * scale_show)
        show_img = cv2.resize(img, (new_w, max_h))
    else:
        show_img = img                  # 原图
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print("[INFO] press 'q' or ESC in the window to quit")
    while True:
        cv2.imshow("Face Detection", show_img)
        k = cv2.waitKey(200) & 0xFF
        if k == ord('q') or k == 27:          # 27 = ESC
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_faces(r"img/Three.jpg")
