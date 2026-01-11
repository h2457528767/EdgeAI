import cv2
from pathlib import Path
import urllib.request
import sys

# ---------- 1. 模型路径 & 下载 ----------
MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
MODEL_PATH = Path("model/face_detection_yunet_2023mar.onnx")

if not MODEL_PATH.exists():
    print("首次使用，正在下载 YuNet 权重...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("下载完成.")

# ---------- 2. 初始化摄像头 ----------
def initialize_camera():
    """初始化USB摄像头"""
    print("=== USB摄像头初始化 ===")
    
    # USB摄像头的设备索引
    usb_camera_indices = [5, 6]
    
    for camera_index in usb_camera_indices:
        print(f"尝试打开 /dev/video{camera_index}...")
        
        try:
            cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✓ 成功打开 USB摄像头 /dev/video{camera_index}")
                    return cap, camera_index
                else:
                    cap.release()
                    
        except Exception as e:
            print(f"  错误: {e}")
            continue
    
    return None, None

cap, camera_index = initialize_camera()
if cap is None:
    print("错误: 无法打开任何摄像头")
    print("请检查:")
    print("1. USB摄像头是否已连接")
    print("2. 摄像头权限: sudo chmod 666 /dev/video5 /dev/video6")
    print("3. 用户组: sudo usermod -a -G video $USER")
    sys.exit(1)

# 读取一帧拿到分辨率
ret, frame = cap.read()
if not ret:
    cap.release()
    sys.exit("无法读取画面")
h, w = frame.shape[:2]
print(f"摄像头分辨率: {w}x{h}")

# ---------- 3. 初始化 YuNet ----------
detector = cv2.FaceDetectorYN_create(
    model=str(MODEL_PATH),
    config="",
    input_size=(w, h),
    score_threshold=0.7,
    nms_threshold=0.4,
    top_k=5000
)

# ---------- 4. 主循环 ----------
print("按 q 退出")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 检测
    detector.setInputSize((w, h))
    faces = detector.detect(frame)[1]  # shape: (N, 15)
    if faces is None:
        faces = []

    for face in faces:
        x, y, w_box, h_box = map(int, face[:4])
        score = float(face[-1])

        # 画框
        cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
        # 写置信度
        label = f"{score:.2f}"
        cv2.putText(frame, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 显示
    cv2.imshow("YuNet USB Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------- 5. 清理 ----------
cap.release()
cv2.destroyAllWindows()
