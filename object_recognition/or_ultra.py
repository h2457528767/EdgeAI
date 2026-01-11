#!/usr/bin/env python3
import cv2
import numpy as np
from ultralytics import YOLO

# ---------- 1. 加载模型 ----------
model = YOLO('./model/yolov8n.pt') # 加载模型
names = model.names          # 动态获取标签

# ---------- 2. 颜色方案（基于 model.names） ----------
def generate_distinct_colors(name_dict):
    """name_dict: {0:'person', 1:'bicycle', ...}"""
    base_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
        (0, 255, 128), (128, 255, 0), (255, 0, 128), (0, 128, 255)
    ]
    fixed_colors = {
        'person': (255, 0, 0),        'car': (0, 255, 0),
        'truck': (255, 255, 0),       'bus': (255, 128, 0),
        'bicycle': (255, 0, 255),     'motorcycle': (128, 0, 255),
        'traffic light': (0, 0, 255), 'stop sign': (0, 0, 255),
        'parking meter': (0, 255, 255)
    }
    colors = []
    for idx in range(len(name_dict)):
        name = name_dict[idx]
        if name in fixed_colors:
            colors.append(fixed_colors[name])
        else:
            color_idx = (idx - len(fixed_colors)) % len(base_colors)
            colors.append(base_colors[color_idx])
    return colors

def get_contrast_text_color(bg):
    r, g, b = bg
    return (0, 0, 0) if 0.2126*r + 0.7152*g + 0.0722*b > 128 else (255, 255, 255)

# 生成颜色表
colors = generate_distinct_colors(names)
txt_colors = [get_contrast_text_color(c) for c in colors]

# ---------- 3. 读图 ----------
img_path = './img/desktop.jpg'
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(img_path)

# ---------- 4. 推理 ----------
results = model(img, conf=0.3, iou=0.45)

# ---------- 5. 绘制 ----------
font_scale = 0.5
thickness = 1
for r in results:
    for b in r.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        conf = float(b.conf[0])
        cls  = int(b.cls[0])
        label = f'{names[cls]} {conf:.2f}'

        color = colors[cls]
        # 半透明填充
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)

        # 标签背景
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(img, (x1, y1-th-8), (x1+tw, y1), color, -1)
        # 文字（自动黑/白）
        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, txt_colors[cls], thickness)

# ---------- 6. 等比例放大显示 ----------
disp_h = 480
h, w = img.shape[:2]
img_show = cv2.resize(img, (int(w * disp_h / h), disp_h),
                      interpolation=cv2.INTER_LINEAR)

cv2.imshow('Colorful YOLO Detection', img_show)
cv2.waitKey(0)
cv2.destroyAllWindows()
