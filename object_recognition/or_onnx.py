#!/usr/bin/env python3
import cv2, numpy as np, onnxruntime as ort, argparse
from pathlib import Path

# ---------- 命令行参数 ----------
parser = argparse.ArgumentParser(description='YOLOv5n-ONNX 物体识别')
parser.add_argument('image', nargs='?', default='./img/bus.jpg',
                    help='待检测图片路径，缺省为 ./img/bus.jpg')
args = parser.parse_args()
IMG_PATH = args.image

MODEL_PATH = "model/yolov5n.onnx"
#IMG_PATH   = "img/bus.jpg"
# ---------- 物体类别 ------------
COCO_NAMES = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
              "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
              "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
              "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
              "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
              "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
              "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
              "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
              "hair drier","toothbrush"]
# -------- 标签颜色 ------------
def generate_distinct_colors(num_colors):
    colors = []
    
    # 使用预定义的明显不同的颜色
    base_colors = [
        (255, 0, 0),    # 红色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 蓝色
        (255, 255, 0),  # 黄色
        (255, 0, 255),  # 紫色
        (0, 255, 255),  # 青色
        (255, 128, 0),  # 橙色
        (128, 0, 255),  # 深紫色
        (0, 255, 128),  # 青绿色
        (128, 255, 0),  # 黄绿色
        (255, 0, 128),  # 粉红色
        (0, 128, 255),  # 天蓝色
    ]
    
    # 为常见类别设置固定颜色
    fixed_colors = {
        'person': (255, 0, 0),       # 红色
        'car': (0, 255, 0),          # 绿色
        'truck': (255, 255, 0),      # 黄色
        'bus': (255, 128, 0),        # 橙色
        'bicycle': (255, 0, 255),    # 紫色
        'motorcycle': (128, 0, 255), # 深紫色
        'traffic light': (0, 0, 255), # 蓝色
        'stop sign': (0, 0, 255),    # 蓝色
        'parking meter': (0, 255, 255), # 青色
    }
    
    # 生成所有类别的颜色
    for i, name in enumerate(COCO_NAMES):
        if name in fixed_colors:
            colors.append(fixed_colors[name])
        else:
            color_idx = (i - len(fixed_colors)) % len(base_colors)
            colors.append(base_colors[color_idx])
    
    return colors
# --------- 文字颜色 ------------
def get_contrast_text_color(background_color):
    """
    根据背景色返回对比度最高的文字颜色（黑色或白色）
    使用相对亮度公式计算
    """
    r, g, b = background_color
    
    # 计算相对亮度（ITU-R BT.709标准）
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    
    # 如果亮度大于128，使用黑色文字，否则使用白色文字
    return (0, 0, 0) if luminance > 128 else (255, 255, 255)

CLASS_COLORS = generate_distinct_colors(len(COCO_NAMES))

# ---------- 1. 加载模型 ----------
sess = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
in_name  = sess.get_inputs()[0].name
out_name = sess.get_outputs()[0].name

print(f"模型输入类型: {sess.get_inputs()[0].type}")
print(f"模型输出类型: {sess.get_outputs()[0].type}")

# ---------- 2. 读图 ----------
img0 = cv2.imread(IMG_PATH)
if img0 is None: 
    raise FileNotFoundError(f"无法找到图片: {IMG_PATH}")

# ---------- 3. 预处理 ----------
blob = cv2.resize(img0, (640, 640))
blob = blob.astype(np.float16) / 255.0
blob = blob.transpose(2, 0, 1)[None]

print(f"输入数据形状: {blob.shape}, 数据类型: {blob.dtype}")

# ---------- 4. 推理 ----------
outputs = sess.run([out_name], {in_name: blob})[0]
outputs = np.squeeze(outputs)

print(f"输出形状: {outputs.shape}")
print(f"输出范围: min={np.min(outputs):.6f}, max={np.max(outputs):.6f}")

if outputs.dtype == np.float16:
    outputs = outputs.astype(np.float32)

# ---------- 5. 后处理 ----------
boxes, scores, class_ids = [], [], []
valid_detections = 0

for pred in outputs:
    pred = pred.astype(np.float32)
    x, y, w, h, conf = pred[:5]
    
    if conf < 0.25:
        continue
        
    if not all(np.isfinite([x, y, w, h, conf])):
        continue
        
    if w <= 1 or h <= 1:
        continue
        
    class_probs = pred[5:]
    cls_id = np.argmax(class_probs)
    cls_conf = class_probs[cls_id]
    final_conf = conf * cls_conf
    
    # 坐标转换
    if x > 1.0 or y > 1.0:
        x_rel, y_rel, w_rel, h_rel = x/640.0, y/640.0, w/640.0, h/640.0
    else:
        x_rel, y_rel, w_rel, h_rel = x, y, w, h
    
    valid_detections += 1
    scores.append(float(final_conf))
    class_ids.append(int(cls_id))
    
    # 映射回原图坐标
    x_scale = img0.shape[1] / 640
    y_scale = img0.shape[0] / 640
    
    x1 = int((x_rel - w_rel / 2) * img0.shape[1])
    y1 = int((y_rel - h_rel / 2) * img0.shape[0])
    x2 = int((x_rel + w_rel / 2) * img0.shape[1])
    y2 = int((y_rel + h_rel / 2) * img0.shape[0])
    
    x1 = max(0, min(x1, img0.shape[1]))
    y1 = max(0, min(y1, img0.shape[0]))
    x2 = max(0, min(x2, img0.shape[1]))
    y2 = max(0, min(y2, img0.shape[0]))
    
    width = x2 - x1
    height = y2 - y1
    
    if width > 0 and height > 0:
        boxes.append([x1, y1, width, height])

print(f"总检测数: {len(outputs)}, 有效检测: {valid_detections}, 最终边界框: {len(boxes)}")

# ---------- 6. NMS和绘制结果 ----------
if boxes:
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.25, nms_threshold=0.45)
    
    if indices is not None and len(indices) > 0:
        print(f"NMS后保留 {len(indices)} 个检测结果")
        
        # 存储检测结果用于排序输出
        detection_results = []
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            cls_id = class_ids[i]
            score = scores[i]
            class_name = COCO_NAMES[cls_id]
            detection_results.append((class_name, score, [x, y, w, h]))
            
            # 获取类别颜色
            bg_color = CLASS_COLORS[cls_id]
            
            # 计算对比文字颜色
            text_color = get_contrast_text_color(bg_color)
            
            # 绘制边界框（细线）
            cv2.rectangle(img0, (x, y), (x + w, y + h), bg_color, 1)
            
            # 创建标签文本
            label = f"{class_name} {score:.2f}"
            
            # 计算文本尺寸
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # 标签背景位置
            text_x = x
            text_y = y - 5 if y - 5 > text_height else y + text_height + 5
            
            # 确保标签不超出图像上边界
            if text_y - text_height - baseline < 0:
                text_y = y + text_height + 5
            
            # 绘制标签背景（填充矩形）
            cv2.rectangle(img0, 
                         (text_x, text_y - text_height - baseline),
                         (text_x + text_width, text_y + baseline),
                         bg_color, -1)
            
            # 绘制标签文本（自适应颜色）
            cv2.putText(img0, label, (text_x, text_y), font, font_scale, 
                       text_color, thickness, cv2.LINE_AA)
        
        # 按置信度排序并输出
        detection_results.sort(key=lambda x: x[1], reverse=True)
        for class_name, score, bbox in detection_results:
            x, y, w, h = bbox
            print(f"检测到: {class_name} ({score:.3f}) at [{x}, {y}, {w}, {h}]")
            
    else:
        print("NMS后没有保留任何检测结果")
else:
    print("没有有效的检测边界框")

# ---------- 7. 显示结果 ----------
cv2.imshow("YOLOv5n_ONNX", img0)
cv2.waitKey(0)
cv2.destroyAllWindows()