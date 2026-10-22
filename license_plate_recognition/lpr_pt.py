import warnings
warnings.filterwarnings("ignore")          # 必须早于 ultralytics/matplotlib 导入

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from ocr_rec import TextRecognizer, init_args

MODEL = "./model/y11n-pose_plate_best.pt"

def DrawPlateNum(img, text, x1, y1, y2=None):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 40)
    l, t, r, b = ImageDraw.Draw(img_pil).textbbox((0, 0), text, font=font)
    tx, ty = x1 - l, y1 - b - 6
    if ty < 0 and y2 is not None:                  # 太靠上则翻到框下方
        ty = y2 + 6
    overlay = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rectangle(
        [tx - 4, ty - 2, tx + (r - l) + 4, ty + b + 2], fill=(0, 0, 0, 160))
    img_pil = Image.alpha_composite(img_pil.convert("RGBA"), overlay).convert("RGB")
    ImageDraw.Draw(img_pil).text((tx, ty), text, font=font, fill=(0, 255, 0))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

class PlateRecognizer:
    def __init__(self, model_path=MODEL):
        self.det = YOLO(model_path)
        self.ocr = TextRecognizer(init_args().parse_args())

    def recognize(self, img):
        objs = []
        for box in self.det(img, verbose=False)[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            plate = img[y1:y2, x1:x2]
            if plate.size == 0:                     # 空框直接跳过，避免 OCR 报错
                continue
            try:
                res, _ = self.ocr([plate])
            except Exception as e:
                print(e)
                continue                            # 原来 exit() 会终止整个程序
            if res[0][0]:
                objs.append({"text": res[0][0], "score_text": res[0][1],
                             "bbox": [x1, y1, x2, y2], "score_bbox": float(box.conf)})
        return objs

def demo(args):
    img = cv2.imread(args.image_path)
    if img is None:
        print("Error: no image found:", args.image_path)
        return
    objs = PlateRecognizer().recognize(img)
    print(objs)
    for o in objs:
        x1, y1, x2, y2 = o["bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        img = DrawPlateNum(img, o["text"], x1, y1, y2)
    cv2.imshow("LPR", img)
    cv2.waitKey(0)

if __name__ == "__main__":
    demo(init_args().parse_args())
