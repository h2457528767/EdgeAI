#!/usr/bin/env python3
import cv2

# 1. 读图
img = cv2.imread('./img/friends.jpg')
if img is None:
    raise FileNotFoundError('找不到 ./img/*.jpg，请确认路径！')

# 2. 加载 Haar 人脸检测模型（OpenCV 自带）
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# 3. 灰度化 + 检测
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

# 4. 画框
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# 5. 弹窗显示
cv2.imshow('Face Detection - press any key to quit', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
