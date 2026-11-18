
ESP32 开发板上传 MicroPython 固件；

将 index.html 和 weight_cnn.json 上传至板端根目录；

Thonny IDE 运行 mnist_web.py 程序；

终端打印

```bash
ESP32 手写数字识别
权重就绪，空闲 132176
连接 WiFi...
IP: 192.168.31.109
空闲 87472
服务: http://192.168.31.109/
  -> 2 (0.986) 89ms
  -> 1 (0.918) 89ms
  -> 0 (0.994) 88ms
```

主机浏览器打开网址 `192.168.31.109`

手写数字并识别。
