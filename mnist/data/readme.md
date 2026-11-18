
## MNIST 数据集

需解压使用。

```python
import numpy as np, json, os, struct
from tensorflow import keras
from tensorflow.keras import layers


def load_mnist(d='data'):
    names = {'xtr': 'train-images.idx3-ubyte', 'ytr': 'train-labels.idx1-ubyte',
             'xte': 't10k-images.idx3-ubyte', 'yte': 't10k-labels.idx1-ubyte'}
    p = {k: os.path.join(d, v) for k, v in names.items()}
    if not all(os.path.exists(v) for v in p.values()):
        print('缺少 MNIST 文件:', [v for v in p.values() if not os.path.exists(v)])
        raise SystemExit(1)
```

原版下载

```python
import urllib.request
import os

data_folder = os.path.join(os.getcwd(), 'data')
os.makedirs(data_folder, exist_ok=True)

urllib.request.urlretrieve('https://azureopendatastorage.blob.core.windows.net/mnist/train-images-idx3-ubyte.gz',
                           filename=os.path.join(data_folder, 'train-images.gz'))
urllib.request.urlretrieve('https://azureopendatastorage.blob.core.windows.net/mnist/train-labels-idx1-ubyte.gz',
                           filename=os.path.join(data_folder, 'train-labels.gz'))
urllib.request.urlretrieve('https://azureopendatastorage.blob.core.windows.net/mnist/t10k-images-idx3-ubyte.gz',
                           filename=os.path.join(data_folder, 'test-images.gz'))
urllib.request.urlretrieve('https://azureopendatastorage.blob.core.windows.net/mnist/t10k-labels-idx1-ubyte.gz',
                           filename=os.path.join(data_folder, 'test-labels.gz'))
```
