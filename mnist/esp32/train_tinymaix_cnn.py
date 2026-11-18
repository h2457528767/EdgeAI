# train_cnn.py —— 复现 TinyMaix mnist 结构，导出 float JSON
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

    def imgs(fn):
        with open(fn, 'rb') as f:
            _, n, r, c = struct.unpack('>4i', f.read(16))
            return np.frombuffer(f.read(), np.uint8).reshape(n, r, c, 1)

    def labs(fn):
        with open(fn, 'rb') as f:
            _, n = struct.unpack('>2i', f.read(8))
            return np.frombuffer(f.read(), np.uint8)

    return (imgs(p['xtr']), labs(p['ytr'])), (imgs(p['xte']), labs(p['yte']))


def build():
    return keras.Sequential([
        keras.Input((28, 28, 1)),
        layers.Conv2D(1, 3, strides=2, padding='valid', use_bias=True),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(3, 3, strides=2, padding='valid', use_bias=True),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(6, 3, strides=2, padding='valid', use_bias=True),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Flatten(),          # 2*2*6 = 24
        layers.Dense(10),
        layers.Softmax(),
    ])


def fold_bn(model):
    """把 BN 折叠进前一层 Conv，推理时只需 conv+bias"""
    L = model.layers
    out = []
    for i, lay in enumerate(L):
        if isinstance(lay, layers.Conv2D):
            w, b = lay.get_weights()
            bn = L[i + 1]                       # Conv 后面紧跟 BN
            g, be, mu, var = bn.get_weights()
            std = np.sqrt(var + bn.epsilon)
            w = w * (g / std)[None, None, None, :]
            b = (b - mu) * g / std + be
            out.append((w.astype(np.float32), b.astype(np.float32)))
    return out


def save_json(model, fn='weights_cnn.json'):
    convs = fold_bn(model)
    dense = [l for l in model.layers if isinstance(l, layers.Dense)][0]
    dw, db = dense.get_weights()

    obj = {}
    for i, (w, b) in enumerate(convs):
        kh, kw, cin, cout = w.shape
        # 转成 [cout][kh][kw][cin] 扁平，板子端内层连续访问
        wt = np.transpose(w, (3, 0, 1, 2)).reshape(-1)
        obj['w%d' % i] = [round(float(v), 5) for v in wt]
        obj['b%d' % i] = [round(float(v), 5) for v in b]
        obj['s%d' % i] = [kh, kw, cin, cout]
    obj['wf'] = [round(float(v), 5) for v in dw.reshape(-1)]   # [24][10]
    obj['bf'] = [round(float(v), 5) for v in db]

    with open(fn, 'w') as f:
        json.dump(obj, f, separators=(',', ':'))

    n = sum(len(v) for k, v in obj.items() if isinstance(v, list) and k[0] in 'wb')
    print('✅ %s  %d 参数  %d 字节 (%.1f KB)'
          % (fn, n, os.path.getsize(fn), os.path.getsize(fn) / 1024))


if __name__ == '__main__':
    (xtr, ytr), (xte, yte) = load_mnist()
    xtr = xtr.astype(np.float32) / 255.0
    xte = xte.astype(np.float32) / 255.0
    print('训练集', xtr.shape, '测试集', xte.shape)

    m = build()
    m.summary()
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    m.fit(xtr, ytr, batch_size=128, epochs=8,
          validation_data=(xte, yte), verbose=1)

    loss, acc = m.evaluate(xte, yte, verbose=0)
    print('🎯 测试准确率: %.4f' % acc)
    save_json(m, 'weights_cnn.json')
