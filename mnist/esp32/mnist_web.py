# main.py —— ESP32 网页手写数字识别
import network, socket, gc, math, time, os
import ujson
from array import array

SSID, PWD = 'D106-108', 'kobe2016'
WFILE, HFILE = 'weights_cnn.json', 'index.html'
INV255 = 1.0 / 255.0


# ---------- 卷积 ----------
def conv(x, ih, iw, ic, oc, w, b, k, s, xf):
    oh, ow = (ih - k) // s + 1, (iw - k) // s + 1
    out = [0.0] * (oh * ow * oc)
    oi = 0
    for y in range(oh):
        for xx in range(ow):
            for o in range(oc):
                wo, acc, idx = w[o], 0.0, 0
                for ky in range(k):
                    p0 = ((y * s + ky) * iw + xx * s) * ic
                    for kx in range(k):
                        p = p0 + kx * ic
                        for ci in range(ic):
                            acc += x[p + ci] * wo[idx]
                            idx += 1
                v = acc * xf + b[o]
                out[oi] = v if v > 0 else 0.0
                oi += 1
    return out, oh, ow


# ---------- 模型 ----------
class Net:
    def __init__(self, path=WFILE):
        gc.collect()
        f = open(path)
        d = ujson.load(f)
        f.close()
        gc.collect()

        self.c = []
        for i in range(3):
            kh, kw, cin, cout = d['s%d' % i]
            fl, st = d['w%d' % i], kh * kw * cin
            self.c.append((kh, cin, cout,
                           [array('f', fl[o * st:(o + 1) * st]) for o in range(cout)],
                           array('f', d['b%d' % i])))
        self.wf = array('f', d['wf'])      # [24][10]
        self.bf = array('f', d['bf'])
        del d
        gc.collect()
        print('权重就绪，空闲 %d' % gc.mem_free())

    def predict(self, px):
        x, h, w = px, 28, 28
        for i, (k, cin, cout, wl, b) in enumerate(self.c):
            x, h, w = conv(x, h, w, cin, cout, wl, b, k, 2,
                           INV255 if i == 0 else 1.0)

        wf, bf = self.wf, self.bf
        o = [bf[k] for k in range(10)]
        for j in range(len(x)):
            xj = x[j]
            if xj:
                for k in range(10):
                    o[k] += xj * wf[j * 10 + k]

        m = max(o)
        e = [math.exp(v - m) for v in o]
        t = sum(e)
        p = [v / t for v in e]
        return p.index(max(p)), max(p), p


# ---------- WiFi ----------
def wifi():
    try:
        network.WLAN(network.AP_IF).active(False)
    except Exception:
        pass
    w = network.WLAN(network.STA_IF)
    w.active(True)
    try:
        w.config(pm=network.WLAN.PM_NONE)
    except Exception:
        pass
    if not w.isconnected():
        print('连接 WiFi...')
        w.connect(SSID, PWD)
        for _ in range(40):
            if w.isconnected():
                break
            time.sleep(0.5)
    if w.isconnected():
        ip = w.ifconfig()[0]
        print('IP:', ip)
        return ip
    print('WiFi 失败')
    return None


# ---------- HTTP ----------
def reply(cli, ctype, body):
    cli.sendall(('HTTP/1.1 200 OK\r\nContent-Type: %s\r\n'
                 'Content-Length: %d\r\nCache-Control: no-store\r\n'
                 'Connection: close\r\n\r\n' % (ctype, len(body))).encode() + body)


def reply_file(cli, path, ctype):
    try:
        size = os.stat(path)[6]
    except OSError:
        reply(cli, 'text/plain; charset=utf-8', b'404 not found')
        return
    cli.sendall(('HTTP/1.1 200 OK\r\nContent-Type: %s\r\n'
                 'Content-Length: %d\r\nCache-Control: no-store\r\n'
                 'Connection: close\r\n\r\n' % (ctype, size)).encode())
    f = open(path, 'rb')
    try:
        while True:
            d = f.read(1024)
            if not d:
                break
            cli.sendall(d)
    finally:
        f.close()


def handle(cli, net):
    buf = b''
    while b'\r\n\r\n' not in buf:
        d = cli.recv(256)
        if not d:
            return
        buf += d
        if len(buf) > 2048:
            break

    head, _, rest = buf.partition(b'\r\n\r\n')
    lines = head.decode('utf-8', 'ignore').split('\r\n')
    parts = lines[0].split(' ')
    method = parts[0] if parts else ''
    path = parts[1] if len(parts) > 1 else '/'

    # 识别接口
    if method == 'POST' and path.startswith('/rec'):
        clen = 0
        for ln in lines[1:]:
            if ln.lower().startswith('content-length:'):
                try:
                    clen = int(ln.split(':', 1)[1])
                except Exception:
                    clen = 0
                break

        body = rest
        while len(body) < clen:
            d = cli.recv(min(512, clen - len(body)))
            if not d:
                break
            body += d

        if len(body) < 784:
            obj = {'e': '像素不足(%d)' % len(body), 'd': -1, 'c': 0, 'p': [0] * 10}
        else:
            t0 = time.ticks_ms()
            d, c, p = net.predict(body)
            print('  -> %d (%.3f) %dms'
                  % (d, c, time.ticks_diff(time.ticks_ms(), t0)))
            obj = {'d': d, 'c': c, 'p': [round(v, 4) for v in p]}

        reply(cli, 'application/json; charset=utf-8', ujson.dumps(obj).encode())
        return

    if path.startswith('/favicon'):
        cli.sendall(b'HTTP/1.1 204 No Content\r\nConnection: close\r\n\r\n')
        return

    reply_file(cli, HFILE, 'text/html; charset=utf-8')


def serve(net, ip):
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('', 80))
    srv.listen(5)
    srv.settimeout(10)
    print('服务: http://%s/' % ip)

    while True:
        try:
            gc.collect()
            cli, _ = srv.accept()
        except OSError:
            continue          # 超时/中断，静默重试
        cli.settimeout(15)
        try:
            handle(cli, net)
        except Exception as e:
            print('err:', e)
        finally:
            try:
                cli.close()
            except Exception:
                pass


# ---------- 入口 ----------
def main():
    print('ESP32 手写数字识别')
    gc.collect()

    net = Net()                    # 加载权重
    gc.collect()

    ip = wifi()
    if not ip:
        ap = network.WLAN(network.AP_IF)
        ap.active(True)
        ap.config(essid='ESP32-MNIST', password='12345678')
        time.sleep(1)
        ip = ap.ifconfig()[0]
        print('AP 模式: http://%s/' % ip)

    gc.collect()
    print('空闲 %d' % gc.mem_free())
    serve(net, ip)


main()
