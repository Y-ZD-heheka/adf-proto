"""STL 软光栅化渲染器: numpy z-buffer,看边缘碎面。

用法: python render_edge.py <stl> <输出.png> <中心x,y,z> <半径> [方位角 仰角] [尺寸]
"""
import sys
import numpy as np

def load_stl(path):
    with open(path, "rb") as f:
        data = f.read()
    n = int(np.frombuffer(data[80:84], dtype="<u4")[0])
    raw = np.frombuffer(data, dtype=np.uint8, count=n * 50, offset=84).reshape(n, 50)
    v = np.frombuffer(raw[:, 12:48].tobytes(), dtype="<f4").reshape(n, 3, 3)
    return v.astype(np.float64)

def render(tris, out_png, center, radius, azim_deg=35.0, elev_deg=25.0, size=1100):
    az, el = np.radians(azim_deg), np.radians(elev_deg)
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.array([-np.sin(az), np.cos(az), 0.0])
    up = np.cross(right, -fwd)

    rel = tris - center
    x = rel @ right
    y = rel @ up
    z = rel @ (-fwd)  # 越大越近

    keep_v = (z > 0.02) & (z < radius * 2.5) & (np.abs(x) < radius * 1.7) & (np.abs(y) < radius * 1.7)
    idx = np.where(keep_v.any(axis=1))[0]
    x, y, z = x[idx], y[idx], z[idx]
    t = tris[idx]

    scale = size / (2.0 * radius)
    px = (x + radius) * scale
    py = (radius - y) * scale

    # 法线与光照(背面不剔除:双面渲染,便于看褶皱背面)
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    nl = np.linalg.norm(n, axis=1)
    ok = nl > 1e-12
    n[ok] /= nl[ok][:, None]
    light = np.array([0.35, 0.25, 0.9])
    light /= np.linalg.norm(light)
    lam = np.abs(n @ light)  # 双面:abs
    shade = np.clip(0.18 + 0.78 * lam, 0.0, 1.0)

    zbuf = np.full((size, size), -1e18)
    img = np.ones((size, size, 3))

    for i in range(len(px)):
        xs, ys, zs = px[i], py[i], z[i]
        x0, x1 = int(max(0, xs.min())), int(min(size - 1, xs.max()) + 1)
        y0, y1 = int(max(0, ys.min())), int(min(size - 1, ys.max()) + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        d = (xs[1] - xs[0]) * (ys[2] - ys[0]) - (xs[2] - xs[0]) * (ys[1] - ys[0])
        if abs(d) < 1e-12:
            continue
        w0 = ((xs[1] - gx) * (ys[2] - gy) - (xs[2] - gx) * (ys[1] - gy)) / d
        w1 = ((xs[2] - gx) * (ys[0] - gy) - (xs[0] - gx) * (ys[2] - gy)) / d
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        zi = w0 * zs[0] + w1 * zs[1] + w2 * zs[2]
        sub_z = zbuf[y0:y1, x0:x1]
        upd = inside & (zi > sub_z)
        if not upd.any():
            continue
        sub_z[upd] = zi[upd]
        c = shade[i]
        sub_i = img[y0:y1, x0:x1]
        sub_i[upd] = [c * 0.92, c * 0.93, c]

    out = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    import struct, zlib

    def write_png(path, arr):
        h, w = arr.shape[:2]
        raw = b"".join(b"\x00" + arr[r].tobytes() for r in range(h))

        def chunk(tag, data):
            c = struct.pack(">I", len(data)) + tag + data
            return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

        png = b"\x89PNG\r\n\x1a\n"
        png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        png += chunk(b"IDAT", zlib.compress(raw, 6))
        png += chunk(b"IEND", b"")
        with open(path, "wb") as f:
            f.write(png)

    write_png(out_png, out)
    print(f"渲染 {len(px)} 三角 -> {out_png}")

if __name__ == "__main__":
    path = sys.argv[1]
    out = sys.argv[2]
    center = np.array([float(v) for v in sys.argv[3].split(",")])
    radius = float(sys.argv[4])
    az = float(sys.argv[5]) if len(sys.argv) > 5 else 35.0
    el = float(sys.argv[6]) if len(sys.argv) > 6 else 25.0
    tris = load_stl(path)
    print(f"载入 {len(tris)} 三角")
    render(tris, out, center, radius, az, el)
