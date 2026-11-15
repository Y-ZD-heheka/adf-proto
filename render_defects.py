"""缺陷高亮渲染:薄片(高度<0.01mm)标红,短锐边(<0.3mm 且二面角>25°)邻接三角标橙。

用法: python render_defects.py <stl> <输出.png> <中心x,y,z> <半径> [方位角 仰角]
"""
import sys
import numpy as np
from collections import defaultdict


def load_stl(path):
    with open(path, "rb") as f:
        data = f.read()
    n = int(np.frombuffer(data[80:84], dtype="<u4")[0])
    raw = np.frombuffer(data, dtype=np.uint8, count=n * 50, offset=84).reshape(n, 50)
    v = np.frombuffer(raw[:, 12:48].tobytes(), dtype="<f4").reshape(n, 3, 3)
    return v.astype(np.float64)


def classify(tris):
    """返回 (sliver_mask, short_sharp_mask):布尔数组,按三角索引。"""
    n = len(tris)
    nn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    area2 = np.linalg.norm(nn, axis=1)
    tri_n = nn / (area2[:, None] + 1e-30)
    e = np.stack(
        [
            np.linalg.norm(tris[:, 1] - tris[:, 0], axis=1),
            np.linalg.norm(tris[:, 2] - tris[:, 1], axis=1),
            np.linalg.norm(tris[:, 0] - tris[:, 2], axis=1),
        ],
        axis=1,
    )
    hmin = np.where(e.max(axis=1) > 0, area2 / np.maximum(e.max(axis=1), 1e-30), 0.0)
    sliver = hmin < 0.01

    allv = tris.reshape(-1, 3)
    uniq, inv = np.unique(allv.round(6), axis=0, return_inverse=True)
    inv = inv.reshape(-1, 3)
    edges = defaultdict(list)
    for ti, (a, b, c) in enumerate(inv):
        for x, y in ((a, b), (b, c), (c, a)):
            edges[(min(x, y), max(x, y))].append(ti)
    sharp = np.zeros(n, dtype=bool)
    n_short_sharp = 0
    for (a, b), ts in edges.items():
        if len(ts) == 2:
            d = np.dot(tri_n[ts[0]], tri_n[ts[1]])
            L = np.linalg.norm(uniq[a] - uniq[b])
            if d < np.cos(np.radians(25)) and L < 0.3:
                sharp[ts[0]] = True
                sharp[ts[1]] = True
                n_short_sharp += 1
    print(f"缺陷统计: 薄片(高度<0.01) {sliver.sum()} | 短锐边(<0.3mm,>25°) {n_short_sharp} 条,涉及三角 {sharp.sum()}")
    return sliver, sharp


def render(tris, sliver, sharp, out_png, center, radius, azim_deg=35.0, elev_deg=25.0, size=1100):
    az, el = np.radians(azim_deg), np.radians(elev_deg)
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.array([-np.sin(az), np.cos(az), 0.0])
    up = np.cross(right, -fwd)

    rel = tris - center
    x = rel @ right
    y = rel @ up
    z = rel @ (-fwd)

    keep_v = (z > 0.02) & (z < radius * 2.5) & (np.abs(x) < radius * 1.7) & (np.abs(y) < radius * 1.7)
    idx = np.where(keep_v.any(axis=1))[0]
    x, y, z = x[idx], y[idx], z[idx]
    t = tris[idx]
    sl = sliver[idx]
    sh = sharp[idx] & ~sl

    scale = size / (2.0 * radius)
    px = (x + radius) * scale
    py = (radius - y) * scale

    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    nl = np.linalg.norm(n, axis=1)
    ok = nl > 1e-12
    n[ok] /= nl[ok][:, None]
    light = np.array([0.35, 0.25, 0.9])
    light /= np.linalg.norm(light)
    lam = np.abs(n @ light)
    shade = np.clip(0.18 + 0.78 * lam, 0.0, 1.0)

    # 颜色:正常 = 蓝灰;短锐边邻接 = 橙;薄片 = 红
    colors = np.stack([shade * 0.92, shade * 0.93, shade], axis=1)
    colors[sh] = np.stack([shade[sh], shade[sh] * 0.55, shade[sh] * 0.15], axis=1)
    colors[sl] = np.stack([shade[sl], shade[sl] * 0.12, shade[sl] * 0.1], axis=1)

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
        sub_i = img[y0:y1, x0:x1]
        sub_i[upd] = colors[i]

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
    sliver, sharp = classify(tris)
    render(tris, sliver, sharp, out, center, radius, az, el)
