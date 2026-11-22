"""原始 STL vs ADF 转出 STL:多角度同视角对比渲染 + 缺陷着色(P23)。

纯 numpy + Pillow,无需 OpenGL。点采样光栅化 + z-buffer + 漏点回填。
着色:红 = 真折叠(相邻面法线点积 < -0.5,即翻折/面内翻转三角),
      黄 = 针状薄片(最小高度 < 0.01mm),绿 = 非流形边邻接三角。
注意:原始 CAD 的刀刃状薄叶顶缘本身就会被标红,判读时以左右对比为准。

用法:
  python qa_render.py <原始.stl> <ADF输出.stl> <输出目录> [--up y|z] [--tag 名称]
例:
  python qa_render.py stls/Turbo_Impeller.stl roundtrip.stl _qa_render --up y --tag imp
  python qa_render.py stls/桌腿固定器.stl rt_leg.stl _qa_render --up z --tag leg

输出:<tag>_<视角>.png(左原始 / 右输出),另自动对输出网格中折叠最密的 3 处做特写。
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw


def load_stl(path):
    b = open(path, "rb").read()
    n = int.from_bytes(b[80:84], "little")
    dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    return np.frombuffer(b, dtype=dt, count=n, offset=84)["v"].astype(np.float64)


def classify(T):
    V, inv = np.unique(T.reshape(-1, 3), axis=0, return_inverse=True)
    F = inv.reshape(-1, 3)
    n = len(F)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    fid = np.tile(np.arange(n), 3)
    key = np.sort(e, 1)
    k = key[:, 0] * len(V) + key[:, 1]
    _, ii, cnt = np.unique(k, return_inverse=True, return_counts=True)
    o = np.argsort(ii, kind="stable")
    si, sf = ii[o], fid[o]
    m2 = cnt[si] == 2
    pairs = sf[m2].reshape(-1, 2)
    cr = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    ar = np.linalg.norm(cr, axis=1)
    nr = cr / np.maximum(ar, 1e-30)[:, None]
    d = (nr[pairs[:, 0]] * nr[pairs[:, 1]]).sum(1)
    fold = np.zeros(n, bool)
    fp = pairs[d < -0.5]
    fold[fp[:, 0]] = True
    fold[fp[:, 1]] = True
    nonman = np.zeros(n, bool)
    nonman[sf[cnt[si] > 2]] = True
    L = np.stack([np.linalg.norm(T[:, 1] - T[:, 0], axis=1),
                  np.linalg.norm(T[:, 2] - T[:, 1], axis=1),
                  np.linalg.norm(T[:, 0] - T[:, 2], axis=1)], 1)
    sliver = (ar / np.maximum(L.max(1), 1e-30) < 0.01) & ~fold
    openE = int((cnt == 1).sum())
    stats = dict(tris=n, open_edges=openE, nonmanifold_edges=int((cnt > 2).sum()),
                 fold_tris=int(fold.sum()), sliver_tris=int(sliver.sum()))
    return (fold, sliver, nonman), stats


_rng = np.random.default_rng(0)


def render(T, flags, center, radius, az, el, up_axis, W=720, smax=30_000_000):
    fold, sliver, nm = flags
    az, el = np.radians(az), np.radians(el)
    U = np.eye(3)[up_axis]
    a1 = np.eye(3)[(up_axis + 1) % 3]
    a2 = np.eye(3)[(up_axis + 2) % 3]
    fwd = np.cos(el) * (np.cos(az) * a1 + np.sin(az) * a2) + np.sin(el) * U
    upw = U if abs(np.sin(el)) < 0.97 else a1
    right = np.cross(upw, fwd)
    right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    rel = T - center
    X, Y, Z = rel @ right, rel @ up, -(rel @ fwd)
    s = W / (2 * radius)
    px, py = (X + radius) * s, (radius - Y) * s
    keep = (px.max(1) > 0) & (px.min(1) < W) & (py.max(1) > 0) & (py.min(1) < W)
    px, py, Z, Tk = px[keep], py[keep], Z[keep], T[keep]
    fo, sl, nmk = fold[keep], sliver[keep], nm[keep]
    n = np.cross(Tk[:, 1] - Tk[:, 0], Tk[:, 2] - Tk[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-30)[:, None]
    nv = np.stack([n @ right, n @ up, n @ fwd], 1)
    nn = nv * np.where(nv[:, 2] >= 0, 1, -1)[:, None]
    L1 = np.array([0.35, 0.55, 0.75]); L1 /= np.linalg.norm(L1)
    L2 = np.array([-0.7, -0.1, 0.4]); L2 /= np.linalg.norm(L2)
    shade = 0.12 + 0.68 * np.clip(nn @ L1, 0, 1) + 0.22 * np.clip(nn @ L2, 0, 1)
    base = np.tile(np.array([0.80, 0.84, 0.92]), (len(Tk), 1))
    base[sl] = [1.0, 0.85, 0.1]
    base[fo] = [1.0, 0.1, 0.1]
    base[nmk] = [0.2, 1.0, 0.3]
    rgb = np.clip(shade[:, None] * base, 0, 1)
    A = 0.5 * np.abs((px[:, 1] - px[:, 0]) * (py[:, 2] - py[:, 0]) - (px[:, 2] - px[:, 0]) * (py[:, 1] - py[:, 0]))
    ns = np.clip(np.ceil(A * 5).astype(int) + 4, 4, 20000)
    if ns.sum() > smax:
        ns = np.maximum(4, (ns * smax / ns.sum()).astype(int))
    idx = np.repeat(np.arange(len(ns)), ns)
    r1, r2 = _rng.random(len(idx)), _rng.random(len(idx))
    sq = np.sqrt(r1)
    w0, w1, w2 = 1 - sq, sq * (1 - r2), sq * r2
    sx = w0 * px[idx, 0] + w1 * px[idx, 1] + w2 * px[idx, 2]
    sy = w0 * py[idx, 0] + w1 * py[idx, 1] + w2 * py[idx, 2]
    sz = w0 * Z[idx, 0] + w1 * Z[idx, 1] + w2 * Z[idx, 2]
    sz = sz + np.where(fo[idx] | sl[idx] | nmk[idx], -radius * 0.004, 0.0)  # 缺陷面压在共面面之上
    ix, iy = sx.astype(np.int64), sy.astype(np.int64)
    ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < W)
    pix, z, ti = iy[ok] * W + ix[ok], sz[ok], idx[ok]
    o = np.lexsort((z, pix))
    pix, ti, z = pix[o], ti[o], z[o]
    first = np.ones(len(pix), bool)
    first[1:] = pix[1:] != pix[:-1]
    img = np.tile(np.array([0.16, 0.17, 0.2]), (W * W, 1))
    dep = np.full(W * W, np.inf)
    img[pix[first]] = rgb[ti[first]]
    dep[pix[first]] = z[first]
    img, dep = img.reshape(W, W, 3), dep.reshape(W, W)
    tol = radius * 0.01
    for _ in range(2):  # 漏点回填:比 3x3 邻域明显更远的像素取邻域值
        best, bimg = dep.copy(), img.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                sd = np.roll(np.roll(dep, dy, 0), dx, 1)
                si = np.roll(np.roll(img, dy, 0), dx, 1)
                m = sd < best - tol
                best = np.where(m, sd, best)
                bimg = np.where(m[..., None], si, bimg)
        img, dep = bimg, best
    return (img * 255).astype(np.uint8)


def fold_hotspots(T, fold, k=3, cell=4.0):
    if not fold.any():
        return []
    c = T[fold].mean(1)
    g = np.floor(c / cell).astype(np.int64)
    keys, inv, cnt = np.unique(g, axis=0, return_inverse=True, return_counts=True)
    out = []
    for j in np.argsort(-cnt)[:k]:
        out.append((c[inv == j].mean(0), int(cnt[j])))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("orig")
    ap.add_argument("out")
    ap.add_argument("dir")
    ap.add_argument("--up", default="y", choices=["x", "y", "z"])
    ap.add_argument("--tag", default="cmp")
    a = ap.parse_args(argv)
    os.makedirs(a.dir, exist_ok=True)
    up = "xyz".index(a.up)
    A, B = load_stl(a.orig), load_stl(a.out)
    fa, sa = classify(A)
    fb, sb = classify(B)
    print("原始:", sa)
    print("输出:", sb)
    P = A.reshape(-1, 3)
    ctr = (P.min(0) + P.max(0)) / 2
    rad = np.linalg.norm(P.max(0) - P.min(0)) * 0.55
    views = {"iso1": (ctr, rad, 30, 35), "iso2": (ctr, rad, 150, 20), "iso3": (ctr, rad, 260, 25),
             "top": (ctr, rad, 0, 89), "bottom": (ctr, rad, 30, -60), "side": (ctr, rad, 75, 2)}
    for i, (p, n) in enumerate(fold_hotspots(B, fb[0])):
        views[f"hot{i + 1}_{n}folds"] = (p, rad * 0.12, 40, 35)
    for vn, (c, r, az, el) in views.items():
        ims = [render(T, f, np.asarray(c, float), r, az, el, up) for T, f in ((A, fa), (B, fb))]
        im = Image.fromarray(np.concatenate([ims[0], np.full((ims[0].shape[0], 6, 3), 255, np.uint8), ims[1]], 1))
        d = ImageDraw.Draw(im)
        d.text((8, 8), f"ORIGINAL [{vn}]  red=fold yellow=sliver green=nonmanifold", fill=(255, 255, 0))
        d.text((ims[0].shape[1] + 14, 8), f"ADF OUTPUT  folds={sb['fold_tris']} slivers={sb['sliver_tris']}", fill=(255, 255, 0))
        f = os.path.join(a.dir, f"{a.tag}_{vn}.png")
        im.save(f)
        print("写出", f)


if __name__ == "__main__":
    main()
