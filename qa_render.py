"""原始 STL vs ADF 转出 STL:多角度同视角对比渲染 + 缺陷着色(P23)。

纯 numpy + Pillow,无需 OpenGL。点采样光栅化 + z-buffer + 漏点回填。
着色:红 = 真缺陷折叠(输出侧:相邻面法线点积 < -0.5,且该三角与最近原始面反向或离原始面 > 0.05mm),
      紫 = 锐棱折叠(几何上折叠,但与最近原始面同向且贴面 —— 忠实复现原始刀刃状薄缘,不是缺陷;P25),
      黄 = 针状薄片(最小高度 < 0.01mm),绿 = 非流形边邻接三角。
原始侧没有参照,折叠一律标红(原始 CAD 的刀刃状薄叶顶缘本身就是这种几何)。

为什么要分两类(P25 实测):叶轮 roundtrip 的 8,470 个"折叠三角"中 8,425 个(99.5%)与最近原始面
同向、离原始面 < 0.02mm —— 是刃口楔角 < 60° 处的忠实锐棱;只有 16 个真翻面。单看
"相邻面法线点积"会把忠实还原锐棱的网格判成劣化。

适用范围:只对**无切削的 roundtrip** 有效。切削演示的新加工面在原始 STL 里没有对应面,
离原始面 > 0.05mm 会被一律判成"真缺陷"(实测 stl_out.stl 423 个,大多是这类误判),
对它只看左右对比与 fold_tris 的相对变化。

用法:
  python qa_render.py <原始.stl> <ADF输出.stl> <输出目录> [--up y|z] [--tag 名称]
例:
  python qa_render.py stls/Turbo_Impeller.stl roundtrip.stl _qa_render --up y --tag imp
  python qa_render.py stls/桌腿固定器.stl rt_leg.stl _qa_render --up z --tag leg

输出:<tag>_<视角>.png(左原始 / 右输出),另自动对输出网格中**真缺陷折叠**最密的 3 处做特写
(没有真缺陷时退回锐棱折叠最密处)。
"""
import argparse
import os
from collections import defaultdict

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


def true_fold_mask(orig, out, fold, off_tol=0.05, cell=0.5):
    """把输出网格的几何折叠三角分成真缺陷 / 忠实锐棱(P25)。

    对每个折叠三角取质心,找原始网格上离它最近的面(含距离在最近距离 +0.005mm 内的
    并列面,以处理锐棱两侧的歧义),与其中最同向者比较法线:
      反向(点积 < -0.5)或 质心离原始面 > off_tol  → 真缺陷;否则 → 忠实锐棱。
    返回 (true_mask, sharp_mask, info)。
    """
    n_out = len(out)
    true_m = np.zeros(n_out, bool)
    sharp_m = np.zeros(n_out, bool)
    idx_f = np.where(fold)[0]
    if len(idx_f) == 0:
        return true_m, sharp_m, dict(true_fold_tris=0, sharp_fold_tris=0)
    sn = np.cross(orig[:, 1] - orig[:, 0], orig[:, 2] - orig[:, 0])
    sa = np.linalg.norm(sn, axis=1)
    sn = sn / np.maximum(sa, 1e-30)[:, None]
    amn = orig.reshape(-1, 3).min(0)
    lo = np.floor((orig.min(1) - amn) / cell).astype(np.int64)
    hi = np.floor((orig.max(1) - amn) / cell).astype(np.int64)
    buck = defaultdict(list)
    for i in range(len(orig)):
        for x in range(lo[i, 0], hi[i, 0] + 1):
            for y in range(lo[i, 1], hi[i, 1] + 1):
                for z in range(lo[i, 2], hi[i, 2] + 1):
                    buck[(x, y, z)].append(i)
    on = np.cross(out[:, 1] - out[:, 0], out[:, 2] - out[:, 0])
    on = on / np.maximum(np.linalg.norm(on, axis=1), 1e-30)[:, None]

    def seg(p, p0, p1):
        d = p1 - p0
        u = np.clip(((p - p0) * d).sum(-1) / np.maximum((d * d).sum(-1), 1e-30), 0, 1)
        return np.linalg.norm(p - (p0 + u[:, None] * d), axis=1)

    for t in idx_f:
        p = out[t].mean(0)
        g = np.floor((p - amn) / cell).astype(np.int64)
        cand = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    cand.update(buck.get((g[0] + dx, g[1] + dy, g[2] + dz), ()))
        if not cand:
            true_m[t] = True
            continue
        ci = np.fromiter(cand, np.int64)
        a, b, c = orig[ci, 0], orig[ci, 1], orig[ci, 2]
        nn = sn[ci]
        dpl = ((p - a) * nn).sum(1)
        q = p - dpl[:, None] * nn
        side = lambda u, v: (np.cross(v - u, q - u) * nn).sum(1)
        ins = (side(a, b) >= 0) & (side(b, c) >= 0) & (side(c, a) >= 0) & (sa[ci] > 1e-14)
        ds = np.where(ins, np.abs(dpl), np.minimum(np.minimum(seg(p, a, b), seg(p, b, c)), seg(p, c, a)))
        dmin = ds.min()
        agree = (nn[ds <= dmin + 0.005] @ on[t]).max()
        if agree < -0.5 or dmin > off_tol:
            true_m[t] = True
        else:
            sharp_m[t] = True
    return true_m, sharp_m, dict(true_fold_tris=int(true_m.sum()), sharp_fold_tris=int(sharp_m.sum()))


_rng = np.random.default_rng(0)


def render(T, flags, center, radius, az, el, up_axis, W=720, smax=30_000_000):
    fold, sliver, nm = flags[:3]
    sharp = flags[3] if len(flags) > 3 else np.zeros(len(T), bool)
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
    fo, sl, nmk, shp = fold[keep], sliver[keep], nm[keep], sharp[keep]
    n = np.cross(Tk[:, 1] - Tk[:, 0], Tk[:, 2] - Tk[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-30)[:, None]
    nv = np.stack([n @ right, n @ up, n @ fwd], 1)
    nn = nv * np.where(nv[:, 2] >= 0, 1, -1)[:, None]
    L1 = np.array([0.35, 0.55, 0.75]); L1 /= np.linalg.norm(L1)
    L2 = np.array([-0.7, -0.1, 0.4]); L2 /= np.linalg.norm(L2)
    shade = 0.12 + 0.68 * np.clip(nn @ L1, 0, 1) + 0.22 * np.clip(nn @ L2, 0, 1)
    base = np.tile(np.array([0.80, 0.84, 0.92]), (len(Tk), 1))
    base[sl] = [1.0, 0.85, 0.1]
    base[shp] = [0.72, 0.45, 0.95]
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
    true_b, sharp_b, info = true_fold_mask(A, B, fb[0])
    sb.update(info)
    print("原始:", sa)
    print("输出:", sb)
    print(f"输出折叠分类:真缺陷 {info['true_fold_tris']} / 忠实锐棱 {info['sharp_fold_tris']}"
          f"(共 {sb['fold_tris']};判据见 true_fold_mask)")
    fb_render = (true_b, fb[1], fb[2], sharp_b)
    P = A.reshape(-1, 3)
    ctr = (P.min(0) + P.max(0)) / 2
    rad = np.linalg.norm(P.max(0) - P.min(0)) * 0.55
    views = {"iso1": (ctr, rad, 30, 35), "iso2": (ctr, rad, 150, 20), "iso3": (ctr, rad, 260, 25),
             "top": (ctr, rad, 0, 89), "bottom": (ctr, rad, 30, -60), "side": (ctr, rad, 75, 2)}
    hot_src, hot_kind = (true_b, "true") if true_b.any() else (sharp_b, "sharp")
    for i, (p, n) in enumerate(fold_hotspots(B, hot_src)):
        views[f"hot{i + 1}_{n}{hot_kind}folds"] = (p, rad * 0.12, 40, 35)
    for vn, (c, r, az, el) in views.items():
        ims = [render(T, f, np.asarray(c, float), r, az, el, up) for T, f in ((A, fa), (B, fb_render))]
        im = Image.fromarray(np.concatenate([ims[0], np.full((ims[0].shape[0], 6, 3), 255, np.uint8), ims[1]], 1))
        d = ImageDraw.Draw(im)
        d.text((8, 8), f"ORIGINAL [{vn}]  red=fold yellow=sliver green=nonmanifold", fill=(255, 255, 0))
        d.text((ims[0].shape[1] + 14, 8),
               f"ADF OUTPUT  red=true-fold {info['true_fold_tris']}  purple=sharp-edge {info['sharp_fold_tris']}"
               f"  slivers={sb['sliver_tris']}", fill=(255, 255, 0))
        f = os.path.join(a.dir, f"{a.tag}_{vn}.png")
        im.save(f)
        print("写出", f)


if __name__ == "__main__":
    main()
