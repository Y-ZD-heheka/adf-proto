"""STL 折叠/拓扑快速体检:按边邻接统计 180° 翻折对、开放边、非流形边。

用法: python fold_count.py <a.stl> [b.stl ...]
判据与 repair_folds 对齐:相邻两面法线点积 < 0.5 记疑似折叠,< -0.5 记真折叠。
"""
import struct
import sys
from collections import defaultdict

import numpy as np


def load_stl(path):
    with open(path, "rb") as f:
        head = f.read(84)
        n = struct.unpack("<I", head[80:84])[0]
        raw = f.read(n * 50)
    rec = np.frombuffer(raw, dtype=np.uint8).reshape(n, 50)
    tri = rec[:, 12:48].copy().view("<f4").reshape(n, 3, 3).astype(np.float64)
    return tri


def weld(tri, q=1e4):
    flat = tri.reshape(-1, 3)
    key = np.round(flat * q).astype(np.int64)
    _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    verts = flat[idx]
    return verts, inv.reshape(-1, 3)


def stats(path):
    tri = load_stl(path)
    if tri.shape[0] == 0:
        print(f"[{path}] 空")
        return
    verts, tf = weld(tri)
    n = tri.shape[0]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(nrm, axis=1)
    nrm = np.divide(nrm, ln[:, None], out=np.zeros_like(nrm), where=ln[:, None] > 0)

    e = np.concatenate(
        [tf[:, [0, 1]], tf[:, [1, 2]], tf[:, [2, 0]]], axis=0
    )
    e.sort(axis=1)
    owner = np.tile(np.arange(n), 3)
    order = np.lexsort((e[:, 1], e[:, 0]))
    e = e[order]
    owner = owner[order]
    same = np.all(e[1:] == e[:-1], axis=1)
    start = np.flatnonzero(
        np.concatenate([[True], ~same])
    )
    cnt = np.diff(np.concatenate([start, [len(e)]]))
    open_e = int((cnt == 1).sum())
    nonman = int((cnt > 2).sum())
    # 恰好 2 面的边:法线点积
    two = np.flatnonzero(cnt == 2)
    i0 = start[two]
    a = owner[i0]
    b = owner[i0 + 1]
    d = (nrm[a] * nrm[b]).sum(axis=1)
    fold = int((d < -0.5).sum())
    sharp = int((d < 0.5).sum())
    deg = int((ln < 1e-12).sum())
    print(
        f"[{path}] 三角 {n} | 顶点 {len(verts)} | 开放边 {open_e} | 非流形边 {nonman}"
        f" | 真折叠 {fold} | >60° 邻接 {sharp} | 零面积 {deg}"
    )


if __name__ == "__main__":
    for p in sys.argv[1:]:
        stats(p)
