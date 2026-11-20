//! 边缘锯齿量化:锐边顶点沿边缘线的共线性残差(二阶差分)。
//!
//! 锐边顶点 = 邻接三角法线最大夹角 > 阈值的顶点;对边缘上相邻三点 (a,b,c),
//! 残差 = b 到线 ac 的距离。锯齿大 → 残差大;光滑边缘 → 残差趋近 0。
//!
//! 用法: cargo run --release --example check_edge -- <重建.stl> [角度阈值°]

use adf_proto::*;
use std::collections::HashMap;

fn load_stl(path: &str) -> Vec<[[Real; 3]; 3]> {
    let tm = stlio::load_stl(path).expect("STL 加载失败");
    tm.tris
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| "stl_out_roundtrip_impeller.stl".into());
    let thresh: Real = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(25.0);
    let tris = load_stl(&path);
    println!("[edge] {path} 三角 {}", tris.len());

    // 焊接顶点
    let q = |x: Real| (x * 1e4).round() as i64;
    let mut vid: HashMap<[i64; 3], u32> = HashMap::new();
    let mut verts: Vec<[Real; 3]> = Vec::new();
    let mut tri_vid: Vec<[u32; 3]> = Vec::with_capacity(tris.len());
    for t in &tris {
        let mut ids = [0u32; 3];
        for (k, v) in t.iter().enumerate() {
            let key = [q(v[0]), q(v[1]), q(v[2])];
            let id = *vid.entry(key).or_insert_with(|| {
                verts.push(*v);
                (verts.len() - 1) as u32
            });
            ids[k] = id;
        }
        tri_vid.push(ids);
    }
    let n = verts.len();
    println!("[edge] 去重顶点 {n}");

    // 邻接三角 + 面法线
    let mut tri_n: Vec<[Real; 3]> = Vec::with_capacity(tris.len());
    for t in &tris {
        let ab = [t[1][0] - t[0][0], t[1][1] - t[0][1], t[1][2] - t[0][2]];
        let ac = [t[2][0] - t[0][0], t[2][1] - t[0][1], t[2][2] - t[0][2]];
        let cr = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        let l = (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt();
        tri_n.push(if l > 0.0 { [cr[0] / l, cr[1] / l, cr[2] / l] } else { [0.0; 3] });
    }
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (ti, tv) in tri_vid.iter().enumerate() {
        for &v in tv {
            adj[v as usize].push(ti as u32);
        }
    }

    // 锐边顶点:邻接面法线最大夹角
    let mut edge_v = vec![false; n];
    let mut edge_count = 0usize;
    for (v, ts) in adj.iter().enumerate() {
        if ts.len() < 2 {
            continue;
        }
        let mut max_dot: Real = 2.0;
        for i in 0..ts.len() {
            for j in i + 1..ts.len() {
                let d = tri_n[ts[i] as usize][0] * tri_n[ts[j] as usize][0]
                    + tri_n[ts[i] as usize][1] * tri_n[ts[j] as usize][1]
                    + tri_n[ts[i] as usize][2] * tri_n[ts[j] as usize][2];
                max_dot = max_dot.min(d.clamp(-1.0, 1.0));
            }
        }
        let ang = max_dot.acos().to_degrees();
        if ang > thresh {
            edge_v[v] = true;
            edge_count += 1;
        }
    }
    println!("[edge] 锐边顶点(邻面夹角>{thresh}°) {edge_count} ({:.1}%)", 100.0 * edge_count as f64 / n as f64);

    // 边缘顶点的一环邻接(只统计也是锐边顶点的邻居)
    let mut vadj: Vec<Vec<u32>> = vec![Vec::new(); n];
    for tv in &tri_vid {
        for k in 0..3 {
            let (a, b) = (tv[k] as usize, tv[(k + 1) % 3] as usize);
            if edge_v[a] && edge_v[b] {
                vadj[a].push(b as u32);
                vadj[b].push(a as u32);
            }
        }
    }

    // 共线性残差:边缘顶点 b 的两个邻居 a,c(b 在中间),b 到线 ac 距离
    // 只统计近似直线段上的点对(cos(ac 方向) < -0.5,避免端点/分叉干扰)
    let mut residuals: Vec<Real> = Vec::new();
    for (b, ns) in vadj.iter().enumerate() {
        if ns.len() < 2 {
            continue;
        }
        let pb = verts[b];
        for i in 0..ns.len() {
            for j in i + 1..ns.len() {
                let (pa, pc) = (verts[ns[i] as usize], verts[ns[j] as usize]);
                let ab = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
                let cb = [pb[0] - pc[0], pb[1] - pc[1], pb[2] - pc[2]];
                let la = (ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2]).sqrt();
                let lc = (cb[0] * cb[0] + cb[1] * cb[1] + cb[2] * cb[2]).sqrt();
                if la < 1e-9 || lc < 1e-9 {
                    continue;
                }
                let cos_ab_cb = (ab[0] * cb[0] + ab[1] * cb[1] + ab[2] * cb[2]) / (la * lc);
                if cos_ab_cb > -0.5 {
                    continue; // b 不是近似在 a—c 中间(端点/分叉处)
                }
                let ac = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
                let lac = (ac[0] * ac[0] + ac[1] * ac[1] + ac[2] * ac[2]).sqrt();
                if lac < 1e-9 {
                    continue;
                }
                let ap = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
                let cr = [
                    ap[1] * ac[2] - ap[2] * ac[1],
                    ap[2] * ac[0] - ap[0] * ac[2],
                    ap[0] * ac[1] - ap[1] * ac[0],
                ];
                let dist = (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt() / lac;
                residuals.push(dist);
            }
        }
    }
    residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = residuals.len();
    if m > 0 {
        let pct = |p: usize| residuals[(p * (m - 1)) / 100];
        println!(
            "[edge] 锯齿残差 mm: p50 {:.4} | p90 {:.4} | p99 {:.4} | max {:.4} (样本 {m})",
            pct(50),
            pct(90),
            pct(99),
            residuals[m - 1]
        );
    } else {
        println!("[edge] 无有效直线段样本");
    }
}
