//! 顶点离面偏差检查:重建网格的每个顶点到**原始**网格表面的距离。
//!
//! 用法: cargo run --release --example check_proj -- <原始.stl> <重建.stl>
//! 输出:偏差分位数(p50/p90/p99/max)与 >0.25/0.5/1mm 的顶点占比。

use adf_proto::*;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let orig = args.get(1).cloned().unwrap_or_else(|| "stls/桌腿固定器.stl".into());
    let rebuilt = args.get(2).cloned().unwrap_or_else(|| "roundtrip.stl".into());
    let tm_in = stlio::load_stl(&orig).expect("原始 STL 加载失败");
    let tm_out = stlio::load_stl(&rebuilt).expect("重建 STL 加载失败");
    println!(
        "[check] 原始 {orig} 三角 {} | 重建 {rebuilt} 三角 {}",
        tm_in.tris.len(),
        tm_out.tris.len()
    );

    let t0 = Instant::now();
    let field = meshfield::MeshField::new(tm_in.tris);
    println!("[check] 原始场构建 {:?}", t0.elapsed());

    // 收集重建网格去重顶点
    let mut verts: Vec<[Real; 3]> = Vec::new();
    let mut vmap: std::collections::HashMap<[u32; 3], usize> = std::collections::HashMap::new();
    let q = |x: Real| (x * 1e4).round() as i64 as u32;
    for t in &tm_out.tris {
        for v in t {
            let key = [q(v[0]), q(v[1]), q(v[2])];
            vmap.entry(key).or_insert_with(|| {
                verts.push(*v);
                verts.len() - 1
            });
        }
    }
    println!("[check] 重建去重顶点 {}", verts.len());

    // 逐顶点查带符号距离,取绝对值
    let t1 = Instant::now();
    let devs: Vec<Real> = verts.iter().map(|v| field.signed_distance(*v).abs()).collect();
    println!("[check] 顶点采样耗时 {:?}", t1.elapsed());
    // 注意:分位数统计用排序副本;"最差顶点"的索引必须对**未排序** devs,
    // 否则坐标和偏差错配(上一版就是在这里把 0.04mm 的点标成了 1.24mm)
    let mut sorted = devs.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let pct = |p: usize| sorted[(p * (n - 1)) / 100];
    let over = |lim: Real| sorted.iter().filter(|d| **d > lim).count();
    println!(
        "[check] 偏差 mm: p50 {:.4} | p90 {:.4} | p99 {:.4} | max {:.4}",
        pct(50),
        pct(90),
        pct(99),
        sorted[n - 1]
    );
    println!(
        "[check] 离面顶点: >0.25mm {:.2}% | >0.5mm {:.2}% | >1mm {:.2}%",
        100.0 * over(0.25) as f64 / n as f64,
        100.0 * over(0.5) as f64 / n as f64,
        100.0 * over(1.0) as f64 / n as f64,
    );
    // 最差 20 个顶点:定位尖刺聚集区
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_unstable_by(|&a, &b| devs[b].partial_cmp(&devs[a]).unwrap());
    println!("[check] 最差 20 顶点 (偏差mm | 坐标):");
    for &i in idx.iter().take(20) {
        let v = verts[i];
        println!("  {:7.4} | ({:8.3}, {:8.3}, {:8.3})", devs[i], v[0], v[1], v[2]);
    }
    // 复核第 1 名:全精度坐标 + 无符号/带符号双查询
    if !idx.is_empty() {
        let v = verts[idx[0]];
        println!("[check] 复核最差点全精度 {v:?}");
        println!(
            "[check]   unsigned={:.5} signed={:.5}",
            field.unsigned_distance(v),
            field.signed_distance(v)
        );
    }
}
