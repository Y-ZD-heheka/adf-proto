//! ADF 场误差空间结构诊断:在可疑点邻域采样 ADF vs 精确场,找误差边界。
//!
//! 用法: cargo run --release --example diag_field -- <原始.stl> x y z

use adf_proto::*;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let orig = args.get(1).cloned().unwrap_or_else(|| "stls/Turbo_Impeller.stl".into());
    let p: [Real; 3] = [
        args.get(2).and_then(|s| s.parse().ok()).unwrap_or(26.9415),
        args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8.2408),
        args.get(4).and_then(|s| s.parse().ok()).unwrap_or(25.3465),
    ];
    let tm = stlio::load_stl(&orig).expect("STL 加载失败");
    let field = meshfield::MeshField::new(tm.tris);
    let (amn, amx) = (field.aabb_min, field.aabb_max);
    let extent = (amx[0] - amn[0]).max((amx[1] - amn[1]).max(amx[2] - amn[2]));
    let margin = extent * 0.06 + 1.0;
    let min = [amn[0] - margin, amn[1] - margin, amn[2] - margin];
    let size = extent + 2.0 * margin;
    let h_target = 0.2;
    let max_depth = (((size / h_target).log2().ceil()) as u8).clamp(6, 10);
    let adf = Adf::build_with(
        Composite { base: BaseField::Mesh(field), cuts: Vec::new() },
        min,
        size,
        0.002,
        0.5,
        max_depth,
    );
    let base = &adf.field.base;
    let h_min = size / (1u64 << max_depth) as Real;
    println!("[diag-field] 点 {p:?} | h_min {h_min:.4}");

    // 1) 点所在叶节点尺寸:沿 -x 方向步进,观察 eval 突变间隔
    //    (自适应网格突变处 = 叶边界)
    let sd_mesh = |q: [Real; 3]| match base {
        BaseField::Mesh(m) => m.signed_distance(q),
        _ => f64::NAN,
    };
    println!("[diag-field] 沿 -x 步进 0.05mm: eval / mesh_sd / 误差");
    for i in 0..40 {
        let t = i as f64 * 0.05;
        let q = [p[0] - t, p[1], p[2]];
        let ev = adf.eval(q) as f64;
        let mm = sd_mesh(q);
        println!("  x-{t:5.2} eval={ev:9.5} mesh={mm:9.5} err={:9.5}", ev - mm);
    }

    // 2) 邻域 7³ 采样:最大 |ADF-精确| 误差及位置
    let mut worst = (0.0f64, [0.0f64; 3]);
    for i in -3..=3i32 {
        for j in -3..=3i32 {
            for k in -3..=3i32 {
                let q = [
                    p[0] + i as f64 * 0.1,
                    p[1] + j as f64 * 0.1,
                    p[2] + k as f64 * 0.1,
                ];
                let ev = adf.eval(q) as f64;
                let mm = sd_mesh(q);
                let e = (ev - mm).abs();
                if e > worst.0 {
                    worst = (e, q);
                }
            }
        }
    }
    println!("[diag-field] ±0.3mm 邻域最大 |eval-mesh| = {:.5} @ {:?}", worst.0, worst.1);

    // 3) bake 粗格点级对比:p 周围 5×5×5 格点,bake 值 vs mesh 真值
    if let Some(bg) = &adf.bake {
        println!("[diag-field] bake n={} h={:.4} min={:?}", bg.n, bg.h, bg.min);
        let gi = |x: Real, a: usize| (((x - bg.min[a]) / bg.h).round() as i32).max(0) as usize;
        let ci = gi(p[0], 0);
        let cj = gi(p[1], 1);
        let ck = gi(p[2], 2);
        println!("[diag-field] 最近格点索引 ({ci},{cj},{ck}) | bake 格点值 vs mesh 真值:");
        for dk in -2i32..=2 {
            for dj in -2i32..=2 {
                let mut line = String::new();
                for di in -2i32..=2 {
                    let (i, j, k) =
                        ((ci as i32 + di) as usize, (cj as i32 + dj) as usize, (ck as i32 + dk) as usize);
                    let q = [bg.min[0] + i as Real * bg.h, bg.min[1] + j as Real * bg.h, bg.min[2] + k as Real * bg.h];
                    let bv = bg.data[(k * bg.n + j) * bg.n + i];
                    let mv = sd_mesh(q);
                    line.push_str(&format!("{bv:+7.3}/{mv:+7.3} "));
                }
                println!("  k{dk:+} j{dj:+}: {line}");
            }
        }
        // 4) 翻转格点判别:全局最近查询 + 伪法线判号 vs bake 值
        //    若伪法线判号给 +X ⟹ bake 走了无种子/EDT 路径(洪泛错号);
        //    若给 -X ⟹ 种子 phi 本身判号错。
        println!("[diag-field] 翻转格点判别(全局最近 + 伪法线):");
        for (di, dj, dk) in [(-2i32, 1, 0), (-1, 0, 1), (-1, 0, 2)] {
            let (i, j, k) = ((ci as i32 + di) as usize, (cj as i32 + dj) as usize, (ck as i32 + dk) as usize);
            let q = [bg.min[0] + i as Real * bg.h, bg.min[1] + j as Real * bg.h, bg.min[2] + k as Real * bg.h];
            let bv = bg.data[(k * bg.n + j) * bg.n + i];
            let mv = sd_mesh(q);
            let m = match &adf.field.base {
                BaseField::Mesh(m) => m,
                _ => continue,
            };
            let (d, t, feat, cp) = m.closest_features(q);
            let sp = m.signed_from_feature(q, t, feat, cp);
            let parity = m.parity_inside_vote(q);
            // 方向剖析:伪法线 vs p−cp
            let (n, verts_of_t) = m.debug_feature_normal(t, feat);
            let rel = [q[0] - cp[0], q[1] - cp[1], q[2] - cp[2]];
            let dot = n[0] * rel[0] + n[1] * rel[1] + n[2] * rel[2];
            println!(
                "  格({i},{j},{k}) bake={bv:+.6} mesh={mv:+.6} | d={d:.6} feat={feat} 判号={sp:+.6} 射线内={parity}"
            );
            println!(
                "    伪法线 n=[{:.4},{:.4},{:.4}] p-cp=[{:.4},{:.4},{:.4}] dot={dot:+.6} 三角顶点={verts_of_t:?}",
                n[0], n[1], n[2], rel[0], rel[1], rel[2]
            );
        }
    }
}
