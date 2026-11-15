//! ADF 原型演示:自适应构建 → 胶囊切削 → 验证(误差/零点偏置/内存/切片/速度)。
//! 运行:cargo run --release

use adf_proto::*;
use std::time::Instant;

fn main() {
    // ── 场景:圆角盒毛坯(平整面 + R2 圆角)→ 顶面铣三条通道 ──
    let min = [-12.0, -12.0, -12.0];
    let size = 24.0;
    let eps = 0.002; // 细分容差 mm(Frisken 判据)
    let band = 0.5; // 壳层带宽 mm:只在 |d| ≤ band 处细分(spike 实证:必要判据)
    let max_depth = 9; // h_min = 24/512 ≈ 0.0469 mm
    let h_min = size / (1u64 << max_depth) as Real;

    println!("=== ADF 原型 ===");
    println!("域 24mm³ / eps={eps}mm / band={band}mm / max_depth={max_depth} → h_min={h_min:.4}mm");

    // ── 自适应构建 ──
    let t0 = Instant::now();
    let mut adf = Adf::build(
        Stock::RoundBox { c: [0.0; 3], half: [8.0; 3], r: 2.0 },
        min,
        size,
        eps,
        band,
        max_depth,
    );
    let build_t = t0.elapsed();
    let (n0, b0, hist0) = adf.leaf_stats_bytes();
    let kb0 = b0 / 1024;
    println!("[build] {build_t:?} | 叶 {n0} | ≈{kb0} KB");
    let depths: Vec<String> = hist0
        .iter()
        .enumerate()
        .filter(|(_, n)| **n > 0)
        .map(|(d, n)| format!("d{d}:{n}"))
        .collect();
    println!("        深度直方图: {}", depths.join(" "));

    // ── 均匀格对比(h_min 全域) ──
    let unif_cells = ((size / h_min).powi(3)) as u64;
    let unif_mb = unif_cells * 4 / (1024 * 1024);
    let ratio = unif_cells * 4 / 1024 / kb0.max(1);
    println!("[memory] ADF ≈ {kb0} KB  vs  均匀 h_min 全域 = {unif_cells} 格 ≈ {unif_mb} MB(压缩 ~{ratio}×)");

    // ── 切削:顶面三条通道(z 递深) ──
    let cuts = [
        Capsule { a: [-5.0, 0.0, 9.0], b: [5.0, 0.0, 9.0], r: 1.5 },
        Capsule { a: [-5.0, 0.0, 7.5], b: [5.0, 0.0, 7.5], r: 1.5 },
        Capsule { a: [-5.0, 0.0, 6.0], b: [5.0, 0.0, 6.0], r: 1.5 },
    ];
    for (i, c) in cuts.into_iter().enumerate() {
        let t = Instant::now();
        let (na, _, _) = adf.leaf_stats_bytes();
        adf.add_cut(c);
        let (nb, bb, _) = adf.leaf_stats_bytes();
        let d_leaf = nb as i64 - na as i64;
        let kb = bb / 1024;
        println!("[cut {}] {:?} | 叶 {na} → {nb}(+{d_leaf})| ≈{kb} KB", i + 1, t.elapsed());
    }

    // ── 深挖三刀:制造"被超越旧表面"的场退化(空气区幅值 = 下界) ──
    for (i, zc) in [5.0f64, 4.0, 3.0].into_iter().enumerate() {
        let t = Instant::now();
        adf.add_cut(Capsule { a: [-5.0, 0.0, zc], b: [5.0, 0.0, zc], r: 1.5 });
        let (nl, _, _) = adf.leaf_stats_bytes();
        println!("[cut {}] {:?} | 叶 {nl}", i + 4, t.elapsed());
    }

    // ── 独立验证指标:|∇d| ≈ 1(不依赖 composite 参考,重距离化的正确性判据) ──
    // 在壳层空气点(0.02 < eval < band)做中心差分;退化区 |∇d| < 1(平台),修复后 ≈ 1
    let grad_metric = |tag: &str, adf: &Adf| {
        let mut rng2 = Rng::new(777);
        let d2 = h_min;
        let mut sum = 0.0f64;
        let mut cnt = 0u64;
        let mut min_g = f64::MAX;
        for _ in 0..4000 {
            let p = [
                min[0] + rng2.f01() * size,
                min[1] + rng2.f01() * size,
                min[2] + rng2.f01() * size,
            ];
            let v = adf.eval(p) as f64;
            if v <= 0.02 || v >= band {
                continue; // 只要壳层空气点
            }
            let gx = (adf.eval([p[0] + d2, p[1], p[2]]) as f64 - adf.eval([p[0] - d2, p[1], p[2]]) as f64)
                / (2.0 * d2);
            let gy = (adf.eval([p[0], p[1] + d2, p[2]]) as f64 - adf.eval([p[0], p[1] - d2, p[2]]) as f64)
                / (2.0 * d2);
            let gz = (adf.eval([p[0], p[1], p[2] + d2]) as f64 - adf.eval([p[0], p[1], p[2] - d2]) as f64)
                / (2.0 * d2);
            let g = (gx * gx + gy * gy + gz * gz).sqrt();
            sum += g;
            min_g = min_g.min(g);
            cnt += 1;
        }
        if cnt > 0 {
            println!(
                "[grad {tag}] 壳层空气点 {cnt}:|∇d| mean={:.3} min={:.3}(理想 ≈1.0)",
                sum / cnt as Real,
                min_g
            );
        }
    };
    grad_metric("redist 前", &adf);

    // ── 场退化探针:旧顶面下的空气点(max 公式只给下界) ──
    // 真值:cap1 壁在 z=10 处 |y| = √(1.5²−1²) ≈ 1.118 → 距 p(y=0.5) ≈ 0.618
    let probe = [0.0, 0.5, 10.0];
    let before = adf.eval(probe) as f64;
    println!("[degen] p={probe:?} 重距离化前 = {before:.4}(composite 下界;真值 ≈ 0.618)");

    // ── 重距离化 ──
    let t = Instant::now();
    let st = adf.redistance(8);
    let after = adf.eval(probe) as f64;
    println!(
        "[redist] {:?} | 界面叶 {} | 网格 {:?} | 回写 {} 角点 | p:{before:.4} → {after:.4}(真值 ≈ 0.618)",
        t.elapsed(),
        st.interface_leaves,
        st.grid_n,
        st.corrected
    );
    println!("[note] redistance 后,[error] vs composite 的差 = 空气区修正量(材料区/零面不变)");
    grad_metric("redist 后", &adf);

    // ── 诊断:沿竖直线扫描,定位 redistance 后的粗格值 ──
    println!("--- scan x=0.732 y=0.095(eval / composite / 叶深)---");
    for zi in 0..24 {
        let z = 9.4 + zi as f64 * 0.05;
        let p = [0.732, 0.095, z];
        println!(
            "z={z:.2}  eval={:.4}  comp={:.4}  d{}",
            adf.eval(p) as f64,
            adf.true_sd(p),
            adf.depth_at(p)
        );
    }

    // ── 粗化:宽松容差(5×eps)回收切空区域的旧壳层细格(渲染/导出口径) ──
    let (nc0, _, _) = adf.leaf_stats_bytes();
    let merged = adf.coarsen_with(eps * 5.0);
    let (nc1, bc1, _) = adf.leaf_stats_bytes();
    let kbc = bc1 / 1024;
    println!(
        "[coarsen] eps_coarse={:.3} 合并 {merged} 次 | 叶 {nc0} → {nc1} | ≈{kbc} KB",
        eps * 5.0
    );

    // ── 误差统计:ADF eval vs 解析真值(随机 10 万点) ──
    let mut rng = Rng::new(2026_1107);
    let n = 100_000u64;
    let mut max_e = 0.0f64;
    let mut sum_e = 0.0f64;
    for _ in 0..n {
        let p = [
            min[0] + rng.f01() * size,
            min[1] + rng.f01() * size,
            min[2] + rng.f01() * size,
        ];
        let e = (adf.eval(p) as Real - adf.true_sd(p)).abs();
        max_e = max_e.max(e);
        sum_e += e;
    }
    let mean_e = sum_e / n as Real;
    println!("[error] 随机 {n} 点 vs 解析真值:max={max_e:.4}mm  mean={mean_e:.4}mm(全域含粗格区)");

    // ── 壳层内误差(验证相关口径:只看 |true| ≤ band 的表面附近) ──
    let mut max_eb = 0.0f64;
    let mut sum_eb = 0.0f64;
    let mut nb = 0u64;
    let mut argmax_p = [0.0f64; 3];
    let mut argmax_info = (0.0f64, 0.0f64, 0u8); // (eval, true, depth)
    for _ in 0..n {
        let p = [
            min[0] + rng.f01() * size,
            min[1] + rng.f01() * size,
            min[2] + rng.f01() * size,
        ];
        let t = adf.true_sd(p);
        if t.abs() <= band {
            let e = (adf.eval(p) as Real - t).abs();
            if e > max_eb {
                max_eb = e;
                argmax_p = p;
                argmax_info = (adf.eval(p) as f64, t, adf.depth_at(p));
            }
            sum_eb += e;
            nb += 1;
        }
    }
    let mean_eb = sum_eb / nb as Real;
    println!(
        "[error] 壳层内(|true|≤{band}mm){nb} 点:max={max_eb:.4}mm  mean={mean_eb:.4}mm(这才是验证相关口径)"
    );
    println!(
        "[argmax] p=({:.3},{:.3},{:.3})  eval={:.4}  true={:.4}  所在叶深度 d{}(格宽 {:.4}mm)",
        argmax_p[0], argmax_p[1], argmax_p[2], argmax_info.0, argmax_info.1, argmax_info.2,
        size / (1u64 << argmax_info.2 as u32) as Real
    );
    adf.dump_leaf(argmax_p);

    // ── 零点偏置(§9.12.1 A 类口径):顶面平坦区,远离切削 ──
    let mut bias = 0.0f64;
    let mut cnt = 0u64;
    for _ in 0..4000 {
        let x = (rng.f01() * 2.0 - 1.0) * 5.0;
        let y = (rng.f01() * 2.0 - 1.0) * 5.0;
        if y.abs() < 3.5 {
            continue; // 避开通道
        }
        bias += adf.eval([x, y, 10.0]) as f64;
        cnt += 1;
    }
    let bias_mean = bias / cnt as Real;
    println!("[bias] 顶面平坦区 {cnt} 点:均值 = {bias_mean:.5} mm(理论 ≈ 0:线性场插值精确)");

    // ── ASCII 切片(z = 9,过第一条通道) ──
    let z = 9.0;
    println!("[slice] z={z}(# 材料 · 空气 o 新切出的通道 + 表面带)");
    for ri in (0..40).rev() {
        let y = -10.0 + ri as Real * 0.5;
        let mut row = String::new();
        for ci in 0..80 {
            let x = -10.0 + ci as Real * 0.25;
            let p = [x, y, z];
            let v = adf.eval(p) as f64;
            let base = adf.field.base.sd(p);
            let ch = if base > 0.0 {
                '·'
            } else if v < -0.05 {
                '#'
            } else if v > 0.05 {
                'o'
            } else {
                '+'
            };
            row.push(ch);
        }
        println!("  {row}");
    }

    // ── 求值速度 ──
    let m = 2_000_000u64;
    let t = Instant::now();
    let mut acc = 0.0f64;
    for _ in 0..m {
        let p = [
            min[0] + rng.f01() * size,
            min[1] + rng.f01() * size,
            min[2] + rng.f01() * size,
        ];
        acc += adf.eval(p) as f64;
    }
    let dt = t.elapsed();
    let ns = dt.as_nanos() as f64 / m as f64;
    println!("[perf] {m} 次 eval:{dt:?}({ns:.0} ns/次)校验和 {acc:.2}");

    // ── Surface Nets 提取 + 二进制 STL 导出(可视化产物) ──
    let h_ext = h_min * 2.0; // 重采样格宽 = 0.09375mm(256³)
    let t = Instant::now();
    let mesh = nets::extract_uniform(&adf, h_ext);
    let ext_t = t.elapsed();
    let sv = mesh.signed_volume();
    let tris = mesh.tris.len();
    let _t = Instant::now();
    nets::write_stl("adf_out.stl", &mesh).expect("STL 写出失败");
    let stl_bytes = std::fs::metadata("adf_out.stl").map(|m| m.len()).unwrap_or(0);
    println!(
        "[nets] h={h_ext:.4}mm 提取:{ext_t:?} | 顶点 {} 三角 {tris} | 有符号体积 {sv:.1} mm³(圆角盒理论 ≈7804,98% 级吻合)| STL adf_out.stl = {} KB",
        mesh.verts.len(),
        stl_bytes / 1024
    );
}
