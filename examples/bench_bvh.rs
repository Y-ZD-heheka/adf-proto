//! BVH 查询微基准 —— 定位构建期 tasks 阶段耗时的构成(P1/P2 前置测量)。
//!
//! 三个问题:
//! 1. 单次最近点查询的真实成本(近表面 vs 深场,带种子 vs 无种子)?
//! 2. 24 工人下每查询成本是否随线程数膨胀(原子计时器争用)?
//! 3. 纯遍历成本 vs 每调用固定开销(Instant::now ×2 + 3 个全局 fetch_add)?
//!
//! 运行:cargo run --release --example bench_bvh

use adf_proto::*;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| "stls/Turbo_Impeller.stl".to_string());
    let n_pts: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let tm = stlio::load_stl(&path).expect("STL 加载失败");
    let field = meshfield::MeshField::new(tm.tris);
    let (amn, amx) = (field.aabb_min, field.aabb_max);
    let extent = (amx[0] - amn[0]).max((amx[1] - amn[1]).max(amx[2] - amn[2]));
    let margin = extent * 0.06 + 1.0;
    let min = [amn[0] - margin, amn[1] - margin, amn[2] - margin];
    let size = extent + 2.0 * margin;
    let h_c = size / 128.0; // BakeGrid 粗格步长
    println!(
        "[domain] size={size:.1}mm | 三角 {} | 粗格 h_c={h_c:.3}mm",
        field.triangle_count()
    );

    // ── 采样点:全域随机(覆盖近表面与深场)──
    let mut rng = Rng::new(20261115);
    let pts: Vec<[Real; 3]> = (0..n_pts)
        .map(|_| {
            [
                min[0] + rng.f01() * size,
                min[1] + rng.f01() * size,
                min[2] + rng.f01() * size,
            ]
        })
        .collect();

    // ── 1) 单线程:无种子 vs 紧种子,按 |d| 分桶 ──
    let t = Instant::now();
    let unseeded: Vec<(Real, f64)> = pts
        .iter()
        .map(|&p| {
            let s = Instant::now();
            let d = field.unsigned_distance(p);
            (d, s.elapsed().as_secs_f64())
        })
        .collect();
    let t_uns = t.elapsed();
    println!(
        "[单线程] 无种子 {} 次 | 总 {:?} | 均 {:.1}µs",
        n_pts,
        t_uns,
        t_uns.as_secs_f64() * 1e6 / n_pts as f64
    );

    // 紧种子 = 真值 + 2.5h_c(与 ExactSrc::Seeded 同口径)
    let t = Instant::now();
    let mut buckets = [(0u64, 0f64); 5]; // (次数, 累计秒):|d| 分桶
    let mut ck: u64 = 0;
    for (i, &p) in pts.iter().enumerate() {
        let (d, _) = unseeded[i];
        let sd = d + 2.5 * h_c;
        let s = Instant::now();
        let (_dd, _t, _f, _cp) = field.closest_features_seeded(p, sd * sd);
        let el = s.elapsed().as_secs_f64();
        let b = if d < 0.5 {
            0
        } else if d < 2.0 {
            1
        } else if d < 10.0 {
            2
        } else if d < 30.0 {
            3
        } else {
            4
        };
        buckets[b].0 += 1;
        buckets[b].1 += el;
        ck += 1;
    }
    let t_seed = t.elapsed();
    println!(
        "[单线程] 紧种子 {} 次 | 总 {:?} | 均 {:.1}µs",
        ck,
        t_seed,
        t_seed.as_secs_f64() * 1e6 / ck as f64
    );
    let names = ["|d|<0.5(表面带)", "0.5–2", "2–10", "10–30", ">30(深场)"];
    for (i, (n, s)) in buckets.iter().enumerate() {
        if *n > 0 {
            println!(
                "  {:<16} {:>6} 次 | 均 {:>7.2}µs | 占比 {:.0}%",
                names[i],
                n,
                s * 1e6 / *n as f64,
                s / t_seed.as_secs_f64() * 100.0
            );
        }
    }

    // ── 2) 多线程扩展:同一批查询,拆给 T 个工人 ──
    println!("[扩展性] 同一批 {} 查询 × 不同工人数:", n_pts);
    for nt in [1usize, 4, 8, 16, 24] {
        let t = Instant::now();
        let chunk = n_pts.div_ceil(nt);
        std::thread::scope(|sc| {
            for w in 0..nt {
                let (s0, s1) = (w * chunk, ((w + 1) * chunk).min(n_pts));
                if s0 >= s1 {
                    continue;
                }
                let pts = &pts;
                let un = &unseeded;
                let field = &field;
                sc.spawn(move || {
                    for i in s0..s1 {
                        let (d, _) = un[i];
                        let sd = d + 2.5 * h_c;
                        let _ = field.closest_features_seeded(pts[i], sd * sd);
                    }
                });
            }
        });
        let el = t.elapsed().as_secs_f64();
        println!(
            "  {nt:>2} 工人 | 墙钟 {:>7.3}s | 每查询 {:>7.2}µs | 相对单线程 {:.2}×",
            el,
            el * 1e6 / n_pts as f64,
            (t_seed.as_secs_f64() / el)
        );
    }
}
