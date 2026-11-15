//! 场体检:判断"提取丢料"还是"场本身缺料"。
//! - 蒙特卡洛统计 `adf.eval < 0` 的体积占比 → 场认为的材料体积;
//! - 与输入网格体积对比;
//! - 另抽表面邻域点,统计场 vs 精确 sd 的符号/幅值误差。
//!
//! 用法:cargo run --release --example probe_field -- stls/Turbo_Impeller.stl 0.2

use adf_proto::*;

fn mesh_volume_abs(tris: &[[[Real; 3]; 3]]) -> Real {
    let mut v = 0.0;
    for t in tris {
        let (a, b, c) = (t[0], t[1], t[2]);
        let ab = sub(b, a);
        let ac = sub(c, a);
        v += dot(a, [ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0]]) / 6.0;
    }
    v.abs()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(|| "stls/Turbo_Impeller.stl".to_string());
    let h_target: Real = args.next().and_then(|v| v.parse().ok()).unwrap_or(0.2);

    let tm = stlio::load_stl(&path).expect("STL 加载失败");
    let vol_in = mesh_volume_abs(&tm.tris);
    let field = meshfield::MeshField::new(tm.tris);
    let (amn, amx) = (field.aabb_min, field.aabb_max);
    let extent = (amx[0] - amn[0]).max((amx[1] - amn[1]).max(amx[2] - amn[2]));
    let margin = extent * 0.06 + 1.0;
    let min = [amn[0] - margin, amn[1] - margin, amn[2] - margin];
    let size = extent + 2.0 * margin;
    let max_depth = (((size / h_target).log2().ceil()) as u8).clamp(6, 10);
    let band: Real = std::env::var("ADF_BAND").ok().and_then(|v| v.parse().ok()).unwrap_or(0.5);
    let eps: Real = 0.002;
    println!("[probe] {path} | 深度 {max_depth} | band {band} | 输入体积 {vol_in:.1}");

    let adf = Adf::build_with(
        Composite { base: BaseField::Mesh(field), cuts: Vec::new() },
        min,
        size,
        eps,
        band,
        max_depth,
    );

    // ── ① 场的材料体积(蒙特卡洛)──
    let mut rng = Rng::new(12345);
    let n = 400_000;
    let (mut neg, mut tot) = (0u64, 0u64);
    for _ in 0..n {
        let p = [
            min[0] + rng.f01() * size,
            min[1] + rng.f01() * size,
            min[2] + rng.f01() * size,
        ];
        if adf.eval(p) < 0.0 {
            neg += 1;
        }
        tot += 1;
    }
    let dv = size * size * size;
    let vol_field = neg as f64 / tot as f64 * dv;
    println!(
        "[probe.volume] 场材料 {vol_field:.1} vs 输入 {vol_in:.1} mm³({:+.1}%)| 域 {dv:.0}",
        (vol_field - vol_in) / vol_in * 100.0
    );

    // ── ② 表面邻域点的符号/幅值误差(只在 |真值| ≤ 0.6mm 内)──
    let mut rng = Rng::new(999);
    let (mut cnt, mut sign_bad, mut sum_e, mut max_e) = (0u64, 0u64, 0.0f64, 0.0f64);
    for _ in 0..400_000 {
        let p = [
            min[0] + rng.f01() * size,
            min[1] + rng.f01() * size,
            min[2] + rng.f01() * size,
        ];
        let t = adf.field.base.sd(p) as f64;
        if t.abs() > 0.6 {
            continue;
        }
        let g = adf.eval(p) as f64;
        if (g < 0.0) != (t < 0.0) {
            sign_bad += 1;
        }
        let e = (g - t).abs();
        sum_e += e;
        max_e = max_e.max(e);
        cnt += 1;
    }
    // ── 局部壁厚:从表面点沿法线向内走到另一侧 ──
    {
        let mut rng = Rng::new(4242);
        let mut th: Vec<f64> = Vec::new();
        while th.len() < 400 {
            let p = [
                min[0] + rng.f01() * size,
                min[1] + rng.f01() * size,
                min[2] + rng.f01() * size,
            ];
            let d = adf.field.base.sd(p) as f64;
            // 只取"表面内一侧"的点(否则射线先穿过自己脚下那层表面)
            if d > -0.005 || d < -0.1 {
                continue;
            }
            // 数值梯度当法线
            let h = 0.01;
            let mut g = [0.0f64; 3];
            for k in 0..3 {
                let mut q = p;
                q[k] += h;
                g[k] = (adf.field.base.sd(q) as f64 - d) / h;
            }
            let gl = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
            if gl < 0.5 {
                continue;
            }
            let n = [g[0] / gl, g[1] / gl, g[2] / gl];
            // 永远朝"材料内部"(梯度反方向)走,找对面那层表面
            let sgn = -1.0;
            let mut t = 0.0;
            let mut hit = -1.0;
            while t < 3.0 {
                t += 0.02;
                let q = [
                    p[0] + sgn * n[0] * t,
                    p[1] + sgn * n[1] * t,
                    p[2] + sgn * n[2] * t,
                ];
                if (adf.field.base.sd(q) as f64) * d < 0.0 {
                    hit = t;
                    break;
                }
            }
            if hit > 0.0 {
                th.push(hit);
            }
        }
        th.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "[probe.thick] 局部壁厚 {} 点: p10 {:.3} 中位 {:.3} p90 {:.3} max {:.3} mm",
            th.len(),
            th[th.len() / 10],
            th[th.len() / 2],
            th[th.len() * 9 / 10],
            th.last().copied().unwrap_or(0.0)
        );
    }
    println!(
        "[probe.near] 表面邻域(|d|≤0.6mm){cnt} 点:符号错 {sign_bad}({:.2}%)| max {max_e:.4} mean {:.4}",
        sign_bad as f64 / cnt.max(1) as f64 * 100.0,
        sum_e / cnt.max(1) as f64
    );

    // ── ③ 未切削直接提取:提取本身丢不丢料 ──
    let h_min = size / (1u64 << max_depth) as Real;
    for mult in [2.0, 1.0] {
        let m = nets::extract_uniform(&adf, h_min * mult);
        let (open, nonman, comps) = m.edge_stats();
        let v = m.signed_volume().abs();
        // 顶点到真表面的距离:表面质量的直接度量(尖刺/塌陷都会顶出来)
        let dev_of = |m: &nets::Mesh| -> Vec<f64> {
            let mut v: Vec<f64> = Vec::new();
            for (i, vtx) in m.verts.iter().enumerate() {
                if i % 5 != 0 {
                    continue;
                }
                let p = [vtx[0] as Real, vtx[1] as Real, vtx[2] as Real];
                v.push(adf.field.base.sd(p).abs() as f64);
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v
        };
        let mut devs: Vec<f64> = Vec::new();
        for (i, vtx) in m.verts.iter().enumerate() {
            if i % 5 != 0 {
                continue; // 抽样 20%
            }
            let p = [vtx[0] as Real, vtx[1] as Real, vtx[2] as Real];
            devs.push(adf.field.base.sd(p).abs() as f64);
        }
        devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = devs.iter().sum::<f64>() / devs.len() as f64;
        let p99 = devs[(devs.len() as f64 * 0.99) as usize];
        let bad = devs.iter().filter(|d| **d > 0.5).count();
        if mult == 2.0 {
            let mut shown = 0;
            for (i, vtx) in m.verts.iter().enumerate() {
                if i % 5 != 0 {
                    continue;
                }
                let p = [vtx[0] as Real, vtx[1] as Real, vtx[2] as Real];
                let d = adf.field.base.sd(p).abs() as f64;
                if d > 0.5 && shown < 12 {
                    if shown == 0 {
                        println!(
                            "   >> 该点 eval={:.4} 真值={:.4}",
                            adf.eval(p),
                            adf.field.base.sd(p)
                        );
                        adf.dump_leaf(p);
                        // 该点所在的**提取格**(与叶网格同对齐)8 角点,看符号是否全同
                        let step = h_min * mult;
                        let g = [
                            ((p[0] - min[0]) / step).floor() as usize,
                            ((p[1] - min[1]) / step).floor() as usize,
                            ((p[2] - min[2]) / step).floor() as usize,
                        ];
                        print!("   >> 提取格 {g:?} 角点:");
                        for k in 0..8 {
                            let c = [
                                min[0] + (g[0] + (k & 1)) as Real * step,
                                min[1] + (g[1] + ((k >> 1) & 1)) as Real * step,
                                min[2] + (g[2] + ((k >> 2) & 1)) as Real * step,
                            ];
                            print!(" {:.3}", adf.eval(c));
                        }
                        println!();
                    }
                    shown += 1;
                    println!(
                        "   坏点 ({:.2},{:.2},{:.2}) 偏离 {:.3}mm 深度 {} 格宽 {:.3}",
                        p[0], p[1], p[2], d, adf.depth_at(p), adf.cell_size_at(p)
                    );
                }
            }
        }
        println!(
            "[probe.nets] h={h:.4}({mult}x h_min) 三角 {tri} | 体积 {v:.1}({dv:+.1}%)| 开放边 {open} 非流形 {nonman} 分量 {comps}",
            h = h_min * mult,
            tri = m.tris.len(),
            dv = (v - vol_in) / vol_in * 100.0
        );
        println!(
            "  顶点偏离真表面: mean {mean:.4} p99 {p99:.4} max {mx:.4} mm | >0.5mm {bad}/{n} ({pct:.2}%)",
            mx = devs.last().copied().unwrap_or(0.0),
            n = devs.len(),
            pct = bad as f64 / devs.len() as f64 * 100.0
        );
        // 平滑 + 投影后再测一遍(评估后处理收益)
        let mut m2 = m;
        let t2 = std::time::Instant::now();
        let (dm, dx) = m2.smooth_and_project(&adf, 3, h_min as f32 * 1.0);
        let el = t2.elapsed();
        let d2 = dev_of(&m2);
        let mean2 = d2.iter().sum::<f64>() / d2.len() as f64;
        let bad2 = d2.iter().filter(|x| **x > 0.5).count();
        let v2 = m2.signed_volume().abs();
        let (o2, nm2, c2) = m2.edge_stats();
        println!(
            "  → 平滑+投影({el:?},位移 mean {dm:.4} max {dx:.4}):mean {mean2:.4} p99 {:.4} max {:.4} mm | >0.5mm {bad2}/{n} ({pct2:.2}%)",
            d2[(d2.len() as f64 * 0.99) as usize],
            d2.last().copied().unwrap_or(0.0),
            n = d2.len(),
            pct2 = bad2 as f64 / d2.len() as f64 * 100.0
        );
        println!(
            "  → 体积 {v2:.1}({:+.2}%)| 开放边 {o2} 非流形 {nm2} 分量 {c2}",
            (v2 - vol_in) / vol_in * 100.0
        );
    }
}
