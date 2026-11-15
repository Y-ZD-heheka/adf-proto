//! 重距离化基准:精确 EDT(生产路径)vs 串行八序扫掠(参考路径)。
//! 回答两件事:EDT 快多少、精度差多少(以解析平面距离为地面真值)。
//!
//! 运行:cargo run --release --example bench_redist

use adf_proto::*;
use std::time::Instant;

/// 斜面种子:法线 (0.6,0.8,0),种子取 |d|<1 格的格点并写入真值距离。
fn plane_case(gn: [usize; 3], hf: f32) -> (Vec<f32>, Vec<f64>) {
    let (nx, ny, nz) = (gn[0], gn[1], gn[2]);
    let nrm = [0.6f64, 0.8, 0.0];
    let p0 = [nx as f64 * 0.5, ny as f64 * 0.5, nz as f64 * 0.5];
    let mut d = vec![f32::INFINITY; nx * ny * nz];
    let mut truth = vec![0.0f64; nx * ny * nz];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let q = [i as f64, j as f64, k as f64];
                let dist = (nrm[0] * (q[0] - p0[0]) + nrm[1] * (q[1] - p0[1])).abs();
                let c = (k * ny + j) * nx + i;
                truth[c] = dist * hf as f64;
                if dist < 1.0 {
                    d[c] = (dist * hf as f64) as f32;
                }
            }
        }
    }
    (d, truth)
}

fn stats(got: &[f32], truth: &[f64], hf: f32) -> (f64, f64) {
    let (mut maxdev, mut sum) = (0.0f64, 0.0f64);
    let mut n = 0u64;
    for (g, t) in got.iter().zip(truth.iter()) {
        if g.is_finite() {
            let e = (*g as f64 - *t).abs();
            maxdev = maxdev.max(e);
            sum += e;
            n += 1;
        }
    }
    (maxdev, sum / n.max(1) as f64)
}

fn main() {
    let gn = [389usize, 201, 905]; // 桌腿固定器真实 redist 网格尺寸
    let hf = 0.0676f32;
    println!("[case] 网格 {gn:?} = {} 格 | hf={hf}", gn[0] * gn[1] * gn[2]);
    let (base, truth) = plane_case(gn, hf);

    // ── 生产路径:EDT ──
    let mut e = base.clone();
    let t = Instant::now();
    edt_redist(&mut e, gn, hf, 0); // nt=0 → 用默认并行度
    let t_edt = t.elapsed();
    let (mx, mean) = stats(&e, &truth, hf);
    println!(
        "[EDT ] {t_edt:?} | 真值 max {mx:.4}mm({:.2} 格) mean {mean:.4}mm",
        mx / hf as f64
    );

    // ── 参考路径:串行八序扫掠 ──
    for r in [8u32, 16, 32] {
        let mut s = base.clone();
        let t = Instant::now();
        eikonal_sweep_serial(&mut s, gn, hf, r);
        let el = t.elapsed();
        let (mx, mean) = stats(&s, &truth, hf);
        println!(
            "[扫掠{r:>2}] {el:?} | 真值 max {mx:.4}mm({:.2} 格) mean {mean:.4}mm | 相对 EDT {:.2}×",
            mx / hf as f64,
            el.as_secs_f64() / t_edt.as_secs_f64()
        );
    }

    // ── EDT 起步 + 扫掠精修(扫掠单调下降,正好抹掉种子离散化的鼓包)──
    println!("[EDT + 扫掠精修]");
    for k in [1u32, 2, 4] {
        let mut s = base.clone();
        let t = Instant::now();
        edt_redist(&mut s, gn, hf, 0);
        let t_edt = t.elapsed();
        eikonal_sweep_serial(&mut s, gn, hf, k);
        let el = t.elapsed();
        let (mx, mean) = stats(&s, &truth, hf);
        println!(
            "  EDT+{k} 轮: {el:?}(其中 EDT {t_edt:?}) | 真值 max {mx:.4}mm({:.2} 格) mean {mean:.4}mm",
            mx / hf as f64
        );
    }

    // ── EDT 并行扩展性 ──
    println!("[EDT 扩展性]");
    let mut baseline = 0.0f64;
    for nt in [1usize, 2, 4, 8, 16, 24] {
        let mut d = base.clone();
        let t = Instant::now();
        edt_redist(&mut d, gn, hf, nt);
        let el = t.elapsed().as_secs_f64();
        if nt == 1 {
            baseline = el;
        }
        println!("  {nt:>2} 工人 | {el:>7.3}s | 加速 {:.2}×", baseline / el);
    }
}
