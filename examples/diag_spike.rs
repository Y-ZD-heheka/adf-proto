//! 点级诊断:对离面顶点逐点打印场值 / 差分梯度 / 牛顿步剖面。
//!
//! 用法: cargo run --release --example diag_spike -- <原始.stl> x y z

use adf_proto::*;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let orig = args.get(1).cloned().unwrap_or_else(|| "stls/Turbo_Impeller.stl".into());
    let p: [Real; 3] = [
        args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.184),
        args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1.566),
        args.get(4).and_then(|s| s.parse().ok()).unwrap_or(39.985),
    ];
    let tm = stlio::load_stl(&orig).expect("STL 加载失败");
    let t0 = Instant::now();
    let field = meshfield::MeshField::new(tm.tris);
    println!("[diag] 场构建 {:?} | 点 {p:?}", t0.elapsed());

    // ADF 用与 roundtrip 相同的域参数重建(h_target=0.2, depth 由公式定)
    let (amn, amx) = (field.aabb_min, field.aabb_max);
    let extent = (amx[0] - amn[0]).max((amx[1] - amn[1]).max(amx[2] - amn[2]));
    let margin = extent * 0.06 + 1.0;
    let min = [amn[0] - margin, amn[1] - margin, amn[2] - margin];
    let size = extent + 2.0 * margin;
    let h_target = 0.2;
    let max_depth = (((size / h_target).log2().ceil()) as u8).clamp(6, 10);
    let h_min = size / (1u64 << max_depth) as Real;
    println!("[diag] depth {max_depth} | h_min {h_min:.4}");

    let adf = Adf::build_with(
        Composite { base: BaseField::Mesh(field), cuts: Vec::new() },
        min,
        size,
        0.002,
        0.5,
        max_depth,
    );

    let d0 = adf.eval(p) as f64;
    println!("[diag] adf.eval = {d0:.5}");

    // 精确场对照
    let dm = match &adf.field.base {
        BaseField::Mesh(m) => m.signed_distance(p),
        _ => f64::NAN,
    };
    println!("[diag] 精确 mesh signed_distance = {dm:.5}");

    // ADF 差分梯度(与 smooth_and_project 相同参数)
    let hstep = 1e-3 * size;
    let mut g = [0.0f64; 3];
    for k in 0..3 {
        let mut qq = p;
        qq[k] += hstep;
        g[k] = (adf.eval(qq) as f64 - d0) / hstep;
    }
    let g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
    println!("[diag] ADF 差分梯度 g = {g:?} | g2 = {g2:.4} (skip 判据 g2<0.25)");

    // 精确场差分梯度对照
    let mut gm = [0.0f64; 3];
    for k in 0..3 {
        let mut qq = p;
        qq[k] += hstep;
        gm[k] = (match &adf.field.base {
            BaseField::Mesh(m) => m.signed_distance(qq),
            _ => f64::NAN,
        } - dm) / hstep;
    }
    let gm2 = gm[0] * gm[0] + gm[1] * gm[1] + gm[2] * gm[2];
    println!("[diag] 精确场梯度 gm = {gm:?} | gm2 = {gm2:.4}");

    // 沿 -g 方向的 eval 剖面:看牛顿步会走到哪
    println!("[diag] 沿 -g/|g| 方向 eval 剖面:");
    let gl = g2.sqrt();
    if gl > 1e-9 {
        let dir = [-g[0] / gl, -g[1] / gl, -g[2] / gl];
        for t in [0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0] {
            let q = [p[0] + dir[0] * t, p[1] + dir[1] * t, p[2] + dir[2] * t];
            println!("  t={t:5.2} eval={:8.5}", adf.eval(q) as f64);
        }
    }
    // 沿精确梯度 -gm 方向剖面
    let gml = gm2.sqrt();
    if gml > 1e-9 {
        let dir = [-gm[0] / gml, -gm[1] / gml, -gm[2] / gml];
        println!("[diag] 沿精确梯度 -gm/|gm| 方向 eval 剖面:");
        for t in [0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0] {
            let q = [p[0] + dir[0] * t, p[1] + dir[1] * t, p[2] + dir[2] * t];
            println!("  t={t:5.2} eval={:8.5} mesh_sd={:8.5}", adf.eval(q) as f64, match &adf.field.base {
                BaseField::Mesh(m) => m.signed_distance(q),
                _ => f64::NAN,
            });
        }
    }
}
