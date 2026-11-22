//! P23 切削区"过切"分段诊断:同一组探针点在 ①切削前 ②切削后 ③重距离化后 三个阶段的
//! ADF 值 vs 复合场真值,定位"孔壁外 0.3–4mm 的材料被判成空气"发生在哪一层。
//!
//! 背景(多角度渲染 + 点云比对,stl_out.stl @ 2026-11-20):刀路邻域 25,203 个顶点中
//! 4,918 个离任何真表面 >0.5mm,全部位于**原材料内、胶囊外 0.3–4mm**,其中 4,094 个
//! 连成一片 —— 场错误而非后处理尖刺。与 README `[error.cut] max 4.3mm` 同源。
//!
//! 用法: cargo run --release --example diag_cut -- [stl] [h_target] [x y z ...]
//! 不给点时使用 P23 记录的代表点。

use adf_proto::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| "stls/Turbo_Impeller.stl".into());
    let h_target: Real = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.08);
    let mut probes: Vec<[Real; 3]> = args[3.min(args.len())..]
        .chunks(3)
        .filter(|c| c.len() == 3)
        .filter_map(|c| Some([c[0].parse().ok()?, c[1].parse().ok()?, c[2].parse().ok()?]))
        .collect();
    if probes.is_empty() {
        probes = vec![
            [-17.37, 12.26, 1.31],
            [-5.64, 28.73, -1.24],
            [-1.43, 26.76, -6.43],
            [-1.34, 26.85, 6.39],
            [-19.39, 12.18, -2.30],
            [1.25, 29.81, -0.52],
        ];
    }

    let tm = stlio::load_stl(&path).expect("STL 加载失败");
    let field = meshfield::MeshField::new(tm.tris);
    let (amn, amx) = (field.aabb_min, field.aabb_max);
    let extent = (amx[0] - amn[0]).max((amx[1] - amn[1]).max(amx[2] - amn[2]));
    let margin = extent * 0.06 + 1.0;
    let min = [amn[0] - margin, amn[1] - margin, amn[2] - margin];
    let size = extent + 2.0 * margin;
    let max_depth = (((size / h_target).log2().ceil()) as u8).clamp(6, 10);
    let h_min = size / (1u64 << max_depth) as Real;
    println!("[diag_cut] depth {max_depth} | h_min {h_min:.4} | 探针 {} 个", probes.len());

    let mut adf = Adf::build_with(
        Composite { base: BaseField::Mesh(field), cuts: Vec::new() },
        min,
        size,
        0.002,
        0.5,
        max_depth,
    );
    // 与 stl-demo 完全相同的两条胶囊
    let c = [(amn[0] + amx[0]) * 0.5, (amn[1] + amx[1]) * 0.5, (amn[2] + amx[2]) * 0.5];
    let r = extent * 0.1;
    let cuts = [
        Capsule { a: [min[0] - 1.0, c[1], c[2]], b: [min[0] + size + 1.0, c[1], c[2]], r },
        Capsule { a: [c[0], c[1], min[2] - 1.0], b: [c[0], c[1], min[2] + size + 1.0], r: r * 0.7 },
    ];

    let stage0: Vec<f32> = probes.iter().map(|p| adf.eval(*p)).collect();
    for cap in cuts {
        adf.add_cut(cap);
    }
    let stage1: Vec<f32> = probes.iter().map(|p| adf.eval(*p)).collect();
    let st = adf.redistance(8);
    println!("[diag_cut] redistance 写回 {} 角点", st.corrected);
    let stage2: Vec<f32> = probes.iter().map(|p| adf.eval(*p)).collect();

    println!("  点                          | base真值 | 复合真值 | 切前eval | 切后eval | 重距后eval | 叶深 | 判定");
    for (i, p) in probes.iter().enumerate() {
        let base = adf.field.base.sd(*p);
        let truth = adf.true_sd(*p);
        let verdict = |v: f32| (v as Real >= 0.0) != (truth >= 0.0);
        let tag = if verdict(stage1[i]) {
            "切削刷新(add_cut)已错号"
        } else if verdict(stage2[i]) {
            "重距离化引入错号"
        } else if (stage2[i] as Real - truth).abs() > 0.25 {
            "幅值偏差>0.25mm"
        } else {
            "OK(场正确 → 查提取/后处理)"
        };
        println!(
            "  ({:7.2},{:7.2},{:7.2}) | {base:8.3} | {truth:8.3} | {:8.3} | {:8.3} | {:10.3} | {:4} | {tag}",
            p[0], p[1], p[2], stage0[i], stage1[i], stage2[i], adf.depth_at(*p)
        );
        if verdict(stage1[i]) || verdict(stage2[i]) {
            adf.dump_leaf(*p);
        }
    }
}
