//! 真实 STL 端到端演示:加载 → 网格场 ADF → 切削 → 重距离化 → 提取 STL → 体积对账。
//! 运行:cargo run --release --bin stl-demo -- [stl路径] [h目标mm]

use adf_proto::*;
use std::time::Instant;

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
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| "stls/桌腿固定器.stl".to_string());
    let h_target: Real = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.08);

    // ── 1) 加载 + 水密检查 ──
    let t = Instant::now();
    let tm = stlio::load_stl(&path).expect("STL 加载失败");
    println!("[load] {path} | 三角 {} | {:?}", tm.tris.len(), t.elapsed());
    let (nt, watertight) = stlio::watertight_report(&tm.tris);
    println!("[watertight] 水密={watertight}(三角 {nt},奇数边检查;非水密 → 符号局部不可靠)");
    let vol_in = mesh_volume_abs(&tm.tris);
    println!("[volume] 输入网格体积(divergence)= {vol_in:.1} mm³");

    // ── 2) 网格场 + BVH ──
    let t = Instant::now();
    let field = meshfield::MeshField::new(tm.tris);
    println!("[bvh] 构建 {:?} | 三角 {}", t.elapsed(), field.triangle_count());
    let (amn, amx) = (field.aabb_min, field.aabb_max);
    let extent = (amx[0] - amn[0]).max((amx[1] - amn[1]).max(amx[2] - amn[2]));
    let margin = extent * 0.06 + 1.0;
    let min = [amn[0] - margin, amn[1] - margin, amn[2] - margin];
    let size = extent + 2.0 * margin;
    let max_depth = (((size / h_target).log2().ceil()) as u8).clamp(6, 10);
    let h_min = size / (1u64 << max_depth) as Real;
    let eps = 0.002;
    let band = 0.5;
    println!("[domain] 尺寸 {size:.1}mm | eps={eps} band={band} depth={max_depth} → h_min={h_min:.4}mm");

    // ── 3) 自适应构建(网格场:BVH 最近点 + 射线奇偶符号) ──
    let t = Instant::now();
    let mut adf = Adf::build_with(
        Composite { base: BaseField::Mesh(field), cuts: Vec::new() },
        min,
        size,
        eps,
        band,
        max_depth,
    );
    let (nl, nb, hist) = adf.leaf_stats_bytes();
    let kb = nb / 1024;
    println!("[build] {:?} | 叶 {nl} | ≈{kb} KB", t.elapsed());
    {
        let rows: Vec<String> = (0..17u8)
            .filter(|d| hist[*d as usize] > 0)
            .map(|d| format!("d{d}:{}", hist[d as usize]))
            .collect();
        println!("[leaves] 按深度 {}", rows.join(" "));
    }
    if let Some(la) = &adf.lattice {
        let (comp, pages, bytes) = la.stats();
        let mb = bytes / (1024 * 1024);
        println!("[cache] 格点缓存:BVH 计算 {comp} | 物化页 {pages} | {mb} MB");
    }

    // ── 4) 场精度抽查:ADF eval vs 直接网格查询(随机点) ──
    let mut rng = Rng::new(4242);
    let mut max_e = 0.0f64;
    let mut sum_e = 0.0f64;
    let mut cnt = 0u64;
    // 拆分:符号相反(拓扑/定号问题) vs 符号一致但幅值有偏差(插值/分辨率问题)
    let (mut n_sign, mut max_sign, mut sum_sign) = (0u64, 0.0f64, 0.0f64);
    let (mut n_mag, mut max_mag, mut sum_mag) = (0u64, 0.0f64, 0.0f64);
    for _ in 0..2000 {
        let p = [
            min[0] + rng.f01() * size,
            min[1] + rng.f01() * size,
            min[2] + rng.f01() * size,
        ];
        let direct = adf.field.base.sd(p) as Real;
        let got = adf.eval(p) as Real;
        let e = (got - direct).abs();
        max_e = max_e.max(e);
        sum_e += e;
        cnt += 1;
        if (got < 0.0) != (direct < 0.0) {
            n_sign += 1;
            max_sign = max_sign.max(e);
            sum_sign += e;
        } else {
            n_mag += 1;
            max_mag = max_mag.max(e);
            sum_mag += e;
        }
    }
    println!(
        "[error] 随机 {cnt} 点:ADF vs 网格直查 max={max_e:.4} mean={:.4} mm(粗格区允许偏差)",
        sum_e / cnt as Real
    );
    println!(
        "[error.split] 符号相反 {:>4} 点({:.1}%) max={max_sign:.3} mean={:.3} | 符号一致 {:>4} 点 max={max_mag:.3} mean={:.3}",
        n_sign,
        n_sign as f64 / cnt as f64 * 100.0,
        sum_sign / n_sign.max(1) as Real,
        n_mag,
        sum_mag / n_mag.max(1) as Real
    );

    // ── 5) 切削:两条胶囊穿过模型(沿 X / 沿 Z) ──
    let c = [(amn[0] + amx[0]) * 0.5, (amn[1] + amx[1]) * 0.5, (amn[2] + amx[2]) * 0.5];
    let r = extent * 0.1;
    let demo_cuts = [
        Capsule { a: [min[0] - 1.0, c[1], c[2]], b: [min[0] + size + 1.0, c[1], c[2]], r },
        Capsule { a: [c[0], c[1], min[2] - 1.0], b: [c[0], c[1], min[2] + size + 1.0], r: r * 0.7 },
    ];
    for (i, cap) in demo_cuts.into_iter().enumerate() {
        let t = Instant::now();
        let (n0, _, _) = adf.leaf_stats_bytes();
        adf.add_cut(cap);
        let (n1, b1, _) = adf.leaf_stats_bytes();
        let d = n1 as i64 - n0 as i64;
        let kb = b1 / 1024;
        println!("[cut {}] {:?} | 叶 {n0} → {n1}({d:+})| ≈{kb} KB", i + 1, t.elapsed());
    }

    // ── 6) 重距离化(切削空气区下界修复) ──
    let t = Instant::now();
    let st = adf.redistance(8);
    println!(
        "[redist] {:?} | 界面叶 {} | 网格 {:?} | 回写 {} 角点",
        t.elapsed(),
        st.interface_leaves,
        st.grid_n,
        st.corrected
    );
    println!(
        "[redist.phase] 符号场 {:?} | 棱种子 {:?} | 松弛 {:?} | 写回 {:?}",
        st.t_eval, st.t_seed, st.t_relax, st.t_write
    );

    // ── 6b) 场精度复核(切削后,限定刀路区表面邻域):ADF vs 复合场真值 ──
    //      这是重距离化路径的直接质量指标:比体积对账更贴近"加工面准不准"。
    {
        let mut rng = Rng::new(7777);
        let (mut max_e, mut sum_e) = (0.0f64, 0.0f64);
        let mut cnt = 0u64;
        for _ in 0..40000 {
            let p = [
                min[0] + rng.f01() * size,
                min[1] + rng.f01() * size,
                min[2] + rng.f01() * size,
            ];
            let t = adf.true_sd(p);
            if t.abs() > 2.0 {
                continue; // 只看表面邻域(深场由粗格承担,不属重距离化职责)
            }
            let e = (adf.eval(p) as Real - t).abs();
            max_e = max_e.max(e);
            sum_e += e;
            cnt += 1;
        }
        if cnt > 0 {
            println!(
                "[error.cut] 刀路区表面邻域 {cnt} 点(|真值|≤2mm):ADF vs 复合场 max={max_e:.4} mean={:.4} mm",
                sum_e / cnt as Real
            );
        }
    }

    // ── 7) Surface Nets 提取 + STL 导出 + 体积对账 ──
    // 提取格宽倍率:环境变量可覆盖(实验用;默认 2×h_min)
    let hf_mult: Real = std::env::var("ADF_H_EXT").ok().and_then(|v| v.parse().ok()).unwrap_or(2.0);
    let h_ext = h_min * hf_mult;
    let t = Instant::now();
    let mut mesh = nets::extract_uniform(&adf, h_ext);
    println!(
        "[nets] h={h_ext:.4}({hf_mult}×h_min) 提取 {:?} | 顶点 {} 三角 {}",
        t.elapsed(),
        mesh.verts.len(),
        mesh.tris.len()
    );
    {
        let (open, nonman, comps) = mesh.edge_stats();
        println!(
            "[nets.topology] 开放边(洞) {open} | 非流形边 {nonman} | 连通分量 {comps}{}",
            match (open, nonman) {
                (0, 0) => " → 闭合流形(可打印)",
                (0, _) => " → 闭合但含非流形边(切片软件一般可容错)",
                _ => " → **有开口**",
            }
        );
    }
    // 折叠/薄片修复:边缘"破损"的真身,塌缩最短边消除
    if std::env::var("ADF_REPAIR").is_ok() {
        let t = Instant::now();
        let (nc, nr) = mesh.repair_folds(&adf, 0.5, 0.004, 0.002, h_ext as f64 * 4.0, 20);
        println!("[repair] {:?} | 塌缩 {nc} 边,删三角 {nr}", t.elapsed());
    }
    let vol_out = mesh.signed_volume().abs();
    let dev = if vol_in > 0.0 { (vol_out - vol_in) / vol_in * 100.0 } else { 0.0 };
    println!(
        "[volume] 提取 {vol_out:.1} vs 输入 {vol_in:.1} mm³(偏差 {dev:+.1}% —— 含切削移除与重采样误差)"
    );
    let t = Instant::now();
    nets::write_stl("stl_out.stl", &mesh).expect("STL 写出失败");
    println!(
        "[stl] stl_out.stl 写出 {:?} | {} KB",
        t.elapsed(),
        std::fs::metadata("stl_out.stl").map(|m| m.len() / 1024).unwrap_or(0)
    );
}
