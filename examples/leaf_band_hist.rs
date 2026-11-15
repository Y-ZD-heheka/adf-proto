//! 八叉树叶节点深度分布调研(自适应提取潜力评估)。
//!
//! 统计**表面带内**(8 角点符号混合,即穿过等值面的叶)叶节点的深度分布,
//! 对照均匀提取(extract_uniform 以固定 h_ext 重采样)估算自适应提取
//! (Octree Dual Contouring)的三角形数量收益:
//!   均匀提取格数 ≈ 带内 (h_ext)³ 格数;自适应提取顶点数 ≈ 带内叶数。
//!
//! 用法: cargo run --release --example leaf_band_hist -- <stl> [h_target]

use adf_proto::*;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| "stls/Turbo_Impeller.stl".to_string());
    let h_target: Real = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.1);

    let tm = stlio::load_stl(&path).expect("STL 加载失败");
    let field = meshfield::MeshField::new(tm.tris);
    let (amn, amx) = (field.aabb_min, field.aabb_max);
    let extent = (amx[0] - amn[0]).max((amx[1] - amn[1]).max(amx[2] - amn[2]));
    let margin = extent * 0.06 + 1.0;
    let min = [amn[0] - margin, amn[1] - margin, amn[2] - margin];
    let size = extent + 2.0 * margin;
    let max_depth = (((size / h_target).log2().ceil()) as u8).clamp(6, 10);
    let h_min = size / (1u64 << max_depth) as Real;
    println!("[domain] 尺寸 {size:.2} | depth {max_depth} | h_min {h_min:.4}");

    let t = Instant::now();
    let adf = Adf::build_with(
        Composite { base: BaseField::Mesh(field), cuts: Vec::new() },
        min,
        size,
        0.002,
        0.5,
        max_depth,
    );
    println!("[build] {:?}", t.elapsed());

    // 全部叶深度直方图 + 表面带内叶深度直方图
    let mut all_hist = [0u64; 17];
    let mut band_hist = [0u64; 17];
    let mut band_cells_uniform = [0u64; 17]; // 若把该叶细分到 depth17?不需要——均匀提取按 h_ext
    let mut stack = vec![(adf.root, 0u8, 0.0f64, 0.0f64, 0.0f64, size)];
    while let Some((idx, d, x, y, z, s)) = stack.pop() {
        match adf.nodes[idx as usize] {
            Node::Leaf(c8) => {
                all_hist[d as usize] += 1;
                let (mut neg, mut pos) = (false, false);
                for &v in c8.iter() {
                    if v < 0.0 { neg = true } else { pos = true }
                }
                if neg && pos {
                    band_hist[d as usize] += 1;
                    // 该叶覆盖的均匀提取(h_ext = 1×h_min)格数
                    let cells = (s / h_min).powi(3).round() as u64;
                    band_cells_uniform[d as usize] += cells;
                }
                let _ = (x, y, z);
            }
            Node::Internal(ch) => {
                let hs = s * 0.5;
                for (ci, &c) in ch.iter().enumerate() {
                    if c == EMPTY {
                        continue;
                    }
                    let (dx, dy, dz) = (ci & 1, (ci >> 1) & 1, (ci >> 2) & 1);
                    stack.push((
                        c,
                        d + 1,
                        x + dx as f64 * hs,
                        y + dy as f64 * hs,
                        z + dz as f64 * hs,
                        hs,
                    ));
                }
            }
        }
    }
    println!("\n[leaf] 深度直方图(全部叶 / 表面带叶 / 带叶折算均匀格数):");
    let mut tot_all = 0u64;
    let mut tot_band = 0u64;
    let mut tot_uni = 0u64;
    for d in 0..17 {
        if all_hist[d] == 0 && band_hist[d] == 0 {
            continue;
        }
        println!(
            "  d{d:2}: all {:>9} | band {:>8} | uniform-cells {:>10}",
            all_hist[d], band_hist[d], band_cells_uniform[d]
        );
        tot_all += all_hist[d];
        tot_band += band_hist[d];
        tot_uni += band_cells_uniform[d];
    }
    println!("\n[leaf] 总叶 {tot_all} | 表面带叶 {tot_band} | 带叶折算均匀提取格数 {tot_uni}");
    println!("[est] 均匀提取顶点 ≈ {tot_uni}(1×h_min 格) → 自适应提取顶点 ≈ {tot_band}(带叶)");
    if tot_uni > 0 {
        println!(
            "[est] 理论顶点压缩比 ≈ {:.1}×(自适应顶点数 / 均匀格数 = {:.1}%)",
            tot_uni as f64 / tot_band as f64,
            100.0 * tot_band as f64 / tot_uni as f64
        );
    }
}
