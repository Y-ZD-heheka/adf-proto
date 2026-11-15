//! 构建阶段重复基准:同一件重复 N 次 build_with,便于 VTune 采样(单次构建太短)。
//! 运行:cargo run --release --example bench_build -- [stl] [h_target] [重复次数]

use adf_proto::*;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| "stls/桌腿固定器.stl".to_string());
    let h_target: Real = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.08);
    let reps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);

    // 域参数与 stl-demo 完全一致,便于与全管线数字对照
    let tm = stlio::load_stl(&path).expect("STL 加载失败");
    let (amn, amx) = {
        let f = meshfield::MeshField::new(tm.tris.clone());
        (f.aabb_min, f.aabb_max)
    };
    let extent = (amx[0] - amn[0]).max((amx[1] - amn[1]).max(amx[2] - amn[2]));
    let margin = extent * 0.06 + 1.0;
    let min = [amn[0] - margin, amn[1] - margin, amn[2] - margin];
    let size = extent + 2.0 * margin;
    let max_depth = (((size / h_target).log2().ceil()) as u8).clamp(6, 10);
    println!(
        "[bench] {path} | 三角 {} | size={size:.1} depth={max_depth} | {reps} 次",
        tm.tris.len()
    );

    let mut total = std::time::Duration::ZERO;
    let mut leaves = 0u64;
    for i in 0..reps {
        let field = meshfield::MeshField::new(tm.tris.clone());
        let t = Instant::now();
        let adf = Adf::build_with(
            Composite { base: BaseField::Mesh(field), cuts: Vec::new() },
            min,
            size,
            0.002,
            0.5,
            max_depth,
        );
        let el = t.elapsed();
        total += el;
        leaves = adf.leaf_stats_bytes().0;
        println!("  [{i}] build {el:?} | 叶 {leaves}");
    }
    println!(
        "[bench] 合计 {total:?} | 均 {:?} | 叶 {leaves}",
        total / reps as u32
    );
}
