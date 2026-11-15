//! STL → ADF → STL 往返转换(**不做任何切削**)。
//!
//! 用途:把输入网格重建为等值面网格,用于检查场的重建质量与网格可打印性。
//! 与 `stl-demo` 的区别:后者会为演示切削能力打两个胶囊孔,本工具**只重建**。
//!
//! 用法:
//!   cargo run --release --bin stl-roundtrip -- <输入.stl> [h_target] [输出.stl] [提取倍率]
//! 例:
//!   cargo run --release --bin stl-roundtrip -- stls/Turbo_Impeller.stl 0.2 out.stl 2

use adf_proto::*;
use std::time::Instant;

fn mesh_volume_abs(tris: &[[[Real; 3]; 3]]) -> Real {
    let mut v = 0.0;
    for t in tris {
        let (a, b, c) = (t[0], t[1], t[2]);
        let ab = sub(b, a);
        let ac = sub(c, a);
        v += dot(
            a,
            [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ],
        ) / 6.0;
    }
    v.abs()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "stls/桌腿固定器.stl".to_string());
    let h_target: Real = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.08);
    let out = args.get(3).cloned().unwrap_or_else(|| "roundtrip.stl".to_string());
    let mult: Real = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(2.0);

    // ── 1) 读入 ──
    let t = Instant::now();
    let tm = stlio::load_stl(&path).expect("STL 加载失败");
    let (nt, watertight) = stlio::watertight_report(&tm.tris);
    let vol_in = mesh_volume_abs(&tm.tris);
    println!(
        "[load] {path} | 三角 {nt} | 水密 {watertight} | 体积 {vol_in:.1} mm³ | {:?}",
        t.elapsed()
    );

    // ── 2) 网格场 + 域 ──
    let field = meshfield::MeshField::new(tm.tris);
    let (amn, amx) = (field.aabb_min, field.aabb_max);
    let extent = (amx[0] - amn[0]).max((amx[1] - amn[1]).max(amx[2] - amn[2]));
    let margin = extent * 0.06 + 1.0;
    let min = [amn[0] - margin, amn[1] - margin, amn[2] - margin];
    let size = extent + 2.0 * margin;
    let max_depth = (((size / h_target).log2().ceil()) as u8).clamp(6, 10);
    let h_min = size / (1u64 << max_depth) as Real;
    println!("[domain] 尺寸 {size:.2}mm | depth {max_depth} | h_min {h_min:.4}mm");

    // ── 3) 建 ADF(不切削)──
    let t = Instant::now();
    let adf = Adf::build_with(
        Composite { base: BaseField::Mesh(field), cuts: Vec::new() },
        min,
        size,
        0.002,
        0.5,
        max_depth,
    );
    let (nl, nb, _) = adf.leaf_stats_bytes();
    println!("[build] {:?} | 叶 {nl} | ≈{} KB", t.elapsed(), nb / 1024);

    // ── 4) 提取 + 体检 + 导出 ──
    let h_ext = h_min * mult;
    let t = Instant::now();
    let mut mesh = nets::extract_uniform(&adf, h_ext);
    // 碎屑清理:丢掉 <50 三角、或体积 <0.01% 总体的孤立分量
    // (场里的小符号岛,在查看器里表现为浮在表面附近的斑点)
    let (nd, nt) = mesh.drop_small_components(50, 1e-4);
    if nt > 0 {
        println!("[clean] 丢弃碎屑分量 {nd} 个 / 三角 {nt}");
    }
    // 平滑 + 投影回零等值面:削掉叶边处的扇贝锯齿(位移上限 = 半格)
    let t2 = Instant::now();
    let (dm, dx) = mesh.smooth_and_project(&adf, 3, h_ext as f32 * 0.5);
    println!(
        "[smooth] {:?} | 位移 mean {dm:.4} max {dx:.4} mm",
        t2.elapsed()
    );
    let (open, nonman, comps) = mesh.edge_stats();
    let vol = mesh.signed_volume().abs();
    println!(
        "[nets] h={h_ext:.4}({mult}×h_min) {:?} | 顶点 {} 三角 {}",
        t.elapsed(),
        mesh.verts.len(),
        mesh.tris.len()
    );
    println!(
        "[nets.topology] 开放边(洞) {open} | 非流形边 {nonman} | 连通分量 {comps}{}",
        match (open, nonman) {
            (0, 0) => " → 闭合流形(可打印)",
            (0, _) => " → 闭合但含非流形边(切片软件一般可容错)",
            _ => " → **有开口**",
        }
    );
    println!(
        "[volume] 重建 {vol:.1} vs 输入 {vol_in:.1} mm³({:+.2}%)",
        (vol - vol_in) / vol_in * 100.0
    );
    let t = Instant::now();
    nets::write_stl(&out, &mesh).expect("STL 写出失败");
    println!("[stl] {out} 写出 {:?}", t.elapsed());
}
