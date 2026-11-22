//! STL → ADF → STL 往返转换(**不做任何切削**)。
//!
//! 用途:把输入网格重建为等值面网格,用于检查场的重建质量与网格可打印性。
//! 与 `stl-demo` 的区别:后者会为演示切削能力打两个胶囊孔,本工具**只重建**。
//!
//! 用法:
//!   cargo run --release --bin stl_roundtrip -- <输入.stl> [h_target] [输出.stl] [提取倍率]
//! 例:
//!   cargo run --release --bin stl_roundtrip -- stls/Turbo_Impeller.stl 0.2 out.stl 2

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
    let closed = field.has_closed_surface();
    let (amn, amx) = (field.aabb_min, field.aabb_max);
    let extent = (amx[0] - amn[0]).max((amx[1] - amn[1]).max(amx[2] - amn[2]));
    let margin = extent * 0.06 + 1.0;
    let min = [amn[0] - margin, amn[1] - margin, amn[2] - margin];
    let size = extent + 2.0 * margin;
    // 深度上限(P42 自动,按薄特征):默认 10。只有 h_target 要求的深度被上限截住(格宽 > h_target),且零件
    // 表面"一格宽内另有一片表面"的面积占比 > 5%(薄壁/窄缝;ADF_THIN_FRAC 可调)时才取 11。依据(P41/P42 实测):
    // 网格垫子(薄特征 22%)depth 10 → 源点 max 0.226mm、0.47% > 0.05mm,11 → 0.063 / 0.13%;爪子刀(≈12%)
    // 0.18 → 0.038;花瓶、翅膀(≤ 2%)10 已达标(0.023 / 0.050),升 11 只换来 2–4× 耗时与更大文件。
    // ADF_MAX_DEPTH 显式指定时优先(叶/缓存/提取内存每级 ×4~8)。
    let need_depth = ((size / h_target).log2().ceil()) as u8;
    let depth_cap: u8 = match std::env::var("ADF_MAX_DEPTH").ok().and_then(|v| v.parse::<u8>().ok()).filter(|&d| d > 0) {
        Some(d) => d.clamp(6, 12),
        None if need_depth > 10 => {
            let h10 = size / 1024.0;
            let frac_max: Real = std::env::var("ADF_THIN_FRAC").ok().and_then(|v| v.parse().ok()).unwrap_or(0.05);
            let t = Instant::now();
            let thin = field.thin_fraction(h10, 4000);
            let cap = if thin > frac_max { 11 } else { 10 };
            println!(
                "[depth.auto] 薄特征占比 {:.1}%(探距 {h10:.3}mm,阈值 {:.0}%)→ 上限 {cap} | {:?}",
                thin * 100.0,
                frac_max * 100.0,
                t.elapsed()
            );
            cap
        }
        None => 10,
    };
    let max_depth = need_depth.clamp(6, depth_cap);
    let h_min = size / (1u64 << max_depth) as Real;
    println!("[domain] 尺寸 {size:.2}mm | 最大边长 {extent:.1}mm | depth {max_depth}(上限 {depth_cap}) | h_min {h_min:.4}mm");

    let h_ext = h_min * mult;
    let t = Instant::now();
    let layers = field.into_roundtrip_layers(h_ext);
    let preserve_parts = closed && (!watertight || layers.len() > 1);
    if layers.len() > 1 {
        println!("[nets.layers] {} 组独立实体共用原域、格距和提取倍率", layers.len());
    }
    if !closed {
        println!("[reference] 输入网格含未闭合边，使用距离场交点及后处理");
    } else if !watertight {
        println!("[reference] 闭合多重壳以穿越奇偶定号，采用原始三角面交点");
    }
    let mut parts = layers.into_iter().map(|field|
        reconstruct_layer(field, min, size, max_depth, h_ext, closed, preserve_parts)
    );
    let mut mesh = parts.next().expect("至少一个网格场");
    for part in parts {
        let start = u32::try_from(mesh.verts.len()).expect("网格顶点数超过 u32 索引范围");
        mesh.verts.extend(part.verts);
        mesh.tris.extend(part.tris.into_iter().map(|t|
            t.map(|i| i.checked_add(start).expect("网格顶点索引溢出"))
        ));
    }
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

fn reconstruct_layer(
    field: meshfield::MeshField, min: [Real; 3], size: Real, max_depth: u8, h_ext: Real,
    closed: bool, preserve_parts: bool,
) -> nets::Mesh {
    // 多壳件(P34):整件不水密/分多层时 preserve_parts 为真,旧版因此整层跳过折痕约束路径
    // (P26–P33 全部不生效,输出又大又糙)。这里按**层**判:本层自身水密就照常走折痕路径。
    // ADF_NO_SHELL_CREASES=1 恢复旧行为。
    // 本层闭合即可(成对反向边,或可用穿越奇偶定号的多重关联边,如妙蛙种子主体 1,245 条
    // "两片表面沿同一条边相接"的四面边)。折痕索引本身只取恰好成对的边,其余边不参与约束。
    let layer_tight = field.has_closed_surface();
    let crease_ok = !preserve_parts || (layer_tight && std::env::var("ADF_NO_SHELL_CREASES").is_err());
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

    // ── 4) 提取和后处理 ──
    let t = Instant::now();
    let mut mesh = nets::extract_mesh_roundtrip(&adf, h_ext);
    let t_extract = t.elapsed();
    // 多壳 STL 中小的孤立分量也可能是源模型细节。单壳沿用已有碎屑清理。
    let (nd, nt) = if preserve_parts { (0, 0) } else { mesh.drop_small_components(50, 1e-4) };
    if nt > 0 {
        println!("[clean] 丢弃碎屑分量 {nd} 个 / 三角 {nt}");
    }
    println!("[extract] 提取 {t_extract:?} | 碎屑清理 {:?} | 三角 {}", t.elapsed() - t_extract, mesh.tris.len());
    // 平滑 + 投影回零等值面:削掉叶边处的扇贝锯齿(位移上限 = 半格)
    // 参考三角面已提供几何约束。切削演示仍用距离场做平滑和吸附；
    // 在无切削模式中再次投向近似场会把已经恢复的棱角拉回粗场。
    let field_post = !closed || ["ADF_FIELD_POST", "ADF_REPAIR", "ADF_SNAP_LIM", "ADF_SNAP_STR"]
        .iter().any(|key| std::env::var_os(key).is_some());
    if field_post && std::env::var("ADF_NO_SMOOTH").is_err() {
        let t2 = Instant::now();
        let (dm, dx) = mesh.smooth_and_project(&adf, 3, h_ext as f32 * 0.5);
        println!(
            "[smooth] {:?} | 位移 mean {dm:.4} max {dx:.4} mm",
            t2.elapsed()
        );
    }
    // 折叠/薄片修复:边缘"破损"的真身(真折叠 + 针状薄片),塌缩最短边消除
    if std::env::var("ADF_REPAIR").is_ok() {
        let t3 = Instant::now();
        let (nc, nr) = mesh.repair_folds(&adf, 0.5, 0.004, 0.002, h_ext as f64 * 4.0, 20);
        println!(
            "[repair] {:?} | 塌缩 {nc} 边,删三角 {nr}",
            t3.elapsed()
        );
    }
    // QEM 自适应简化:平坦区大三角、缘带密三角("按需面片")。
    // ADF_ERR2=平方误差上限 mm²(默认 4e-4 ≈ 0.02mm 弦差);ADF_NO_DEC=1 跳过。
    if std::env::var("ADF_NO_DEC").is_err() {
        let err2: f64 = std::env::var("ADF_ERR2").ok().and_then(|v| v.parse().ok()).unwrap_or(1e-4);
        let t4 = Instant::now();
        let nt0 = mesh.tris.len();
        // 特征保护角(ADF_FEAT_DEG,默认 25°):planar-pair 约束的边阈值
        let feat_deg: f64 = std::env::var("ADF_FEAT_DEG").ok().and_then(|v| v.parse().ok()).unwrap_or(25.0);
        let creases = match &adf.field.base {
            BaseField::Mesh(source) if closed && !field_post && crease_ok =>
                Some(nets::ReferenceCreases::new(source, h_ext * 1.2, feat_deg.max(35.0))),
            _ => None,
        };
        let before_crease_fit = creases.as_ref().map(|_| mesh.clone());
        // P37:大网格分段并行 QEM(ADF_DECIMATE_SERIAL=1 退回串行)
        let nc = mesh.decimate_with_creases_parallel(err2, feat_deg, creases.as_ref());
        println!(
            "[decimate] {:?} | 塌缩 {nc} 边 | 三角 {nt0} → {} | 文件 ≈{} MB",
            t4.elapsed(),
            mesh.tris.len(),
            mesh.tris.len() * 50 / 1024 / 1024
        );
        // 简化把凸曲面顶点按弦高内缩(QEM 最优在曲面内侧),圆缘会变成
        // 多段内凹折线。投影回零面把弦高归零(平面区幂等)。
        if field_post && std::env::var("ADF_NO_PROJ").is_err() {
            let (pm, px) = mesh.project_to_zero(&adf, 3, h_ext as f64);
            println!("[project] 首轮位移 mean {pm:.4} max {px:.4} mm");
        }
        // 脊线重吸附:decimate 对锐边放宽阈值合并锯齿阶梯(否则满屏短黑线),
        // 合并点残余偏移由强吸附拉回原始缘线 → 脊线成长段光滑折线。
        // 吸附参数可调:lim 必须够大才能把 decimate 挤出去的脊线顶点拉回
        // (0.5mm 对 40mm 半径的圆缘偏小,残余台阶 ~1mm 会留成"牙齿")
        let snap_lim: f64 = std::env::var("ADF_SNAP_LIM")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.5);
        let snap_str: f64 = std::env::var("ADF_SNAP_STR")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.95);
        if field_post && std::env::var("ADF_NO_SNAP2").is_err() {
            let t5 = Instant::now();
            let ns = mesh.snap_sharp_to_rim(&adf, snap_lim, snap_str);
            // 吸附把锐边顶点拉回缘线,邻接窄带三角被挤扁 → 清理针状薄片
            let nsl = mesh.collapse_slivers_ex(0.02, 6, 0.08, 0.3);
            println!("[snap2] {:?} | 重吸附 {ns} 锐边顶点,清薄片 {nsl}", t5.elapsed());
            // 真折叠(边缘多余面片)修复:场梯度仲裁(折叠面法线逆梯度,
            // 合法薄板带顺梯度)——塌缩真折叠,合并点投影回零面;修复点
            // 的残余偏移由随后的重吸附拉回缘线。
            if std::env::var("ADF_NO_REPAIR2").is_err() {
                let t6 = Instant::now();
                let (nc2, nr2) = mesh.repair_folds(&adf, 0.5, 0.004, 0.002, h_ext as f64 * 4.0, 10);
                let nsl2 = mesh.collapse_slivers_ex(0.02, 6, 0.08, 0.3);
                let ns2 = mesh.snap_sharp_to_rim(&adf, snap_lim, snap_str);
                println!(
                    "[repair2] {:?} | 折叠塌缩 {nc2},删面 {nr2},清薄片 {nsl2},重吸附 {ns2}",
                    t6.elapsed()
                );
            }
        }
        if std::env::var("ADF_NO_PROJ").is_err()
            && let BaseField::Mesh(reference) = &adf.field.base
        {
            let t = Instant::now();
            mesh.fit_reference(reference, h_ext * 0.5, closed, creases.as_ref());
            println!("[reference] 原始表面投影 {:?}", t.elapsed());
        }
        // 折痕毛刺清理(P26/P27):投影收尾后,翻边恢复缺失的折痕边,再把停在折痕旁
        // 0.005–1.2h 的"半吸附"顶点塌缩/吸附到折痕上,消掉主棱线旁的小毛刺/阶梯/平行短线。只在折痕约束路径上跑;ADF_NO_SPUR=1 可关。
        if let Some(grid) = creases.as_ref()
            && std::env::var("ADF_NO_SPUR").is_err()
        {
            let t = Instant::now();
            let nt0 = mesh.tris.len();
            let (nf, nc, nm) = mesh.merge_crease_spurs(grid, 30.0, 0.005, h_ext * 1.2, h_ext * 3.0, 0.02, 2);
            println!(
                "[spur] {:?} | 翻边 {nf} / 塌缩 {nc} / 吸附 {nm} | 三角 {nt0} → {}",
                t.elapsed(),
                mesh.tris.len()
            );
        }
        // 按源网格复核修补(P28):面级缺陷(斜跨折痕、退化、偏离源面的小鳍片)用翻边/塌缩消除。
        // 只在折痕约束路径上跑;ADF_NO_REFFIX=1 可关。
        if let Some(grid) = creases.as_ref()
            && std::env::var("ADF_NO_REFFIX").is_err()
            && let BaseField::Mesh(reference) = &adf.field.base
        {
            let t = Instant::now();
            let (b0, nf, nc, b1) = mesh.repair_reference_defects(reference, grid, 30.0, 4);
            println!("[reffix] {:?} | 可见缺陷面 {b0} → {b1} | 翻边 {nf} / 塌缩 {nc}", t.elapsed());
        }
        // 以源网格为准的二次简化(P29):顶点已在源面上,按源网格判误差继续合并(平面区零误差),
        // 随后再做一次面级复核。默认开启;ADF_NO_REFDEC=1 可关,ADF_REF_TOL 调容差(默认 0.01mm)。
        if let Some(grid) = creases.as_ref()
            && std::env::var("ADF_NO_REFDEC").is_err()
            && let BaseField::Mesh(reference) = &adf.field.base
        {
            // 默认容差 0.02mm(P36):0.01 时曲面边隐含弦高中位数只有 0.004–0.006mm,三角数主要由容差决定;
            // 0.02 仍远低于 FDM 层高(0.1)和光固化 XY 精度(≈0.035)。ADF_REF_TOL=0.01 取最高保真。
            let tol: Real = std::env::var("ADF_REF_TOL").ok().and_then(|v| v.parse().ok()).unwrap_or(0.02);
            let t = Instant::now();
            let nt0 = mesh.tris.len();
            // 各子步骤耗时(P36):定位慢在哪一步
            let mut lap = Instant::now();
            let mut laps: Vec<(&str, std::time::Duration)> = Vec::new();
            let mut tick = |name: &'static str, lap: &mut Instant| {
                laps.push((name, lap.elapsed()));
                *lap = Instant::now();
            };
            let mut n = mesh.decimate_to_reference_parallel(reference, grid, tol, 0.005);
            tick("简化", &mut lap);
            // 源折痕嵌入(P30):两条折痕汇合处(如圆柱与方板外棱相切)窄于一格的窄条被提取抹成斜坡,
            // 在这里把缺失的折痕边切回来,再清理切边留下的短边并重新简化。ADF_NO_EMBED=1 可关。
            let (mut nk, mut ncut) = (0, 0);
            if std::env::var("ADF_NO_EMBED").is_err() {
                (nk, ncut) = mesh.embed_reference_creases(grid, 0.005, h_ext * 1.8);
            }
            tick("嵌入", &mut lap);
            // 交替收尾(P32):二次简化的"共面即精确"分支会把顶点并到一个本来就斜跨折痕的面上,
            // 重新造出一条斜跨外棱的对角线。每次简化后都再做一轮折痕翻边/塌缩/吸附,直到两者都不再
            // 改动(至多 4 轮)。P36:补充简化只从锐边带起步(嵌入/清理只改折痕附近),不再全网格重试。
            // P38:大网格分段并行(alternate_crease_cleanup 内部判断规模)。
            n += mesh.alternate_crease_cleanup(reference, grid, tol, h_ext);
            tick("交替收尾", &mut lap);
            let (b0, nf, nc, b1) = mesh.repair_reference_defects(reference, grid, 30.0, 2);
            // 面级复核的翻边/塌缩也可能留下斜跨折痕的面:最后再做一次折痕翻边(不再简化)。
            mesh.merge_crease_spurs(grid, 30.0, 0.005, h_ext * 1.2, h_ext * 3.0, 0.02, 1);
            tick("复核", &mut lap);
            // 视觉毛刺清理(P33):塌缩造成"源件上没有的多余线"的微面。ADF_NO_VISFIX=1 可关。
            let nvis = if std::env::var("ADF_NO_VISFIX").is_err() { mesh.clean_visual_spurs(reference, 4) } else { 0 };
            tick("视觉清理", &mut lap);
            println!(
                "[refdec] {:?} | tol {tol}mm | 塌缩 {n} | 折痕嵌入 角点 {nk} / 切边 {ncut} | 视觉清理 {nvis} | 三角 {nt0} → {} | 文件 ≈{} KB | 复核 {b0} → {b1}(翻边 {nf} / 塌缩 {nc})",
                t.elapsed(),
                mesh.tris.len(),
                (84 + 50 * mesh.tris.len()) / 1024
            );
            println!(
                "[refdec.time] {}",
                laps.iter().map(|(k, d)| format!("{k} {:.1}s", d.as_secs_f64())).collect::<Vec<_>>().join(" | ")
            );
        }
        // 不同索引精确落到同一 f32 导出坐标时，STL 读回会改变顶点关联。
        // 在原有提取结果上重做常规简化，保留薄壁原有的连接关系。
        if let Some(original) = before_crease_fit
            && mesh.has_coincident_export_vertices()
            && let BaseField::Mesh(reference) = &adf.field.base
        {
            mesh = original;
            mesh.decimate(err2, feat_deg, 0);
            if std::env::var("ADF_NO_PROJ").is_err() {
                mesh.project_with_face_samples(reference, h_ext * 0.5, closed);
            }
            println!("[reference] 独立索引投影至重合坐标，已采用曲面拟合");
        }
    }
    mesh
}
