//! ADF 内存核算:用计数分配器把"建树前后堆增量"分清,而不是靠 RSS 猜。
//! 运行:cargo run --release --example mem_report -- [stl] [h_target]
//!
//! 口径:
//! - `[heap]` 行 = 全局堆当前占用(仅进程堆,不含栈/映射文件/代码段);
//! - 每个阶段打印 `Δ` = 该阶段净增,即该对象**活着**占的字节;
//! - `[peak]` = 全程堆峰值(ADF 与提取网格不同时存在,故峰值≈构建期);
//! - 树结构另按 `size_of::<Node>() × nodes.len()` 给出"理论字节",与分配器口径互校。

use adf_proto::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

struct Counting;

static CUR: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

#[inline]
fn add(n: usize) {
    let now = CUR.fetch_add(n, Ordering::Relaxed) + n;
    PEAK.fetch_max(now, Ordering::Relaxed);
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            add(l.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        CUR.fetch_sub(l.size(), Ordering::Relaxed);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, l, new) };
        if !q.is_null() {
            if new >= l.size() {
                add(new - l.size());
            } else {
                CUR.fetch_sub(l.size() - new, Ordering::Relaxed);
            }
        }
        q
    }
}

#[global_allocator]
static A: Counting = Counting;

fn heap() -> usize {
    CUR.load(Ordering::Relaxed)
}

fn mb(b: usize) -> f64 {
    b as f64 / (1024.0 * 1024.0)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| "stls/Turbo_Impeller.stl".to_string());
    let h_target: Real = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.1);
    let mult: Real = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1.0);

    println!("[cfg] 进程堆基线 {} KB", heap() / 1024);

    // ── 1) STL ──
    let base0 = heap();
    let t = Instant::now();
    let tm = stlio::load_stl(&path).expect("STL 加载失败");
    let ntri = tm.tris.len();
    println!(
        "[stl] {ntri} 三角 | 堆 {} KB (Δ {:.1} MB) | {:?}",
        heap() / 1024,
        mb(heap() - base0),
        t.elapsed()
    );

    // ── 2) 网格场(三角 + BVH + 伪法线) ──
    let base1 = heap();
    let field = meshfield::MeshField::new(tm.tris);
    let field_bytes = heap() - base1;
    println!(
        "[field] 三角+BVH+伪法线+边法线表 | 堆 {} KB | Δ {:.2} MB ({:.0} B/三角)",
        heap() / 1024,
        mb(field_bytes),
        field_bytes as f64 / ntri as f64
    );

    let (amn, amx) = (field.aabb_min, field.aabb_max);
    let extent = (amx[0] - amn[0]).max((amx[1] - amn[1]).max(amx[2] - amn[2]));
    let margin = extent * 0.06 + 1.0;
    let min = [amn[0] - margin, amn[1] - margin, amn[2] - margin];
    let size = extent + 2.0 * margin;
    let max_depth = (((size / h_target).log2().ceil()) as u8).clamp(6, 10);
    let h_min = size / (1u64 << max_depth) as Real;
    let h_ext = h_min * mult;
    println!(
        "[domain] size {size:.2}mm | depth {max_depth} | h_min {h_min:.4} | h_ext {h_ext:.4}"
    );

    // ── 3) 建 ADF(含烘焙网格 + 格点缓存页 + 八叉树) ──
    let base2 = heap();
    let peak0 = PEAK.load(Ordering::Relaxed);
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
    let adf_bytes = heap() - base2;
    let peak_build = PEAK.load(Ordering::Relaxed).saturating_sub(peak0);
    let (nl, nb_tree, hist) = adf.leaf_stats_bytes();
    let node_sz = std::mem::size_of::<Node>();
    let mem = adf.mem_bytes();
    println!("[adf.mem] {}", mem.summary());
    println!(
        "[check] 分配器实测 {:.1} MB(建树 Δ {:.1} + 网格场 {:.1}) vs 核算 {:.1} MB(差 {:+.1}%)",
        mb(adf_bytes + field_bytes),
        mb(adf_bytes),
        mb(field_bytes),
        mb(mem.total()),
        (mem.total() as f64 / (adf_bytes + field_bytes).max(1) as f64 - 1.0) * 100.0
    );
    println!(
        "[adf] build {el:?} | 堆 {} KB | Δ {:.1} MB | 构建期峰值 {:.1} MB",
        heap() / 1024,
        mb(adf_bytes),
        mb(peak_build)
    );
    println!(
        "[tree] 叶 {nl} | 节点 {} | size_of::<Node>() {node_sz} B | 树占用 {:.2} MB (核算式 {:.2} MB)",
        adf.nodes.len(),
        mb(adf.nodes.len() * node_sz),
        mb(nb_tree as usize)
    );
    let deep: u64 = (0..17).map(|d| hist[d]).sum::<u64>();
    let _ = deep;
    let mut acc = String::new();
    for d in 6..=max_depth.min(16) {
        if hist[d as usize] > 0 {
            acc.push_str(&format!("d{d}:{} ", hist[d as usize]));
        }
    }
    println!("[tree.depth] {acc}");
    println!(
        "[adf.total] ADF 自身(树+烘焙+格点页,不含场)≈ {:.1} MB | 含场 ≈ {:.1} MB",
        mb(adf_bytes - field_bytes.max(0)),
        mb(heap() - base0)
    );

    // ── 4) 释放顺序验证(ADF 拥有场,drop 后应回到 STL 之后) ──
    drop(adf);
    println!("[drop] 释放 ADF 后堆 {} KB(基线 {} KB)", heap() / 1024, base0 / 1024);
    println!(
        "[peak] 进程堆峰值 {:.1} MB",
        mb(PEAK.load(Ordering::Relaxed))
    );
}
