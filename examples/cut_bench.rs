//! P44/P45 切削吞吐基准(零依赖)。
//!
//! 在零件顶部生成往复铣刀路(球头刀沿短直线段扫掠 = 胶囊),按不同批大小送入 ADF 增量刷新,
//! 测每秒处理的刀路段数与每批延迟;模拟"刀路按固定速率到达、每 200ms 处理一批"的实时节奏;
//! 再测切削区 32³ 块的提取耗时(实时显示"按块重提取"的单价)。
//!
//! 运行(项目根目录):
//!   cargo run --release --example cut_bench -- [stl] [段数=3000] [刀半径mm=2] [段长mm=0.5]
//! 结果同时写入 out/cut_bench_<模型名>.txt。
//! 环境变量:ADF_MAX_DEPTH(深度上限,默认 10)、ADF_CUT_CELL(刀路索引格宽 mm)、
//!          ADF_REFRESH_UNIT_DEPTH(刷新单元深度,默认 6)、ADF_BENCH_K(三模式对照段数,默认 300)。

use adf_proto::*;
use std::fmt::Write as _;
use std::time::{Duration, Instant};

static LOG: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());

macro_rules! log {
    ($($t:tt)*) => {{
        let line = format!($($t)*);
        println!("{line}");
        let mut g = LOG.lock().unwrap();
        let _ = writeln!(g, "{line}");
    }};
}

fn arg<T: std::str::FromStr>(args: &[String], i: usize, default: T) -> T {
    args.get(i).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn f3(p: [Real; 3]) -> String {
    format!("[{:.1}, {:.1}, {:.1}]", p[0], p[1], p[2])
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// 与 stl_roundtrip 相同的域:外扩 6% + 1mm,深度按 h_target = 0.1 取,上限 ADF_MAX_DEPTH(默认 10)。
fn build(tris: &[[[Real; 3]; 3]], quiet: bool) -> Adf {
    let field = meshfield::MeshField::new(tris.to_vec());
    let (amn, amx) = (field.aabb_min, field.aabb_max);
    let extent = (amx[0] - amn[0]).max((amx[1] - amn[1]).max(amx[2] - amn[2]));
    let margin = extent * 0.06 + 1.0;
    let min = [amn[0] - margin, amn[1] - margin, amn[2] - margin];
    let size = extent + 2.0 * margin;
    let cap: u8 = std::env::var("ADF_MAX_DEPTH").ok().and_then(|v| v.parse().ok()).unwrap_or(10).clamp(6, 12);
    let max_depth = (((size / 0.1).log2().ceil()) as u8).clamp(6, cap);
    let t = Instant::now();
    let adf = Adf::build_with(Composite { base: BaseField::Mesh(field), cuts: Vec::new() }, min, size, 0.002, 0.5, max_depth);
    if !quiet {
        let (nl, _, _) = adf.leaf_stats_bytes();
        log!(
            "[build] {:.2}s | depth {} | h_min {:.4}mm | 叶 {nl} | 节点表 {}",
            t.elapsed().as_secs_f64(),
            adf.max_depth,
            adf.size / (1u64 << adf.max_depth) as Real,
            adf.nodes.len()
        );
    }
    adf
}

/// 往复铣刀路:XY 取包围盒中间 70%,行距 = 刀半径,每层下切 0.5mm(首层刀底 = 顶面下 0.5mm);
/// 层间沿 Y 反向走,换层只是一段 0.5mm 的下插。
fn toolpath(amn: [Real; 3], amx: [Real; 3], r: Real, seg: Real, n: usize) -> Vec<Capsule> {
    let lerp = |a: Real, b: Real, t: Real| a + (b - a) * t;
    let (x0, x1) = (lerp(amn[0], amx[0], 0.15), lerp(amn[0], amx[0], 0.85));
    let (y0, y1) = (lerp(amn[1], amx[1], 0.15), lerp(amn[1], amx[1], 0.85));
    let rows = (((y1 - y0) / r).floor() as usize).max(1) + 1;
    let k = (((x1 - x0) / seg).ceil() as usize).max(1);
    let mut pts: Vec<[Real; 3]> = Vec::new();
    let mut fwd = true;
    let mut layer = 0usize;
    while pts.len() <= n {
        let z = amx[2] - 0.5 * (layer + 1) as Real + r;
        for ri in 0..rows {
            let row = if layer % 2 == 0 { ri } else { rows - 1 - ri };
            let y = (y0 + row as Real * r).min(y1);
            for i in 0..=k {
                let t = i as Real / k as Real;
                let x = if fwd { lerp(x0, x1, t) } else { lerp(x1, x0, t) };
                pts.push([x, y, z]);
            }
            fwd = !fwd;
        }
        layer += 1;
    }
    pts.windows(2).take(n).map(|w| Capsule { a: w[0], b: w[1], r }).collect()
}

struct Stats {
    label: String,
    batch: usize,
    segs: usize,
    total: Duration,
    lat: Vec<Duration>,
}

fn feed(adf: &mut Adf, caps: &[Capsule], batch: usize, mode: CutMode, label: &str) -> Stats {
    let batch = batch.max(1);
    let mut lat = Vec::with_capacity(caps.len() / batch + 1);
    let t0 = Instant::now();
    for ch in caps.chunks(batch) {
        let t = Instant::now();
        adf.add_cuts_mode(ch, mode);
        lat.push(t.elapsed());
    }
    Stats { label: label.to_string(), batch, segs: caps.len(), total: t0.elapsed(), lat }
}

fn report(s: &Stats, adf: &Adf) {
    if s.lat.is_empty() {
        log!("[{}] 无数据", s.label);
        return;
    }
    let mut l = s.lat.clone();
    l.sort();
    let q = |f: f64| ms(l[((l.len() - 1) as f64 * f).round() as usize]);
    // 前 1/4 与后 1/4 批次的每段平均耗时:看刀路越切越多时是否变慢
    let quarter = (s.lat.len() / 4).max(1);
    let per_seg = |part: &[Duration]| part.iter().sum::<Duration>().as_secs_f64() / (part.len() * s.batch) as f64 * 1e6;
    let (nl, bytes, _) = adf.leaf_stats_bytes();
    log!(
        "[{}] 段 {} | 总 {:.2}s | {:.0} 段/s | 每批 p50 {:.2}ms p99 {:.2}ms max {:.2}ms | 每段 前1/4 {:.0}µs 后1/4 {:.0}µs | 叶 {nl}({} KB)| 节点表 {}",
        s.label,
        s.segs,
        s.total.as_secs_f64(),
        s.segs as f64 / s.total.as_secs_f64(),
        q(0.5),
        q(0.99),
        ms(*l.last().unwrap()),
        per_seg(&s.lat[..quarter]),
        per_seg(&s.lat[s.lat.len() - quarter..]),
        bytes / 1024,
        adf.nodes.len()
    );
}

fn sample_box(rng: &mut Rng, mn: [Real; 3], mx: [Real; 3]) -> [Real; 3] {
    [
        mn[0] + rng.f01() * (mx[0] - mn[0]),
        mn[1] + rng.f01() * (mx[1] - mn[1]),
        mn[2] + rng.f01() * (mx[2] - mn[2]),
    ]
}

/// 两棵树在刀路区的对照:近表面(|a| < band)逐位不同数与最大差、符号不同数(|真值| > 2h)。
fn compare(name: &str, a: &Adf, b: &Adf, mn: [Real; 3], mx: [Real; 3]) {
    let h = a.size / (1u64 << a.max_depth) as Real;
    let mut rng = Rng::new(9);
    let (mut near, mut diff, mut sign) = (0u32, 0u32, 0u32);
    let mut dmax = 0.0f64;
    for _ in 0..20000 {
        let p = sample_box(&mut rng, mn, mx);
        let (x, y) = (a.eval(p), b.eval(p));
        if a.true_sd(p).abs() > 2.0 * h && (x < 0.0) != (y < 0.0) {
            sign += 1;
        }
        if x.abs() < a.band as f32 {
            near += 1;
            if x.to_bits() != y.to_bits() {
                diff += 1;
                dmax = dmax.max((x - y).abs() as Real);
            }
        }
    }
    log!(
        "[对照] {name}:树节点 {} vs {} | 近表面 {near} 点中不同 {diff}(最大差 {dmax:.4}mm)| 符号不同(|真值| > 2h){sign}",
        a.nodes.len(),
        b.nodes.len()
    );
}

/// 场精度:刀路区随机点,ADF 插值 vs 复合场真值(base + 全部刀路)。
fn accuracy(a: &Adf, mn: [Real; 3], mx: [Real; 3]) -> String {
    let h = a.size / (1u64 << a.max_depth) as Real;
    let mut rng = Rng::new(77);
    let (mut emax, mut esum, mut cnt, mut bad) = (0.0f64, 0.0f64, 0u32, 0u32);
    for _ in 0..20000 {
        let p = sample_box(&mut rng, mn, mx);
        let t = a.true_sd(p);
        let e = a.eval(p) as Real;
        if t.abs() < a.band {
            let d = (e - t).abs();
            emax = emax.max(d);
            esum += d;
            cnt += 1;
        }
        if t.abs() > 2.0 * h && (e < 0.0) != (t < 0.0) {
            bad += 1;
        }
    }
    format!("近表面 {cnt} 点误差 max {emax:.4} mean {:.4}mm | 符号错(|真值| > 2h) {bad}", esum / cnt.max(1) as Real)
}

/// 实时节奏模拟:刀路以 `rate` 段/s 匀速到达;每个 `tick` 边界处理"已到达未处理"的全部段为一批,
/// 上一批没做完就做完再接着取。延迟 = 一段到达 → 它所在批次刷新完成(只含 ADF 刷新,不含提取/上传)。
fn simulate(tris: &[[[Real; 3]; 3]], caps: &[Capsule], rate: f64, tick: f64) {
    let mut a = build(tris, true);
    let n = caps.len();
    let (mut done, mut t, mut busy) = (0usize, 0.0f64, 0.0f64);
    let (mut lat_max, mut lat_sum, mut lat_cnt) = (0.0f64, 0.0f64, 0usize);
    let mut sizes: Vec<usize> = Vec::new();
    let mut dur: Vec<f64> = Vec::new();
    while done < n {
        let arrived = ((t * rate).floor() as usize).min(n);
        if arrived <= done {
            t = ((t / tick).floor() + 1.0) * tick;
            continue;
        }
        let t0 = Instant::now();
        a.add_cuts(&caps[done..arrived]);
        let d = t0.elapsed().as_secs_f64();
        let finish = t + d;
        for i in done..arrived {
            let l = finish - i as f64 / rate;
            lat_max = lat_max.max(l);
            lat_sum += l;
            lat_cnt += 1;
        }
        sizes.push(arrived - done);
        dur.push(d);
        busy += d;
        done = arrived;
        t = ((finish / tick).floor() + 1.0) * tick;
    }
    let span = n as f64 / rate;
    dur.sort_by(|x, y| x.total_cmp(y));
    log!(
        "[实时 {rate:.0} 段/s, 每 {:.0}ms 一批] 批 {} 个,平均 {:.0} 段 | 每批耗时 p50 {:.0}ms max {:.0}ms | 刷新占用 {:.0}% | 延迟 平均 {:.0}ms 最大 {:.0}ms | {}",
        tick * 1e3,
        sizes.len(),
        n as f64 / sizes.len() as f64,
        dur[dur.len() / 2] * 1e3,
        dur.last().unwrap() * 1e3,
        busy / span * 100.0,
        lat_sum / lat_cnt as f64 * 1e3,
        lat_max * 1e3,
        if busy <= span { "跟得上" } else { "跟不上(积压)" }
    );
}

/// 切削区 32³ 块逐块提取(实时显示按块重提取的单价)。
fn blocks(a: &Adf, mn: [Real; 3], mx: [Real; 3]) {
    const B: usize = 32;
    let h = a.size / (1u64 << a.max_depth) as Real;
    let n = (a.size / h).round() as usize;
    let lo: [usize; 3] = std::array::from_fn(|k| (((mn[k] - a.min[k]) / h).floor().max(0.0) as usize).min(n - 1) / B);
    let hi: [usize; 3] = std::array::from_fn(|k| (((mx[k] - a.min[k]) / h).ceil().max(0.0) as usize).min(n - 1) / B);
    nets::set_quiet(true);
    let mut times: Vec<Duration> = Vec::new();
    let (mut tris, mut verts, mut empty) = (0usize, 0usize, 0usize);
    let t_all = Instant::now();
    for bz in lo[2]..=hi[2] {
        for by in lo[1]..=hi[1] {
            for bx in lo[0]..=hi[0] {
                let off = [bx * B, by * B, bz * B];
                let shape: [usize; 3] = std::array::from_fn(|k| B.min(n - off[k]));
                let t = Instant::now();
                let m = nets::extract_box(a, h, off, shape);
                times.push(t.elapsed());
                tris += m.tris.len();
                verts += m.verts.len();
                if m.tris.is_empty() {
                    empty += 1;
                }
            }
        }
    }
    let total = t_all.elapsed();
    nets::set_quiet(false);
    times.sort();
    let q = |f: f64| ms(times[((times.len() - 1) as f64 * f).round() as usize]);
    log!(
        "[blocks] 切削区 {B}³ 块 {} 个(空 {empty})| 串行合计 {:.1}ms | 每块 p50 {:.2}ms p90 {:.2}ms max {:.2}ms | 三角 {tris} 顶点 {verts}",
        times.len(),
        ms(total),
        q(0.5),
        q(0.9),
        ms(*times.last().unwrap())
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| "stls/桌腿固定器.stl".to_string());
    let nseg: usize = arg(&args, 2, 3000);
    let r: Real = arg(&args, 3, 2.0);
    let seg: Real = arg(&args, 4, 0.5);
    let k: usize = std::env::var("ADF_BENCH_K").ok().and_then(|v| v.parse().ok()).unwrap_or(300);

    let tris = stlio::load_stl(&path).expect("STL 加载失败").tris;
    let (amn, amx) = {
        let probe = meshfield::MeshField::new(tris.clone());
        (probe.aabb_min, probe.aabb_max)
    };
    let caps = toolpath(amn, amx, r, seg, nseg);
    let k = k.min(caps.len());
    let (mut pmn, mut pmx) = ([Real::INFINITY; 3], [Real::NEG_INFINITY; 3]);
    for c in &caps {
        let (a, b) = c.aabb();
        for i in 0..3 {
            pmn[i] = pmn[i].min(a[i]);
            pmx[i] = pmx[i].max(b[i]);
        }
    }
    log!("[path] {path} | 段 {} | 刀半径 {r}mm 段长 {seg}mm | 刀路包围盒 {} – {}", caps.len(), f3(pmn), f3(pmx));
    // 抽样区:刀路包围盒外扩 1mm
    let smn = [pmn[0] - 1.0, pmn[1] - 1.0, pmn[2] - 1.0];
    let smx = [pmx[0] + 1.0, pmx[1] + 1.0, pmx[2] + 1.0];

    // ① 三种模式逐段处理同样的前 k 段:速度与一致性
    let mut lin = build(&tris, false);
    let st = feed(&mut lin, &caps[..k], 1, CutMode::Linear, "全量遍历 逐段");
    report(&st, &lin);
    let mut idx = build(&tris, true);
    let st = feed(&mut idx, &caps[..k], 1, CutMode::Indexed, "索引 逐段");
    report(&st, &idx);
    compare("全量遍历 vs 索引", &lin, &idx, smn, smx);
    drop(idx);
    let mut inc = build(&tris, true);
    let st = feed(&mut inc, &caps[..k], 1, CutMode::Incremental, "增量 逐段");
    report(&st, &inc);
    compare("全量遍历 vs 增量", &lin, &inc, smn, smx);
    drop(lin);

    // ② 增量 逐段走完全部刀路:看越切越多是否变慢
    let st = feed(&mut inc, &caps[k..], 1, CutMode::Incremental, "增量 逐段(续)");
    report(&st, &inc);
    log!("[精度] 增量 逐段:{}", accuracy(&inc, smn, smx));
    drop(inc);

    // ③ 批量:模拟"每 200ms 攒一批"
    let mut last: Option<Adf> = None;
    for b in [16usize, 128, 256, 1024] {
        drop(last.take()); // 先释放上一组再建下一组,免得同时占两份树内存
        let mut a = build(&tris, true);
        let st = feed(&mut a, &caps, b, CutMode::Incremental, &format!("增量 每批 {b}"));
        report(&st, &a);
        log!("[精度] 每批 {b}:{}", accuracy(&a, smn, smx));
        last = Some(a);
    }
    let a = last.expect("至少一组批量");
    log!("[index] 已登记刀路 {} | 索引格宽 {:.2}mm", a.cut_index.len(), a.cut_index.cell());

    // ④ 实时节奏:刀路按固定速率到达,每 200ms 处理一批
    for rate in [1000.0, 3000.0, 10000.0] {
        simulate(&tris, &caps, rate, 0.2);
    }

    // ⑤ 提取:切削区逐块 vs 整件
    blocks(&a, smn, smx);
    let h = a.size / (1u64 << a.max_depth) as Real;
    let t = Instant::now();
    let m = nets::extract_uniform(&a, h);
    log!("[nets] 整件提取 {:.0}ms | 三角 {}", ms(t.elapsed()), m.tris.len());

    let stem = std::path::Path::new(&path).file_stem().and_then(|s| s.to_str()).unwrap_or("model").to_string();
    let out = format!("out/cut_bench_{stem}.txt");
    let _ = std::fs::create_dir_all("out");
    match std::fs::write(&out, LOG.lock().unwrap().as_bytes()) {
        Ok(()) => println!("[saved] {out}"),
        Err(e) => println!("[saved] 写 {out} 失败:{e}"),
    }
}
