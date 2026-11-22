//! # adf-proto —— 自适应采样距离场(ADF)最小原型
//!
//! 对应《rust-cnc-cam-research.md》§8.10(多分辨率仿真)与 §5.15(spike 建议)。
//! 目标:在 Rust 中验证 Frisken 2000 判据的自适应细分 + composite 布尔减 +
//! 增量刷新的正确性、内存收益与精度上限,并**量化跨分辨率边界的实际误差**
//! (已知 T-junction 风险,spike 的意义就是把它的量级测出来)。
//!
//! ## 范围(2 周 spike 的最小闭环)
//! 1. 八叉树 + 角点采样(f32)+ Frisken 细分判据(中心真值 vs 三线性插值);
//! 2. 解析毛坯(圆角盒/球)+ 胶囊扫掠切削(composite ADF:true = max(stock, −cuts));
//! 3. 增量刷新(新扫掠 AABB 剪枝 → 判据重估 → 按需细分;不做粗化——已知限制);
//! 4. 验证台:误差统计、零点偏置(§9.12.1 A 类口径)、内存 vs 均匀格、ASCII 切片。
//!
//! ## 原型边界(均为报告已声明的研究级难题,不在本 spike 内)
//! - 角点按叶存储,不做全局顶点共享(跨叶连续性靠"同位置同值"保证);
//! - 只细分不粗化;
//! - 不做跨分辨率表面提取(验证走 `eval`,不输出网格);
//! - 单线程(确定性优先;并行方案见 §24.5.8 hp-adaptive-SDF 参照)。

mod creases;

pub type Real = f64;

// ─────────────────────────── 基础向量工具 ───────────────────────────

#[inline]
pub fn sub(a: [Real; 3], b: [Real; 3]) -> [Real; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
pub fn add(a: [Real; 3], b: [Real; 3]) -> [Real; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
#[inline]
pub fn scale(a: [Real; 3], k: Real) -> [Real; 3] {
    [a[0] * k, a[1] * k, a[2] * k]
}
#[inline]
pub fn dot(a: [Real; 3], b: [Real; 3]) -> Real {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
pub fn len(a: [Real; 3]) -> Real {
    dot(a, a).sqrt()
}
#[inline]
fn clamp(x: Real, lo: Real, hi: Real) -> Real {
    x.max(lo).min(hi)
}

// ─────────────────────────── 场原语 ───────────────────────────

/// 胶囊体 = 刀具沿直线段扫掠的扫掠体(§8.4:直线扫掠 = 刀具 ⊕ 线段)。
#[derive(Clone, Copy, Debug)]
pub struct Capsule {
    pub a: [Real; 3],
    pub b: [Real; 3],
    pub r: Real,
}

impl Capsule {
    pub fn sd(&self, p: [Real; 3]) -> Real {
        let ab = sub(self.b, self.a);
        let ap = sub(p, self.a);
        let ab2 = dot(ab, ab);
        let t = if ab2 <= 0.0 { 0.0 } else { clamp(dot(ap, ab) / ab2, 0.0, 1.0) };
        let closest = add(self.a, scale(ab, t));
        len(sub(p, closest)) - self.r
    }
    /// 扫掠 AABB(含半径)——refresh 剪枝用。
    pub fn aabb(&self) -> ([Real; 3], [Real; 3]) {
        let mut mn = [0.0 as Real; 3];
        let mut mx = [0.0 as Real; 3];
        for k in 0..3 {
            mn[k] = self.a[k].min(self.b[k]) - self.r;
            mx[k] = self.a[k].max(self.b[k]) + self.r;
        }
        (mn, mx)
    }
}

/// 毛坯(解析),符号约定:**负在材料内、正在空气**(§9.12.1)。
#[derive(Clone, Copy, Debug)]
pub enum Stock {
    Sphere { c: [Real; 3], r: Real },
    RoundBox { c: [Real; 3], half: [Real; 3], r: Real },
}

impl Stock {
    pub fn sd(&self, p: [Real; 3]) -> Real {
        match *self {
            Stock::Sphere { c, r } => len(sub(p, c)) - r,
            Stock::RoundBox { c, half, r } => {
                let q = sub(p, c);
                let d = [q[0].abs() - half[0], q[1].abs() - half[1], q[2].abs() - half[2]];
                let out = len([d[0].max(0.0), d[1].max(0.0), d[2].max(0.0)]);
                let inside = d[0].max(d[1]).max(d[2]).min(0.0);
                out + inside - r
            }
        }
    }
}

/// 场源基座:解析毛坯 或 三角网格(STL)。
pub enum BaseField {
    Stock(Stock),
    Mesh(meshfield::MeshField),
}

impl BaseField {
    #[inline]
    pub fn sd(&self, p: [Real; 3]) -> Real {
        match self {
            BaseField::Stock(s) => s.sd(p),
            BaseField::Mesh(m) => m.sd(p),
        }
    }
    /// 无符号距离:Stock 解析(廉价);Mesh 走 BVH 最近点(无射线)。
    /// (诊断/统计路径使用;构建与切削的有符号路径见 LatticeCache)
    #[inline]
    pub fn sd_unsigned(&self, p: [Real; 3]) -> Real {
        match self {
            BaseField::Stock(s) => s.sd(p).abs(),
            BaseField::Mesh(m) => m.unsigned_distance(p),
        }
    }
    /// 有符号距离,**无锁直查**(绕开 MeshField 的全局 Mutex 位置缓存——
    /// 24 线程构建/切削的每次直查都过那把锁会串行化一切;格点缓存才是
    /// 正牌的共享缓存,这里是"缓存未覆盖的中粗格查询"的快速通道)。
    #[inline]
    pub fn sd_unlocked(&self, p: [Real; 3]) -> Real {
        match self {
            BaseField::Stock(s) => s.sd(p),
            BaseField::Mesh(m) => m.signed_distance_pn(p),
        }
    }
}

/// Composite ADF(Sullivan 2012 思想):true(p) = max(base(p), −min_cuts(sd_cut(p)))。
/// 切削 = 布尔减;新增刀路只是往 cuts 里追加,旧样本值不需要重推导。
/// (不含 Clone:MeshField 携带 RefCell 缓存)
pub struct Composite {
    pub base: BaseField,
    pub cuts: Vec<Capsule>,
}

impl Composite {
    /// max 公式必须**全场一致**计算,禁止"毛坯外早退":
    /// 刀路穿出毛坯时(如通道边缘),早退会让场在毛坯面处产生 ~0.5mm 的人为间断
    /// (材料侧角点 +0.49、空气侧角点 +0.005),三线性插值把它抹成 0.3 的假误差。
    /// 语义:空气区幅值是下界(符号恒正确)——composite ADF 的标准口径(Sullivan 2012)。
    #[inline]
    pub fn sd(&self, p: [Real; 3]) -> Real {
        let mut m = self.base.sd(p);
        for c in &self.cuts {
            m = m.max(-c.sd(p));
        }
        m
    }
}

// ─────────────────────────── 格点距离缓存(烘焙)───────────────────────────

/// 页边 2^PAGE_BITS 个 h_min 格点;页 = 稠密 f32 立方(4³×4B = 256B)。
///
/// PAGE_BITS 是"覆盖密度 vs 页内浪费"的旋钮,实测(2026-11-15):
/// | PAGE_BITS | 桌腿页数/内存 | 叶轮页数/内存 | 桌腿构建 |
/// |---|---|---|---|
/// | 4 (16³) | 13169 / 205MB | 3417 / 53MB | 1.138s |
/// | 3 (8³)  | 49323 / 96MB  | 17831 / 34MB | 1.154s |
/// | 2 (4³)  | 219512 / 53MB | 102424 / 25MB | 1.157s |
/// 壳层是**曲面**,页利用率只有 ~38%(PB=4 时更低),故小页反而更省:
/// PB=2 让内存降 3.9×,代价是构建 ~+1.7%。切削路径页数同量级,一并受益。
/// 若将来改用"远场稀疏 + 近表面稠密"的混合粒度,PB=4 在大块稠密区更划算。
const PAGE_BITS: u32 = 2;
const PAGE_SIDE: usize = 1 << PAGE_BITS;
const PAGE_CELLS: usize = PAGE_SIDE * PAGE_SIDE * PAGE_SIDE;

/// 网格场的 h_min 格点**有符号**距离缓存(VTune 实证:构建+切削 ~85% 耗在
/// BVH 最近点 + 奇偶射线)。
/// 页式稀疏稠密:细格(s ≤ 2·h_min)查询触发的页发布为全 NaN 哨兵(AtomicU32
/// 位模式,严格无数据竞争),格点值**惰性计算**(首次查询该格点 = 一次 BVH
/// 最近点 + 一次射线奇偶,之后 O(1) 读)——总计算量 = 实际被查的唯一格点。
/// 有符号语义统一三者:构建判据(取绝对值)、叶角点直存(免独立符号回填)、
/// add_cut 复合场的 base 项(O(1))。页数有界:细格只存在于壳层。
/// 值语义:符号与幅值都在**格点精确位置**处 → 决策调度无关、逐位可复现。
pub struct LatticeCache {
    min: [Real; 3],
    h_min: Real,
    /// 每轴格点数 = 2^max_depth + 1
    nkey: u32,
    npage: u32,
    pages: Vec<std::sync::atomic::AtomicPtr<std::sync::atomic::AtomicU32>>,
    locks: Vec<std::sync::Mutex<()>>,
    computes: std::sync::atomic::AtomicU64,
}

/// f32 NaN 的规范位模式,作"未计算"哨兵(格点真值永不为 NaN)。
const NAN_BITS: u32 = 0x7fc0_0000;
/// 贪心播种的网格规模阈值(小网格堆展开少,播种纯开销)
/// 保留:贪心播种路径已删除(教训 5),此阈值不再被消费,留作口径记录。
#[allow(dead_code)]
const SEED_MIN_TRIS: usize = 2048;
/// 距离场 INF 哨兵(Fast Sweeping 未到达)
const INF32: f32 = f32::INFINITY;

impl LatticeCache {
    fn new(min: [Real; 3], h_min: Real, max_depth: u8) -> Self {
        let nkey = (1u32 << max_depth) + 1;
        let npage = nkey.div_ceil(PAGE_SIDE as u32);
        let n = (npage * npage * npage) as usize;
        Self {
            min,
            h_min,
            nkey,
            npage,
            pages: (0..n)
                .map(|_| std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()))
                .collect(),
            locks: (0..n).map(|_| std::sync::Mutex::new(())).collect(),
            computes: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// 缓存统计:(BVH 计算次数, 物化页数, 页数据字节)。
    pub fn stats(&self) -> (u64, u64, u64) {
        let pages = self
            .pages
            .iter()
            .filter(|p| !p.load(std::sync::atomic::Ordering::Relaxed).is_null())
            .count() as u64;
        (
            self.computes.load(std::sync::atomic::Ordering::Relaxed),
            pages,
            pages * PAGE_CELLS as u64 * 4,
        )
    }

    #[inline]
    fn key_of(&self, p: [Real; 3]) -> [u32; 3] {
        // 手写"四舍五入 + 钳位",不用 f64::round():x86-64 基线下 round 是 libm
        // 导入调用(VTune:round + thunk 合计占构建 CPU ~5%),而 h_min 侧每次查询
        // 都要算 3 次。语义等价:非负输入下 `+0.5 再截断` 即 round();负数经
        // float→int 饱和转换落到 0,与原 `.max(0.0)` 一致。
        let hi = self.nkey - 1;
        let mut k = [0u32; 3];
        for a in 0..3 {
            let t = (p[a] - self.min[a]) / self.h_min + 0.5;
            k[a] = if t >= hi as Real + 0.5 { hi } else { t as u32 };
        }
        k
    }
    #[inline]
    fn page_idx(&self, k: [u32; 3]) -> usize {
        let px = (k[0] >> PAGE_BITS) as usize;
        let py = (k[1] >> PAGE_BITS) as usize;
        let pz = (k[2] >> PAGE_BITS) as usize;
        (pz * self.npage as usize + py) * self.npage as usize + px
    }
    #[inline]
    fn cell_idx(&self, k: [u32; 3]) -> usize {
        let m = (PAGE_SIDE - 1) as u32;
        ((k[2] & m) as usize * PAGE_SIDE + (k[1] & m) as usize) * PAGE_SIDE + (k[0] & m) as usize
    }

    /// 内存分项:(页目录字节, 已发布页载荷字节, 已发布页数)。
    fn mem_bytes(&self) -> (usize, usize, usize) {
        use std::mem::size_of;
        let dir = self.pages.capacity() * size_of::<std::sync::atomic::AtomicPtr<std::sync::atomic::AtomicU32>>()
            + self.locks.capacity() * size_of::<std::sync::Mutex<()>>();
        let n = self
            .pages
            .iter()
            .filter(|p| !p.load(std::sync::atomic::Ordering::Relaxed).is_null())
            .count();
        (dir, n * PAGE_CELLS * 4, n)
    }

    /// 页发布(页锁内双检,只发生一次):分配全 NaN 哨兵页,读路径此后无锁。
    fn publish(&self, page: usize) -> *mut std::sync::atomic::AtomicU32 {
        {
            let p = self.pages[page].load(std::sync::atomic::Ordering::Acquire);
            if !p.is_null() {
                return p;
            }
        }
        let _g = self.locks[page].lock().unwrap();
        let p = self.pages[page].load(std::sync::atomic::Ordering::Acquire);
        if !p.is_null() {
            return p;
        }
        let data: Vec<_> = (0..PAGE_CELLS)
            .map(|_| std::sync::atomic::AtomicU32::new(NAN_BITS))
            .collect();
        let raw = Box::into_raw(data.into_boxed_slice()) as *mut std::sync::atomic::AtomicU32;
        self.pages[page].store(raw, std::sync::atomic::Ordering::Release);
        raw
    }

    /// 带门限查表(add_cut 用):命中 → 精确;未命中时网格近似 |v| ≤ gate
    /// (表面邻域)才做精确计算并入缓存;否则直接返回网格值(页保持壳层有界)。
    /// 语义安全:max(base, −cut) 的零等值面只依赖 base 的**符号**与小幅值区,
    /// 门限 2mm > 网格误差 0.9h,符号与小幅值区恒由精确值承担。
    pub fn lookup_signed_gated(
        &self,
        p: [Real; 3],
        src: &ExactSrc,
        grid: Option<&BakeGrid>,
        gate: Real,
    ) -> Real {
        let k = self.key_of(p);
        let page = self.page_idx(k);
        let ptr = {
            let ptr = self.pages[page].load(std::sync::atomic::Ordering::Acquire);
            if ptr.is_null() {
                self.publish(page)
            } else {
                ptr
            }
        };
        let cell = unsafe { &*ptr.add(self.cell_idx(k)) };
        let bits = cell.load(std::sync::atomic::Ordering::Relaxed);
        if bits != NAN_BITS {
            return f32::from_bits(bits) as Real;
        }
        if let Some(g) = grid {
            let gv = g.sample_nearest(p);
            if gv.abs() > gate {
                return gv;
            }
        }
        let lp = [
            self.min[0] + k[0] as Real * self.h_min,
            self.min[1] + k[1] as Real * self.h_min,
            self.min[2] + k[2] as Real * self.h_min,
        ];
        let v = src.signed(lp) as f32;
        cell.store(v.to_bits(), std::sync::atomic::Ordering::Relaxed);
        self.computes.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        v as Real
    }

    /// 查表:页未发布则发布(全 NaN);格点未算则惰性计算一次(原子写,
    /// 同格点并发双方算出同一位模式,结果确定)。
    #[inline]
    fn lookup_signed(&self, p: [Real; 3], src: &ExactSrc) -> Real {
        let k = self.key_of(p);
        let page = self.page_idx(k);
        let ptr = {
            let ptr = self.pages[page].load(std::sync::atomic::Ordering::Acquire);
            if ptr.is_null() {
                self.publish(page)
            } else {
                ptr
            }
        };
        let cell = unsafe { &*ptr.add(self.cell_idx(k)) };
        let bits = cell.load(std::sync::atomic::Ordering::Relaxed);
        if bits == NAN_BITS {
            let lp = [
                self.min[0] + k[0] as Real * self.h_min,
                self.min[1] + k[1] as Real * self.h_min,
                self.min[2] + k[2] as Real * self.h_min,
            ];
            let v = src.signed(lp) as f32;
            cell.store(v.to_bits(), std::sync::atomic::Ordering::Relaxed);
            self.computes.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            v as Real
        } else {
            f32::from_bits(bits) as Real
        }
    }
}

impl Drop for LatticeCache {
    fn drop(&mut self) {
        for p in &mut self.pages {
            let raw = *p.get_mut();
            if !raw.is_null() {
                unsafe { drop(Box::from_raw(raw as *mut [std::sync::atomic::AtomicU32; PAGE_CELLS])) };
            }
        }
    }
}

// ─────────────────────────── 粗距烘焙网格(距离变换)───────────────────────────

/// 全域粗距网格(有符号,n=129 点/轴,h=size/128,8MB):O(N) 距离变换烘焙。
/// 三步:①三角形 AABB 光栅化为格点候选表(按 k-板并行,无锁)
/// ②候选点局部精确最近(伪法线符号)③双向 Fast Sweeping(Godunov,复用
/// eikonal_update)传播到全格。
/// 动机(实测):15.5 万网格上粗格远点 BVH 查询 100–500µs/次 × 数百万次
/// → 构建 1093s;烘焙后远场/近测试 = O(1) 查表。
/// 职责边界:**近测试 + 远叶角点**;近格(表面邻域)角点仍走精确路径
/// (LatticeCache/解析)——网格量化 ±0.9h 绝不进表面区。
/// 已知限制:完全封闭的内部空腔没有负号种子(符号靠传播,不带拓扑),
/// 腔内会退化为正距离;demo 网格无此形态,封闭腔件需奇偶/绕数补种。
pub struct BakeGrid {
    pub min: [Real; 3],
    pub n: usize,
    pub h: Real,
    pub data: Vec<f32>, // 有符号
}

impl BakeGrid {
    #[inline]
    fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        (k * self.n + j) * self.n + i
    }
    /// 最近邻采样(域外钳制到边界值)
    #[inline]
    pub fn sample_nearest(&self, p: [Real; 3]) -> Real {
        // 同 key_of:避免 libm round 调用(每次查询 3 次)
        let nmax = self.n - 1;
        let mut g = [0usize; 3];
        for a in 0..3 {
            let t = (p[a] - self.min[a]) / self.h + 0.5;
            g[a] = if t >= nmax as Real + 0.5 { nmax } else { t as usize };
        }
        self.data[self.idx(g[0], g[1], g[2])] as Real
    }

    /// 三线性采样(域外先钳位到边界格);诊断用。
    #[inline]
    pub fn sample_trilinear(&self, p: [Real; 3]) -> f32 {
        let nmax = self.n - 1;
        let mut g = [0usize; 3];
        let mut f = [0.0f64; 3];
        for a in 0..3 {
            let tc = ((p[a] - self.min[a]) / self.h).max(0.0).min(nmax as Real);
            let i = (tc as usize).min(nmax.saturating_sub(1));
            g[a] = i;
            f[a] = (tc - i as Real).clamp(0.0, 1.0);
        }
        let (i1, j1, k1) = ((g[0] + 1).min(nmax), (g[1] + 1).min(nmax), (g[2] + 1).min(nmax));
        let d = &self.data;
        let v = |i: usize, j: usize, k: usize| d[self.idx(i, j, k)] as Real;
        let (u, vv, w) = (f[0], f[1], f[2]);
        let c00 = v(g[0], g[1], g[2]) * (1.0 - u) + v(i1, g[1], g[2]) * u;
        let c10 = v(g[0], j1, g[2]) * (1.0 - u) + v(i1, j1, g[2]) * u;
        let c01 = v(g[0], g[1], k1) * (1.0 - u) + v(i1, g[1], k1) * u;
        let c11 = v(g[0], j1, k1) * (1.0 - u) + v(i1, j1, k1) * u;
        let c0 = c00 * (1.0 - vv) + c10 * vv;
        let c1 = c01 * (1.0 - vv) + c11 * vv;
        (c0 * (1.0 - w) + c1 * w) as f32
    }
}

/// 1D 平方距离变换(Felzenszwalb & Huttenlocher 2012 思想,自实现):
/// 输入每点 (平方距离, 站点) —— 站点 = 有限 sq 的格;输出每点到最近站的
/// (平方距离, 站点)。下抛物线包络 O(n);非站格不生成抛物线,只被覆盖。
fn edt_1d(sq: &[f64], st: &[u32]) -> (Vec<f64>, Vec<u32>) {
    let n = sq.len();
    let mut out = vec![f64::INFINITY; n];
    let mut out_arg = vec![u32::MAX; n];
    let Some(f0) = (0..n).find(|&q| sq[q].is_finite()) else {
        return (out, out_arg); // 整线无站
    };
    let mut v = vec![0usize; n]; // 包络抛物线顶(格位置,均为有限站)
    let mut z = vec![0.0f64; n + 1];
    let mut k = 0usize;
    v[0] = f0;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    let inter = |q: usize, pk: usize| {
        ((sq[q] + (q * q) as f64) - (sq[pk] + (pk * pk) as f64)) / (2.0 * (q - pk) as f64)
    };
    for q in (f0 + 1)..n {
        if !sq[q].is_finite() {
            continue;
        }
        let mut s = inter(q, v[k]);
        while s <= z[k] {
            k -= 1;
            s = inter(q, v[k]);
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = f64::INFINITY;
    }
    let mut e = 0usize;
    for q in 0..n {
        while z[e + 1] < q as f64 {
            e += 1;
        }
        let dx = q as f64 - v[e] as f64;
        out[q] = sq[v[e]] + dx * dx;
        out_arg[q] = st[v[e]];
    }
    (out, out_arg)
}

/// 三轴 1D EDT 合成(轴 0/1/2 = i/j/k);axis 段的线基址与步长见内部分解。
fn edt_axis(d2: &mut [f64], arg: &mut [u32], n: usize, axis: usize) {
    let npts = n * n * n;
    let mut sq = vec![0.0f64; n];
    let mut st = vec![0u32; n];
    for l in 0..npts / n {
        let (base, step) = match axis {
            0 => (l * n, 1usize),           // l = k*n + j(线沿 i)
            1 => ((l / n) * n * n + (l % n), n), // l = k*n + i(线沿 j)
            _ => (l, n * n),                // l = j*n + i(线沿 k)
        };
        for q in 0..n {
            sq[q] = d2[base + q * step];
            st[q] = arg[base + q * step];
        }
        let (out, out_arg) = edt_1d(&sq, &st);
        for q in 0..n {
            d2[base + q * step] = out[q];
            arg[base + q * step] = out_arg[q];
        }
    }
}

/// 候选网格:三角形 → 光栅化格桶(h_coarse 粒度,与 BakeGrid 同格)。
/// 壳层键(|d| ≤ band+0.3)的精确计算走 7³ 邻域线性扫描 ——
/// 实测:15.5 万网格上 BVH 遍历(带贪心种子)对薄叶片几何仍 30–100µs/键,
/// 邻域候选 ~30 条 × ~0.1µs ≈ 3µs;高曲率件(叶片)细键数 ~20× 于平板件。
/// **参考实现(已不在生产路径)**:种子 BVH 落地后,7³ 邻域候选扫描被反超
/// (教训 4),`bake_field` 仍返回它但调用方已丢弃。保留代码与注释供对照。
#[allow(dead_code)]
pub struct CandGrid {
    pub min: [Real; 3],
    pub n: usize,
    pub h: Real,
    pub cells: Vec<Vec<u32>>,
}

#[allow(dead_code)]
impl CandGrid {
    fn empty() -> Self {
        Self { min: [0.0; 3], n: 0, h: 1.0, cells: Vec::new() }
    }
    #[inline]
    fn cell_of(&self, p: [Real; 3]) -> [usize; 3] {
        let mut g = [0usize; 3];
        for a in 0..3 {
            let t = ((p[a] - self.min[a]) / self.h).floor();
            g[a] = t.clamp(0.0, (self.n - 1) as Real) as usize;
        }
        g
    }
    /// (2r+1)³ 邻域候选汇集(r=2;键 |d| ≤ band+0.3 < 2h_c 时完备;
    /// 教训:r=3 在叶片密集区一次拉进 ~3.7 万候选(5ms/查),r=2 ≈1.3 万(2µs))
    fn neighborhood(&self, p: [Real; 3], r: usize, out: &mut Vec<u32>) {
        if self.cells.is_empty() {
            return;
        }
        let g = self.cell_of(p);
        let n = self.n;
        for dk in -(r as i32)..=(r as i32) {
            let k = g[2] as i32 + dk;
            if k < 0 || k >= n as i32 {
                continue;
            }
            for dj in -(r as i32)..=(r as i32) {
                let j = g[1] as i32 + dj;
                if j < 0 || j >= n as i32 {
                    continue;
                }
                for di in -(r as i32)..=(r as i32) {
                    let i = g[0] as i32 + di;
                    if i < 0 || i >= n as i32 {
                        continue;
                    }
                    out.extend_from_slice(&self.cells[(k as usize * n + j as usize) * n + i as usize]);
                }
            }
        }
    }
}

/// 精确求值源:纯 base(解析)或 网格种子 BVH(粗距作上界种子,堆剪枝)。
/// 教训:候选邻域线性扫描在密集几何(叶片 7³ 邻域 ~1.5 万三角)上反超 BVH,
/// 且无种子 BVH 对薄叶片 30–100µs/键——种子(粗距 + 2.5h_c 上界)让堆剪枝
/// 从首次比较生效,个位数 µs;种子来源 BakeGrid 的 EDT 界 d−h ≲ grid ≲ d+1.8h。
pub enum ExactSrc<'a> {
    Base(&'a BaseField),
    Seeded { base: &'a BaseField, grid: &'a BakeGrid },
}

impl ExactSrc<'_> {
    /// 有符号精确值
    #[inline]
    fn signed(&self, p: [Real; 3]) -> Real {
        match self {
            ExactSrc::Base(b) => b.sd_unlocked(p),
            ExactSrc::Seeded { base, grid } => {
                DBG_CAND.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let m = match &base {
                    BaseField::Mesh(m) => m,
                    BaseField::Stock(_) => unreachable!("Seeded 仅网格场"),
                };
                let g = grid.sample_nearest(p).abs();
                let seed_d2 = (g + 2.5 * grid.h) as f64;
                let seed_d2 = seed_d2 * seed_d2;
                let (_d, t, feat, cp) = m.closest_features_seeded(p, seed_d2);
                m.signed_from_feature(p, t, feat, cp)
            }
        }
    }
}

/// [诊断] 候选/种子路径计数
pub static DBG_CAND: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static DBG_CAND_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static DBG_CAND_FB: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
fn bake_field(base: &BaseField, min: [Real; 3], size: Real) -> (BakeGrid, CandGrid) {
    let n = 129usize;
    let h = size / (n - 1) as Real;
    let npts = n * n * n;
    let mut data = vec![0f32; npts];
    let stok = match base {
        BaseField::Stock(st) => Some(*st),
        BaseField::Mesh(_) => None,
    };
    if let Some(st) = stok {
        // 解析场:逐点直接(串行 ~40ms,免并行借用)
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let p = [
                        min[0] + i as Real * h,
                        min[1] + j as Real * h,
                        min[2] + k as Real * h,
                    ];
                    data[(k * n + j) * n + i] = st.sd(p) as f32;
                }
            }
        }
        return (BakeGrid { min, n, h, data }, CandGrid::empty());
    }
    let m = match base {
        BaseField::Mesh(m) => m,
        BaseField::Stock(_) => unreachable!(),
    };
    // ── ① 候选光栅化(按 k-板并行,线程本地表,无锁;三角形 AABB 外扩 2 格)──
    // 外扩 E 格 ⟹ 离三角 d < E·h 的格点必有候选。站点带要 1.5h(见②),
    // E=1 只保 d<h:带上 d∈[h,1.5h] 处出现无种子孔洞,材料内部的洪泛区
    // 从洞漏进叶片缝隙,把缝隙标成"材料内"(叶轮实测 ADF 零面外扩 1.15mm)。
    let clampi = |t: Real| t.clamp(0.0, (n - 1) as Real) as usize;
    let nthreads = 12usize;
    let kslab = n.div_ceil(nthreads);
    let slots: Vec<std::sync::Mutex<Option<std::collections::HashMap<usize, Vec<u32>>>>> =
        (0..nthreads).map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|sc| {
        for w in 0..nthreads {
            let slot = &slots[w];
            sc.spawn(move || {
                let (k0, k1) = (w * kslab, ((w + 1) * kslab).min(n));
                if k0 >= k1 {
                    return;
                }
                let mut map: std::collections::HashMap<usize, Vec<u32>> = std::collections::HashMap::new();
                for (ti, tri) in m.tris.iter().enumerate() {
                    let (mut cmn, mut cmx) = ([f64::MAX; 3], [f64::MIN; 3]);
                    for v in tri {
                        for a in 0..3 {
                            cmn[a] = cmn[a].min(v[a]);
                            cmx[a] = cmx[a].max(v[a]);
                        }
                    }
                    let i0 = clampi(((cmn[0] - min[0]) / h).floor() - 2.0);
                    let i1 = clampi(((cmx[0] - min[0]) / h).ceil() + 2.0);
                    let j0 = clampi(((cmn[1] - min[1]) / h).floor() - 2.0);
                    let j1 = clampi(((cmx[1] - min[1]) / h).ceil() + 2.0);
                    let kk0 = clampi(((cmn[2] - min[2]) / h).floor() - 2.0).max(k0);
                    let kk1 = (clampi(((cmx[2] - min[2]) / h).ceil() + 2.0) + 1).min(k1);
                    for k in kk0..kk1 {
                        for j in j0..=j1 {
                            for i in i0..=i1 {
                                map.entry((k * n + j) * n + i).or_default().push(ti as u32);
                            }
                        }
                    }
                }
                *slot.lock().unwrap() = Some(map);
            });
        }
    });
    let mut cells: Vec<Vec<u32>> = vec![Vec::new(); npts];
    for slot in &slots {
        if let Some(map) = slot.lock().unwrap().take() {
            for (cell, v) in map {
                cells[cell].extend_from_slice(&v);
            }
        }
    }
    let cg = CandGrid { min, n, h, cells };
    let cands = &cg.cells;
    // ── ② 局部精确初始化(自格候选;外扩 1 格已保证 |d| ≲ 1.5h 有种子)──
    //    逐格点独立、无共享写 → 按 k 平面切带并行(结果与工人数无关)。
    let mut phi = vec![INF32; npts];
    {
        let plane = n * n;
        let ppw = n.div_ceil(crate::nthreads()).max(1);
        let chunks: Vec<&mut [f32]> = phi.chunks_mut(ppw * plane).collect();
        let cands: &[Vec<u32>] = cands;
        std::thread::scope(|sc| {
            for (w, chunk) in chunks.into_iter().enumerate() {
                let k0 = w * ppw;
                sc.spawn(move || {
                    for (idx, slot) in chunk.iter_mut().enumerate() {
                        let k = k0 + idx / plane;
                        let rem = idx % plane;
                        let (j, i) = (rem / n, rem % n);
                        let list = &cands[(k * n + j) * n + i];
                        if list.is_empty() {
                            continue;
                        }
                        let p = [
                            min[0] + i as Real * h,
                            min[1] + j as Real * h,
                            min[2] + k as Real * h,
                        ];
                        let (_d, t, feat, cp) = m.closest_among(p, list);
                        let v = m.signed_from_feature(p, t, feat, cp) as f32;
                        // 站点过滤:扩展 AABB 的盒角格离三角形可达数格(带厚达 ~8 格),
                        // 只留 |phi| ≤ 1.5h 的贴表面薄带作站点 —— EDT 量的才是"到表面距离"。
                        *slot = if v.abs() <= 1.5 * h as f32 { v } else { INF32 };
                    }
                });
            }
        });
    }
    // ── ③ 精确欧氏距离变换(Felzenszwalb & Huttenlocher 2012,自实现)──
    // 站点 = 种子带格点(phi 有限);三轴 1D 下抛物线包络合成,O(N) 且精确
    // (弃用扫掠:一阶 Godunov 对角各向异性给点源/凸面源远场带来 ~+30% 系统误差)。
    // 误差界:站点带贴表面 ~1 格 → d − h ≲ EDT ≲ d + 1.8h(双侧,近测试可保守)。
    // 符号:带内取**精确** phi 符号;带外由洪泛界定(见 flood_outside_sign)。
    // 旧做法"取最近站的 phi 符号"有两处结构性错误:
    //   ① 薄壁件:壁厚 < 站点带间距时,壁内侧的最近站落在壁的另一侧 → 符号取反
    //      (叶轮实测 [error.split] 符号相反 31/2000,max 61.8mm);
    //   ② 封闭腔:腔内没有任何负站点 → 整个腔退化成正距离。
    // 洪泛只依赖拓扑(带是闭合隔板),与壁厚、站距都无关,并顺带补上腔体负号。
    let sign = flood_outside_sign(&phi, n);
    let mut d2 = vec![0.0f64; npts];
    let mut arg = vec![0u32; npts];
    let inv_h = 1.0 / h;
    for c in 0..npts {
        if phi[c].is_finite() {
            // 种子保留**亚格距离**(phi/h)²:EDT 于是求 min(dist² + f),
            // 等价于"到半径为 |phi| 的球面"的平方距离 —— 带内的亚格精度不再被丢掉。
            // 旧写法置 0,等于只量"到最近种子格点"的距离,系统高估(≤1.5h)。
            let t = phi[c] as f64 * inv_h;
            d2[c] = t * t;
            arg[c] = c as u32;
        } else {
            d2[c] = f64::INFINITY;
            arg[c] = u32::MAX;
        }
    }
    edt_axis(&mut d2, &mut arg, n, 0);
    edt_axis(&mut d2, &mut arg, n, 1);
    edt_axis(&mut d2, &mut arg, n, 2);
    let mut n_flip = 0u64;
    for c in 0..npts {
        // EDT 在格单位下工作 → × h 转毫米
        let mag = if d2[c].is_finite() {
            (d2[c].max(0.0).sqrt() * h as f64) as f32
        } else {
            0.0
        };
        data[c] = sign[c] as f32 * mag;
        if d2[c].is_finite() && arg[c] != u32::MAX {
            let old = if phi[arg[c] as usize] < 0.0 { -1.0f32 } else { 1.0 };
            if old != sign[c] as f32 {
                n_flip += 1;
            }
        }
    }
    eprintln!("[bake.sign] 洪泛定号 vs 最近站定号:不同 {n_flip} / {npts} 格");
    (BakeGrid { min, n, h, data }, cg)
}

/// 带外符号:①域边界可达的非带区 = 外部(+1);②其余非带区按**连通分量**处理,
/// 每个分量由"相邻带内节点的精确 phi 符号"投票定号 —— 这区分了两类不可达区:
/// 材料内部(相邻带点为负)与**封闭空腔**(相邻带点为正,空腔是空气)。
/// ③带内返回 0,由调用方用精确 phi 符号覆盖;整体 +1/−1。
///
/// 正确性前提:①域比零件 bbox 大出一圈(调用方保证 2mm 余量),故"零件外"
/// 必与域边界连通;②带(|phi| ≤ 1.5h)构成闭合隔板,洪泛不穿墙。
/// 若带上有洞(种子缺失)会漏填成"外部",表现为远场符号错——由 [error.split] 守门。
fn flood_outside_sign(phi: &[f32], n: usize) -> Vec<i8> {
    let mut sign = vec![0i8; phi.len()];
    let mut seen = vec![false; phi.len()];
    let plane = n * n;
    // ── ① 从域边界洪泛:域外的空气必然与边界连通 ──
    let mut queue: Vec<u32> = Vec::new();
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                if !(i == 0 || j == 0 || k == 0 || i + 1 == n || j + 1 == n || k + 1 == n) {
                    continue;
                }
                let c = (k * n + j) * n + i;
                if phi[c].is_finite() || seen[c] {
                    continue;
                }
                seen[c] = true;
                sign[c] = 1;
                queue.push(c as u32);
            }
        }
    }
    // 6 邻(面连通:斜向缺口不允许泄漏,薄壁的带因此仍是隔板)
    let nb = |c: usize| -> [usize; 6] {
        let i = c % n;
        let j = (c / n) % n;
        let k = c / plane;
        [
            if i > 0 { c - 1 } else { usize::MAX },
            if i + 1 < n { c + 1 } else { usize::MAX },
            if j > 0 { c - n } else { usize::MAX },
            if j + 1 < n { c + n } else { usize::MAX },
            if k > 0 { c - plane } else { usize::MAX },
            if k + 1 < n { c + plane } else { usize::MAX },
        ]
    };
    let mut head = 0usize;
    while head < queue.len() {
        let c = queue[head] as usize;
        head += 1;
        for cc in nb(c) {
            if cc != usize::MAX && !seen[cc] && !phi[cc].is_finite() {
                seen[cc] = true;
                sign[cc] = 1;
                queue.push(cc as u32);
            }
        }
    }
    // ── ② 剩余非带区:逐连通分量投票(材料内部 / 封闭空腔)──
    let mut comp: Vec<u32> = Vec::new();
    for s in 0..phi.len() {
        if seen[s] || phi[s].is_finite() {
            continue;
        }
        comp.clear();
        comp.push(s as u32);
        seen[s] = true;
        let (mut pos, mut neg) = (0u32, 0u32);
        let mut head = 0usize;
        while head < comp.len() {
            let c = comp[head] as usize;
            head += 1;
            for cc in nb(c) {
                if cc == usize::MAX {
                    continue;
                }
                if phi[cc].is_finite() {
                    // 带内节点是精确的:它的符号就是本分量该取的方向
                    if phi[cc] < 0.0 {
                        neg += 1;
                    } else {
                        pos += 1;
                    }
                } else if !seen[cc] {
                    seen[cc] = true;
                    comp.push(cc as u32);
                }
            }
        }
        // 平票或无边可依时按外部处理(空网格等退化情形保持旧行为)
        let sg: i8 = if neg > pos { -1 } else { 1 };
        for &c in &comp {
            sign[c as usize] = sg;
        }
    }
    // ── ③ 带内:精确 phi 符号 ──
    for c in 0..phi.len() {
        if phi[c].is_finite() {
            sign[c] = if phi[c] < 0.0 { -1 } else { 1 };
        }
    }
    sign
}

// ─────────────────────────── ADF 树 ───────────────────────────

/// 叶节点角点顺序:k 的位 = (x = k&1, y = k>>1&1, z = k>>2&1)(与子节点同约定)。
#[derive(Clone, Copy)]
pub enum Node {
    Leaf([f32; 8]),
    Internal([u32; 8]),
}

pub const EMPTY: u32 = u32::MAX;

/// ADF 常驻内存分项(字节)。
///
/// 为什么单独列出 `lattice_dir`:页目录是 **(2^depth+1)/2^PAGE_BITS 的三次方**
/// 个指针 + 同数量的锁,深度 10、PAGE_BITS=2 时是 257³×2 个槽 —— 与树本身
/// 同量级。只统计"已发布页载荷"会低估一个数量级。
#[derive(Default, Clone, Copy, Debug)]
pub struct AdfMem {
    /// 八叉树节点(容量口径)
    pub tree: usize,
    /// 格点缓存页目录(指针表 + 锁表)
    pub lattice_dir: usize,
    /// 格点缓存已发布页的载荷(f32 稠密立方)
    pub lattice_pages: usize,
    /// 已发布页数
    pub lattice_page_count: usize,
    /// 粗距烘焙网格
    pub bake: usize,
    /// 源网格场(三角 / BVH / AABB / 法线 / 边表)
    pub field: usize,
    /// 切削胶囊
    pub cuts: usize,
}

impl AdfMem {
    pub fn total(&self) -> usize {
        self.tree + self.lattice_dir + self.lattice_pages + self.bake + self.field + self.cuts
    }
    /// 单行摘要(日志/CLI 用)。
    pub fn summary(&self) -> String {
        let m = |b: usize| b as f64 / (1024.0 * 1024.0);
        format!(
            "合计 {:.1} MB = 树 {:.1} + 页目录 {:.1} + 页载荷 {:.1}({} 页) + 烘焙 {:.1} + 网格场 {:.1} + 切削 {:.1}",
            m(self.total()),
            m(self.tree),
            m(self.lattice_dir),
            m(self.lattice_pages),
            self.lattice_page_count,
            m(self.bake),
            m(self.field),
            m(self.cuts)
        )
    }
}

#[derive(Default, Clone, Copy, Debug)]
pub struct BuildStats {    pub subdivided: u64,
    pub leaves: u64,
    pub sd_evals: u64,
    pub leaf_depth: [u64; 17],
}

#[derive(Default, Clone, Copy, Debug)]
pub struct RedistStats {
    pub interface_leaves: u64,
    pub grid_n: [usize; 3],
    pub corrected: u64,
    /// 分段耗时(性能定位用):符号场初值 / 棱种子 / 松弛扫掠 / 写回
    pub t_eval: std::time::Duration,
    pub t_seed: std::time::Duration,
    pub t_relax: std::time::Duration,
    pub t_write: std::time::Duration,
}

#[inline]
fn old_signum_positive(_old: f32) -> f32 {
    1.0 // 写回仅用于空气侧(old > 0.5h),符号恒为正
}

pub struct Adf {
    pub min: [Real; 3],
    pub size: Real,
    /// Frisken 细分容差(mm):中心真值 vs 三线性插值的允许偏差
    pub eps: Real,
    /// 壳层带宽(mm):只在 |d| ≤ band 的表面附近细分(spike 实证:全场判据会被
    /// 材料内部的非线性场吃穿,叶数 292 万/131MB;带宽限制后回落 ~1 个数量级)
    pub band: Real,
    pub max_depth: u8,
    pub nodes: Vec<Node>,
    pub root: u32,
    pub field: Composite,
    /// 格点距离缓存(仅网格场;留给 add_cut 增量刷新复用)
    pub lattice: Option<std::sync::Arc<LatticeCache>>,
    /// 粗距烘焙网格(近测试 + 深场 + 种子;仅网格场)
    pub bake: Option<BakeGrid>,
}

#[inline]
fn corner_pos(o: [Real; 3], s: Real, k: usize) -> [Real; 3] {
    [
        o[0] + ((k & 1) as Real) * s,
        o[1] + (((k >> 1) & 1) as Real) * s,
        o[2] + (((k >> 2) & 1) as Real) * s,
    ]
}

/// 单元格内三线性场的解析梯度(供 QEF 的切平面法线用)。
/// `u` 属于 [0,1]^3 为格内坐标,`inv_step` = 1/格宽,返回世界坐标梯度(mm^-1)。
fn tri_grad(cv: &[f32; 8], u: [f64; 3], inv_step: f64) -> [f64; 3] {
    let mut g = [0.0f64; 3];
    for (i, &c) in cv.iter().enumerate() {
        let dx = (i & 1) as usize;
        let dy = ((i >> 1) & 1) as usize;
        let dz = ((i >> 2) & 1) as usize;
        let wx = if dx == 1 { u[0] } else { 1.0 - u[0] };
        let wy = if dy == 1 { u[1] } else { 1.0 - u[1] };
        let wz = if dz == 1 { u[2] } else { 1.0 - u[2] };
        let sx = if dx == 1 { 1.0 } else { -1.0 };
        let sy = if dy == 1 { 1.0 } else { -1.0 };
        let sz = if dz == 1 { 1.0 } else { -1.0 };
        let c = c as f64;
        g[0] += c * sx * wy * wz;
        g[1] += c * wx * sy * wz;
        g[2] += c * wx * wy * sz;
    }
    [g[0] * inv_step, g[1] * inv_step, g[2] * inv_step]
}

#[inline]
fn child_origin(o: [Real; 3], s: Real, i: usize) -> [Real; 3] {
    let h = s * 0.5;
    [
        o[0] + ((i & 1) as Real) * h,
        o[1] + (((i >> 1) & 1) as Real) * h,
        o[2] + (((i >> 2) & 1) as Real) * h,
    ]
}

#[inline]
fn child_index(q: [Real; 3], o: [Real; 3], s: Real) -> usize {
    let h = s * 0.5;
    ((q[0] >= o[0] + h) as usize)
        | (((q[1] >= o[1] + h) as usize) << 1)
        | (((q[2] >= o[2] + h) as usize) << 2)
}

#[inline]
fn trilinear(u: [Real; 3], c: &[f32; 8]) -> f32 {
    let (x0, y0, z0) = (1.0 - u[0], 1.0 - u[1], 1.0 - u[2]);
    let w = [
        x0 * y0 * z0,
        u[0] * y0 * z0,
        x0 * u[1] * z0,
        u[0] * u[1] * z0,
        x0 * y0 * u[2],
        u[0] * y0 * u[2],
        x0 * u[1] * u[2],
        u[0] * u[1] * u[2],
    ];
    let mut acc = 0.0f64;
    for k in 0..8 {
        acc += w[k] * c[k] as f64;
    }
    acc as f32
}

#[inline]
fn aabb_overlap(o: [Real; 3], s: Real, mn: [Real; 3], mx: [Real; 3]) -> bool {
    o[0] < mx[0] && o[0] + s > mn[0] && o[1] < mx[1] && o[1] + s > mn[1] && o[2] < mx[2] && o[2] + s > mn[2]
}

/// 3D Eikonal 的 Godunov 更新(级联:1 轴 → 2 轴 → 3 轴)。
/// a[] = 每轴两侧邻居的较小距离(可为 INF);h = 格宽。返回新的距离解。
/// 教训 5 的修复:简单 min(邻居)+h 收敛到 L1 曼哈顿距离,必须用多轴二次方程。
fn eikonal_update(a: &mut [f32; 3], h: f32) -> f32 {
    a.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let (a1, a2, a3) = (a[0], a[1], a[2]);
    let t = a1 + h; // 1 轴解
    if a2.is_finite() && t > a2 {
        // 2 轴:(t−a1)² + (t−a2)² = h²
        let disc2 = 2.0 * h * h - (a1 - a2) * (a1 - a2);
        if disc2 >= 0.0 {
            let t2 = 0.5 * (a1 + a2) + 0.5 * disc2.sqrt();
            if !a3.is_finite() || t2 <= a3 {
                return t2;
            }
            // 3 轴:(t−a1)² + (t−a2)² + (t−a3)² = h²
            let disc3 = 3.0 * h * h
                - (a1 - a2) * (a1 - a2)
                - (a1 - a3) * (a1 - a3)
                - (a2 - a3) * (a2 - a3);
            if disc3 >= 0.0 {
                return (a1 + a2 + a3) / 3.0 + (disc3 / 3.0).sqrt();
            }
        }
    }
    t
}

/// 并行度:按可用逻辑核数。注意本机 16 物理 / 24 逻辑,BVH 查询实测 16 工人
/// 已略优于 24(超线程在 Cache 密集遍历上互相挤占);此处取逻辑核数保持不变,
/// 需要时再按阶段细分。
fn nthreads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).max(1)
}

/// 串行八序 Gauss-Seidel 松弛(**参考实现**,供并行版逐位对账;不参与生产路径)。
pub fn eikonal_sweep_serial(d: &mut [f32], gn: [usize; 3], hf: f32, rounds: u32) {
    let (nx, ny, nz) = (gn[0], gn[1], gn[2]);
    let inf = f32::INFINITY;
    let mut cand = [inf; 3];
    for r in 0..rounds {
        let (fx, fy, fz) = (r & 1 == 0, r & 2 == 0, r & 4 == 0);
        for kz in 0..nz {
            let k = if fz { kz } else { nz - 1 - kz };
            for jy in 0..ny {
                let j = if fy { jy } else { ny - 1 - jy };
                for ix in 0..nx {
                    let i = if fx { ix } else { nx - 1 - ix };
                    let c = (k * ny + j) * nx + i;
                    let cur = d[c];
                    if cur <= hf {
                        continue;
                    }
                    // 每轴取两侧邻居的较小值(Godunov 的 per-axis 候选)
                    cand[0] = inf;
                    if i > 0 { cand[0] = cand[0].min(d[c - 1]); }
                    if i + 1 < nx { cand[0] = cand[0].min(d[c + 1]); }
                    cand[1] = inf;
                    if j > 0 { cand[1] = cand[1].min(d[c - nx]); }
                    if j + 1 < ny { cand[1] = cand[1].min(d[c + nx]); }
                    cand[2] = inf;
                    if k > 0 { cand[2] = cand[2].min(d[c - nx * ny]); }
                    if k + 1 < nz { cand[2] = cand[2].min(d[c + nx * ny]); }
                    let t = eikonal_update(&mut cand, hf);
                    if t < cur {
                        d[c] = t;
                    }
                }
            }
        }
    }
}

/// 复用缓冲的 1D EDT(**与 edt_1d 算法完全一致**,只是省掉每线 4 次分配)。
/// 不产出 arg(重距离化只要距离)。
fn edt_1d_into(sq: &[f64], out: &mut [f64], v: &mut [usize], z: &mut [f64]) {
    let n = sq.len();
    out[..n].fill(f64::INFINITY);
    let Some(f0) = (0..n).find(|&q| sq[q].is_finite()) else {
        return; // 整线无站
    };
    let mut k = 0usize;
    v[0] = f0;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    let inter = |q: usize, pk: usize| {
        ((sq[q] + (q * q) as f64) - (sq[pk] + (pk * pk) as f64)) / (2.0 * (q - pk) as f64)
    };
    for q in (f0 + 1)..n {
        if !sq[q].is_finite() {
            continue;
        }
        let mut s = inter(q, v[k]);
        while s <= z[k] {
            k -= 1;
            s = inter(q, v[k]);
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = f64::INFINITY;
    }
    let mut e = 0usize;
    for q in 0..n {
        while z[e + 1] < q as f64 {
            e += 1;
        }
        let dx = q as f64 - v[e] as f64;
        out[q] = sq[v[e]] + dx * dx;
    }
}

/// 矩形网格单轴 1D EDT,按线并行。
/// **安全性**:同一轴上的不同线,其元素索引集合两两不交(轴分解的定义),
/// 且 `l ≡ w (mod nworkers)` 保证每条线只归一个工人 ⟹ 无别名写。
/// **确定性**:每条线的输出只由该线输入决定,与线序、工人数无关 ⟹ 逐位可复现。
fn edt_axis_par(d: &mut [f32], gn: [usize; 3], axis: usize, nt: usize) {
    let (nx, ny, nz) = (gn[0], gn[1], gn[2]);
    let plane = nx * ny;
    let (nline, len, step) = match axis {
        0 => (ny * nz, nx, 1usize),      // 线沿 i:l = k*ny + j
        1 => (nx * nz, ny, nx),          // 线沿 j:l = k*nx + i
        _ => (nx * ny, nz, plane),       // 线沿 k:l = j*nx + i
    };
    if nline == 0 || len == 0 {
        return;
    }
    let base_of = move |l: usize| -> usize {
        match axis {
            0 => l * nx,
            1 => (l / nx) * plane + (l % nx),
            _ => l,
        }
    };
    let ptr = d.as_mut_ptr() as usize; // 仅用于跨工人共享;写集合互斥见上
    let nworkers = nt.clamp(1, nline);
    std::thread::scope(|sc| {
        for w in 0..nworkers {
            sc.spawn(move || {
                let mut sq = vec![0.0f64; len];
                let mut out = vec![0.0f64; len];
                let mut v = vec![0usize; len];
                let mut z = vec![0.0f64; len + 1];
                let p = ptr as *mut f32;
                let mut l = w;
                while l < nline {
                    let base = base_of(l);
                    // SAFETY:本条线的索引集合 {base + q*step} 与其他线不交(见函数注释),
                    // 且 q*step 上界 = (len-1)*step + base < plane*nz。
                    unsafe {
                        for q in 0..len {
                            sq[q] = *p.add(base + q * step) as f64;
                        }
                        edt_1d_into(&sq, &mut out, &mut v, &mut z);
                        for q in 0..len {
                            *p.add(base + q * step) = out[q] as f32;
                        }
                    }
                    l += nworkers;
                }
            });
        }
    });
}

/// 重距离化的精确 EDT:种子(符号翻转棱,含沿棱的子格距离)不变,
/// 用三轴距离变换求 min over 种子 q 的 sqrt(v_q² + |p−q|²)。
///
/// 为什么用 EDT 而不是八序扫掠:
/// - 扫掠是**一阶迎风**格式,对点源/凸面源有 ~+30% 对角各向异性(教训 7,bake 已弃用);
///   EDT 无此偏差,且在带内与三线性插值相容。
/// - 扫掠需要 `rounds` 轮才有足够传播,且**固定点依赖更新顺序**——实测条带并行
///   会收敛到与串行不同的不动点(偏差 1.8–3.5 格),因此"并行 + 逐位一致"不可兼得;
///   EDT 是 O(N)、无迭代、无收敛参数,天然可并行且逐位可复现。
pub fn edt_redist(d: &mut [f32], gn: [usize; 3], hf: f32, nt: usize) {
    if nt == 0 {
        return edt_redist(d, gn, hf, nthreads());
    }
    let inv = 1.0 / hf;
    // mm → 格单位平方(距离上限约百格,平方在 f32 精确可表示区间内)
    for v in d.iter_mut() {
        *v = if v.is_finite() {
            let t = *v * inv;
            t * t
        } else {
            f32::INFINITY
        };
    }
    for axis in 0..3 {
        edt_axis_par(d, gn, axis, nt);
    }
    for v in d.iter_mut() {
        *v = if v.is_finite() { v.sqrt() * hf } else { f32::INFINITY };
    }
}

/// 多点判据:中心 + 6 面心的插值偏差最大值(角点插值恒等于角点值,零信息,不查)。
/// 教训 3 的修复:中心单点判据对 |x|=|y| 折线/鞍点盲(实测壳层 max 0.294mm)。
/// 返回(最大偏差, 中心真值)——中心真值供 1-Lipschitz 壳层判定复用。
fn cell_deviation(
    sd: &mut dyn FnMut([Real; 3]) -> Real,
    o: [Real; 3],
    s: Real,
    corners: &[f32; 8],
    stats: &mut BuildStats,
) -> (f32, f32) {
    let ctr = [o[0] + s * 0.5, o[1] + s * 0.5, o[2] + s * 0.5];
    let c_true = sd(ctr) as f32;
    stats.sd_evals += 1;
    let mut dev = (c_true - trilinear([0.5, 0.5, 0.5], corners)).abs();
    for fi in 0..6usize {
        let axis = fi / 2;
        let side = fi % 2;
        let mut p = [o[0] + s * 0.5, o[1] + s * 0.5, o[2] + s * 0.5];
        p[axis] = if side == 1 { o[axis] + s } else { o[axis] };
        let f_true = sd(p) as f32;
        stats.sd_evals += 1;
        let mut u = [0.5 as Real; 3];
        u[axis] = side as Real;
        dev = dev.max((f_true - trilinear(u, corners)).abs());
    }
    (dev, c_true)
}

/// 远叶构造:近判定已排除表面;角点 = 网格(深场 ±2.7h)或解析。
/// 符号钳制:Lipschitz 界 ⟹ 远叶全部角点 |d| > band 且与中心同号 ——
/// 网格噪声(±2.7h)若把角点翻过零会制造虚假零穿越(提取出伪几何),
/// 故按中心符号把 |角点| 钳到 ≥ band(解析路径值精确,钳制不生效)。
#[allow(clippy::too_many_arguments)]
fn far_leaf(
    nodes: &mut Vec<Node>,
    src: &ExactSrc,
    grid: Option<&BakeGrid>,
    sign: Real,
    band: Real,
    o: [Real; 3],
    s: Real,
    depth: u8,
    stats: &mut BuildStats,
) -> u32 {
    let mut corners = [0f32; 8];
    // 开口网格可能无法一致定号，不能由中心推断全叶。
    let certified = match src {
        ExactSrc::Base(BaseField::Mesh(mesh)) | ExactSrc::Seeded { base: BaseField::Mesh(mesh), .. } => mesh.has_closed_surface(),
        _ => true,
    };
    for k in 0..8 {
        let p = corner_pos(o, s, k);
        let v = match grid {
            Some(g) => g.sample_nearest(p) as f32,
            None => src.signed(p) as f32,
        };
        stats.sd_evals += 1;
        corners[k] = if certified || v.abs() < band as f32 {
            v.abs().max(band as f32).copysign(sign as f32)
        } else {
            v
        };
    }
    stats.leaves += 1;
    stats.leaf_depth[depth as usize] += 1;
    nodes.push(Node::Leaf(corners));
    (nodes.len() - 1) as u32
}

/// 递归构建(refresh 路径,复合场闭包;保持旧行为逐位一致):
/// 8 角点采样 → 多点判据(中心+6面心)→ 壳层判定 → 按需细分。
fn build_cell_rebuild(
    nodes: &mut Vec<Node>,
    sd: &mut dyn FnMut([Real; 3]) -> Real,
    o: [Real; 3],
    s: Real,
    depth: u8,
    eps: Real,
    band: Real,
    max_depth: u8,
    stats: &mut BuildStats,
) -> u32 {
    let mut corners = [0f32; 8];
    for k in 0..8 {
        corners[k] = sd(corner_pos(o, s, k)) as f32;
        stats.sd_evals += 1;
    }
    let (dev, c_true) = cell_deviation(sd, o, s, &corners, stats);
    // 壳层判定(SDF 1-Lipschitz 的保守界):|sd(格心)| ≤ band + 半对角线
    let half_diag = s * 0.75_f64.sqrt();
    let near = c_true.abs() as Real <= band + half_diag;
    if depth < max_depth && near && dev > eps as f32 {
        stats.subdivided += 1;
        let mut ch = [0u32; 8];
        let hs = s * 0.5;
        for i in 0..8 {
            // 内层子索引必须打 LOCAL_TAG:本函数只在 refresh_walk 的局部 vec
            // 里建子树,未打标的索引在合并阶段会被当成既有全局索引 → 指向域角落
            // 的远场空气叶,刀路穿过粗叶时会把假空气接进材料(P23 根因)。
            ch[i] = tag_local(build_cell_rebuild(nodes, sd, child_origin(o, s, i), hs, depth + 1, eps, band, max_depth, stats));
        }
        nodes.push(Node::Internal(ch));
    } else {
        stats.leaves += 1;
        stats.leaf_depth[depth as usize] += 1;
        nodes.push(Node::Leaf(corners));
    }
    (nodes.len() - 1) as u32
}

/// 递归构建(构建路径):中心先行的三路采样。
/// 路由:①细格(s ≤ 2·h_min)全走 LatticeCache(精确;壳层有界页)
/// ②粗格中心 = 烘焙网格(近似)→ 近测试加 3h 裕量(1-Lipschitz 保守,不漏面)
///   → 远叶角点 = 网格(深场 ±0.9h 不可见);近格角点/判据 = base 精确直查
/// ③近大格(s ≥ 7·h_min):场变化 ≥ hd − band ≫ eps ⟹ 必细分,免角点采样。
/// 关键教训:网格量化值若进近格角点,会破坏"平面三线性精确"性质
/// (表面位置误差 ±0.9h)——网格只服务近测试与深场,绝不进表面区。
///
/// **为何 `fine` 阈值不放大(P10 的否定结论)**:理论上 depth ≤ max_depth 的格点
/// 是 `min + i·2^(max_depth−depth)·h_min`(h_min 整数倍),而 `key_of` 正是取整键,
/// 所以缓存取值无插值误差、可安全覆盖任意尺寸近格。但实测把阈值放宽到 `s < 7·h_min`
/// (即纳入 s = 4·h_min)**反而更慢**:桌腿查询 2422 万 → 1153 万(−52%),
/// 唯一 BVH 计算 542 万 → 1119 万(+106%),页 13169 → 34136,缓存 205 → 533MB,
/// 构建 1.14 → 1.17s。原因:那 1800 万次"直查"并非同一批格点的重复,而是
/// ①每格 7 个偏差采样点(面心/中心,本就互不相同)与②不再细分的叶的角点;
/// 把它们塞进缓存等于用稀疏覆盖换唯一计算翻倍,页利用率掉到 8%。
/// 结论:缓存阈值由"覆盖密度"决定,不是越宽越好的参数。
fn build_cell(
    nodes: &mut Vec<Node>,
    src: &ExactSrc,
    cache: Option<&LatticeCache>,
    grid: Option<&BakeGrid>,
    h_min: Real,
    o: [Real; 3],
    s: Real,
    depth: u8,
    eps: Real,
    band: Real,
    max_depth: u8,
    stats: &mut BuildStats,
) -> u32 {
    let fine = cache.is_some() && s <= 2.0 * h_min;
    let lazy_sub = s >= 7.0 * h_min;
    let half_diag = s * 0.75_f64.sqrt();
    let ctr = [o[0] + s * 0.5, o[1] + s * 0.5, o[2] + s * 0.5];
    // 中心值(路由决策)
    let (c_center, center_exact) = if fine {
        (cache.unwrap().lookup_signed(ctr, src), true)
    } else if let Some(g) = grid {
        (g.sample_nearest(ctr), false)
    } else {
        (src.signed(ctr), true)
    };
    // 裕量须覆盖 EDT 误差界(d − h ≲ v ≲ d + 1.8h)+ 最近邻采样 ±0.87h → 3h
    let margin = if center_exact { 0.0 } else { 3.0 * grid.unwrap().h };
    let near = c_center.abs() <= band + half_diag + margin;
    if !near {
        // 远叶:表面已被 1-Lipschitz + 裕量排除;符号取网格中心值
        return far_leaf(nodes, src, grid, c_center.signum(), band, o, s, depth, stats);
    }
    if lazy_sub && depth < max_depth {
        // 近大格:网格裕量可能误报(EDT 高估侧 ~2.7h)→ 精确中心复核(1 查)
        // 把误报环带格立即打成远叶(网格角点),阻断细分级联;含表面格 |c|≤hd 必通过
        if grid.is_some() {
            let c_ex = src.signed(ctr);
            stats.sd_evals += 1;
            if c_ex.abs() > band + half_diag {
                return far_leaf(nodes, src, grid, c_ex.signum(), band, o, s, depth, stats);
            }
        }
        // 必细分,角点由子格承担(免采样)
        stats.subdivided += 1;
        let mut ch = [0u32; 8];
        let hs = s * 0.5;
        for i in 0..8 {
            ch[i] = build_cell(
                nodes,
                src,
                cache,
                grid,
                h_min,
                child_origin(o, s, i),
                hs,
                depth + 1,
                eps,
                band,
                max_depth,
                stats,
            );
        }
        nodes.push(Node::Internal(ch));
        return (nodes.len() - 1) as u32;
    }
    // 近小格:先精确中心复核(1 查;网格裕量误报的环带格在此早退成远叶),
    // 通过后才全采样(细格→缓存;粗格→base 直查)
    if !center_exact {
        let c_ex = src.signed(ctr);
        stats.sd_evals += 1;
        if c_ex.abs() > band + half_diag {
            return far_leaf(nodes, src, grid, c_ex.signum(), band, o, s, depth, stats);
        }
    }
    let mut cached_sd = cache.map(|c| move |p: [Real; 3]| -> Real { c.lookup_signed(p, src) });
    let mut direct_sd = |p: [Real; 3]| -> Real { src.signed(p) };
    let sd: &mut dyn FnMut([Real; 3]) -> Real = if fine { cached_sd.as_mut().unwrap() } else { &mut direct_sd };
    let mut corners = [0f32; 8];
    for k in 0..8 {
        corners[k] = sd(corner_pos(o, s, k)) as f32;
        stats.sd_evals += 1;
    }
    let (dev, c_true) = cell_deviation(sd, o, s, &corners, stats);
    let near_exact = c_true.abs() as Real <= band + half_diag;
    if depth < max_depth && near_exact && dev > eps as f32 {
        stats.subdivided += 1;
        let mut ch = [0u32; 8];
        let hs = s * 0.5;
        for i in 0..8 {
            ch[i] = build_cell(
                nodes,
                src,
                cache,
                grid,
                h_min,
                child_origin(o, s, i),
                hs,
                depth + 1,
                eps,
                band,
                max_depth,
                stats,
            );
        }
        nodes.push(Node::Internal(ch));
    } else {
        stats.leaves += 1;
        stats.leaf_depth[depth as usize] += 1;
        nodes.push(Node::Leaf(corners));
    }
    (nodes.len() - 1) as u32
}

/// 增量刷新:只访问与新扫掠 AABB 相交的单元格;
/// 叶 → 更新角点 → 重估判据(可能细分);内部 → 递归命中子格。
/// 已知限制:不做粗化;不处理"细分后子格又满足合并"的回摆。
/// 刷新子树产出(只读并行走树的结果):
/// - Untouched:与新刀路无交集,原样
/// - Leaf:既有叶的角点更新(槽位原位改写)
/// - Internal:既有子树根被改写为新 Internal;ch 为 8 子新索引,
///   local_mask 标记哪些槽位是 local 局部节点(合并时 +off 修正)
#[derive(Default)]
enum RefreshOutcome {
    #[default]
    Untouched,
    Leaf([f32; 8]),
    /// ch 的槽位:高位打标 = 局部 vec 索引(合并时 +off 转全局);未打标 = 既有全局索引
    Internal { ch: [u32; 8], local: Vec<Node> },
}

/// 局部索引打标位(u32 最高位;真实树索引远小于 2^31)
const LOCAL_TAG: u32 = 0x8000_0000;
#[inline]
fn tag_local(i: u32) -> u32 {
    i | LOCAL_TAG
}
#[inline]
fn is_local(i: u32) -> bool {
    i & LOCAL_TAG != 0
}
#[inline]
fn untag(i: u32) -> u32 {
    i & !LOCAL_TAG
}

/// 只读刷新走树:不写任何全局节点,新节点累积到本单元局部 vec(local 索引 = 位置)。
/// 与 refresh_cell 同语义:剪枝 → 叶重算角点/判据 → 按需细分(build_cell_rebuild 复合场)。
fn refresh_walk(
    global: &[Node],
    sd: &(dyn Fn([Real; 3]) -> Real + Sync),
    new_cuts: &[Capsule],
    idx: u32,
    o: [Real; 3],
    s: Real,
    depth: u8,
    eps: Real,
    band: Real,
    max_depth: u8,
) -> RefreshOutcome {
    let mut hit = false;
    for c in new_cuts {
        let (amn, amx) = c.aabb();
        let (mn, mx) = (
            [amn[0] - band, amn[1] - band, amn[2] - band],
            [amx[0] + band, amx[1] + band, amx[2] + band],
        );
        if aabb_overlap(o, s, mn, mx) {
            hit = true;
            break;
        }
    }
    if !hit {
        return RefreshOutcome::Untouched;
    }
    match global[idx as usize] {
        Node::Leaf(_) => {
            let mut adapter = |p: [Real; 3]| sd(p);
            let mut corners = [0f32; 8];
            for k in 0..8 {
                corners[k] = adapter(corner_pos(o, s, k)) as f32;
            }
            let (dev, c_true) = cell_deviation(&mut adapter, o, s, &corners, &mut BuildStats::default());
            let half_diag = s * 0.75_f64.sqrt();
            let near = c_true.abs() as Real <= band + half_diag;
            if depth < max_depth && near && dev > eps as f32 {
                let mut local: Vec<Node> = Vec::new();
                let mut ch = [0u32; 8];
                for i in 0..8 {
                    let li = build_cell_rebuild(
                        &mut local,
                        &mut adapter,
                        child_origin(o, s, i),
                        s * 0.5,
                        depth + 1,
                        eps,
                        band,
                        max_depth,
                        &mut BuildStats::default(),
                    );
                    ch[i] = tag_local(li);
                }
                RefreshOutcome::Internal { ch, local }
            } else {
                RefreshOutcome::Leaf(corners)
            }
        }
        Node::Internal(ch0) => {
            let mut ch = ch0;
        let mut local: Vec<Node> = Vec::new();
        let mut changed = false;
            for i in 0..8 {
                let sub = refresh_walk(
                    global,
                    sd,
                    new_cuts,
                    ch0[i],
                    child_origin(o, s, i),
                    s * 0.5,
                    depth + 1,
                    eps,
                    band,
                    max_depth,
                );
                match sub {
                    RefreshOutcome::Untouched => {}
                    RefreshOutcome::Leaf(c) => {
                        ch[i] = tag_local(local.len() as u32);
                        local.push(Node::Leaf(c));
                        changed = true;
                    }
                    RefreshOutcome::Internal { ch: cch, local: l } => {
                        let off2 = local.len() as u32;
                        for mut n in l {
                            if let Node::Internal(ref mut cc) = n {
                                for c2 in cc {
                                    if is_local(*c2) {
                                        // 子局部索引 → 父局部索引(保持打标)
                                        *c2 = tag_local(untag(*c2) + off2);
                                    }
                                    // 未打标 = 既有全局索引,原样保留
                                }
                            }
                            local.push(n);
                        }
                        let mut cc = cch;
                        for i2 in 0..8 {
                            if is_local(cc[i2]) {
                                cc[i2] = tag_local(untag(cc[i2]) + off2);
                            }
                        }
                        ch[i] = tag_local(local.len() as u32);
                        local.push(Node::Internal(cc));
                        changed = true;
                    }
                }
            }
            if changed {
                RefreshOutcome::Internal { ch, local }
            } else {
                RefreshOutcome::Untouched
            }
        }
    }
}

impl Adf {
    /// 自适应构建:域为 [min, min+size]³ 的立方体。
    /// band:壳层带宽 mm(只在 |d| ≤ band 处细分;spike 实证此为必要判据)。
    pub fn build(stock: Stock, min: [Real; 3], size: Real, eps: Real, band: Real, max_depth: u8) -> Adf {
        Self::build_with(
            Composite { base: BaseField::Stock(stock), cuts: Vec::new() },
            min,
            size,
            eps,
            band,
            max_depth,
        )
    }

    /// 从任意场源(解析毛坯 或 网格)构建。
    /// 网格场:全局格点**有符号**距离缓存(页式稀疏稠密 + 惰性计算),细格查询
    /// = O(1) 查表,叶角点直存缓存值——无独立符号回填,决策调度无关、可复现。
    pub fn build_with(field: Composite, min: [Real; 3], size: Real, eps: Real, band: Real, max_depth: u8) -> Adf {
        let h_min = size / (1u64 << max_depth) as Real;
        let h2 = size / 8.0;
        // 只有网格场缓存+烘焙(解析场一次求值 ~20ns,皆纯开销)
        let t_bake = std::time::Instant::now();
        let (cache, grid) = match &field.base {
            BaseField::Mesh(_) => {
                let (g, _c) = bake_field(&field.base, min, size);
                (
                    Some(std::sync::Arc::new(LatticeCache::new(min, h_min, max_depth))),
                    Some(g),
                )
            }
            BaseField::Stock(_) => (None, None),
        };
        eprintln!("[phase] bake {:?}", t_bake.elapsed());
        let t_tasks = std::time::Instant::now();
        let base = &field.base;
        let src = match (base, grid.as_ref()) {
            (b, Some(g)) => ExactSrc::Seeded { base: b, grid: g },
            (b, None) => ExactSrc::Base(b),
        };
        // 任务粒度 = depth-4 单元(4096 个):叶片穿越子树与角落空子树工作量差数百倍,
        // depth-3(512 任务)尾部失衡严重;depth-4 尾部 ~1/8,原子队列吃满 24 工人
        let h4 = h2 * 0.5;
        let tasks: Vec<([Real; 3], Real)> = (0..4096usize)
            .map(|t| {
                let (ix, iy, iz) = ((t & 15) as Real, ((t >> 4) & 15) as Real, ((t >> 8) & 15) as Real);
                ([min[0] + ix * h4, min[1] + iy * h4, min[2] + iz * h4], h4)
            })
            .collect();
        let results: Vec<std::sync::Mutex<Option<(Vec<Node>, u32)>>> =
            (0..4096).map(|_| std::sync::Mutex::new(None)).collect();
        let next = std::sync::atomic::AtomicUsize::new(0);
        let nworkers = 24.min(512);
        std::thread::scope(|sc| {
            for _ in 0..nworkers {
                sc.spawn(|| {
                    loop {
                        let t = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if t >= 4096 {
                            break;
                        }
                        let (o, s) = tasks[t];
                        let mut sub_nodes = Vec::new();
                        let sub_root = build_cell(
                            &mut sub_nodes,
                            &src,
                            cache.as_deref(),
                            grid.as_ref(),
                            h_min,
                            o,
                            s,
                            4,
                            eps,
                            band,
                            max_depth,
                            &mut BuildStats::default(),
                        );
                        *results[t].lock().unwrap() = Some((sub_nodes, sub_root));
                    }
                });
            }
        });
        eprintln!(
            "[phase] tasks {:?} | 候选查询 {} | 均候选 {:.0} | BVH后备 {}",
            t_tasks.elapsed(),
            DBG_CAND.load(std::sync::atomic::Ordering::Relaxed),
            DBG_CAND_N.load(std::sync::atomic::Ordering::Relaxed) as f64
                / DBG_CAND.load(std::sync::atomic::Ordering::Relaxed).max(1) as f64,
            DBG_CAND_FB.load(std::sync::atomic::Ordering::Relaxed)
        );
        let t_asm = std::time::Instant::now();
        // 组装(三层):512 棵 depth-3 子树 → 64 个 depth-2 Internal → 8 个 depth-1 Internal → root
        // 确定性:子树按 t3 序 append;子树内部索引 +off;层间子节点 = 对应子树根
        let mut nodes: Vec<Node> = Vec::new();
        let mut d4_root = [0u32; 4096];
        for t4 in 0..4096usize {
            let (sub_nodes, sub_root) = results[t4].lock().unwrap().take().unwrap();
            let off = nodes.len() as u32;
            let llocal = sub_nodes.len() as u32;
            for mut n in sub_nodes {
                if let Node::Internal(ref mut ch) = n {
                    for c in ch {
                        *c += off;
                    }
                }
                nodes.push(n);
            }
            // [dbg] 每任务校验:子索引必须落在本任务块 [off, off+L)
            for (ni, n) in nodes[off as usize..].iter().enumerate() {
                if let Node::Internal(ch) = n {
                    for c in ch {
                        if *c < off || *c >= off + llocal {
                            panic!(
                                "[dbg] 任务 {t4}:off={off} L={llocal} 节点 {} 子 {c} 出块",
                                off as usize + ni
                            );
                        }
                    }
                }
            }
            d4_root[t4] = sub_root + off;
        }
        // depth-3 层:512 个 Internal,子 = 8 个 d4 根(t4 位序 (2ix3+lx)|(2iy3+ly)<<4|(2iz3+lz)<<8)
        let mut d3_root = [0u32; 512];
        for t3 in 0..512usize {
            let (ix3, iy3, iz3) = (t3 & 7, (t3 >> 3) & 7, (t3 >> 6) & 7);
            let mut ch = [0u32; 8];
            for c in 0..8usize {
                let (lx, ly, lz) = (c & 1, (c >> 1) & 1, (c >> 2) & 1);
                let t4 = (2 * ix3 + lx) | ((2 * iy3 + ly) << 4) | ((2 * iz3 + lz) << 8);
                ch[c] = d4_root[t4 as usize];
            }
            nodes.push(Node::Internal(ch));
            d3_root[t3] = (nodes.len() - 1) as u32;
        }
        let _ = h4;
        // depth-2 层:64 个 Internal;t2 的 8 个 d3 子 = (2ix2+lx, 2iy2+ly, 2iz2+lz)
        let mut d2_children = [[0u32; 8]; 64];
        for t2 in 0..64usize {
            let (ix2, iy2, iz2) = (t2 & 3, (t2 >> 2) & 3, (t2 >> 4) & 3);
            for c in 0..8usize {
                let (lx, ly, lz) = (c & 1, (c >> 1) & 1, (c >> 2) & 1);
                let t3 = (2 * ix2 + lx) | ((2 * iy2 + ly) << 3) | ((2 * iz2 + lz) << 6);
                d2_children[t2][c] = d3_root[t3];
            }
            nodes.push(Node::Internal(d2_children[t2]));
            d3_root[t2] = (nodes.len() - 1) as u32; // 复用:此时存 depth-2 节点索引
        }
        // depth-1 层:8 个 Internal;t1 的 8 个 d2 子 = (2ix1+lx, 2iy1+ly, 2iz1+lz)
        let mut d1_children = [[0u32; 8]; 8];
        for t1 in 0..8usize {
            let (ix1, iy1, iz1) = (t1 & 1, (t1 >> 1) & 1, (t1 >> 2) & 1);
            for c in 0..8usize {
                let (lx2, ly2, lz2) = (c & 1, (c >> 1) & 1, (c >> 2) & 1);
                let t2 = (2 * ix1 + lx2) | ((2 * iy1 + ly2) << 2) | ((2 * iz1 + lz2) << 4);
                d1_children[t1][c] = d3_root[t2]; // 此处 d3_root[t2] 已存 depth-2 节点索引
            }
            nodes.push(Node::Internal(d1_children[t1]));
            d3_root[t1] = (nodes.len() - 1) as u32; // 复用:存 depth-1 节点索引
        }
        // 根
        nodes.push(Node::Internal(d3_root[0..8].try_into().unwrap()));
        let root = (nodes.len() - 1) as u32;
        eprintln!("[phase] assemble {:?}", t_asm.elapsed());
        // 符号已在缓存惰性计算中一并确定(有符号值直存叶角点),无独立 Phase B
        Adf {
            min,
            size,
            eps,
            band,
            max_depth,
            nodes,
            root,
            field,
            lattice: cache,
            bake: grid,
        }
    }

    /// 追加一条刀路扫掠并增量刷新(只触碰相交区域 + band 邻域)。
    /// base 项走格点缓存(有符号,O(1));刀路 SDF 解析(廉价)逐条复合。
    pub fn add_cut(&mut self, cap: Capsule) {
        self.field.cuts.push(cap);
        let new_from = self.field.cuts.len() - 1;
        let (min, size, eps, band, max_depth, root) =
            (self.min, self.size, self.eps, self.band, self.max_depth, self.root);
        let base = &self.field.base;
        let cuts = &self.field.cuts;
        let lat = &self.lattice;
        let bake = &self.bake;
        let gate = band + 2.0;
        let src = match (base, self.bake.as_ref()) {
            (b, Some(g)) => ExactSrc::Seeded { base: b, grid: g },
            (b, None) => ExactSrc::Base(b),
        };
        let new_cuts: Vec<Capsule> = self.field.cuts[new_from..].to_vec();
        // 复合场闭包:无内部可变状态(Fn + Sync),24 工人共享只读调用
        let sd_s = |p: [Real; 3]| -> Real {
            let mut m = match (lat, bake) {
                (Some(c), _) => c.lookup_signed_gated(p, &src, bake.as_ref(), gate),
                (None, Some(g)) => g.sample_nearest(p),
                _ => src.signed(p),
            };
            for c in cuts {
                m = m.max(-c.sd(p));
            }
            m
        };
        // ── 阶段 1(并行,只读):层级走根收集 4096 个 depth-4 单元 → Outcome ──
        let nodes_ref: &[Node] = &self.nodes;
        let root_ch = match nodes_ref[root as usize] {
            Node::Internal(c) => c,
            _ => unreachable!("根必 Internal"),
        };
        let mut units: Vec<(u32, [Real; 3], Real)> = Vec::with_capacity(4096);
        for (i1, &n1) in root_ch.iter().enumerate() {
            let o1 = child_origin(min, size, i1);
            let c1 = match nodes_ref[n1 as usize] {
                Node::Internal(c) => c,
                _ => unreachable!("层级构建保证 Internal"),
            };
            for (i2, &n2) in c1.iter().enumerate() {
                let o2 = child_origin(o1, size * 0.5, i2);
                let c2 = match nodes_ref[n2 as usize] {
                    Node::Internal(c) => c,
                    _ => unreachable!(),
                };
                for (i3, &n3) in c2.iter().enumerate() {
                    let o3 = child_origin(o2, size * 0.25, i3);
                    let c3 = match nodes_ref[n3 as usize] {
                        Node::Internal(c) => c,
                        _ => unreachable!(),
                    };
                    for (i4, &n4) in c3.iter().enumerate() {
                        let o4 = child_origin(o3, size * 0.125, i4);
                        units.push((n4, o4, size * 0.0625)); // D4 单元尺寸 = h4(size/16)
                    }
                }
            }
        }
        let results: Vec<std::sync::Mutex<Option<RefreshOutcome>>> =
            (0..units.len()).map(|_| std::sync::Mutex::new(None)).collect();
        let next = std::sync::atomic::AtomicUsize::new(0);
        let nworkers = 24.min(units.len());
        std::thread::scope(|sc| {
            for _ in 0..nworkers {
                sc.spawn(|| loop {
                    let t = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if t >= units.len() {
                        break;
                    }
                    let (idx, o, s) = units[t];
                    let out = refresh_walk(
                        nodes_ref,
                        &sd_s,
                        &new_cuts,
                        idx,
                        o,
                        s,
                        4,
                        eps,
                        band,
                        max_depth,
                    );
                    *results[t].lock().unwrap() = Some(out);
                });
            }
        });
        // ── 阶段 2(串行合并):槽位原位复用 + 局部节点追加(+off 修正)──
        // d3 及以上层级槽位不变(原位改写),层级索引稳定,无需向上重连
        for (t, (slot, _, _)) in units.iter().enumerate() {
            let mut out = results[t].lock().unwrap().take().unwrap();
            match &mut out {
                RefreshOutcome::Untouched => {}
                RefreshOutcome::Leaf(c) => {
                    self.nodes[*slot as usize] = Node::Leaf(*c);
                }
                RefreshOutcome::Internal { ch, local } => {
                    let off = self.nodes.len() as u32;
                    for mut n in local.drain(..) {
                        if let Node::Internal(ref mut cc) = n {
                            for c2 in cc {
                                if is_local(*c2) {
                                    // 局部 → 全局(去标)
                                    *c2 = untag(*c2) + off;
                                }
                            }
                        }
                        self.nodes.push(n);
                    }
                    for i in 0..8 {
                        if is_local(ch[i]) {
                            ch[i] = untag(ch[i]) + off;
                        }
                    }
                    self.nodes[*slot as usize] = Node::Internal(*ch);
                }
            }
        }
    }

    /// 自适应求值:下降到包含 p 的叶,角点三线性插值。
    /// 已知限制:跨分辨率边界处插值基不一致,存在 T-junction 误差(spike 要测的就是它)。
    pub fn eval(&self, p: [Real; 3]) -> f32 {
        let (corners, u, _) = self.leaf_coordinates(self.clamp_to_domain(p));
        trilinear(u, corners)
    }

    /// 投影使用 eval 的域外常量外延：被钳位的轴上偏导为零，其余轴保留叶内偏导。
    fn eval_with_gradient(&self, p: [Real; 3]) -> (Real, [Real; 3]) {
        let q = self.clamp_to_domain(p);
        let (corners, u, width) = self.leaf_coordinates(q);
        let mut gradient = tri_grad(corners, u, 1.0 / width);
        for axis in 0..3 {
            if q[axis] != p[axis] {
                gradient[axis] = 0.0;
            }
        }
        (trilinear(u, corners) as Real, gradient)
    }

    /// 输入已钳位到域内的坐标，取所在叶、局部坐标及格宽。
    #[inline]
    fn leaf_coordinates(&self, q: [Real; 3]) -> (&[f32; 8], [Real; 3], Real) {
        let mut o = self.min;
        let mut s = self.size;
        let mut idx = self.root;
        loop {
            match &self.nodes[idx as usize] {
                Node::Leaf(c) => {
                    let u = [(q[0] - o[0]) / s, (q[1] - o[1]) / s, (q[2] - o[2]) / s];
                    return (c, u, s);
                }
                Node::Internal(ch) => {
                    let i = child_index(q, o, s);
                    o = child_origin(o, s, i);
                    s *= 0.5;
                    idx = ch[i];
                }
            }
        }
    }

    /// 解析真值(验证基准)。
    pub fn true_sd(&self, p: [Real; 3]) -> Real {
        self.field.sd(p)
    }

    /// p 所在叶的深度(诊断用)。
    pub fn depth_at(&self, p: [Real; 3]) -> u8 {
        let q = self.clamp_to_domain(p);
        let mut o = self.min;
        let mut s = self.size;
        let mut idx = self.root;
        let mut d = 0u8;
        loop {
            match self.nodes[idx as usize] {
                Node::Leaf(_) => return d,
                Node::Internal(ch) => {
                    let i = child_index(q, o, s);
                    o = child_origin(o, s, i);
                    s *= 0.5;
                    idx = ch[i];
                    d += 1;
                }
            }
        }
    }

    /// p 所在叶的格宽(诊断用)。
    pub fn cell_size_at(&self, p: [Real; 3]) -> Real {
        self.size / (1u64 << self.depth_at(p) as u32) as Real
    }

    /// 诊断:转储 p 所在叶的 8 个角点(位置/值)。
    pub fn dump_leaf(&self, p: [Real; 3]) {
        // 先下降到叶,记录 origin/size
        let q = self.clamp_to_domain(p);
        let mut o = self.min;
        let mut s = self.size;
        let mut idx = self.root;
        loop {
            match self.nodes[idx as usize] {
                Node::Leaf(c) => {
                    println!("  [leaf] origin=({:.4},{:.4},{:.4}) size={s:.4}", o[0], o[1], o[2]);
                    for k in 0..8 {
                        let cp = corner_pos(o, s, k);
                        println!(
                            "    c{k} ({:.4},{:.4},{:.4}) = {:.4}",
                            cp[0], cp[1], cp[2], c[k]
                        );
                    }
                    return;
                }
                Node::Internal(ch) => {
                    let i = child_index(q, o, s);
                    o = child_origin(o, s, i);
                    s *= 0.5;
                    idx = ch[i];
                }
            }
        }
    }

    /// 重距离化(§8.4/§9.12):多次布尔减后,**空气区幅值是下界**(被后续刀路超越
    /// 的旧表面参与 max,导致低估),材料区恒精确、零等值面恒精确。
    /// 做法:在"界面叶(角点符号变化)∩ 刀路区"+ band 上建 h_min 均匀网格,
    /// 以符号翻转棱为种子(距离 ≤ t·h),6 向×8 序 Gauss-Seidel 松弛解 |∇d|=1,
    /// 写回只修正**空气侧且偏离 > 0.25h** 的角点(保持平坦面精确值、符号、零面)。
    pub fn redistance(&mut self, rounds: u32) -> RedistStats {
        let h = self.size / (1u64 << self.max_depth) as Real;
        let band = self.band;
        // 0) 刀路总 AABB(场退化只发生在刀路区;远离刀路的平坦面保持精确)
        let (mut c0, mut c1) = ([f64::MAX; 3], [f64::MIN; 3]);
        for c in &self.field.cuts {
            let (mn, mx) = c.aabb();
            for k in 0..3 {
                c0[k] = c0[k].min(mn[k]);
                c1[k] = c1[k].max(mx[k]);
            }
        }
        // 1) 界面叶:角点符号变化 且 与刀路区相交
        let mut iface: Vec<(u32, [Real; 3], Real)> = Vec::new();
        Self::collect_interface(&self.nodes, self.root, self.min, self.size, c0, c1, &mut iface);
        let mut st = RedistStats { interface_leaves: iface.len() as u64, ..Default::default() };
        if iface.is_empty() {
            return st;
        }
        // 2) 网格范围 = 界面叶 AABB 并集,外扩 band,对齐 h_min 格
        let (mut b0, mut b1) = ([f64::MAX; 3], [f64::MIN; 3]);
        for (_, o, s) in &iface {
            for k in 0..3 {
                b0[k] = b0[k].min(o[k]);
                b1[k] = b1[k].max(o[k] + s);
            }
        }
        let mut gmin = [0.0; 3];
        let mut gn = [0usize; 3];
        for k in 0..3 {
            let lo = ((b0[k] - band - self.min[k]) / h).floor().max(0.0);
            gmin[k] = self.min[k] + lo * h;
            let hi_cells = ((b1[k] + band - gmin[k]) / h).ceil() as usize + 1;
            let nmax = ((self.size / h).ceil() as usize + 1).min(1usize << self.max_depth);
            gn[k] = hi_cells.min(nmax - lo as usize);
        }
        st.grid_n = gn;
        let (nx, ny, nz) = (gn[0], gn[1], gn[2]);
        let gi = |i: usize, j: usize, k: usize| (k * ny + j) * nx + i;

        // 3) 符号场(退化场,符号恒正确)
        let t_eval = std::time::Instant::now();
        let mut sv = vec![0f32; nx * ny * nz];
        // 逐点 eval 是纯函数、彼此独立 → 按 k 平面切带并行(结果与线程数无关,
        // 因为每个格点的值只由 (i,j,k) 位置决定,不依赖访问顺序)。
        {
            let this: &Adf = &*self;
            let ppw = nz.div_ceil(nthreads()).max(1);
            let band = ppw * nx * ny;
            let chunks: Vec<&mut [f32]> = sv.chunks_mut(band).collect();
            std::thread::scope(|sc| {
                for (w, chunk) in chunks.into_iter().enumerate() {
                    let k0 = w * ppw;
                    sc.spawn(move || {
                        for (idx, slot) in chunk.iter_mut().enumerate() {
                            let k = k0 + idx / (nx * ny);
                            let rem = idx % (nx * ny);
                            let (j, i) = (rem / nx, rem % nx);
                            *slot = this.eval([
                                gmin[0] + i as Real * h,
                                gmin[1] + j as Real * h,
                                gmin[2] + k as Real * h,
                            ]);
                        }
                    });
                }
            });
        }

        // 4) 距离格:INF 初始化,种子 = 符号翻转棱(端点距离 ≤ t·h)
        st.t_eval = t_eval.elapsed();
        let t_seed = std::time::Instant::now();
        let inf = f32::INFINITY;
        let mut d = vec![inf; nx * ny * nz];
        let hf = h as f32;
        let seed = |i: usize, j: usize, k: usize, dist: Real, d: &mut Vec<f32>| {
            let dv = dist as f32;
            if dv < d[gi(i, j, k)] {
                d[gi(i, j, k)] = dv;
            }
        };
        // x 向棱
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx.saturating_sub(1) {
                    let (sa, sb) = (sv[gi(i, j, k)] as Real, sv[gi(i + 1, j, k)] as Real);
                    if (sa < 0.0) != (sb < 0.0) {
                        let t = sa / (sa - sb);
                        seed(i, j, k, t * h, &mut d);
                        seed(i + 1, j, k, (1.0 - t) * h, &mut d);
                    }
                }
            }
        }
        // y 向棱
        for k in 0..nz {
            for j in 0..ny.saturating_sub(1) {
                for i in 0..nx {
                    let (sa, sb) = (sv[gi(i, j, k)] as Real, sv[gi(i, j + 1, k)] as Real);
                    if (sa < 0.0) != (sb < 0.0) {
                        let t = sa / (sa - sb);
                        seed(i, j, k, t * h, &mut d);
                        seed(i, j + 1, k, (1.0 - t) * h, &mut d);
                    }
                }
            }
        }
        // z 向棱
        for k in 0..nz.saturating_sub(1) {
            for j in 0..ny {
                for i in 0..nx {
                    let (sa, sb) = (sv[gi(i, j, k)] as Real, sv[gi(i, j, k + 1)] as Real);
                    if (sa < 0.0) != (sb < 0.0) {
                        let t = sa / (sa - sb);
                        seed(i, j, k, t * h, &mut d);
                        seed(i, j, k + 1, (1.0 - t) * h, &mut d);
                    }
                }
            }
        }

        // 5) 精确 EDT 重距离化(见 edt_redist 的取舍说明)。
        //    设 ADF_REDIST_SWEEP=1 可切回串行八序扫掠——仅作对账/回归用参考路径,
        //    `rounds` 只在该路径下有意义。
        st.t_seed = t_seed.elapsed();
        let t_relax = std::time::Instant::now();
        if std::env::var_os("ADF_REDIST_SWEEP").is_some() {
            eikonal_sweep_serial(&mut d, gn, hf, rounds);
        } else {
            edt_redist(&mut d, gn, hf, nthreads());
        }
        //    教训 5:简单 min(邻居)+h 松弛收敛到 L1(曼哈顿)距离 —— 实测空气区
        //    eval = L1(如 1.4647 = 30.7 格 × h 的曼哈顿路径);必须用 Godunov 多轴解。

        // 6) 写回:只修正空气侧(old > 0.5h)且偏离 > 0.25h 的角点
        st.t_relax = t_relax.elapsed();
        let t_write = std::time::Instant::now();
        let mut corrected = 0u64;
        let gmax = [gmin[0] + (nx - 1) as Real * h, gmin[1] + (ny - 1) as Real * h, gmin[2] + (nz - 1) as Real * h];
        Self::writeback_walk(
            &mut self.nodes, self.root, self.min, self.size, gmin, gmax, h, gn, &d,
            &self.field.cuts, &mut corrected,
        );
        st.corrected = corrected;
        st.t_write = t_write.elapsed();
        st
    }

    /// 把 p 钳入域内。**逐轴**上界 = `min[k] + size`:域是从 `min` 起、边长 `size`
    /// 的立方体,而 `min` 各轴不同(域是把零件 bbox 各轴按最大边长对齐出来的),
    /// 用 axis-0 的上界会把另一轴的高坐标误钳到边界 —— `eval` 因此落到错误的叶,
    /// 远场误差达数十毫米(P6 的真实根因,2026-11-15 定位)。
    #[inline]
    fn clamp_to_domain(&self, p: [Real; 3]) -> [Real; 3] {
        let mut q = [0.0; 3];
        for k in 0..3 {
            q[k] = clamp(p[k], self.min[k], self.min[k] + self.size - 1e-9);
        }
        q
    }

    fn collect_interface(
        nodes: &[Node],
        idx: u32,
        o: [Real; 3],
        s: Real,
        c0: [Real; 3],
        c1: [Real; 3],
        out: &mut Vec<(u32, [Real; 3], Real)>,
    ) {
        if !aabb_overlap(o, s, c0, c1) {
            return; // 场退化只发生在刀路区
        }
        match nodes[idx as usize] {
            Node::Leaf(c) => {
                let (mut neg, mut pos) = (false, false);
                for v in c {
                    neg |= v < 0.0;
                    pos |= v >= 0.0;
                }
                if neg && pos {
                    out.push((idx, o, s)); // 界面叶:零等值面从格内穿过
                }
            }
            Node::Internal(ch) => {
                for i in 0..8 {
                    Self::collect_interface(nodes, ch[i], child_origin(o, s, i), s * 0.5, c0, c1, out);
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn writeback_walk(
        nodes: &mut Vec<Node>,
        idx: u32,
        o: [Real; 3],
        s: Real,
        gmin: [Real; 3],
        gmax: [Real; 3],
        h: Real,
        gn: [usize; 3],
        d: &[f32],
        cuts: &[Capsule],
        corrected: &mut u64,
    ) {
        if !aabb_overlap(o, s, gmin, gmax) {
            return;
        }
        match nodes[idx as usize] {
            Node::Leaf(_) => {
                if let Node::Leaf(mut c) = nodes[idx as usize] {
                    for k in 0..8 {
                        let p = corner_pos(o, s, k);
                        if c[k] <= 0.5 * h as f32 {
                            continue; // 界面邻域/材料侧:保持(材料区恒精确)
                        }
                        // 精确性判据(P23):空气角点旧值 v 若不大于到所有刀路的距离,
                        // 则它是精确距离,不需要也不应被 EDT 覆盖。证明:v = base(p),
                        // 原表面最近点 q 满足 |p−q| = v;对每条刀路 cap.sd(q) ≥
                        // cap.sd(p) − v ≥ 0,即 q 未被切掉,新表面距离 ≤ v;而切削只
                        // 减少材料,距离只增不减 ⟹ 恰为 v。
                        // 不加此判据时,刀路 AABB 并集(贯穿件的长胶囊 = 整片厚板)内
                        // 所有原始表面的空气角点都会被 EDT 的种子离散化值(±0.25–1 格)
                        // 覆盖,三线性零面随之漂移 —— 远离刀路的原始表面出现"橘皮"。
                        let cut_dist = cuts.iter().map(|cap| cap.sd(p)).fold(Real::INFINITY, Real::min);
                        if (c[k] as Real) <= cut_dist {
                            continue;
                        }
                        let g: [isize; 3] = [
                            ((p[0] - gmin[0]) / h).round() as isize,
                            ((p[1] - gmin[1]) / h).round() as isize,
                            ((p[2] - gmin[2]) / h).round() as isize,
                        ];
                        if g[0] < 0 || g[1] < 0 || g[2] < 0
                            || g[0] >= gn[0] as isize || g[1] >= gn[1] as isize || g[2] >= gn[2] as isize
                        {
                            continue;
                        }
                        let m = d[(g[2] as usize * gn[1] + g[1] as usize) * gn[0] + g[0] as usize];
                        if !m.is_finite() {
                            continue;
                        }
                        let new = old_signum_positive(c[k]) * m; // 空气侧:符号恒正
                        if (new - c[k]).abs() > 0.25 * h as f32 {
                            c[k] = new;
                            *corrected += 1;
                        }
                    }
                    nodes[idx as usize] = Node::Leaf(c);
                }
            }
            Node::Internal(ch) => {
                for i in 0..8 {
                    Self::writeback_walk(nodes, ch[i], child_origin(o, s, i), s * 0.5, gmin, gmax, h, gn, d, cuts, corrected);
                }
            }
        }
    }

    /// 后序粗化:8 子全为叶、**父格不在表面壳层内**、且合并后多点判据仍满足
    /// → 合并为父叶。壳层守卫与 build 同款(1-Lipschitz):防止把含折线/开口缘的
    /// 空气大格合并掉(spike 实测:无守卫时开口缘 eval 误差 1.06mm)。
    /// 返回本子树发生的合并次数。根(depth 0)不合并。
    fn coarsen_cell(
        nodes: &mut Vec<Node>,
        sd: &mut dyn FnMut([Real; 3]) -> Real,
        idx: u32,
        o: [Real; 3],
        s: Real,
        depth: u8,
        eps: Real,
        band: Real,
    ) -> u64 {
        let ch = match nodes[idx as usize] {
            Node::Leaf(_) => return 0,
            Node::Internal(ch) => ch,
        };
        let mut merges = 0u64;
        for i in 0..8 {
            merges += Self::coarsen_cell(nodes, sd, ch[i], child_origin(o, s, i), s * 0.5, depth + 1, eps, band);
        }
        // 子树合并后重读子节点
        let ch = match nodes[idx as usize] {
            Node::Internal(ch) => ch,
            Node::Leaf(_) => return merges,
        };
        let mut all_leaf = true;
        for c in ch {
            if !matches!(nodes[c as usize], Node::Leaf(_)) {
                all_leaf = false;
                break;
            }
        }
        if !all_leaf || depth == 0 {
            return merges;
        }
        // 父角点 = 第 i 个子叶的第 i 角(位置重合,值即真样本)
        let mut corners = [0f32; 8];
        for k in 0..8 {
            if let Node::Leaf(cc) = nodes[ch[k as usize] as usize] {
                corners[k] = cc[k];
            } else {
                return merges;
            }
        }
        let (dev, c_true) = cell_deviation(sd, o, s, &corners, &mut BuildStats::default());
        // 壳层守卫:父格可能触及 |d| ≤ band 壳层 → 不合并
        let half_diag = s * 0.75_f64.sqrt();
        if c_true.abs() as Real <= band + half_diag {
            return merges;
        }
        if dev <= eps as f32 {
            nodes[idx as usize] = Node::Leaf(corners);
            merges += 1;
        }
        merges
    }

    /// 全树粗化一遍(自底向上),容差 = 自身 eps。返回合并次数。
    pub fn coarsen(&mut self) -> u64 {
        self.coarsen_with(self.eps)
    }

    /// 全树粗化(独立容差):渲染/导出路径可用宽松容差(如 4–5×eps)换叶数,
    /// 验证路径用严格 eps。场值漂移上界 ≈ eps_coarse(壳层内不合并)。
    pub fn coarsen_with(&mut self, eps_coarse: Real) -> u64 {
        let (min, size, band, root) = (self.min, self.size, self.band, self.root);
        let field = &self.field;
        let mut sd_s = |p: [Real; 3]| -> Real { field.sd(p) };
        let nodes = &mut self.nodes;
        Self::coarsen_cell(nodes, &mut sd_s, root, min, size, 0, eps_coarse, band)
    }

    /// 叶统计:(数量, 估算字节, 深度直方图)。
    /// ADF 常驻内存分项(容量口径,含它拥有的源网格场)。
    pub fn mem_bytes(&self) -> AdfMem {
        use std::mem::size_of;
        let mut m = AdfMem {
            tree: self.nodes.capacity() * size_of::<Node>(),
            cuts: self.field.cuts.capacity() * size_of::<Capsule>(),
            ..AdfMem::default()
        };
        if let Some(l) = &self.lattice {
            let (d, p, n) = l.mem_bytes();
            m.lattice_dir = d;
            m.lattice_pages = p;
            m.lattice_page_count = n;
        }
        if let Some(g) = &self.bake {
            m.bake = g.data.capacity() * size_of::<f32>() + size_of::<BakeGrid>();
        }
        if let BaseField::Mesh(f) = &self.field.base {
            m.field = f.mem_bytes();
        }
        m
    }

    pub fn leaf_stats_bytes(&self) -> (u64, u64, [u64; 17]) {
        let mut n = 0u64;
        let mut bytes = 0u64;
        let mut hist = [0u64; 17];
        let mut stack = vec![(self.root, 0u8)];
        while let Some((idx, d)) = stack.pop() {
            match self.nodes[idx as usize] {
                Node::Leaf(_) => {
                    n += 1;
                    bytes += std::mem::size_of::<Node>() as u64; // 实测 36 B([f32;8] + 1 B 标签,对齐 4)
                    hist[d as usize] += 1;
                }
                Node::Internal(ch) => {
                    bytes += std::mem::size_of::<Node>() as u64;
                    for c in ch {
                        stack.push((c, d + 1));
                    }
                }
            }
        }
        (n, bytes, hist)
    }
}

/// 确定性随机数(LCG),验证台用。
pub struct Rng(pub u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    pub fn f01(&mut self) -> Real {
        (self.next_u64() >> 11) as Real / (1u64 << 53) as Real
    }
}

// ─────────────────────────── Surface Nets 表面提取 + STL ───────────────────────────

/// 均匀重采样 + Surface Nets(对偶法,免 LUT;与 CNCSIM 同选型)。
/// 这是 ADF → 可视化网格的务实路线:在目标分辨率上重采样 eval,再做对偶提取。
pub mod nets {
    use super::*;

    /// 整数键专用哈希(零依赖):std 默认 SipHash 对 `(u32,u32)` 边键太慢 ——
    /// 482k 三角的绕向统一要 260ms,换成乘法散列后降到 ~60ms。
    #[derive(Default, Clone, Copy)]
    struct FxHash(u64);
    impl std::hash::Hasher for FxHash {
        #[inline]
        fn write_u64(&mut self, v: u64) {
            self.0 = (self.0 ^ v).wrapping_mul(0x517c_c1b7_2722_0a95);
        }
        #[inline]
        fn write(&mut self, bytes: &[u8]) {
            for &b in bytes {
                self.0 = (self.0.rotate_left(5) ^ b as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
            }
        }
        #[inline]
        fn finish(&self) -> u64 {
            self.0
        }
    }
    #[derive(Default, Clone, Copy)]
    struct FxBuild;
    impl std::hash::BuildHasher for FxBuild {
        type Hasher = FxHash;
        #[inline]
        fn build_hasher(&self) -> FxHash {
            FxHash(0)
        }
    }

    #[derive(Clone)]
    pub struct Mesh {
        pub verts: Vec<[f32; 3]>,
        pub tris: Vec<[u32; 3]>,
    }

    pub use crate::creases::ReferenceCreases;

    fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    }
    fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }
    fn g64(v: [f32; 3]) -> [f64; 3] {
        [v[0] as f64, v[1] as f64, v[2] as f64]
    }

    // ── 折痕毛刺清理(P26/P27)用的小工具 ──
    /// 薄片判据的最小高度(mm),与 qa_render 的 sliver 口径一致。
    const SPUR_SLIVER_H: f64 = 0.01;
    fn spur_edge_key(a: u32, b: u32) -> u64 {
        if a < b { ((a as u64) << 32) | b as u64 } else { ((b as u64) << 32) | a as u64 }
    }
    fn spur_normal(p: [[f64; 3]; 3]) -> [f64; 3] {
        cross3(sub(p[1], p[0]), sub(p[2], p[0]))
    }
    fn spur_emax(p: [[f64; 3]; 3]) -> f64 {
        len(sub(p[1], p[0])).max(len(sub(p[2], p[1]))).max(len(sub(p[0], p[2])))
    }
    /// 归一化质量 2A / 最长边²(等边 0.866)。
    fn spur_quality(p: [[f64; 3]; 3]) -> f64 {
        let e = spur_emax(p);
        if e > 0.0 { len(spur_normal(p)) / (e * e) } else { 0.0 }
    }
    /// 最小高度 2A / 最长边。
    fn spur_height(p: [[f64; 3]; 3]) -> f64 {
        let e = spur_emax(p);
        if e > 0.0 { len(spur_normal(p)) / e } else { 0.0 }
    }
    /// 面法线与质心处最近源折痕两侧法线的最大一致度;附近无折痕时 −1。
    fn spur_agree(grid: &ReferenceCreases, p: [[f64; 3]; 3]) -> f64 {
        let n = spur_normal(p);
        let l = len(n);
        if l < 1e-14 {
            return -1.0;
        }
        let c = scale(add(add(p[0], p[1]), p[2]), 1.0 / 3.0);
        match grid.nearest(c) {
            Some((_, _, [a, b])) => dot3(n, a).max(dot3(n, b)) / l,
            None => -1.0,
        }
    }
    /// P29:新面在源网格上的采样复核。按 ≤0.3mm 间距(每边 2–8 段)取重心网格点(不含 3 个顶点,
    /// 它们本来就在源面上),要求到源面距离 ≤ `tol`,法线与最近源面(或其旁的源折痕两侧)一致度 ≥ `agree`。
    /// P36:半径 `r` 内的最近源面(距离, 三角形号);`r` 内没有则 None。BVH 以 r² 作剪枝初值,
    /// 只展开离 p 不足 r 的子树 —— 结果与无界查询相同(最近点在 r 内时),但快得多:简化/复核里
    /// 绝大多数查询点本就贴着源面(< 0.02mm),无界查询要先下探到别处再回头收紧。
    fn closest_within(reference: &meshfield::MeshField, p: [f64; 3], r: f64) -> Option<(f64, u32)> {
        // 以 (2r)² 作初值:没找到时返回值恰为 2r,与"找到且 ≤ r"无歧义(不依赖 sqrt 的末位舍入)。
        let (d, t, _, _) = reference.closest_features_seeded(p, 4.0 * r * r);
        (d <= r).then_some((d, t))
    }

    /// 到源面的距离与三角形号:先在 0.1mm 内找,找不到再无界查询(结果与 `closest_features` 相同,P37)。
    fn closest_near(reference: &meshfield::MeshField, p: [f64; 3]) -> (f64, u32) {
        match closest_within(reference, p, 0.1) {
            Some(x) => x,
            None => {
                let (d, t, _, _) = reference.closest_features(p);
                (d, t)
            }
        }
    }

    /// 最近源面三角形号:先在 0.1mm 内找,找不到再无界查询(结果与 `closest_features` 相同)。
    fn closest_face_near(reference: &meshfield::MeshField, p: [f64; 3]) -> u32 {
        match closest_within(reference, p, 0.1) {
            Some((_, t)) => t,
            None => reference.closest_features(p).1,
        }
    }

    fn ref_sample_ok(reference: &meshfield::MeshField, grid: &ReferenceCreases, p: [[f64; 3]; 3], tol: f64, agree: f64) -> bool {
        let n = spur_normal(p);
        let l = len(n);
        if l < 1e-14 {
            return false;
        }
        let n = scale(n, 1.0 / l);
        let m = (spur_emax(p) / 0.3).ceil().clamp(2.0, 8.0) as usize;
        for i in 0..=m {
            for j in 0..=(m - i) {
                let k = m - i - j;
                if i == m || j == m || k == m {
                    continue;
                }
                let s = scale(
                    add(add(scale(p[0], i as f64), scale(p[1], j as f64)), scale(p[2], k as f64)),
                    1.0 / m as f64,
                );
                let Some((d, t)) = closest_within(reference, s, tol) else {
                    return false;
                };
                let mut ag = dot3(n, reference.face_normal(t));
                if ag < agree
                    && let Some((dk, _, [a, b])) = grid.nearest(s)
                    && dk <= d + 0.005
                {
                    ag = ag.max(dot3(n, a)).max(dot3(n, b));
                }
                if ag < agree {
                    return false;
                }
            }
        }
        true
    }
    /// 面相对源网格的缺陷判据与代价(P28)。返回 (是否可见缺陷, 代价)。
    /// 可见缺陷:7 个采样点(3 顶点、质心、3 边中点)到源面最大距离 > 0.02mm,或质心处法线与
    /// 最近源面 / 最近源折痕两侧法线的最大一致度 < 0.8,或退化。
    fn ref_face_cost(reference: &meshfield::MeshField, grid: &ReferenceCreases, p: [[f64; 3]; 3]) -> (bool, f64) {
        let c = scale(add(add(p[0], p[1]), p[2]), 1.0 / 3.0);
        let samples = [
            p[0],
            p[1],
            p[2],
            c,
            scale(add(p[0], p[1]), 0.5),
            scale(add(p[1], p[2]), 0.5),
            scale(add(p[2], p[0]), 0.5),
        ];
        let mut dmax: f64 = 0.0;
        for s in samples {
            dmax = dmax.max(closest_near(reference, s).0);
        }
        let n = spur_normal(p);
        let l = len(n);
        let agree = if l < 1e-14 {
            -1.0
        } else {
            let (dc, t) = closest_near(reference, c);
            let mut ag = dot3(n, reference.face_normal(t)) / l;
            if let Some((dk, _, [a, b])) = grid.nearest(c)
                && dk <= dc + 0.005
            {
                ag = ag.max(dot3(n, a).max(dot3(n, b)) / l);
            }
            ag
        };
        let bad = dmax > 0.02 || agree < 0.8;
        let cost = (dmax - 0.01).max(0.0) + (0.9 - agree).max(0.0) * 0.05 + if l < 1e-14 { 0.05 } else { 0.0 };
        (bad, cost)
    }
    /// P30 折痕嵌入的工作状态:边 → 面表、顶点空间哈希(重合检查),以及带守卫的边/面分裂。
    struct CreaseEmbed<'a> {
        verts: &'a mut Vec<[f32; 3]>,
        tris: &'a mut Vec<[u32; 3]>,
        emap: std::collections::HashMap<u64, Vec<u32>, FxBuild>,
        vhash: std::collections::HashMap<[i64; 3], Vec<u32>, FxBuild>,
        cell: f64,
        on_tol: f64,
        /// P36:面的均匀网格(按面包围盒登记;跨格过多的大面放进 `big`,每次查询都带上)。
        /// 旧版每个角点、每次切边都遍历全部面并在切边后从头重扫,翅膀上是 ~10¹⁰ 次面访问。
        fgrid: std::collections::HashMap<[i64; 3], Vec<u32>, FxBuild>,
        big: Vec<u32>,
        fcell: f64,
    }

    impl<'a> CreaseEmbed<'a> {
        fn new(verts: &'a mut Vec<[f32; 3]>, tris: &'a mut Vec<[u32; 3]>, on_tol: f64, fcell: f64) -> Self {
            let cell = (on_tol * 4.0).max(1e-3);
            let mut emap: std::collections::HashMap<u64, Vec<u32>, FxBuild> =
                std::collections::HashMap::with_capacity_and_hasher(tris.len() * 3, FxBuild);
            for (ti, t) in tris.iter().enumerate() {
                for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                    emap.entry(spur_edge_key(a, b)).or_default().push(ti as u32);
                }
            }
            let mut vhash: std::collections::HashMap<[i64; 3], Vec<u32>, FxBuild> =
                std::collections::HashMap::with_capacity_and_hasher(verts.len(), FxBuild);
            for (vi, v) in verts.iter().enumerate() {
                vhash.entry(g64(*v).map(|x| (x / cell).floor() as i64)).or_default().push(vi as u32);
            }
            let mut e = Self { verts, tris, emap, vhash, cell, on_tol, fgrid: Default::default(), big: Vec::new(), fcell };
            for fi in 0..e.tris.len() as u32 {
                e.grid_insert(fi);
            }
            e
        }

        fn fkey(&self, p: [f64; 3]) -> [i64; 3] {
            p.map(|x| (x / self.fcell).floor() as i64)
        }

        fn grid_insert(&mut self, fi: u32) {
            let p = self.tris[fi as usize].map(|i| g64(self.verts[i as usize]));
            let lo = self.fkey([0, 1, 2].map(|k| p[0][k].min(p[1][k]).min(p[2][k])));
            let hi = self.fkey([0, 1, 2].map(|k| p[0][k].max(p[1][k]).max(p[2][k])));
            let span = (0..3).map(|k| (hi[k] - lo[k] + 1) as u64).product::<u64>();
            if span > 512 {
                self.big.push(fi);
                return;
            }
            for x in lo[0]..=hi[0] {
                for y in lo[1]..=hi[1] {
                    for z in lo[2]..=hi[2] {
                        self.fgrid.entry([x, y, z]).or_default().push(fi);
                    }
                }
            }
        }

        /// 包围盒与以 `pts` 中各点为中心、边长 2·(fcell) 的立方体相交的面(去重、升序)。
        fn faces_near(&self, pts: &[[f64; 3]]) -> Vec<u32> {
            let mut out: Vec<u32> = self.big.clone();
            let mut seen: std::collections::HashSet<[i64; 3], FxBuild> = Default::default();
            for &p in pts {
                let k = self.fkey(p);
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        for dz in -1..=1 {
                            let c = [k[0] + dx, k[1] + dy, k[2] + dz];
                            if seen.insert(c)
                                && let Some(list) = self.fgrid.get(&c)
                            {
                                out.extend_from_slice(list);
                            }
                        }
                    }
                }
            }
            out.sort_unstable();
            out.dedup();
            out
        }

        fn pos(&self, i: u32) -> [f64; 3] {
            g64(self.verts[i as usize])
        }

        fn near_vertex(&self, p: [f64; 3]) -> bool {
            let k = p.map(|x| (x / self.cell).floor() as i64);
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(list) = self.vhash.get(&[k[0] + dx, k[1] + dy, k[2] + dz]) {
                            if list.iter().any(|&v| len(sub(self.pos(v), p)) < self.on_tol) {
                                return true;
                            }
                        }
                    }
                }
            }
            false
        }

        fn add_vertex(&mut self, p: [f64; 3]) -> u32 {
            let id = self.verts.len() as u32;
            let pf = p.map(|x| x as f32);
            self.verts.push(pf);
            self.vhash.entry(g64(pf).map(|x| (x / self.cell).floor() as i64)).or_default().push(id);
            id
        }

        fn set_face(&mut self, fi: u32, t: [u32; 3]) {
            let old = self.tris[fi as usize];
            for (a, b) in [(old[0], old[1]), (old[1], old[2]), (old[2], old[0])] {
                if let Some(list) = self.emap.get_mut(&spur_edge_key(a, b)) {
                    list.retain(|&f| f != fi);
                    if list.is_empty() {
                        self.emap.remove(&spur_edge_key(a, b));
                    }
                }
            }
            self.tris[fi as usize] = t;
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                self.emap.entry(spur_edge_key(a, b)).or_default().push(fi);
            }
            // 新顶点可能离原边 ≤ r_cut,面包围盒可能变大:补登记(旧登记保留,查询时按几何再筛)
            self.grid_insert(fi);
        }

        fn push_face(&mut self, t: [u32; 3]) {
            let fi = self.tris.len() as u32;
            self.tris.push(t);
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                self.emap.entry(spur_edge_key(a, b)).or_default().push(fi);
            }
            self.grid_insert(fi);
        }

        /// 子面 `pn` 替换(或部分替换)原面 `po` 时是否可接受。
        fn sub_ok(po: [[f64; 3]; 3], pn: [[f64; 3]; 3], side: Option<[[f64; 3]; 2]>) -> bool {
            let (no, nn) = (spur_normal(po), spur_normal(pn));
            let (lo, ln) = (len(no), len(nn));
            if ln < 1e-12 || lo < 1e-14 {
                return false;
            }
            let agree = side.map_or(-1.0, |[n1, n2]| dot3(nn, n1).max(dot3(nn, n2)) / ln);
            // 子面与某侧源法线一致(≥ 0.9)就接受,不再与原面比较:被切的原面本身可能是斜跨折痕、
            // 朝向已偏的坏面(桌腿 −Y 切点处原面法线 (−0.39, −0.87, −0.3),切出的 +x 子面与它点积
            // −0.39,旧判据拒绝,留下一根伸出棱线的"刺",P31)。
            if dot3(no, nn) < 0.2 * lo * ln && agree < 0.9 {
                return false;
            }
            !(side.is_some() && spur_quality(pn) >= 0.02 && agree < 0.5)
        }

        /// 在边 (a, b) 上点 p 处分裂(所有相邻面各一分为二)。守卫不过则不改动,返回 None。
        fn try_split_edge(&mut self, a: u32, b: u32, p: [f64; 3], side: Option<[[f64; 3]; 2]>) -> Option<u32> {
            let faces = self.emap.get(&spur_edge_key(a, b))?.clone();
            if faces.is_empty() || faces.len() > 2 || self.near_vertex(p) {
                return None;
            }
            let p = p.map(|x| x as f32 as f64);
            let mut plan: Vec<(u32, u32, u32, u32)> = Vec::with_capacity(2);
            for &fi in &faces {
                let t = self.tris[fi as usize];
                let po = t.map(|i| self.pos(i));
                if let Some([n1, n2]) = side {
                    let nf = spur_normal(po);
                    if dot3(nf, n1).max(dot3(nf, n2)) < 0.5 * len(nf) {
                        return None;
                    }
                }
                let i = t.iter().position(|&v| v == a)?;
                let (x, y) = if t[(i + 1) % 3] == b { (a, b) } else { (b, a) };
                let k = t.iter().position(|&v| v == x)?;
                let r = t[(k + 2) % 3];
                for pn in [[self.pos(x), p, self.pos(r)], [p, self.pos(y), self.pos(r)]] {
                    if !Self::sub_ok(po, pn, side) {
                        return None;
                    }
                }
                plan.push((fi, x, y, r));
            }
            let m = self.add_vertex(p);
            for (fi, x, y, r) in plan {
                self.set_face(fi, [x, m, r]);
                self.push_face([m, y, r]);
            }
            Some(m)
        }

        fn try_split_face(&mut self, fi: u32, p: [f64; 3]) -> bool {
            if self.near_vertex(p) {
                return false;
            }
            let p = p.map(|x| x as f32 as f64);
            let [a, b, c] = self.tris[fi as usize];
            let po = [a, b, c].map(|i| self.pos(i));
            for pn in [[po[0], po[1], p], [po[1], po[2], p], [po[2], po[0], p]] {
                if !Self::sub_ok(po, pn, None) {
                    return false;
                }
            }
            let m = self.add_vertex(p);
            self.set_face(fi, [a, b, m]);
            self.push_face([b, c, m]);
            self.push_face([c, a, m]);
            true
        }

        fn insert_corner(&mut self, c: [f64; 3], r_cut: f64) -> bool {
            if self.near_vertex(c) {
                return false;
            }
            let mut best: Option<(f64, u32, [f64; 3])> = None;
            // fcell ≥ 2·r_cut ⟹ 包围盒 ± r_cut 含 c 的面都登记在 c 所在格的 3×3×3 邻域内
            for fi in self.faces_near(&[c]) {
                let t = self.tris[fi as usize];
                let p = t.map(|i| g64(self.verts[i as usize]));
                let lo = [0, 1, 2].map(|k| p[0][k].min(p[1][k]).min(p[2][k]) - r_cut);
                let hi = [0, 1, 2].map(|k| p[0][k].max(p[1][k]).max(p[2][k]) + r_cut);
                if (0..3).any(|k| c[k] < lo[k] || c[k] > hi[k]) {
                    continue;
                }
                let (q, bc) = closest_on_triangle(c, p);
                let d = len(sub(q, c));
                if best.is_none_or(|(bd, _, _)| d < bd) {
                    best = Some((d, fi as u32, bc));
                }
            }
            let Some((d, fi, bc)) = best else { return false };
            if d > r_cut {
                return false;
            }
            let t = self.tris[fi as usize];
            let small: Vec<usize> = (0..3).filter(|&k| bc[k] < 0.02).collect();
            match small.as_slice() {
                [k] => self.try_split_edge(t[(k + 1) % 3], t[(k + 2) % 3], c, None).is_some(),
                [] => self.try_split_face(fi, c),
                _ => false,
            }
        }

        /// 沿一段源折痕切边,返回分裂次数。
        fn cut_along(&mut self, [a0, b0]: [[f64; 3]; 2], normals: [[f64; 3]; 2], r_cut: f64) -> usize {
            let g = sub(normals[0], normals[1]);
            if len(g) < 1e-6 {
                return 0;
            }
            let ab = sub(b0, a0);
            let ab2 = dot3(ab, ab);
            if ab2 <= 0.0 {
                return 0;
            }
            let seg = |p: [f64; 3]| -> ([f64; 3], f64) {
                let t = dot3(sub(p, a0), ab) / ab2;
                (add(a0, scale(ab, t.clamp(0.0, 1.0))), t)
            };
            let lo = [0, 1, 2].map(|k| a0[k].min(b0[k]) - 2.0 * r_cut);
            let hi = [0, 1, 2].map(|k| a0[k].max(b0[k]) + 2.0 * r_cut);
            let inside = |p: [f64; 3]| (0..3).all(|k| p[k] >= lo[k] && p[k] <= hi[k]);
            let mut rejected: std::collections::HashSet<u64, FxBuild> = Default::default();
            // 沿线段每隔半格取点做邻域查询:fcell ≥ 4·r_cut ⟹ 覆盖到线段 2·r_cut 内的全部面。
            let steps = (len(ab) / (0.5 * self.fcell)).ceil().max(1.0) as usize;
            let probes: Vec<[f64; 3]> = (0..=steps).map(|i| add(a0, scale(ab, i as f64 / steps as f64))).collect();
            let mut n = 0usize;
            'scan: for _ in 0..10_000 {
                for fi in self.faces_near(&probes) {
                    let fi = fi as usize;
                    let t = self.tris[fi];
                    if !t.iter().any(|&v| inside(self.pos(v))) {
                        continue;
                    }
                    for k in 0..3 {
                        let (a, b) = (t[k], t[(k + 1) % 3]);
                        let key = spur_edge_key(a, b);
                        if rejected.contains(&key) {
                            continue;
                        }
                        let (pa, pb) = (self.pos(a), self.pos(b));
                        let (sa, sb) = (dot3(sub(pa, a0), g), dot3(sub(pb, a0), g));
                        if sa * sb >= 0.0 {
                            continue;
                        }
                        if len(sub(seg(pa).0, pa)) < self.on_tol || len(sub(seg(pb).0, pb)) < self.on_tol {
                            continue;
                        }
                        let x = add(pa, scale(sub(pb, pa), sa / (sa - sb)));
                        let (q, tr) = seg(x);
                        if len(sub(x, q)) > r_cut || !(-0.02..=1.02).contains(&tr) {
                            continue;
                        }
                        if self.try_split_edge(a, b, q, Some(normals)).is_some() {
                            n += 1;
                            continue 'scan;
                        }
                        rejected.insert(key);
                    }
                }
                break;
            }
            n
        }
    }

    /// 点到三角形的最近点及其重心坐标(Ericson《Real-Time Collision Detection》5.1.5)。
    fn closest_on_triangle(p: [f64; 3], t: [[f64; 3]; 3]) -> ([f64; 3], [f64; 3]) {
        let [a, b, c] = t;
        let (ab, ac, ap) = (sub(b, a), sub(c, a), sub(p, a));
        let (d1, d2) = (dot3(ab, ap), dot3(ac, ap));
        if d1 <= 0.0 && d2 <= 0.0 {
            return (a, [1.0, 0.0, 0.0]);
        }
        let bp = sub(p, b);
        let (d3, d4) = (dot3(ab, bp), dot3(ac, bp));
        if d3 >= 0.0 && d4 <= d3 {
            return (b, [0.0, 1.0, 0.0]);
        }
        let vc = d1 * d4 - d3 * d2;
        if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
            let v = d1 / (d1 - d3);
            return (add(a, scale(ab, v)), [1.0 - v, v, 0.0]);
        }
        let cp = sub(p, c);
        let (d5, d6) = (dot3(ab, cp), dot3(ac, cp));
        if d6 >= 0.0 && d5 <= d6 {
            return (c, [0.0, 0.0, 1.0]);
        }
        let vb = d5 * d2 - d1 * d6;
        if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
            let w = d2 / (d2 - d6);
            return (add(a, scale(ac, w)), [1.0 - w, 0.0, w]);
        }
        let va = d3 * d6 - d5 * d4;
        if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
            let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            return (add(b, scale(sub(c, b), w)), [0.0, 1.0 - w, w]);
        }
        let den = 1.0 / (va + vb + vc);
        let (v, w) = (vb * den, vc * den);
        (add(add(a, scale(ab, v)), scale(ac, w)), [1.0 - v - w, v, w])
    }

    /// P33:一条共享边的视觉代价 max(0, 输出二面角 − 源二面角 − 15°),只计输出二面角 > 20° 的边。
    /// 源二面角取两侧各一个贴边采样点(边中点向面质心走 30%)处最近源面的法线夹角;退化面计 30。
    fn visual_excess(reference: &meshfield::MeshField, p1: [[f64; 3]; 3], p2: [[f64; 3]; 3], pa: [f64; 3], pb: [f64; 3]) -> f64 {
        let (n1, n2) = (spur_normal(p1), spur_normal(p2));
        let (l1, l2) = (len(n1), len(n2));
        if l1 < 1e-14 || l2 < 1e-14 {
            return 30.0;
        }
        let dev = (dot3(n1, n2) / (l1 * l2)).clamp(-1.0, 1.0).acos().to_degrees();
        if dev <= 20.0 {
            return 0.0;
        }
        let mid = scale(add(pa, pb), 0.5);
        let probe = |p: [[f64; 3]; 3]| {
            let c = scale(add(add(p[0], p[1]), p[2]), 1.0 / 3.0);
            reference.face_normal(closest_face_near(reference, add(mid, scale(sub(c, mid), 0.3))))
        };
        let sdev = dot3(probe(p1), probe(p2)).clamp(-1.0, 1.0).acos().to_degrees();
        (dev - sdev - 15.0).max(0.0)
    }

    /// P33:面法线与质心处最近源面法线的夹角(度);退化面 180。
    fn face_source_error(reference: &meshfield::MeshField, p: [[f64; 3]; 3]) -> f64 {
        let n = spur_normal(p);
        let l = len(n);
        if l < 1e-14 {
            return 180.0;
        }
        let c = scale(add(add(p[0], p[1]), p[2]), 1.0 / 3.0);
        let t = closest_face_near(reference, c);
        (dot3(n, reference.face_normal(t)) / l).clamp(-1.0, 1.0).acos().to_degrees()
    }

    /// P33:面上 7 个采样点(3 顶点、质心、3 边中点)到源面的最大距离。调用方只比较是否 > 0.02mm,
    /// 故任一点在 0.0201mm 内找不到源面即返回 ∞ 并提前结束(P36)。
    fn ref_face_max_distance(reference: &meshfield::MeshField, p: [[f64; 3]; 3]) -> f64 {
        let c = scale(add(add(p[0], p[1]), p[2]), 1.0 / 3.0);
        let mut dmax: f64 = 0.0;
        for s in [p[0], p[1], p[2], c, scale(add(p[0], p[1]), 0.5), scale(add(p[1], p[2]), 0.5), scale(add(p[2], p[0]), 0.5)] {
            match closest_within(reference, s, 0.0201) {
                Some((d, _)) => dmax = dmax.max(d),
                None => return f64::INFINITY,
            }
        }
        dmax
    }

    /// P40:按较小顶点号分行的无向边表(CSR)。面 f 的第 k 条边 (t[k], t[k+1]) 记为 (较大顶点号, f·3+k),
    /// 放在较小顶点号那一行;行内按较大顶点号**稳定**排序 ⟹ 同一条边的各次出现保持面序、边序。
    /// 内存 8 字节 / 半边(旧 HashMap 约 16 字节 + 控制字节 / 槽,且容量按 n·3 预分配)。
    struct EdgeRows {
        start: Vec<u32>,
        ent: Vec<(u32, u32)>,
    }

    impl EdgeRows {
        /// 无向边 (a, b) 的全部出现(按面序、边序)。
        fn run(&self, a: u32, b: u32) -> &[(u32, u32)] {
            let (lo, hi) = (a.min(b) as usize, a.max(b));
            let row = &self.ent[self.start[lo] as usize..self.start[lo + 1] as usize];
            let s = row.partition_point(|x| x.0 < hi);
            let e = s + row[s..].partition_point(|x| x.0 <= hi);
            &row[s..e]
        }
    }

    fn edge_rows(tris: &[[u32; 3]], nv: usize) -> EdgeRows {
        let mut start = vec![0u32; nv + 1];
        for t in tris {
            for k in 0..3 {
                start[t[k].min(t[(k + 1) % 3]) as usize + 1] += 1;
            }
        }
        for i in 0..nv {
            start[i + 1] += start[i];
        }
        let mut cur: Vec<u32> = start[..nv].to_vec();
        let mut ent = vec![(0u32, 0u32); tris.len() * 3];
        for (f, t) in tris.iter().enumerate() {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                let c = &mut cur[a.min(b) as usize];
                ent[*c as usize] = (a.max(b), (f * 3 + k) as u32);
                *c += 1;
            }
        }
        drop(cur);
        // 行内稳定排序:按条目数均分到各线程(行边界切分),互不重叠
        let nt = nthreads().max(1);
        let target = ent.len().div_ceil(nt).max(1);
        let mut cuts: Vec<usize> = (0..=nt)
            .map(|w| if w == nt { nv } else { start.partition_point(|&s| (s as usize) < w * target).min(nv) })
            .collect();
        for i in 1..cuts.len() {
            cuts[i] = cuts[i].max(cuts[i - 1]);
        }
        {
            let start = &start;
            std::thread::scope(|sc| {
                let mut rest: &mut [(u32, u32)] = &mut ent;
                let mut off = 0usize;
                for w in 0..nt {
                    let (r0, r1) = (cuts[w], cuts[w + 1]);
                    let e1 = start[r1] as usize;
                    let (mine, tail) = std::mem::take(&mut rest).split_at_mut(e1 - off);
                    rest = tail;
                    let base = off;
                    off = e1;
                    sc.spawn(move || {
                        for r in r0..r1 {
                            let (a, b) = (start[r] as usize - base, start[r + 1] as usize - base);
                            mine[a..b].sort_by_key(|x| x.0);
                        }
                    });
                }
            });
        }
        EdgeRows { start, ent }
    }

    /// 一个面由 `po` 变成 `pn` 是否可接受:不退化;不成为薄片(除非旧面更差);
    /// 旧面非薄片时转角 ≤ 60°。
    fn spur_replace_ok(po: [[f64; 3]; 3], pn: [[f64; 3]; 3], min_quality: f64) -> bool {
        let (no, nn) = (spur_normal(po), spur_normal(pn));
        let (lo, ln) = (len(no), len(nn));
        if ln < 1e-12 {
            return false;
        }
        let (qo, qn) = (spur_quality(po), spur_quality(pn));
        let (ho, hn) = (spur_height(po), spur_height(pn));
        if (qn < min_quality && qn < qo) || (hn < SPUR_SLIVER_H && hn < ho) {
            return false;
        }
        !(qo > 0.02 && dot3(no, nn) < 0.5 * lo * ln)
    }

    impl Mesh {
        /// STL encodes coordinates without vertex indices. Two used vertex IDs at
        /// the same f32 position will falsely meet when the exported STL is read.
        pub fn has_coincident_export_vertices(&self) -> bool {
            let mut visited = vec![false; self.verts.len()];
            let mut coordinates = std::collections::HashSet::with_capacity(self.verts.len());
            for &id in self.tris.iter().flatten() {
                let id = id as usize;
                if visited[id] { continue; }
                visited[id] = true;
                let key = self.verts[id].map(|x| if x == 0.0 { 0 } else { x.to_bits() });
                if !coordinates.insert(key) { return true; }
            }
            false
        }

        fn crease_targets(&self, grid: &ReferenceCreases, radius: Real) -> Vec<Option<[Real;3]>> {
            let nearby: Vec<_> = self.verts.iter().map(|&p| grid.contains_nearby_segment(g64(p))).collect();
            let mut fans = vec![Vec::new(); self.verts.len()];
            for &t in &self.tris {
                if !t.iter().any(|&i| nearby[i as usize]) { continue; }
                let [a,b,c] = t.map(|i| g64(self.verts[i as usize]));
                let n = cross3(sub(b,a),sub(c,a));
                let length = len(n);
                if length > 0.0 {
                    let normal = scale(n, 1. / length);
                    for i in t {
                        if nearby[i as usize] { fans[i as usize].push(normal); }
                    }
                }
            }
            self.verts.iter().zip(fans).map(|(&p, fan)| {
                if fan.is_empty() { None } else { grid.target(g64(p), &fan, radius) }
            }).collect()
        }

        /// **折痕毛刺清理(P26/P27)**:折痕约束路径收尾后,原始折痕线上相邻的两个顶点
        /// 之间常夹着一个"半吸附"顶点 —— `fit_reference` 把它往折痕拉时,验收发现会压扁
        /// 邻面,步长减半后停在离折痕约 0.03mm 处。它和折痕上的邻点组成 ~0.09mm 长、
        /// 二面角 ~120° 的短边,查看器把它画成主棱线旁的小毛刺;连续几个就是"阶梯";
        /// 折痕边本身缺失(被一条斜跨的边代替)时则显示为棱线旁一段平行短线。
        ///
        /// 每轮三步(共 `rounds` 轮),都只动锐边(二面角 > `sharp_deg`)顶点及其 1 环:
        /// 1. **翻边恢复折痕边**:四边形两个对顶点都在源折痕上、且其连线沿折痕时,把斜跨的
        ///    对角线翻成折痕边(新面须与该处源折痕两侧法线之一同向 ≥ 0.9、不成薄片);
        /// 2. **塌缩**:离折痕 `on_tol..off_max` 的顶点沿边并到已在折痕上、边长 ≤ `max_len`
        ///    的邻点(共享面恰 2 + 链接条件、存留面不退化/不翻转/不成薄片);
        /// 3. **吸附**:仍剩下的顶点直接移到折痕上的最近点(同样的形状与翻转守卫,且新面
        ///    与源法线的一致度不得变差)。
        ///
        /// 薄片 = 归一化质量 2A/e²max < `min_quality` 或最小高度 < 0.01mm;新面只有在旧面本来
        /// 更差时才允许低于阈值。返回 (翻边数, 塌缩数, 吸附数)。
        ///
        /// Python 原型实测(叶轮 h=0.1 mult=1,>30° 锐边):毛刺 1,563 → 202、阶梯 497 → 72、
        /// 非流形 22 → 11、薄片 123 → 117、真缺陷折叠 16 → 14(P26 只塌缩时为 410 / 149)。
        pub fn merge_crease_spurs(
            &mut self,
            grid: &ReferenceCreases,
            sharp_deg: f64,
            on_tol: Real,
            off_max: Real,
            max_len: Real,
            min_quality: f64,
            rounds: usize,
        ) -> (usize, usize, usize) {
            self.merge_crease_spurs_locked(grid, sharp_deg, on_tol, off_max, max_len, min_quality, rounds, None, true)
        }

        /// P38:`locked` 中的顶点不参与翻边(四个顶点任一被锁即跳过)、塌缩(两端任一)与吸附;
        /// `compact` 为假时塌缩只删死面、顶点号不变(分段模式)。
        #[allow(clippy::too_many_arguments)]
        fn merge_crease_spurs_locked(
            &mut self,
            grid: &ReferenceCreases,
            sharp_deg: f64,
            on_tol: Real,
            off_max: Real,
            max_len: Real,
            min_quality: f64,
            rounds: usize,
            locked: Option<&[bool]>,
            compact: bool,
        ) -> (usize, usize, usize) {
            let (mut nf, mut nc, mut nm) = (0, 0, 0);
            for _ in 0..rounds {
                let f = self.flip_onto_creases(grid, sharp_deg, on_tol, min_quality, locked);
                let c = self.collapse_onto_creases(grid, sharp_deg, on_tol, off_max, max_len, min_quality, locked, compact);
                let m = self.move_onto_creases(grid, sharp_deg, on_tol, off_max, min_quality, locked);
                nf += f;
                nc += c;
                nm += m;
                if f + c + m == 0 {
                    break;
                }
            }
            (nf, nc, nm)
        }

        /// 锐边(二面角 > `sharp_deg`)两端顶点及其 1 环。
        fn sharp_band(&self, sharp_deg: f64) -> Vec<bool> {
            let nv = self.verts.len();
            let mut emap: std::collections::HashMap<u64, [u32; 3], FxBuild> =
                std::collections::HashMap::with_capacity_and_hasher(self.tris.len() * 3, FxBuild);
            for (ti, t) in self.tris.iter().enumerate() {
                for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                    let e = emap.entry(spur_edge_key(a, b)).or_insert([u32::MAX, u32::MAX, 0]);
                    if e[0] == u32::MAX {
                        e[0] = ti as u32;
                    } else if e[1] == u32::MAX {
                        e[1] = ti as u32;
                    }
                    e[2] += 1;
                }
            }
            let cos_sharp = sharp_deg.to_radians().cos();
            let mut core = vec![false; nv];
            for (key, e) in emap.iter() {
                if e[2] != 2 {
                    continue;
                }
                let n0 = spur_normal(self.tris[e[0] as usize].map(|i| g64(self.verts[i as usize])));
                let n1 = spur_normal(self.tris[e[1] as usize].map(|i| g64(self.verts[i as usize])));
                let (l0, l1) = (len(n0), len(n1));
                if l0 > 0.0 && l1 > 0.0 && dot3(n0, n1) < cos_sharp * l0 * l1 {
                    core[(key >> 32) as usize] = true;
                    core[(key & 0xffff_ffff) as usize] = true;
                }
            }
            let mut band = core.clone();
            for t in &self.tris {
                if t.iter().any(|&i| core[i as usize]) {
                    for &i in t {
                        band[i as usize] = true;
                    }
                }
            }
            band
        }

        /// 第 1 步:翻边恢复缺失的折痕边。
        fn flip_onto_creases(&mut self, grid: &ReferenceCreases, sharp_deg: f64, on_tol: Real, min_quality: f64, locked: Option<&[bool]>) -> usize {
            let band = self.sharp_band(sharp_deg);
            let on: Vec<bool> = (0..self.verts.len())
                .map(|v| band[v] && grid.distance(g64(self.verts[v])).is_some_and(|d| d < on_tol))
                .collect();
            let mut emap: std::collections::HashMap<u64, [u32; 3], FxBuild> =
                std::collections::HashMap::with_capacity_and_hasher(self.tris.len() * 3, FxBuild);
            for (ti, t) in self.tris.iter().enumerate() {
                for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                    let e = emap.entry(spur_edge_key(a, b)).or_insert([u32::MAX, u32::MAX, 0]);
                    if e[0] == u32::MAX {
                        e[0] = ti as u32;
                    } else if e[1] == u32::MAX {
                        e[1] = ti as u32;
                    }
                    e[2] += 1;
                }
            }
            let mut keys: Vec<u64> = emap
                .iter()
                .filter(|(k, e)| e[2] == 2 && (band[(*k >> 32) as usize] || band[(*k & 0xffff_ffff) as usize]))
                .map(|(k, _)| *k)
                .collect();
            keys.sort_unstable();
            let on_crease = |p: [Real; 3]| grid.distance(p).is_some_and(|d| d < on_tol);
            let mut used = vec![false; self.tris.len()];
            let mut n = 0usize;
            for key in keys {
                let Some(e) = emap.get(&key).copied() else { continue };
                if e[2] != 2 {
                    continue;
                }
                let (f1, f2) = (e[0], e[1]);
                if used[f1 as usize] || used[f2 as usize] {
                    continue;
                }
                let (mut w, mut x) = ((key >> 32) as u32, (key & 0xffff_ffff) as u32);
                // f1 须含有向边 w→x(两面绕向一致时 f2 含 x→w)
                let t1o = self.tris[f1 as usize];
                let i = t1o.iter().position(|&v| v == w).unwrap();
                if t1o[(i + 1) % 3] != x {
                    std::mem::swap(&mut w, &mut x);
                }
                let opp = |t: [u32; 3]| *t.iter().find(|&&v| v != w && v != x).unwrap();
                let (a, b) = (opp(self.tris[f1 as usize]), opp(self.tris[f2 as usize]));
                if a == b || !on[a as usize] || !on[b as usize] || emap.contains_key(&spur_edge_key(a, b)) {
                    continue;
                }
                if locked.is_some_and(|l| [w, x, a, b].iter().any(|&v| l[v as usize])) {
                    continue;
                }
                let (pw, px, pa, pb) = (
                    g64(self.verts[w as usize]),
                    g64(self.verts[x as usize]),
                    g64(self.verts[a as usize]),
                    g64(self.verts[b as usize]),
                );
                if on_crease(scale(add(pw, px), 0.5)) {
                    continue; // w–x 已是折痕边
                }
                if ![0.25, 0.5, 0.75].iter().all(|&t| on_crease(add(pa, scale(sub(pb, pa), t)))) {
                    continue; // a–b 不沿折痕
                }
                let (n1, n2) = ([a, w, b], [b, x, a]);
                let ok = [[pa, pw, pb], [pb, px, pa]].into_iter().all(|p| {
                    spur_quality(p) >= min_quality && spur_height(p) >= SPUR_SLIVER_H && spur_agree(grid, p) >= 0.9
                });
                if !ok {
                    continue;
                }
                self.tris[f1 as usize] = n1;
                self.tris[f2 as usize] = n2;
                used[f1 as usize] = true;
                used[f2 as usize] = true;
                emap.remove(&key);
                emap.insert(spur_edge_key(a, b), [f1, f2, 2]);
                n += 1;
            }
            n
        }

        /// 第 2 步:把折痕旁顶点沿边塌缩到折痕上的邻点。
        fn collapse_onto_creases(
            &mut self,
            grid: &ReferenceCreases,
            sharp_deg: f64,
            on_tol: Real,
            off_max: Real,
            max_len: Real,
            min_quality: f64,
            locked: Option<&[bool]>,
            compact: bool,
        ) -> usize {
            let nv = self.verts.len();
            let nt = self.tris.len();
            if nv == 0 || nt == 0 {
                return 0;
            }
            let band = self.sharp_band(sharp_deg);
            let dist: Vec<Real> = (0..nv)
                .map(|v| if band[v] { grid.distance(g64(self.verts[v])).unwrap_or(Real::INFINITY) } else { Real::INFINITY })
                .collect();
            let mut order: Vec<u32> = (0..nv as u32)
                .filter(|&v| dist[v as usize] >= on_tol && dist[v as usize] < off_max)
                .filter(|&v| !locked.is_some_and(|l| l[v as usize]))
                .collect();
            order.sort_by(|a, b| dist[*a as usize].total_cmp(&dist[*b as usize]).then(a.cmp(b)));
            let mut v2t: Vec<Vec<u32>> = vec![Vec::new(); nv];
            for (ti, t) in self.tris.iter().enumerate() {
                for &v in t {
                    v2t[v as usize].push(ti as u32);
                }
            }
            let mut dead = vec![false; nt];
            let nbrs_of = |v2t: &[Vec<u32>], tris: &[[u32; 3]], v: u32| -> Vec<u32> {
                let mut s: Vec<u32> = Vec::with_capacity(8);
                for &ti in &v2t[v as usize] {
                    for &x in &tris[ti as usize] {
                        if x != v && !s.contains(&x) {
                            s.push(x);
                        }
                    }
                }
                s
            };
            let mut n_collapsed = 0usize;
            for &w in &order {
                if v2t[w as usize].is_empty() {
                    continue;
                }
                let pw = g64(self.verts[w as usize]);
                let nw = nbrs_of(&v2t, &self.tris, w);
                let mut ons: Vec<(Real, u32)> = nw
                    .iter()
                    .filter(|&&u| dist[u as usize] < on_tol && !locked.is_some_and(|l| l[u as usize]))
                    .map(|&u| (len(sub(g64(self.verts[u as usize]), pw)), u))
                    .filter(|&(l, _)| l <= max_len)
                    .collect();
                ons.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                'cand: for &(_, u) in &ons {
                    let shared: Vec<u32> = v2t[w as usize]
                        .iter()
                        .copied()
                        .filter(|&ti| self.tris[ti as usize].contains(&u))
                        .collect();
                    if shared.len() != 2 {
                        continue;
                    }
                    let opp = |ti: u32| -> u32 { *self.tris[ti as usize].iter().find(|&&x| x != u && x != w).unwrap() };
                    let want = [opp(shared[0]), opp(shared[1])];
                    let nu = nbrs_of(&v2t, &self.tris, u);
                    let common: Vec<u32> = nw.iter().copied().filter(|x| nu.contains(x)).collect();
                    if common.len() != 2 || !common.contains(&want[0]) || !common.contains(&want[1]) {
                        continue;
                    }
                    let pu = g64(self.verts[u as usize]);
                    for &ti in &v2t[w as usize] {
                        if shared.contains(&ti) {
                            continue;
                        }
                        let t = self.tris[ti as usize];
                        let po = t.map(|i| g64(self.verts[i as usize]));
                        let pn = t.map(|i| if i == w { pu } else { g64(self.verts[i as usize]) });
                        if !spur_replace_ok(po, pn, min_quality) {
                            continue 'cand;
                        }
                    }
                    for &ti in &shared {
                        dead[ti as usize] = true;
                        for x in self.tris[ti as usize] {
                            v2t[x as usize].retain(|&y| y != ti);
                        }
                    }
                    let moved = std::mem::take(&mut v2t[w as usize]);
                    for ti in moved {
                        for x in self.tris[ti as usize].iter_mut() {
                            if *x == w {
                                *x = u;
                            }
                        }
                        v2t[u as usize].push(ti);
                    }
                    n_collapsed += 1;
                    break;
                }
            }
            if n_collapsed > 0 && !compact {
                let mut k2 = 0usize;
                self.tris.retain(|_| {
                    let keep = !dead[k2];
                    k2 += 1;
                    keep
                });
            } else if n_collapsed > 0 {
                let mut remap = vec![u32::MAX; nv];
                let mut verts = Vec::with_capacity(nv);
                let mut tris = Vec::with_capacity(nt);
                for (ti, t) in self.tris.iter().enumerate() {
                    if dead[ti] {
                        continue;
                    }
                    tris.push(t.map(|i| {
                        if remap[i as usize] == u32::MAX {
                            remap[i as usize] = verts.len() as u32;
                            verts.push(self.verts[i as usize]);
                        }
                        remap[i as usize]
                    }));
                }
                self.verts = verts;
                self.tris = tris;
            }
            n_collapsed
        }

        /// 第 3 步:剩下的折痕旁顶点直接移到折痕最近点。
        fn move_onto_creases(
            &mut self,
            grid: &ReferenceCreases,
            sharp_deg: f64,
            on_tol: Real,
            off_max: Real,
            min_quality: f64,
            locked: Option<&[bool]>,
        ) -> usize {
            let nv = self.verts.len();
            let band = self.sharp_band(sharp_deg);
            let mut v2t: Vec<Vec<u32>> = vec![Vec::new(); nv];
            for (ti, t) in self.tris.iter().enumerate() {
                for &v in t {
                    v2t[v as usize].push(ti as u32);
                }
            }
            let mut n = 0usize;
            for w in 0..nv {
                if !band[w] || v2t[w].is_empty() || locked.is_some_and(|l| l[w]) {
                    continue;
                }
                let Some((d, q, _)) = grid.nearest(g64(self.verts[w])) else { continue };
                if !(d >= on_tol && d < off_max) {
                    continue;
                }
                let qf = q.map(|v| v as f32 as f64);
                let ok = v2t[w].iter().all(|&ti| {
                    let t = self.tris[ti as usize];
                    let po = t.map(|i| g64(self.verts[i as usize]));
                    let pn = t.map(|i| if i as usize == w { qf } else { g64(self.verts[i as usize]) });
                    if !spur_replace_ok(po, pn, min_quality) {
                        return false;
                    }
                    let ag = spur_agree(grid, pn);
                    ag >= 0.9 || ag >= spur_agree(grid, po)
                });
                if ok {
                    self.verts[w] = qf.map(|v| v as f32);
                    n += 1;
                }
            }
            n
        }

        /// **按源网格复核修补(P28)**:折痕约束路径的最后一步。前面各步都只看"顶点在不在
        /// 源面/折痕上",而放大后看到的残余瑕疵是**面**的问题:三个顶点都在源面上,面却斜跨
        /// 折痕(法线介于两侧之间)、退化成折痕上的一条线、或成为偏离源面的小鳍片。
        ///
        /// 判据(`ref_face_cost`):面上 7 个采样点(3 顶点、质心、3 边中点)到源面的最大距离
        /// > 0.02mm(QEM err2=1e-4 的弦高本身约 0.01mm,不算),或质心处法线与源面/源折痕两侧
        /// 法线的最大一致度 < 0.8(偏 > 37°),或退化。只扫锐边带(`sharp_band`)上的面。
        ///
        /// 每轮:①对缺陷面逐边试**翻边**(新两面代价更低且不同时为缺陷);②对仍是缺陷的面逐边
        /// 试**塌缩**(两方向,链接条件、不翻转、不新造薄片,且邻域缺陷数不增、代价严格下降)。
        /// 返回 (修前缺陷面数, 翻边数, 塌缩数, 修后缺陷面数)。
        ///
        /// Python 原型实测(叶轮 h=0.1 mult=1,在当前 Rust 输出上):可见缺陷面 129 → 7,
        /// 真缺陷折叠 15 → 0,阶梯 85 → 20,薄片 82 → 72,非流形 22 → 13,输出→原始 max ≤ 0.01mm。
        pub fn repair_reference_defects(
            &mut self,
            reference: &meshfield::MeshField,
            grid: &ReferenceCreases,
            sharp_deg: f64,
            rounds: usize,
        ) -> (usize, usize, usize, usize) {
            let before = self.reference_bad_faces(reference, grid, sharp_deg).len();
            let (mut nf, mut nc) = (0usize, 0usize);
            let mut after = before;
            for _ in 0..rounds {
                if after == 0 {
                    break;
                }
                let bad = self.reference_bad_faces(reference, grid, sharp_deg);
                let f = self.flip_reference_defects(reference, grid, &bad);
                let bad = self.reference_bad_faces(reference, grid, sharp_deg);
                let c = self.collapse_reference_defects(reference, grid, &bad);
                nf += f;
                nc += c;
                after = self.reference_bad_faces(reference, grid, sharp_deg).len();
                if f + c == 0 {
                    break;
                }
            }
            (before, nf, nc, after)
        }

        /// 锐边带上的缺陷面(并行扫描,结果按面号升序,与线程数无关)。
        fn reference_bad_faces(&self, reference: &meshfield::MeshField, grid: &ReferenceCreases, sharp_deg: f64) -> Vec<u32> {
            let band = self.sharp_band(sharp_deg);
            let cand: Vec<u32> = (0..self.tris.len() as u32)
                .filter(|&t| self.tris[t as usize].iter().any(|&v| band[v as usize]))
                .collect();
            if cand.is_empty() {
                return Vec::new();
            }
            let chunk = cand.len().div_ceil(nthreads()).max(1);
            let parts: Vec<Vec<u32>> = std::thread::scope(|scope| {
                let handles: Vec<_> = cand
                    .chunks(chunk)
                    .map(|part| {
                        scope.spawn(move || {
                            part.iter()
                                .copied()
                                .filter(|&t| {
                                    let p = self.tris[t as usize].map(|i| g64(self.verts[i as usize]));
                                    ref_face_cost(reference, grid, p).0
                                })
                                .collect::<Vec<u32>>()
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().expect("缺陷扫描线程 panic")).collect()
            });
            parts.concat()
        }

        /// 第 ① 步:缺陷面逐边试翻边。
        fn flip_reference_defects(&mut self, reference: &meshfield::MeshField, grid: &ReferenceCreases, bad: &[u32]) -> usize {
            let nt = self.tris.len();
            let mut emap: std::collections::HashMap<u64, [u32; 3], FxBuild> =
                std::collections::HashMap::with_capacity_and_hasher(nt * 3, FxBuild);
            for (ti, t) in self.tris.iter().enumerate() {
                for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                    let e = emap.entry(spur_edge_key(a, b)).or_insert([u32::MAX, u32::MAX, 0]);
                    if e[0] == u32::MAX {
                        e[0] = ti as u32;
                    } else if e[1] == u32::MAX {
                        e[1] = ti as u32;
                    }
                    e[2] += 1;
                }
            }
            let mut used = vec![false; nt];
            let mut n = 0usize;
            for &fi in bad {
                if used[fi as usize] {
                    continue;
                }
                let t = self.tris[fi as usize];
                let (bad0, c0) = ref_face_cost(reference, grid, t.map(|i| g64(self.verts[i as usize])));
                if !bad0 {
                    continue;
                }
                for k in 0..3 {
                    let (w, x, a) = (t[k], t[(k + 1) % 3], t[(k + 2) % 3]);
                    let Some(e) = emap.get(&spur_edge_key(w, x)).copied() else { continue };
                    if e[2] != 2 {
                        continue;
                    }
                    let f2 = if e[0] == fi { e[1] } else { e[0] };
                    if f2 == fi || used[f2 as usize] {
                        continue;
                    }
                    let Some(&b) = self.tris[f2 as usize].iter().find(|&&v| v != w && v != x) else { continue };
                    if a == b || emap.contains_key(&spur_edge_key(a, b)) {
                        continue;
                    }
                    let (t1, t2) = ([a, w, b], [b, x, a]);
                    let p1 = t1.map(|i| g64(self.verts[i as usize]));
                    let p2 = t2.map(|i| g64(self.verts[i as usize]));
                    if spur_quality(p1).min(spur_quality(p2)) < 1e-3 {
                        continue;
                    }
                    let old = c0 + ref_face_cost(reference, grid, self.tris[f2 as usize].map(|i| g64(self.verts[i as usize]))).1;
                    let (b1, c1) = ref_face_cost(reference, grid, p1);
                    let (b2, c2) = ref_face_cost(reference, grid, p2);
                    if c1 + c2 < old - 1e-6 && !(b1 && b2) {
                        self.tris[fi as usize] = t1;
                        self.tris[f2 as usize] = t2;
                        used[fi as usize] = true;
                        used[f2 as usize] = true;
                        emap.remove(&spur_edge_key(w, x));
                        emap.insert(spur_edge_key(a, b), [fi, f2, 2]);
                        n += 1;
                        break;
                    }
                }
            }
            n
        }

        /// 第 ② 步:缺陷面逐边试塌缩(短边优先,两方向)。
        fn collapse_reference_defects(&mut self, reference: &meshfield::MeshField, grid: &ReferenceCreases, bad: &[u32]) -> usize {
            let nv = self.verts.len();
            let nt = self.tris.len();
            let mut v2t: Vec<Vec<u32>> = vec![Vec::new(); nv];
            for (ti, t) in self.tris.iter().enumerate() {
                for &v in t {
                    v2t[v as usize].push(ti as u32);
                }
            }
            let nbrs_of = |v2t: &[Vec<u32>], tris: &[[u32; 3]], v: u32| -> Vec<u32> {
                let mut s: Vec<u32> = Vec::with_capacity(8);
                for &ti in &v2t[v as usize] {
                    for &x in &tris[ti as usize] {
                        if x != v && !s.contains(&x) {
                            s.push(x);
                        }
                    }
                }
                s
            };
            let mut dead = vec![false; nt];
            let mut n = 0usize;
            for &fi in bad {
                if dead[fi as usize] {
                    continue;
                }
                let t = self.tris[fi as usize];
                if !ref_face_cost(reference, grid, t.map(|i| g64(self.verts[i as usize]))).0 {
                    continue;
                }
                let mut edges = [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])];
                let elen = |e: (u32, u32), verts: &[[f32; 3]]| len(sub(g64(verts[e.0 as usize]), g64(verts[e.1 as usize])));
                edges.sort_by(|p, q| elen(*p, &self.verts).total_cmp(&elen(*q, &self.verts)));
                'edge: for (a, b) in edges {
                    'dir: for (w, u) in [(a, b), (b, a)] {
                        let shared: Vec<u32> = v2t[w as usize]
                            .iter()
                            .copied()
                            .filter(|&ti| self.tris[ti as usize].contains(&u))
                            .collect();
                        if shared.len() != 2 {
                            continue 'dir;
                        }
                        let opp = |ti: u32| -> u32 { *self.tris[ti as usize].iter().find(|&&x| x != u && x != w).unwrap() };
                        let want = [opp(shared[0]), opp(shared[1])];
                        let nw = nbrs_of(&v2t, &self.tris, w);
                        let nu = nbrs_of(&v2t, &self.tris, u);
                        let common: Vec<u32> = nw.iter().copied().filter(|x| nu.contains(x)).collect();
                        if common.len() != 2 || !common.contains(&want[0]) || !common.contains(&want[1]) {
                            continue 'dir;
                        }
                        let mut aff: Vec<u32> = v2t[w as usize].clone();
                        for &ti in &v2t[u as usize] {
                            if !aff.contains(&ti) {
                                aff.push(ti);
                            }
                        }
                        let pu = g64(self.verts[u as usize]);
                        let (mut old_cost, mut old_bad, mut new_cost, mut new_bad) = (0.0f64, 0usize, 0.0f64, 0usize);
                        for &ti in &aff {
                            let tt = self.tris[ti as usize];
                            let po = tt.map(|i| g64(self.verts[i as usize]));
                            let (bo, co) = ref_face_cost(reference, grid, po);
                            old_cost += co;
                            old_bad += bo as usize;
                            if shared.contains(&ti) {
                                continue;
                            }
                            let pn = tt.map(|i| if i == w { pu } else { g64(self.verts[i as usize]) });
                            let (no, nn) = (spur_normal(po), spur_normal(pn));
                            let (lo, ln) = (len(no), len(nn));
                            if ln < 1e-12 {
                                continue 'dir;
                            }
                            if spur_quality(po) > 0.02 && dot3(no, nn) < 0.5 * lo * ln {
                                continue 'dir;
                            }
                            let (ho, hn) = (spur_height(po), spur_height(pn));
                            if hn < SPUR_SLIVER_H && hn < ho && ho >= SPUR_SLIVER_H {
                                continue 'dir;
                            }
                            let (bn, cn) = ref_face_cost(reference, grid, pn);
                            new_cost += cn;
                            new_bad += bn as usize;
                        }
                        if new_bad > old_bad || new_cost >= old_cost - 1e-6 {
                            continue 'dir;
                        }
                        for &ti in &shared {
                            dead[ti as usize] = true;
                            for x in self.tris[ti as usize] {
                                v2t[x as usize].retain(|&y| y != ti);
                            }
                        }
                        let moved = std::mem::take(&mut v2t[w as usize]);
                        for ti in moved {
                            for x in self.tris[ti as usize].iter_mut() {
                                if *x == w {
                                    *x = u;
                                }
                            }
                            v2t[u as usize].push(ti);
                        }
                        n += 1;
                        break 'edge;
                    }
                }
            }
            if n > 0 {
                self.compact_dead(&dead);
            }
            n
        }

        /// **以源网格为准的二次简化(P29)**:折痕约束路径的收尾。QEM(`decimate_with_creases`)的代价是
        /// "到累计平面的距离平方之和",平面区每并入一个带 ~0.005mm 提取噪声的面,代价就涨一截,到
        /// err2 就停 —— 叶轮 48% 的三角落在源网格本身是平面的区域(边长 p50 0.58mm),桌腿 794 个源
        /// 三角的件输出 34.6 万三角。此时顶点已经投影到源面上,可以直接以源网格为准判误差:
        ///
        /// 半边塌缩 w→u,**顶点一律不移动**(u 已在源面上)。
        /// - 精确判据:u 落在 w 每个存留邻面的平面上(≤ 2e-4mm)⇒ 新面与旧面共面,1 环多边形重新扇形
        ///   三角化,覆盖区域不变,零附加误差,不采样;
        /// - 否则对新面按 ≤0.3mm 间距(每边 2–8 段)采样,到源面距离 ≤ `tol`、法线与源面(或折痕两侧)
        ///   一致度 ≥ 0.95;
        /// - 折痕顶点(离源折痕 < `on_tol`)只能沿折痕并到相邻折痕顶点(折痕度恰为 2,拐角锁定),
        ///   新折痕边的 1/4、1/2、3/4 点仍须在折痕上;
        /// - 守卫:共享面恰 2、链接条件、不翻转(转角 ≤ 60°)、不退化、质量 2A/e²max 不降到 0.01 以下;
        ///   非流形 / 边界边的端点锁定。短边优先(懒失效堆)。
        ///
        /// 返回塌缩次数。Python 原型实测(tol=0.01mm):桌腿 345,750 → 1,554 三角(17.3MB → 78KB,
        /// 闭合流形,双向 max 0.017mm);叶轮 366,130 → 33,668(18.3MB → 1.7MB,源 STL 7.7MB,
        /// 输出→源 p99 0.008 / max 0.010mm,真缺陷折叠 2,薄片 35)。
        pub fn decimate_to_reference(
            &mut self,
            reference: &meshfield::MeshField,
            grid: &ReferenceCreases,
            tol: Real,
            on_tol: Real,
        ) -> usize {
            self.decimate_to_reference_seeded(reference, grid, tol, on_tol, None, None, true)
        }

        /// P37:`decimate_to_reference` 的分段并行版。各段内跨段顶点自动锁定(段界边只有 1 个面,
        /// 本函数本就锁定非 2 面边的端点),且顶点从不移动;合并后只在接缝 2 环内再做一遍并压缩。
        /// 判据对源网格是绝对的(采样距离 ≤ tol),两遍不会累积误差。< 20 万三角或 `ADF_DECIMATE_SERIAL` 时串行。
        pub fn decimate_to_reference_parallel(
            &mut self,
            reference: &meshfield::MeshField,
            grid: &ReferenceCreases,
            tol: Real,
            on_tol: Real,
        ) -> usize {
            let k: usize = std::env::var("ADF_DECIMATE_BLOCKS").ok().and_then(|v| v.parse().ok()).unwrap_or(32);
            if k < 2 || self.tris.len() < 200_000 || std::env::var_os("ADF_DECIMATE_SERIAL").is_some() {
                return self.decimate_to_reference(reference, grid, tol, on_tol);
            }
            self.decimate_to_reference_blocks(reference, grid, tol, on_tol, k)
        }

        /// 分段并行源网格简化本体(无规模门槛,测试直接调用)。
        pub fn decimate_to_reference_blocks(
            &mut self,
            reference: &meshfield::MeshField,
            grid: &ReferenceCreases,
            tol: Real,
            on_tol: Real,
            k: usize,
        ) -> usize {
            let (n, cross) = self.run_in_slabs(k.max(1), |m, lock| {
                m.decimate_to_reference_seeded(reference, grid, tol, on_tol, None, Some(lock), false)
            });
            let near = self.ring_expand(&cross, 2);
            let lock: Vec<bool> = near.iter().map(|&b| !b).collect();
            n + self.decimate_to_reference_seeded(reference, grid, tol, on_tol, Some(near), Some(lock.as_slice()), true)
        }

        /// 与 `decimate_to_reference` 相同,但只从锐边带(二面角 > 30° 的边端点及其 1 环)上的边起步
        /// (塌缩后邻边照常入堆,改动可向外扩散)。P36:嵌入/毛刺清理只改折痕附近,之后的补充简化
        /// 没必要把全网格上一轮已被拒绝的边再试一遍(翅膀每轮 ~150 万次被拒尝试,每次都采样源面)。
        pub fn decimate_to_reference_near_creases(
            &mut self,
            reference: &meshfield::MeshField,
            grid: &ReferenceCreases,
            tol: Real,
            on_tol: Real,
        ) -> usize {
            let seed = self.sharp_band(30.0);
            self.decimate_to_reference_seeded(reference, grid, tol, on_tol, Some(seed), None, true)
        }

        fn decimate_to_reference_seeded(
            &mut self,
            reference: &meshfield::MeshField,
            grid: &ReferenceCreases,
            tol: Real,
            on_tol: Real,
            seed: Option<Vec<bool>>,
            extra_lock: Option<&[bool]>,
            compact: bool,
        ) -> usize {
            const PLANE_EPS: f64 = 2e-4;
            const AGREE: f64 = 0.95;
            const QMIN: f64 = 0.01;
            let nv = self.verts.len();
            let nt = self.tris.len();
            if nv == 0 || nt == 0 {
                return 0;
            }
            // 顶点坐标全程不变(只改连接关系)
            let pos: Vec<[f64; 3]> = self.verts.iter().map(|&v| g64(v)).collect();
            let on: Vec<bool> = pos.iter().map(|&p| grid.distance(p).is_some_and(|d| d < on_tol)).collect();
            let on_crease = |p: [f64; 3]| grid.distance(p).is_some_and(|d| d < on_tol);
            let along_crease = |a: u32, b: u32| -> bool {
                let (pa, pb) = (pos[a as usize], pos[b as usize]);
                [0.25, 0.5, 0.75].iter().all(|&t| on_crease(add(pa, scale(sub(pb, pa), t))))
            };
            let crease_edge = |a: u32, b: u32| on[a as usize] && on[b as usize] && along_crease(a, b);
            let elen = |a: u32, b: u32| len(sub(pos[a as usize], pos[b as usize]));
            let mut ecount: std::collections::HashMap<u64, u32, FxBuild> =
                std::collections::HashMap::with_capacity_and_hasher(nt * 3, FxBuild);
            for t in &self.tris {
                for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                    *ecount.entry(spur_edge_key(a, b)).or_insert(0) += 1;
                }
            }
            let mut locked = vec![false; nv];
            for (&k, &c) in &ecount {
                if c != 2 {
                    locked[(k >> 32) as usize] = true;
                    locked[(k & 0xffff_ffff) as usize] = true;
                }
            }
            if let Some(x) = extra_lock {
                for (l, &e) in locked.iter_mut().zip(x) {
                    *l |= e;
                }
            }
            let mut v2t: Vec<Vec<u32>> = vec![Vec::new(); nv];
            for (ti, t) in self.tris.iter().enumerate() {
                for &v in t {
                    v2t[v as usize].push(ti as u32);
                }
            }
            let nbrs_of = |v2t: &[Vec<u32>], tris: &[[u32; 3]], v: u32| -> Vec<u32> {
                let mut s: Vec<u32> = Vec::with_capacity(8);
                for &ti in &v2t[v as usize] {
                    for &x in &tris[ti as usize] {
                        if x != v && !s.contains(&x) {
                            s.push(x);
                        }
                    }
                }
                s
            };
            let mut keys: Vec<u64> = ecount.keys().copied().collect();
            keys.sort_unstable();
            let mut heap: std::collections::BinaryHeap<std::cmp::Reverse<(u64, u32, u32, u32, u32)>> =
                std::collections::BinaryHeap::with_capacity(keys.len());
            for k in keys {
                let (a, b) = ((k >> 32) as u32, (k & 0xffff_ffff) as u32);
                if let Some(sd) = seed.as_ref()
                    && !(sd[a as usize] || sd[b as usize])
                {
                    continue;
                }
                heap.push(std::cmp::Reverse((elen(a, b).to_bits(), a, b, 0, 0)));
            }
            let mut ver = vec![0u32; nv];
            let mut dead = vec![false; nt];
            let mut n = 0usize;
            while let Some(std::cmp::Reverse((_, a, b, va, vb))) = heap.pop() {
                if ver[a as usize] != va || ver[b as usize] != vb || v2t[a as usize].is_empty() || v2t[b as usize].is_empty() {
                    continue;
                }
                'dir: for (w, u) in [(a, b), (b, a)] {
                    if locked[w as usize] {
                        continue;
                    }
                    let shared: Vec<u32> = v2t[w as usize]
                        .iter()
                        .copied()
                        .filter(|&ti| self.tris[ti as usize].contains(&u))
                        .collect();
                    if shared.len() != 2 {
                        continue;
                    }
                    let opp = |ti: u32| -> u32 { *self.tris[ti as usize].iter().find(|&&x| x != u && x != w).unwrap() };
                    let want = [opp(shared[0]), opp(shared[1])];
                    let nw = nbrs_of(&v2t, &self.tris, w);
                    let nu = nbrs_of(&v2t, &self.tris, u);
                    let common: Vec<u32> = nw.iter().copied().filter(|x| nu.contains(x)).collect();
                    if common.len() != 2 || !common.contains(&want[0]) || !common.contains(&want[1]) {
                        continue;
                    }
                    if on[w as usize] {
                        let cn: Vec<u32> = nw.iter().copied().filter(|&z| crease_edge(w, z)).collect();
                        if cn.len() != 2 || !cn.contains(&u) {
                            continue;
                        }
                        let z = if cn[0] == u { cn[1] } else { cn[0] };
                        if !along_crease(u, z) {
                            continue;
                        }
                    }
                    let pu = pos[u as usize];
                    for &ti in &v2t[w as usize] {
                        if shared.contains(&ti) {
                            continue;
                        }
                        let t = self.tris[ti as usize];
                        let po = t.map(|i| pos[i as usize]);
                        let pn = t.map(|i| if i == w { pu } else { pos[i as usize] });
                        let (no, nn) = (spur_normal(po), spur_normal(pn));
                        let (lo, ln) = (len(no), len(nn));
                        if ln < 1e-12 || dot3(no, nn) <= 0.5 * lo * ln {
                            continue 'dir;
                        }
                        let (qo, qn) = (spur_quality(po), spur_quality(pn));
                        if qn < QMIN && qn < qo {
                            continue 'dir;
                        }
                        let exact = (dot3(no, sub(pu, po[0])) / lo).abs() <= PLANE_EPS;
                        if !exact && !ref_sample_ok(reference, grid, pn, tol, AGREE) {
                            continue 'dir;
                        }
                    }
                    for &ti in &shared {
                        dead[ti as usize] = true;
                        for x in self.tris[ti as usize] {
                            v2t[x as usize].retain(|&y| y != ti);
                        }
                    }
                    let moved = std::mem::take(&mut v2t[w as usize]);
                    for ti in moved {
                        for x in self.tris[ti as usize].iter_mut() {
                            if *x == w {
                                *x = u;
                            }
                        }
                        v2t[u as usize].push(ti);
                    }
                    ver[u as usize] += 1;
                    ver[w as usize] += 1;
                    n += 1;
                    for z in nbrs_of(&v2t, &self.tris, u) {
                        heap.push(std::cmp::Reverse((elen(u, z).to_bits(), u, z, ver[u as usize], ver[z as usize])));
                    }
                    break 'dir;
                }
            }
            if compact {
                // 无塌缩时也压缩:分段合并后的孤点要在这里清掉
                self.compact_dead(&dead);
            } else if n > 0 {
                // P37 分段模式:顶点号不变,只删死面
                let mut k2 = 0usize;
                self.tris.retain(|_| {
                    let keep = !dead[k2];
                    k2 += 1;
                    keep
                });
            }
            n
        }

        /// **源折痕嵌入(P30)**:把源网格的折痕线"刻"进输出网格。
        ///
        /// 前面的步骤只会把顶点吸到**已有**的折痕上;当两条源折痕在一点汇合、两者之间的窄条宽度
        /// 小于提取格宽 h 时(桌腿:圆柱底边与方板外棱相切,窄条宽 w ≈ z²/60),提取把窄条抹成一个
        /// 斜坡,两条棱线在切点前约 2mm 就"连"在一起,后处理也无从恢复。这里直接造出缺失的折痕边:
        ///
        /// 1. **角点**:源折痕图里度 ≠ 2 的顶点(`ReferenceCreases::corners`)。输出网格里没有顶点离它
        ///    < `on_tol` 时,在最近点处插入(落在边上分裂该边,否则 1→3 分裂面),坐标取源角点。
        /// 2. **切边**:对每段源折痕 AB(两侧源面法线 n1, n2),s(x) = (x − A)·(n1 − n2) 在折痕两侧异号。
        ///    输出边两端异号、都不在折痕上(> `on_tol`)、交点离线段 < `r_cut` 且投影参数在 [−0.02, 1.02]
        ///    时,在线段上的最近点处分裂该边。依次分裂后,跨折痕的面自然出现沿折痕的边。
        ///
        /// 守卫:新顶点不与已有顶点重合(> `on_tol`);边两侧的面都须朝向折痕两侧之一(点积 ≥ 0.5,
        /// 挡住薄壁另一面);子面不退化、不反向(与原面点积 ≥ 0.2;或与某侧源法线一致 ≥ 0.9 且不反向 ——
        /// 跨折痕的斜面切开后子面转到两侧源面上,与原斜面可差 90°),非薄片子面须与两侧之一一致 ≥ 0.5。
        ///
        /// 返回 (插入角点数, 切边数)。之后应再跑 `merge_crease_spurs` 与 `decimate_to_reference` 收拾切边
        /// 留下的短边与薄片。Python 原型实测(桌腿 h=0.08 mult=1):源折痕采样点离最近可见锐边
        /// > 0.05mm 的从 334 降到 0(最大 1.18 → 0.023mm),三角 1,456 → 1,390,仍闭合流形、无折叠。
        pub fn embed_reference_creases(&mut self, grid: &ReferenceCreases, on_tol: Real, r_cut: Real) -> (usize, usize) {
            if self.tris.is_empty() {
                return (0, 0);
            }
            // 面网格格宽 ≥ 4·r_cut:角点查询与沿线段查询都只需 3×3×3 邻域(见 faces_near 调用处)
            let mut e = CreaseEmbed::new(&mut self.verts, &mut self.tris, on_tol, (4.0 * r_cut).max(0.5));
            let mut n_corner = 0usize;
            for c in grid.corners() {
                if e.insert_corner(c, r_cut) {
                    n_corner += 1;
                }
            }
            let mut n_cut = 0usize;
            for &(edge, normals) in grid.segments() {
                n_cut += e.cut_along(edge, normals, r_cut);
            }
            (n_corner, n_cut)
        }

        /// **视觉毛刺清理(P33)**:查看器会把二面角超过阈值的边画成线。若输出某条边的二面角比源网格
        /// 同处(两侧各取一个贴边采样点的源面法线夹角)大 15° 以上且 > 20°,它就是一条源件上没有的
        /// "多余线"。叶轮实测 622 条(> 30° 的 272 条),几乎都在折痕旁,由面积 ~0.001mm²、高 ~0.01mm 的
        /// 微面(翻面、斜面、薄片)造成 —— 它们离源面都 < 0.02mm,前面各步的距离/法线判据看不出。
        ///
        /// 对每条多余线(按超出量从大到小),取两侧面中与源法线偏差更大的那个,按边长从短到长、两个方向
        /// 试塌缩;接受条件:共享面恰 2 + 链接条件;新面不退化、7 点采样到源面 ≤ 0.02mm;新面不与原面
        /// 反向(原面本身是薄片或已偏 > 90° 时除外);受影响区域(两端点的 2 环)的视觉代价
        /// Σ max(0, 输出二面角 − 源二面角 − 15°)(只计 > 20° 的边,退化面计 30)严格下降。
        ///
        /// 返回塌缩次数。Python 原型实测(叶轮 h=0.1 mult=1,P32 Rust 输出上 4 轮):多余线 622 → 101、
        /// 其中 > 30° 的 272 → 9;三角 25,578 → 24,644;非流形 22 → 0;真缺陷折叠 4 → 0;输出→源 p99 0.0087mm。
        pub fn clean_visual_spurs(&mut self, reference: &meshfield::MeshField, rounds: usize) -> usize {
            // P38:大网格先分段并行清理(跨段顶点锁定、不压缩),再在接缝 2 环内串行清理(其余锁定)并压缩。
            // 段数固定(`ADF_DECIMATE_BLOCKS`,默认 32)⟹ 与线程数无关。< 20 万三角或 `ADF_DECIMATE_SERIAL` 时串行。
            let k: usize = std::env::var("ADF_DECIMATE_BLOCKS").ok().and_then(|v| v.parse().ok()).unwrap_or(32);
            if k >= 2 && self.tris.len() >= 200_000 && std::env::var_os("ADF_DECIMATE_SERIAL").is_none() {
                return self.clean_visual_spurs_blocks(reference, rounds, k);
            }
            self.clean_visual_spurs_locked(reference, rounds, None, true)
        }

        /// 分段并行视觉清理本体(无规模门槛,测试直接调用)。
        pub fn clean_visual_spurs_blocks(&mut self, reference: &meshfield::MeshField, rounds: usize, k: usize) -> usize {
            let (n, cross) = self.run_in_slabs(k.max(1), |m, lock| m.clean_visual_spurs_locked(reference, rounds, Some(lock), false));
            let near = self.ring_expand(&cross, 2);
            let lock: Vec<bool> = near.iter().map(|&b| !b).collect();
            n + self.clean_visual_spurs_locked(reference, rounds, Some(lock.as_slice()), true)
        }

        /// `locked`:任一端点被锁的边不塌缩(跨段顶点的邻域在本段内不完整,链接条件会漏判);
        /// `compact`:为假时只删死面、顶点号不变(分段模式)。
        fn clean_visual_spurs_locked(&mut self, reference: &meshfield::MeshField, rounds: usize, locked: Option<&[bool]>, compact: bool) -> usize {
            // 带锁时轮间不得压缩(压缩会重编顶点号,`locked` 随即失效),只在最后压缩一次。
            let inner_compact = compact && locked.is_none();
            let mut total = 0usize;
            for _ in 0..rounds {
                let n = self.clean_visual_spurs_once(reference, locked, inner_compact);
                total += n;
                if n == 0 {
                    break;
                }
            }
            if compact && (!inner_compact || total == 0) {
                // 清掉分段合并后的孤点与带锁轮次留下的孤点
                let dead = vec![false; self.tris.len()];
                self.compact_dead(&dead);
            }
            total
        }

        fn clean_visual_spurs_once(&mut self, reference: &meshfield::MeshField, locked: Option<&[bool]>, compact: bool) -> usize {
            let nv = self.verts.len();
            let nt = self.tris.len();
            if nt == 0 {
                return 0;
            }
            let mut v2t: Vec<Vec<u32>> = vec![Vec::new(); nv];
            for (ti, t) in self.tris.iter().enumerate() {
                for &v in t {
                    v2t[v as usize].push(ti as u32);
                }
            }
            let mut emap: std::collections::HashMap<u64, [u32; 3], FxBuild> =
                std::collections::HashMap::with_capacity_and_hasher(nt * 3, FxBuild);
            for (ti, t) in self.tris.iter().enumerate() {
                for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                    let e = emap.entry(spur_edge_key(a, b)).or_insert([u32::MAX, u32::MAX, 0]);
                    if e[0] == u32::MAX {
                        e[0] = ti as u32;
                    } else if e[1] == u32::MAX {
                        e[1] = ti as u32;
                    }
                    e[2] += 1;
                }
            }
            let mut keys: Vec<u64> = emap.iter().filter(|(_, e)| e[2] == 2).map(|(k, _)| *k).collect();
            keys.sort_unstable();
            let pos = |verts: &[[f32; 3]], t: [u32; 3]| t.map(|i| g64(verts[i as usize]));
            // P38:候选扫描是纯函数(每条边 0–2 次源面查询),按排好序的键分块并行、按块序拼接 ⟹ 与串行一致。
            let mut cand: Vec<(f64, u32, u32)> = {
                let (verts, tris, emap) = (&self.verts, &self.tris, &emap);
                let chunk = keys.len().div_ceil(nthreads()).max(1);
                std::thread::scope(|scope| {
                    let handles: Vec<_> = keys
                        .chunks(chunk)
                        .map(|part| {
                            scope.spawn(move || {
                                let mut out = Vec::new();
                                for &key in part {
                                    let e = emap[&key];
                                    let (a, b) = ((key >> 32) as u32, (key & 0xffff_ffff) as u32);
                                    let ex = visual_excess(
                                        reference,
                                        pos(verts, tris[e[0] as usize]),
                                        pos(verts, tris[e[1] as usize]),
                                        g64(verts[a as usize]),
                                        g64(verts[b as usize]),
                                    );
                                    if ex > 0.0 {
                                        out.push((ex, e[0], e[1]));
                                    }
                                }
                                out
                            })
                        })
                        .collect();
                    handles.into_iter().flat_map(|h| h.join().expect("候选扫描线程 panic")).collect()
                })
            };
            cand.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)).then(x.2.cmp(&y.2)));
            let mut dead = vec![false; nt];
            let mut n = 0usize;
            for (_, f1, f2) in cand {
                if dead[f1 as usize] || dead[f2 as usize] {
                    continue;
                }
                let e1 = face_source_error(reference, pos(&self.verts, self.tris[f1 as usize]));
                let e2 = face_source_error(reference, pos(&self.verts, self.tris[f2 as usize]));
                let order = if e1 >= e2 { [f1, f2] } else { [f2, f1] };
                'face: for fw in order {
                    if dead[fw as usize] {
                        continue;
                    }
                    let t = self.tris[fw as usize];
                    let mut edges = [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])];
                    let elen = |e: (u32, u32), verts: &[[f32; 3]]| len(sub(g64(verts[e.0 as usize]), g64(verts[e.1 as usize])));
                    edges.sort_by(|p, q| elen(*p, &self.verts).total_cmp(&elen(*q, &self.verts)));
                    for (x, y) in edges {
                        if locked.is_some_and(|l| l[x as usize] || l[y as usize]) {
                            continue;
                        }
                        for (w, u) in [(x, y), (y, x)] {
                            if self.try_visual_collapse(reference, &mut v2t, &mut dead, w, u) {
                                n += 1;
                                break 'face;
                            }
                        }
                    }
                }
            }
            if n > 0 {
                if compact {
                    self.compact_dead(&dead);
                } else {
                    let mut k2 = 0usize;
                    self.tris.retain(|_| {
                        let keep = !dead[k2];
                        k2 += 1;
                        keep
                    });
                }
            }
            n
        }

        /// 区域 `faces` 内所有内部边的视觉代价;`over` 给出假想状态下被替换的面。
        fn local_visual_cost(&self, reference: &meshfield::MeshField, faces: &[u32], over: &[(u32, [u32; 3])]) -> f64 {
            let get = |f: u32| over.iter().find(|(g, _)| *g == f).map_or(self.tris[f as usize], |x| x.1);
            let mut edges: std::collections::HashMap<u64, Vec<u32>, FxBuild> = Default::default();
            for &f in faces {
                let t = get(f);
                for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                    edges.entry(spur_edge_key(a, b)).or_default().push(f);
                }
            }
            let mut keys: Vec<u64> = edges.keys().copied().collect();
            keys.sort_unstable();
            let mut cost = 0.0;
            for key in keys {
                let fs = &edges[&key];
                if fs.len() != 2 {
                    continue;
                }
                let (a, b) = ((key >> 32) as u32, (key & 0xffff_ffff) as u32);
                let p = |f: u32| get(f).map(|i| g64(self.verts[i as usize]));
                cost += visual_excess(reference, p(fs[0]), p(fs[1]), g64(self.verts[a as usize]), g64(self.verts[b as usize]));
            }
            cost
        }

        /// 试把 w 并入 u(u 不动)。满足 `clean_visual_spurs` 的全部条件才执行并返回 true。
        fn try_visual_collapse(
            &mut self,
            reference: &meshfield::MeshField,
            v2t: &mut [Vec<u32>],
            dead: &mut [bool],
            w: u32,
            u: u32,
        ) -> bool {
            let shared: Vec<u32> = v2t[w as usize].iter().copied().filter(|&ti| self.tris[ti as usize].contains(&u)).collect();
            if shared.len() != 2 {
                return false;
            }
            let nbrs_of = |v: u32| -> Vec<u32> {
                let mut s: Vec<u32> = Vec::with_capacity(8);
                for &ti in &v2t[v as usize] {
                    for &x in &self.tris[ti as usize] {
                        if x != v && !s.contains(&x) {
                            s.push(x);
                        }
                    }
                }
                s
            };
            let opp = |ti: u32| -> u32 { *self.tris[ti as usize].iter().find(|&&x| x != u && x != w).unwrap() };
            let want = [opp(shared[0]), opp(shared[1])];
            let (nw, nu) = (nbrs_of(w), nbrs_of(u));
            let common: Vec<u32> = nw.iter().copied().filter(|x| nu.contains(x)).collect();
            if common.len() != 2 || !common.contains(&want[0]) || !common.contains(&want[1]) {
                return false;
            }
            let mut aff: Vec<u32> = v2t[w as usize].clone();
            for &ti in &v2t[u as usize] {
                if !aff.contains(&ti) {
                    aff.push(ti);
                }
            }
            let mut ring: Vec<u32> = Vec::new();
            for &f in &aff {
                for &v in &self.tris[f as usize] {
                    for &g in &v2t[v as usize] {
                        if !ring.contains(&g) {
                            ring.push(g);
                        }
                    }
                }
            }
            let old = self.local_visual_cost(reference, &ring, &[]);
            if old <= 0.0 {
                return false;
            }
            let mut over: Vec<(u32, [u32; 3])> = Vec::new();
            for &f in &v2t[w as usize] {
                if shared.contains(&f) {
                    continue;
                }
                let t = self.tris[f as usize];
                let tt = t.map(|i| if i == w { u } else { i });
                let po = t.map(|i| g64(self.verts[i as usize]));
                let pn = tt.map(|i| g64(self.verts[i as usize]));
                let (no, nn) = (spur_normal(po), spur_normal(pn));
                if len(nn) < 1e-12 {
                    return false;
                }
                if dot3(no, nn) <= 0.0 && !(spur_quality(po) < 0.05 || face_source_error(reference, po) > 90.0) {
                    return false;
                }
                if ref_face_max_distance(reference, pn) > 0.02 {
                    return false;
                }
                over.push((f, tt));
            }
            let ring2: Vec<u32> = ring.iter().copied().filter(|f| !shared.contains(f)).collect();
            let new = self.local_visual_cost(reference, &ring2, &over);
            if new >= old - 1e-6 {
                return false;
            }
            for &ti in &shared {
                dead[ti as usize] = true;
                for x in self.tris[ti as usize] {
                    v2t[x as usize].retain(|&y| y != ti);
                }
            }
            let moved = std::mem::take(&mut v2t[w as usize]);
            for ti in moved {
                for x in self.tris[ti as usize].iter_mut() {
                    if *x == w {
                        *x = u;
                    }
                }
                v2t[u as usize].push(ti);
            }
            true
        }

        /// P38:嵌入之后的"折痕清理 ↔ 补充简化"交替收尾。大网格时整段循环分 32 段并行(跨段顶点锁定、
        /// 不压缩),再在接缝 2 环内串行重跑一遍(其余锁定)并压缩;小网格直接串行。返回补充简化的塌缩数。
        pub fn alternate_crease_cleanup(
            &mut self,
            reference: &meshfield::MeshField,
            grid: &ReferenceCreases,
            tol: Real,
            h_ext: Real,
        ) -> usize {
            let k: usize = std::env::var("ADF_DECIMATE_BLOCKS").ok().and_then(|v| v.parse().ok()).unwrap_or(32);
            if k >= 2 && self.tris.len() >= 200_000 && std::env::var_os("ADF_DECIMATE_SERIAL").is_none() {
                return self.alternate_crease_cleanup_blocks(reference, grid, tol, h_ext, k);
            }
            self.alternate_crease_cleanup_locked(reference, grid, tol, h_ext, None, true)
        }

        /// 分段并行交替收尾本体(无规模门槛,测试直接调用)。
        pub fn alternate_crease_cleanup_blocks(
            &mut self,
            reference: &meshfield::MeshField,
            grid: &ReferenceCreases,
            tol: Real,
            h_ext: Real,
            k: usize,
        ) -> usize {
            let (n, cross) = self.run_in_slabs(k.max(1), |m, lock| m.alternate_crease_cleanup_locked(reference, grid, tol, h_ext, Some(lock), false));
            let near = self.ring_expand(&cross, 2);
            let lock: Vec<bool> = near.iter().map(|&b| !b).collect();
            n + self.alternate_crease_cleanup_locked(reference, grid, tol, h_ext, Some(lock.as_slice()), true)
        }

        fn alternate_crease_cleanup_locked(
            &mut self,
            reference: &meshfield::MeshField,
            grid: &ReferenceCreases,
            tol: Real,
            h_ext: Real,
            locked: Option<&[bool]>,
            compact: bool,
        ) -> usize {
            // 带锁时步骤间不得压缩(压缩会重编顶点号,`locked` 随即失效),只在最后压缩一次。
            let inner_compact = compact && locked.is_none();
            let mut n = 0usize;
            for _ in 0..4 {
                let (f, c, m) = self.merge_crease_spurs_locked(grid, 30.0, 0.005, h_ext * 1.2, h_ext * 3.0, 0.02, 2, locked, inner_compact);
                let seed = self.sharp_band(30.0);
                let d = self.decimate_to_reference_seeded(reference, grid, tol, 0.005, Some(seed), locked, inner_compact);
                n += d;
                if f + c + m + d == 0 {
                    break;
                }
            }
            if compact {
                let dead = vec![false; self.tris.len()];
                self.compact_dead(&dead);
            }
            n
        }

        /// 删除标记为死的三角与由此孤立的顶点(保持其余顶点/三角的相对顺序)。
        fn compact_dead(&mut self, dead: &[bool]) {
            let nv = self.verts.len();
            let mut remap = vec![u32::MAX; nv];
            let mut verts = Vec::with_capacity(nv);
            let mut tris = Vec::with_capacity(self.tris.len());
            for (ti, t) in self.tris.iter().enumerate() {
                if dead[ti] {
                    continue;
                }
                tris.push(t.map(|i| {
                    if remap[i as usize] == u32::MAX {
                        remap[i as usize] = verts.len() as u32;
                        verts.push(self.verts[i as usize]);
                    }
                    remap[i as usize]
                }));
            }
            self.verts = verts;
            self.tris = tris;
        }

        /// **绕向统一**:从每个三角沿共享边洪泛,凡与已定向邻居"同向穿过同一条边"的
        /// 三角就翻转(交换两个顶点)。最后按有符号体积判整体朝向,必要时整体翻。
        ///
        /// 为什么需要:Surface Nets 的每个 quad 靠 `want`(场值增大的方向)独立定向,
        /// 在薄壁/窄槽等场值不稳处会个别翻反。几何本身没错,但**背面剔除的渲染器
        /// 会把翻反的三角显示成洞**——实测叶轮 966 个三角(0.2%)成 324 处斑点,
        /// 观感即"模型很多空洞"。翻反的三角在 STL 里有符号体积贡献为负,
        /// 与"真的缺面"是两回事。
        ///
        /// 返回翻转的三角数。
        pub fn orient_consistently(&mut self) -> usize {
            let n = self.tris.len();
            // P40:边表由 n·3 容量的 HashMap 换成按较小顶点号分行的 CSR(`edge_rows`),查询 = 行内二分。
            // 结果与旧版逐位一致:旧版槽位按插入依次 [空,空]→[x,空]→[x,y]→(第 3 次)[空,空]→… 循环,
            // 某条边可用 ⟺ 出现次数 c ≡ 2 (mod 3),两槽 = 最后两次出现(行内按面序、边序排列 = 插入序)。
            // 方向位 nf 取自邻面的**原始**朝向:只在邻面未访问时读取,而未访问的面从未被翻转过。
            let rows = edge_rows(&self.tris, self.verts.len());
            let slot_of = |run: &[(u32, u32)]| -> Option<[u32; 2]> {
                let c = run.len();
                (c % 3 == 2).then(|| [run[c - 2].1, run[c - 1].1])
            };
            let mut seen = vec![false; n];
            let mut flipped = 0usize;
            let mut stack: Vec<u32> = Vec::new();
            let mut group: Vec<u32> = Vec::new();
            for s in 0..n {
                if seen[s] {
                    continue;
                }
                seen[s] = true;
                stack.push(s as u32);
                group.clear();
                // ① 沿**流形边**洪泛,把组内绕向调成一致(非流形边不跨越,同旧版)。
                while let Some(ti) = stack.pop() {
                    group.push(ti);
                    let t = self.tris[ti as usize];
                    for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                        let fwd = a < b;
                        let Some(slot) = slot_of(rows.run(a, b)) else { continue };
                        for he in slot {
                            let nt = he / 3;
                            let k = (he % 3) as usize;
                            if nt == ti || seen[nt as usize] {
                                continue;
                            }
                            seen[nt as usize] = true;
                            let o = self.tris[nt as usize];
                            let nf = o[k] < o[(k + 1) % 3];
                            if nf == fwd {
                                self.tris[nt as usize].swap(1, 2);
                                flipped += 1;
                            }
                            stack.push(nt);
                        }
                    }
                }
                // ② 整组朝向:以组首顶点为原点算有符号体积(避免远处大数相消)
                let o = self.verts[self.tris[group[0] as usize][0] as usize];
                let mut vol = 0.0f64;
                for &ti in &group {
                    let t = self.tris[ti as usize];
                    let a = self.verts[t[0] as usize];
                    let b = self.verts[t[1] as usize];
                    let c = self.verts[t[2] as usize];
                    let (ax, ay, az) = (
                        (a[0] - o[0]) as f64,
                        (a[1] - o[1]) as f64,
                        (a[2] - o[2]) as f64,
                    );
                    let (bx, by, bz) = (
                        (b[0] - o[0]) as f64,
                        (b[1] - o[1]) as f64,
                        (b[2] - o[2]) as f64,
                    );
                    let (cx, cy, cz) = (
                        (c[0] - o[0]) as f64,
                        (c[1] - o[1]) as f64,
                        (c[2] - o[2]) as f64,
                    );
                    vol += ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx)
                        + az * (bx * cy - by * cx);
                }
                if vol < 0.0 {
                    for &ti in &group {
                        self.tris[ti as usize].swap(1, 2);
                        flipped += 1;
                    }
                }
            }
            flipped
        }

        /// **平滑 + 投影回零等值面**:削掉 Surface Nets 在薄壁/叶边处的锯齿尖刺。
        ///
        /// 为什么只做"平滑 + 投影"而不只是平滑:尖刺是**顶点定位**误差,场在表面附近
        /// 是准的(实测 mean 0.0023mm),所以平滑把顶点往邻域均值拉之后,再用
        /// Newton 步 `p -= d·∇d/|∇d|²` 把它拉回零等值面 —— 形状不被磨圆掉,
        /// 只把"出格"的顶点收回来。
        ///
        /// 参数:`iters` 次 Taubin(λ=0.5, μ=−0.53,交替做**不收缩**的平滑),
        /// 位移上限 `max_disp`(按顶点原始位置钳住,避免薄壁被抹平)。
        /// 返回(平均位移, 最大位移)mm。
        /// 把每个顶点沿场梯度 Newton 投影回零等值面(**不做**平滑)。
        ///
        /// 为什么简化后必须做:QEM 的代价是"到切平面距离²",在**凸曲面**上
        /// 最优点落在曲面**内侧**(弦高)。合并跨度 2–4° 时 40mm 半径的圆柱缘
        /// 就内缩 0.02–0.2mm,表现为圆缘变成多段内凹折线(查看器逐段画缘线
        /// = 满屏锯齿)。投影按场真值把顶点拉回零面,弦高归零;对平面区幂等。
        ///
        /// `max_step`:单步位移上限(建议 1 个重采样格,防薄壁顶点跨到另一侧)。
        /// 返回 (首轮平均位移 mm, 首轮最大位移 mm)。
        pub fn project_to_zero(&mut self, adf: &Adf, iters: usize, max_step: f64) -> (f64, f64) {
            let n = self.verts.len();
            if n == 0 || iters == 0 {
                return (0.0, 0.0);
            }
            let mut p: Vec<[f64; 3]> = self
                .verts
                .iter()
                .map(|v| [v[0] as f64, v[1] as f64, v[2] as f64])
                .collect();
            let lim = if max_step.is_finite() && max_step > 0.0 { max_step } else { 1.0 };
            let (mut sum, mut cnt, mut maxd) = (0.0f64, 0usize, 0.0f64);
            for it in 0..iters {
                for i in 0..n {
                    let q = p[i];
                    let (d0, g) = adf.eval_with_gradient(q);
                    if !d0.is_finite() || d0.abs() > 8.0 * lim {
                        continue;
                    }
                    let g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
                    if g2 < 0.25 {
                        continue; // 场的奇异处,投影会把点甩远
                    }
                    let mut step = [0.0f64; 3];
                    for k in 0..3 {
                        step[k] = -d0 * g[k] / g2;
                    }
                    let sl = (step[0] * step[0] + step[1] * step[1] + step[2] * step[2]).sqrt();
                    if sl > lim {
                        let f = lim / sl;
                        for k in 0..3 {
                            step[k] *= f;
                        }
                    }
                    for k in 0..3 {
                        p[i][k] = q[k] + step[k];
                    }
                    if it == 0 {
                        sum += sl;
                        cnt += 1;
                        maxd = maxd.max(sl);
                    }
                }
            }
            for (v, q) in self.verts.iter_mut().zip(p.iter()) {
                *v = [q[0] as f32, q[1] as f32, q[2] as f32];
            }
            (sum / cnt.max(1) as f64, maxd)
        }

        /// 无切削 STL 重建的末次投影。每点沿最近面方向移动至多 max_step，
        /// 串行验收移动以计入已移动的邻点，保留邻面原方向上的至少 1/4 面积。
        /// 仅走到投影距离的 90%，避免临近锐边的多个顶点重合。
        pub fn project_to_reference(&mut self, reference: &meshfield::MeshField, max_step: f64) {
            self.project_with_face_samples(reference, max_step, false);
        }

        /// 简化网格在采样原顶点外，也用面内三点估计弦差。
        pub fn project_with_face_samples(&mut self, reference: &meshfield::MeshField, max_step: f64, fit_faces: bool) {
            self.fit_reference(reference, max_step, fit_faces, None);
        }

        pub fn fit_reference(&mut self, reference: &meshfield::MeshField, max_step: f64, fit_faces: bool, creases: Option<&ReferenceCreases>) {
            if self.verts.is_empty() || !max_step.is_finite() || max_step <= 0.0 {
                return;
            }
            let mut incident = vec![Vec::new(); self.verts.len()];
            let mut initial_normals = Vec::with_capacity(self.tris.len());
            for (index, &[a, b, c]) in self.tris.iter().enumerate() {
                let p = g64(self.verts[a as usize]);
                let ab = sub(g64(self.verts[b as usize]), p);
                let ac = sub(g64(self.verts[c as usize]), p);
                initial_normals.push(cross3(ab, ac));
                let face = index as u32;
                incident[a as usize].push((b, c, face));
                incident[b as usize].push((c, a, face));
                incident[c as usize].push((a, b, face));
            }
            let mut targets = self.verts.clone();
            let batch = targets.len().div_ceil(nthreads());
            std::thread::scope(|scope| {
                for chunk in targets.chunks_mut(batch) {
                    scope.spawn(move || {
                        for v in chunk {
                            let p = g64(*v);
                            let (distance, _, _, closest) = reference.closest_features(p);
                            if distance > 0.0 {
                                let fraction = (max_step / distance).min(0.9);
                                *v = add(p, scale(sub(closest, p), fraction)).map(|x| x as f32);
                            }
                        }
                    });
                }
            });
            if fit_faces {
                // P37:逐面采样(每面 3 次源面查询)并行;查询以 (2·max_step)² 作剪枝初值 —— 距离 ≤ max_step 时
                // 结果与无界查询相同,超出时该面本就被跳过。累加仍按面序串行 ⟹ 与串行逐位一致。
                let per_face: Vec<Option<([f64; 3], f64, [f64; 3])>> = {
                    let (tris_ref, targets_ref): (&[[u32; 3]], &[[f32; 3]]) = (&self.tris, &targets);
                    let chunk = tris_ref.len().div_ceil(nthreads()).max(1);
                    std::thread::scope(|scope| {
                        let handles: Vec<_> = tris_ref
                            .chunks(chunk)
                            .map(|part| {
                                scope.spawn(move || {
                                    part.iter()
                                        .map(|&[a, b, c]| {
                                            let [p, q, r] = [a, b, c].map(|i| g64(targets_ref[i as usize]));
                                            let normal = cross3(sub(q, p), sub(r, p));
                                            let area = len(normal);
                                            if area == 0.0 {
                                                return None;
                                            }
                                            let center = scale(add(add(p, q), r), 1.0 / 3.0);
                                            let mut heights = [0.0; 3];
                                            // (2/3, 1/6, 1/6) 的三个重心坐标给出二次精度的三角形积分。
                                            // 每个残差按该样点的重心权重分回三个顶点，区分叶片上不对称的弯曲。
                                            for (i, point) in [p, q, r].into_iter().enumerate() {
                                                let sample = scale(add(center, point), 0.5);
                                                let (distance, _, _, closest) =
                                                    reference.closest_features_seeded(sample, 4.0 * max_step * max_step);
                                                if distance > max_step {
                                                    return None;
                                                }
                                                heights[i] = dot3(sub(closest, sample), normal) / area;
                                            }
                                            Some((normal, area, heights))
                                        })
                                        .collect::<Vec<_>>()
                                })
                            })
                            .collect();
                        handles.into_iter().flat_map(|h| h.join().expect("面采样线程 panic")).collect()
                    })
                };
                let mut weighted = vec![([0.0; 3], 0.0); self.verts.len()];
                for (&[a, b, c], res) in self.tris.iter().zip(&per_face) {
                    let Some((normal, area, heights)) = *res else { continue };
                    let total: Real = heights.iter().sum();
                    for (corner, i) in [a, b, c].into_iter().enumerate() {
                        let (sum, weight) = &mut weighted[i as usize];
                        let height = (total + 3.0 * heights[corner]) / 6.0;
                        *sum = add(*sum, scale(normal, height));
                        *weight += area;
                    }
                }
                for (i, (sum, weight)) in weighted.into_iter().enumerate() {
                    if weight == 0.0 { continue; }
                    let candidate = add(g64(targets[i]), scale(sum, 1.0 / weight));
                    let origin = g64(self.verts[i]);
                    let delta = sub(candidate, origin);
                    targets[i] = add(origin, scale(delta, (max_step / len(delta)).min(1.0))).map(|v| v as f32);
                }
            }
            if let Some(grid) = creases {
                for (target, crease) in targets.iter_mut().zip(self.crease_targets(grid, grid.radius)) {
                    if let Some(point) = crease { *target = point.map(|v| v as f32); }
                }
            }
            for (i, target) in targets.into_iter().enumerate() {
                let original = self.verts[i];
                if original == target {
                    continue;
                }
                let old = g64(original);
                let mut next = g64(target);
                for _ in 0..5 {
                    let valid = incident[i].iter().all(|&(b, c, face)| {
                        let b = g64(self.verts[b as usize]);
                        let c = g64(self.verts[c as usize]);
                        let before = cross3(sub(b, old), sub(c, old));
                        let after = cross3(sub(b, next), sub(c, next));
                        let initial = initial_normals[face as usize];
                        dot3(before, after) > 0.25 * dot3(before, before)
                            && dot3(initial, after) > 0.25 * dot3(initial, initial)
                    });
                    if valid {
                        self.verts[i] = next.map(|x| x as f32);
                        break;
                    }
                    next = scale(add(old, next), 0.5).map(|x| x as f32 as f64);
                }
            }
        }

        pub fn smooth_and_project(&mut self, adf: &Adf, iters: usize, max_disp: f32) -> (f64, f64) {
            let n = self.verts.len();
            if n == 0 || self.tris.is_empty() {
                return (0.0, 0.0);
            }
            // 顶点邻接(CSR):每个三角给三个顶点各贡献 2 个邻居
            let mut deg = vec![0u32; n];
            for t in &self.tris {
                for i in 0..3 {
                    deg[t[i] as usize] += 2;
                }
            }
            let mut off = vec![0u32; n + 1];
            for i in 0..n {
                off[i + 1] = off[i] + deg[i];
            }
            let mut adj = vec![0u32; off[n] as usize];
            let mut cur = off.clone();
            for t in &self.tris {
                for i in 0..3 {
                    let a = t[i] as usize;
                    adj[cur[a] as usize] = t[(i + 1) % 3];
                    adj[cur[a] as usize + 1] = t[(i + 2) % 3];
                    cur[a] += 2;
                }
            }
            let orig: Vec<[f32; 3]> = self.verts.clone();
            let mut p: Vec<[f64; 3]> = self
                .verts
                .iter()
                .map(|v| [v[0] as f64, v[1] as f64, v[2] as f64])
                .collect();
            let mut tmp = p.clone();
            let lap = |p: &Vec<[f64; 3]>, f: f64, out: &mut Vec<[f64; 3]>| {
                for i in 0..n {
                    let (s0, s1) = (off[i] as usize, off[i + 1] as usize);
                    let k = (s1 - s0) as f64;
                    if k == 0.0 {
                        out[i] = p[i];
                        continue;
                    }
                    let (mut ax, mut ay, mut az) = (0.0, 0.0, 0.0);
                    for &j in &adj[s0..s1] {
                        let q = p[j as usize];
                        ax += q[0];
                        ay += q[1];
                        az += q[2];
                    }
                    out[i][0] = p[i][0] + f * (ax / k - p[i][0]);
                    out[i][1] = p[i][1] + f * (ay / k - p[i][1]);
                    out[i][2] = p[i][2] + f * (az / k - p[i][2]);
                }
            };
            for _ in 0..iters {
                lap(&p, 0.5, &mut tmp);
                std::mem::swap(&mut p, &mut tmp);
                lap(&p, -0.53, &mut tmp);
                std::mem::swap(&mut p, &mut tmp);
            }
            // 位移上限(防止薄壁被抹平):按原始位置钳住
            for i in 0..n {
                let d = [
                    p[i][0] - orig[i][0] as f64,
                    p[i][1] - orig[i][1] as f64,
                    p[i][2] - orig[i][2] as f64,
                ];
                let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                if len > max_disp as f64 {
                    let s = max_disp as f64 / len;
                    for k in 0..3 {
                        p[i][k] = orig[i][k] as f64 + d[k] * s;
                    }
                }
            }
            // ── 锐边切向平滑:锯齿(scalloping)是边缘线**切向**的格级波动
            // (QEF 顶点在相邻提取单元间沿锐边摆动 ±半格),法向平滑
            // (Taubin + 投影)对它无效。做法:识别特征边(两面法线夹角 >
            // 阈值)上的"内部"特征顶点(恰 2 条特征边,即链中点),把它向
            // 两个特征邻居的中点收紧(λ=0.5,2 次);端点/分叉不动,避免
            // 磨短边缘。收紧后的位置由下方统一 Newton 投影拉回零等值面,
            // 边缘保持锐利(顶点仍贴两面交界),锯齿被抹平。
            {
                // 边 → 相邻两面(恰 2 个三角才判特征;非流形边跳过)
                let mut face_of: std::collections::HashMap<(u32, u32), [u32; 2]> =
                    std::collections::HashMap::with_capacity(n * 3);
                for (ti, t) in self.tris.iter().enumerate() {
                    for k in 0..3 {
                        let (a, b) = (t[k], t[(k + 1) % 3]);
                        let key = (a.min(b), a.max(b));
                        match face_of.get_mut(&key) {
                            Some(slot) => slot[1] = ti as u32,
                            None => {
                                face_of.insert(key, [ti as u32, u32::MAX]);
                            }
                        }
                    }
                }
                let tri_n: Vec<[f64; 3]> = self
                    .tris
                    .iter()
                    .map(|t| {
                        let (v0, v1, v2) =
                            (self.verts[t[0] as usize], self.verts[t[1] as usize], self.verts[t[2] as usize]);
                        let ab = [
                            v1[0] as f64 - v0[0] as f64,
                            v1[1] as f64 - v0[1] as f64,
                            v1[2] as f64 - v0[2] as f64,
                        ];
                        let ac = [
                            v2[0] as f64 - v0[0] as f64,
                            v2[1] as f64 - v0[1] as f64,
                            v2[2] as f64 - v0[2] as f64,
                        ];
                        let cr = [
                            ab[1] * ac[2] - ab[2] * ac[1],
                            ab[2] * ac[0] - ab[0] * ac[2],
                            ab[0] * ac[1] - ab[1] * ac[0],
                        ];
                        let l = (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt();
                        if l > 1e-12 {
                            [cr[0] / l, cr[1] / l, cr[2] / l]
                        } else {
                            [0.0; 3]
                        }
                    })
                    .collect();
                let edge_angle_cos: f64 = (-25.0f64).to_radians().cos(); // cos(>25°) 上界
                // PCA 主方向切向收紧:锯齿是沿缘方向的格级波动。缘带顶点的
                // **特征边(两面夹角大)主要是横向跨缘边**——缘带内部沿缘边的
                // 两侧是同一面的相邻面片,夹角小,不是特征边——所以不能用
                // 特征邻居选"沿缘对"(实测:按特征邻居收紧,方向全错,残差
                // 不动)。改用**全部网格邻居的 PCA 主轴**:缘带顶点沿缘排成
                // 长链(跨 3-6 格),横向只有短跨——主轴 = 缘方向;把
                // (邻居质心 − 顶点)只取沿主轴分量收紧 = 抹掉沿缘波动,
                // 横向(缘宽)不动。位移再减去 ADF 梯度分量(保持离面距离,
                // 防止投影撤销)。
                let mut all_adj: Vec<Vec<u32>> = vec![Vec::new(); n];
                for t in &self.tris {
                    for k in 0..3 {
                        let (a, b) = (t[k] as usize, t[(k + 1) % 3] as usize);
                        all_adj[a].push(b as u32);
                        all_adj[b].push(a as u32);
                    }
                }
                // 顶点邻面法线最大夹角 > 25° ⟹ 锐边顶点(与吸附块同判据)
                let mut sharp2 = vec![false; n];
                for (key, slot) in face_of.iter() {
                    if slot[1] == u32::MAX {
                        continue;
                    }
                    let d = tri_n[slot[0] as usize][0] * tri_n[slot[1] as usize][0]
                        + tri_n[slot[0] as usize][1] * tri_n[slot[1] as usize][1]
                        + tri_n[slot[0] as usize][2] * tri_n[slot[1] as usize][2];
                    if d < edge_angle_cos {
                        sharp2[key.0 as usize] = true;
                        sharp2[key.1 as usize] = true;
                    }
                }
                const EDGE_ITERS: usize = 3;
                let lambda: f64 = 0.6;
                let mut n_tight = 0u64;
                let mut sum_move = 0.0f64;
                for _ in 0..EDGE_ITERS {
                    for i in 0..n {
                        if !sharp2[i] {
                            continue;
                        }
                        let ns = &all_adj[i];
                        if ns.len() < 3 {
                            continue;
                        }
                        // 邻居质心与协方差
                        let pi = p[i];
                        let mut mean = [0.0f64; 3];
                        for &j in ns {
                            let q = p[j as usize];
                            for k in 0..3 {
                                mean[k] += q[k];
                            }
                        }
                        let inv = 1.0 / ns.len() as f64;
                        for k in 0..3 {
                            mean[k] *= inv;
                        }
                        let mut cov = [[0.0f64; 3]; 3];
                        for &j in ns {
                            let q = p[j as usize];
                            let d = [q[0] - mean[0], q[1] - mean[1], q[2] - mean[2]];
                            for a in 0..3 {
                                for b in 0..3 {
                                    cov[a][b] += d[a] * d[b];
                                }
                            }
                        }
                        // 幂迭代:协方差最大特征向量 = 缘方向
                        let mut v = [0.31, 0.42, 0.85];
                        for _ in 0..10 {
                            let w = [
                                cov[0][0] * v[0] + cov[0][1] * v[1] + cov[0][2] * v[2],
                                cov[1][0] * v[0] + cov[1][1] * v[1] + cov[1][2] * v[2],
                                cov[2][0] * v[0] + cov[2][1] * v[1] + cov[2][2] * v[2],
                            ];
                            let l = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
                            if l < 1e-15 {
                                break;
                            }
                            v = [w[0] / l, w[1] / l, w[2] / l];
                        }
                        // 沿主轴的质心偏差 = 沿缘锯齿分量
                        let d = [mean[0] - pi[0], mean[1] - pi[1], mean[2] - pi[2]];
                        let dt = d[0] * v[0] + d[1] * v[1] + d[2] * v[2];
                        let mut mv = [lambda * dt * v[0], lambda * dt * v[1], lambda * dt * v[2]];
                        // 纯切向:减去沿 ADF 梯度的分量(保持离面距离)
                        {
                            let d0c = adf.eval(pi) as f64;
                            let hs = 1e-3 * adf.size;
                            let mut g = [0.0f64; 3];
                            for k in 0..3 {
                                let mut qq = pi;
                                qq[k] += hs;
                                g[k] = (adf.eval(qq) as f64 - d0c) / hs;
                            }
                            let g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
                            if g2 > 1e-12 {
                                let dp = (mv[0] * g[0] + mv[1] * g[1] + mv[2] * g[2]) / g2;
                                for k in 0..3 {
                                    mv[k] -= dp * g[k];
                                }
                            }
                        }
                        n_tight += 1;
                        sum_move += (mv[0] * mv[0] + mv[1] * mv[1] + mv[2] * mv[2]).sqrt();
                        for k in 0..3 {
                            p[i][k] += mv[k];
                        }
                    }
                }
                if n_tight > 0 {
                    eprintln!(
                        "[edge-smooth] PCA 切向收紧 {n_tight} 点/3 迭代,均 {:.4} mm/次",
                        sum_move / n_tight as f64
                    );
                }
            }
            // Newton 投影回零等值面:8 步(锐边处平滑会把顶点拉离表面 >1mm,
            // 2 步+限幅收不完;对已收敛顶点幂等,多迭代无副作用)
            for _ in 0..8 {
                for i in 0..n {
                    let q = [p[i][0], p[i][1], p[i][2]];
                    let (d0, g) = adf.eval_with_gradient(q);
                    // 只挡真正飞掉的点(碎屑已在投影前清理);阈值太紧会把
                    // 锐边处合法的 1mm 级顶点挡在投影门外(实测卡在 1.25mm)
                    if d0.abs() > 8.0 * max_disp as f64 {
                        continue; // 离面太远,不投影
                    }
                    let g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
                    // |∇d| 应≈1;远小于 1 说明落在场的奇异处,投影会把点甩得很远
                    if g2 < 0.25 {
                        continue;
                    }
                    // 投影步长上限放宽到 4×max_disp:max_disp 是为**平滑**设的,
                    // 用它钳投影会让 0.5mm 级的尖刺永远收不回来(实测卡在 0.91%)。
                    let lim = 4.0 * max_disp as f64;
                    let mut step = [0.0f64; 3];
                    for k in 0..3 {
                        step[k] = -d0 * g[k] / g2;
                    }
                    let sl = (step[0] * step[0] + step[1] * step[1] + step[2] * step[2]).sqrt();
                    if sl > lim {
                        let f = lim / sl;
                        for k in 0..3 {
                            step[k] *= f;
                        }
                    }
                    for k in 0..3 {
                        p[i][k] = q[k] + step[k];
                    }
                }
            }
            // ── 特征吸附:锯齿的根源是 **ADF 零面本身**在锐边处的圆角化波动
            // (三线性插值让相邻提取单元的零面位置交替偏向两侧,零面呈串珠状)
            // ——切向收紧会被投影拉回锯齿零面(实测:收紧力度 ×2.7 残差不动),
            // 法向平滑同样无效。绕过 ADF:输出网格的**锐边顶点**(邻面法线
            // 夹角 > 25°)若其最近**原始**特征是边/顶点(feat≤5,距离 ≤
            // 0.5mm),把顶点 70% 吸附到该特征的真实位置——边缘线回归原始
            // 几何,锯齿消失。部分吸附(0.7)避免侧面顶部顶点与边线顶点
            // 完全重合产生退化三角。非锐边顶点不查询(免误吸 + 省 90% 时延)。
            let no_snap = std::env::var("ADF_NO_SNAP").is_ok();
            if !no_snap && let BaseField::Mesh(m) = &adf.field.base {
                // 边 → 相邻两面
                let mut face_of: std::collections::HashMap<(u32, u32), [u32; 2]> =
                    std::collections::HashMap::with_capacity(n * 3);
                for (ti, t) in self.tris.iter().enumerate() {
                    for k in 0..3 {
                        let (a, b) = (t[k], t[(k + 1) % 3]);
                        let key = (a.min(b), a.max(b));
                        match face_of.get_mut(&key) {
                            Some(slot) => slot[1] = ti as u32,
                            None => {
                                face_of.insert(key, [ti as u32, u32::MAX]);
                            }
                        }
                    }
                }
                let tri_n: Vec<[f64; 3]> = self
                    .tris
                    .iter()
                    .map(|t| {
                        let (v0, v1, v2) = (
                            self.verts[t[0] as usize],
                            self.verts[t[1] as usize],
                            self.verts[t[2] as usize],
                        );
                        let ab = [
                            v1[0] as f64 - v0[0] as f64,
                            v1[1] as f64 - v0[1] as f64,
                            v1[2] as f64 - v0[2] as f64,
                        ];
                        let ac = [
                            v2[0] as f64 - v0[0] as f64,
                            v2[1] as f64 - v0[1] as f64,
                            v2[2] as f64 - v0[2] as f64,
                        ];
                        let cr = [
                            ab[1] * ac[2] - ab[2] * ac[1],
                            ab[2] * ac[0] - ab[0] * ac[2],
                            ab[0] * ac[1] - ab[1] * ac[0],
                        ];
                        let l = (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt();
                        if l > 1e-12 {
                            [cr[0] / l, cr[1] / l, cr[2] / l]
                        } else {
                            [0.0; 3]
                        }
                    })
                    .collect();
                // 顶点邻面法线最大夹角 > 25° ⟹ 锐边顶点
                let mut sharp = vec![false; n];
                for (key, slot) in face_of.iter() {
                    if slot[1] == u32::MAX {
                        continue;
                    }
                    let d = tri_n[slot[0] as usize][0] * tri_n[slot[1] as usize][0]
                        + tri_n[slot[0] as usize][1] * tri_n[slot[1] as usize][1]
                        + tri_n[slot[0] as usize][2] * tri_n[slot[1] as usize][2];
                    if d < (-25.0f64).to_radians().cos() {
                        sharp[key.0 as usize] = true;
                        sharp[key.1 as usize] = true;
                    }
                }
                let lim = 0.5f64;
                let mut n_snap = 0u64;
                for i in 0..n {
                    if !sharp[i] {
                        continue;
                    }
                    let q = [p[i][0], p[i][1], p[i][2]];
                    let (d, _t, feat, cp) = m.closest_features_seeded(q, lim * lim);
                    if feat <= 5 && d <= lim {
                        for k in 0..3 {
                            p[i][k] += 0.7 * (cp[k] - q[k]);
                        }
                        n_snap += 1;
                    }
                }
                eprintln!("[snap] 吸附 {n_snap} 锐边顶点到原始特征(lim={lim})");

                // ── 原始缘点云吸附:锯齿是边缘线的**形状误差**(顶点相对邻居
                // 光滑、相对原始缘线偏 0.1-0.3mm),任何"相对邻居"的滤波都
                // 无效(实测:切向收紧力度 ×2.7、PCA 主轴收紧,残差均不动),
                // 必须引入**外部参照**。原始叶轮边缘是光滑圆缘——原始网格里
                // 它表现为二面角 > 10° 的顶点带(缘点云)。输出锐边顶点吸附到
                // 缘点云的最近点:切向对齐到原始缘线(锯齿消除),横向由
                // "最近点"自然保持(缘带多排点,不压扁)。70% 部分吸附避免
                // 多顶点吸到同一点后完全重合。
                {
                    // 1) 原始缘带顶点:邻面法线夹角 > 10°(光滑缘的二面角峰)
                    // 原始网格顶点焊接(量化 key → id)
                    let qo = |x: Real| (x * 1e4).round() as i64;
                    let mut ovmap: std::collections::HashMap<[i64; 3], u32> =
                        std::collections::HashMap::new();
                    let mut ovs: Vec<[f64; 3]> = Vec::new();
                    let mut otv: Vec<[u32; 3]> = Vec::with_capacity(m.tris.len());
                    for t in m.tris.iter() {
                        let mut ids = [0u32; 3];
                        for (k, v) in t.iter().enumerate() {
                            let key = [qo(v[0]), qo(v[1]), qo(v[2])];
                            let id = *ovmap.entry(key).or_insert_with(|| {
                                ovs.push([v[0] as f64, v[1] as f64, v[2] as f64]);
                                (ovs.len() - 1) as u32
                            });
                            ids[k] = id;
                        }
                        otv.push(ids);
                    }
                    let mut ot_n: Vec<[f64; 3]> = Vec::with_capacity(m.tris.len());
                    for t in m.tris.iter() {
                        let (a, b, c) = (t[0], t[1], t[2]);
                        let ab = [
                            b[0] as f64 - a[0] as f64,
                            b[1] as f64 - a[1] as f64,
                            b[2] as f64 - a[2] as f64,
                        ];
                        let ac = [
                            c[0] as f64 - a[0] as f64,
                            c[1] as f64 - a[1] as f64,
                            c[2] as f64 - a[2] as f64,
                        ];
                        let cr = [
                            ab[1] * ac[2] - ab[2] * ac[1],
                            ab[2] * ac[0] - ab[0] * ac[2],
                            ab[0] * ac[1] - ab[1] * ac[0],
                        ];
                        let l = (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt();
                        ot_n.push(if l > 1e-12 {
                            [cr[0] / l, cr[1] / l, cr[2] / l]
                        } else {
                            [0.0; 3]
                        });
                    }
                    let mut oeid: std::collections::HashMap<(u32, u32), [u32; 2]> =
                        std::collections::HashMap::with_capacity(m.tris.len() * 3);
                    for (ti, t) in otv.iter().enumerate() {
                        for k in 0..3 {
                            let (a, b) = (t[k], t[(k + 1) % 3]);
                            let key = (a.min(b), a.max(b));
                            match oeid.get_mut(&key) {
                                Some(slot) => slot[1] = ti as u32,
                                None => {
                                    oeid.insert(key, [ti as u32, u32::MAX]);
                                }
                            }
                        }
                    }
                    let cos10 = (-10.0f64).to_radians().cos();
                    // 缘边 = 线段集合(连续吸附目标,离散点吸附会产生折痕)
                    let mut rim_segs: Vec<[[f64; 3]; 2]> = Vec::new();
                    for (key, slot) in oeid.iter() {
                        if slot[1] == u32::MAX {
                            continue;
                        }
                        let d = ot_n[slot[0] as usize][0] * ot_n[slot[1] as usize][0]
                            + ot_n[slot[0] as usize][1] * ot_n[slot[1] as usize][1]
                            + ot_n[slot[0] as usize][2] * ot_n[slot[1] as usize][2];
                        if d < cos10 {
                            rim_segs.push([ovs[key.0 as usize], ovs[key.1 as usize]]);
                        }
                    }
                    // 2) 空间哈希(格 0.4mm,按线段两端所属桶+中点桶)
                    let cell = 0.4f64;
                    let key_of = |p: [f64; 3]| -> (i64, i64, i64) {
                        (
                            (p[0] / cell).floor() as i64,
                            (p[1] / cell).floor() as i64,
                            (p[2] / cell).floor() as i64,
                        )
                    };
                    let mut grid: std::collections::HashMap<(i64, i64, i64), Vec<u32>> =
                        std::collections::HashMap::with_capacity(rim_segs.len());
                    for (si, s) in rim_segs.iter().enumerate() {
                        let mut keys = std::collections::HashSet::new();
                        keys.insert(key_of(s[0]));
                        keys.insert(key_of(s[1]));
                        keys.insert(key_of([
                            (s[0][0] + s[1][0]) * 0.5,
                            (s[0][1] + s[1][1]) * 0.5,
                            (s[0][2] + s[1][2]) * 0.5,
                        ]));
                        for k in keys {
                            grid.entry(k).or_default().push(si as u32);
                        }
                    }
                    // 点到线段最近点
                    let seg_closest = |q: [f64; 3], s: [[f64; 3]; 2]| -> [f64; 3] {
                        let ab = [s[1][0] - s[0][0], s[1][1] - s[0][1], s[1][2] - s[0][2]];
                        let ap = [q[0] - s[0][0], q[1] - s[0][1], q[2] - s[0][2]];
                        let ab2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
                        let tt = if ab2 > 1e-18 {
                            ((ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / ab2).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        [
                            s[0][0] + ab[0] * tt,
                            s[0][1] + ab[1] * tt,
                            s[0][2] + ab[2] * tt,
                        ]
                    };
                    // 3) 输出锐边顶点 → 最近缘线段最近点(70% 吸附)
                    let mut n_snap2 = 0u64;
                    for i in 0..n {
                        if !sharp[i] {
                            continue;
                        }
                        let q = p[i];
                        let (ki, kj, kk) = key_of(q);
                        let (mut best_d2, mut best_pt) = (f64::INFINITY, [0.0f64; 3]);
                        for da in -1i64..=1 {
                            for db in -1i64..=1 {
                                for dc in -1i64..=1 {
                                    if let Some(list) = grid.get(&(ki + da, kj + db, kk + dc)) {
                                        for &si in list {
                                            let cpt = seg_closest(q, rim_segs[si as usize]);
                                            let d2 = (cpt[0] - q[0]).powi(2)
                                                + (cpt[1] - q[1]).powi(2)
                                                + (cpt[2] - q[2]).powi(2);
                                            if d2 < best_d2 {
                                                best_d2 = d2;
                                                best_pt = cpt;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        if best_d2.is_finite() && best_d2 <= lim * lim {
                            for k in 0..3 {
                                p[i][k] += 0.7 * (best_pt[k] - q[k]);
                            }
                            n_snap2 += 1;
                        }
                    }
                    eprintln!(
                        "[snap] 缘线段 {} 条,吸附 {} 锐边顶点(lim={lim})",
                        rim_segs.len(),
                        n_snap2
                    );
                }
            }
            let (mut sum, mut mx) = (0.0f64, 0.0f64);
            for i in 0..n {
                let d = [
                    p[i][0] - orig[i][0] as f64,
                    p[i][1] - orig[i][1] as f64,
                    p[i][2] - orig[i][2] as f64,
                ];
                let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                sum += len;
                mx = mx.max(len);
                self.verts[i] = [p[i][0] as f32, p[i][1] as f32, p[i][2] as f32];
            }
            (sum / n as f64, mx)
        }

        /// **决断后脊线重吸附**:原始 CAD 的锐边在 h 采样下变成锯齿阶梯——
        /// 每个阶梯段(≈h 长)二面角 >25°,查看器逐段画线 = 满屏短黑线。
        /// QEM 不合并锯齿:合并代价 = 垂直抖动²(±0.05mm → ~2.5e-3 mm²)
        /// 超出 err2 上限;decimate 因此对锐边放宽阈值(见其注释)。合并点
        /// 落在原始缘线附近但残余偏移仍在——本方法把锐边顶点(邻面夹角
        /// >25°)强吸附到**原始网格**缘线段(邻面法线夹角>10° 的边,空间
        /// 哈希查询),脊线成长段光滑折线,查看器只画一条特征线。
        /// 返回吸附顶点数。
        pub fn snap_sharp_to_rim(&mut self, adf: &Adf, lim: f64, strength: f64) -> usize {
            let m = match &adf.field.base {
                BaseField::Mesh(m) => m,
                _ => return 0,
            };
            // 1) 原始缘线段:邻面法线夹角 > 10° 的边(顶点焊接 → 边 → 两面)
            let qo = |x: Real| (x * 1e4).round() as i64;
            let mut ovmap: std::collections::HashMap<[i64; 3], u32> = std::collections::HashMap::new();
            let mut ovs: Vec<[f64; 3]> = Vec::new();
            let mut otv: Vec<[u32; 3]> = Vec::with_capacity(m.tris.len());
            for t in m.tris.iter() {
                let mut ids = [0u32; 3];
                for (k, v) in t.iter().enumerate() {
                    let key = [qo(v[0]), qo(v[1]), qo(v[2])];
                    let id = *ovmap.entry(key).or_insert_with(|| {
                        ovs.push([v[0] as f64, v[1] as f64, v[2] as f64]);
                        (ovs.len() - 1) as u32
                    });
                    ids[k] = id;
                }
                otv.push(ids);
            }
            let mut ot_n: Vec<[f64; 3]> = Vec::with_capacity(m.tris.len());
            for t in m.tris.iter() {
                let (a, b, c) = (t[0], t[1], t[2]);
                let ab = [
                    b[0] as f64 - a[0] as f64,
                    b[1] as f64 - a[1] as f64,
                    b[2] as f64 - a[2] as f64,
                ];
                let ac = [
                    c[0] as f64 - a[0] as f64,
                    c[1] as f64 - a[1] as f64,
                    c[2] as f64 - a[2] as f64,
                ];
                let cr = [
                    ab[1] * ac[2] - ab[2] * ac[1],
                    ab[2] * ac[0] - ab[0] * ac[2],
                    ab[0] * ac[1] - ab[1] * ac[0],
                ];
                let l = (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt();
                ot_n.push(if l > 1e-12 {
                    [cr[0] / l, cr[1] / l, cr[2] / l]
                } else {
                    [0.0; 3]
                });
            }
            let mut oeid: std::collections::HashMap<(u32, u32), [u32; 2]> =
                std::collections::HashMap::with_capacity(m.tris.len() * 3);
            for (ti, t) in otv.iter().enumerate() {
                for k in 0..3 {
                    let (a, b) = (t[k], t[(k + 1) % 3]);
                    let key = (a.min(b), a.max(b));
                    match oeid.get_mut(&key) {
                        Some(slot) => slot[1] = ti as u32,
                        None => {
                            oeid.insert(key, [ti as u32, u32::MAX]);
                        }
                    }
                }
            }
            let cos10 = (-10.0f64).to_radians().cos();
            let mut rim_segs: Vec<[[f64; 3]; 2]> = Vec::new();
            for (key, slot) in oeid.iter() {
                if slot[1] == u32::MAX {
                    continue;
                }
                let d = ot_n[slot[0] as usize][0] * ot_n[slot[1] as usize][0]
                    + ot_n[slot[0] as usize][1] * ot_n[slot[1] as usize][1]
                    + ot_n[slot[0] as usize][2] * ot_n[slot[1] as usize][2];
                if d < cos10 {
                    rim_segs.push([ovs[key.0 as usize], ovs[key.1 as usize]]);
                }
            }
            if rim_segs.is_empty() {
                return 0;
            }
            // 2) 空间哈希(格 0.4mm,端点桶 + 中点桶)
            let cell = 0.4f64;
            let key_of = |p: [f64; 3]| -> (i64, i64, i64) {
                (
                    (p[0] / cell).floor() as i64,
                    (p[1] / cell).floor() as i64,
                    (p[2] / cell).floor() as i64,
                )
            };
            let mut grid: std::collections::HashMap<(i64, i64, i64), Vec<u32>> =
                std::collections::HashMap::with_capacity(rim_segs.len());
            for (si, s) in rim_segs.iter().enumerate() {
                let mut keys = std::collections::HashSet::new();
                keys.insert(key_of(s[0]));
                keys.insert(key_of(s[1]));
                keys.insert(key_of([
                    (s[0][0] + s[1][0]) * 0.5,
                    (s[0][1] + s[1][1]) * 0.5,
                    (s[0][2] + s[1][2]) * 0.5,
                ]));
                for k in keys {
                    grid.entry(k).or_default().push(si as u32);
                }
            }
            let seg_closest = |q: [f64; 3], s: [[f64; 3]; 2]| -> [f64; 3] {
                let ab = [s[1][0] - s[0][0], s[1][1] - s[0][1], s[1][2] - s[0][2]];
                let ap = [q[0] - s[0][0], q[1] - s[0][1], q[2] - s[0][2]];
                let ab2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
                let tt = if ab2 > 1e-18 {
                    ((ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / ab2).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                [
                    s[0][0] + ab[0] * tt,
                    s[0][1] + ab[1] * tt,
                    s[0][2] + ab[2] * tt,
                ]
            };
            // 3) 输出锐边顶点(邻面夹角 > 25°)
            let n = self.verts.len();
            let mut face_of: std::collections::HashMap<(u32, u32), [u32; 2]> =
                std::collections::HashMap::with_capacity(n * 3);
            for (ti, t) in self.tris.iter().enumerate() {
                for k in 0..3 {
                    let (a, b) = (t[k], t[(k + 1) % 3]);
                    let key = (a.min(b), a.max(b));
                    match face_of.get_mut(&key) {
                        Some(slot) => slot[1] = ti as u32,
                        None => {
                            face_of.insert(key, [ti as u32, u32::MAX]);
                        }
                    }
                }
            }
            let mut tri_n: Vec<[f64; 3]> = Vec::with_capacity(self.tris.len());
            for t in self.tris.iter() {
                let (a, b, c) = (
                    self.verts[t[0] as usize],
                    self.verts[t[1] as usize],
                    self.verts[t[2] as usize],
                );
                let (a, b, c) = (
                    [a[0] as f64, a[1] as f64, a[2] as f64],
                    [b[0] as f64, b[1] as f64, b[2] as f64],
                    [c[0] as f64, c[1] as f64, c[2] as f64],
                );
                let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                let cr = [
                    ab[1] * ac[2] - ab[2] * ac[1],
                    ab[2] * ac[0] - ab[0] * ac[2],
                    ab[0] * ac[1] - ab[1] * ac[0],
                ];
                let l = (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt();
                tri_n.push(if l > 1e-12 {
                    [cr[0] / l, cr[1] / l, cr[2] / l]
                } else {
                    [0.0; 3]
                });
            }
            let cos25 = (-25.0f64).to_radians().cos();
            let mut sharp = vec![false; n];
            for (key, slot) in face_of.iter() {
                if slot[1] == u32::MAX {
                    continue;
                }
                let nn0 = tri_n[slot[0] as usize];
                let nn1 = tri_n[slot[1] as usize];
                let d = nn0[0] * nn1[0] + nn0[1] * nn1[1] + nn0[2] * nn1[2];
                if d < cos25 {
                    sharp[key.0 as usize] = true;
                    sharp[key.1 as usize] = true;
                }
            }
            // 4) 强吸附(P23:翻转守卫 + 缘线存活判据)
            //    ① 翻转守卫:旧版无任何守卫,平盘面/孔口处锐边顶点被拉过对边,整片
            //       三角 180° 翻面(叶轮底盘 617 对折叠中 88% 法线完全反向)。现要求
            //       每个邻面法线转角 ≤60°,否则步长减半重试(至多 4 次),仍不行则放弃。
            //    ② 缘线存活:切削后被挖掉的原始缘线不在当前零面上(eval ≫ 0),不再
            //       吸引新加工面上的点。要求目标 |eval| ≤ 0.5·h_min;无切削时不受影响。
            fn snap_keeps_orientation(
                verts: &[[f32; 3]],
                tris: &[[u32; 3]],
                ring: &[u32],
                vi: usize,
                cand: [f64; 3],
            ) -> bool {
                for &ti in ring {
                    let t = tris[ti as usize];
                    let mut po = [[0.0f64; 3]; 3];
                    let mut pn = [[0.0f64; 3]; 3];
                    for k in 0..3 {
                        let w = t[k] as usize;
                        let p = verts[w];
                        po[k] = [p[0] as f64, p[1] as f64, p[2] as f64];
                        pn[k] = if w == vi { cand } else { po[k] };
                    }
                    let crs = |q: &[[f64; 3]; 3]| -> [f64; 3] {
                        let ab = [q[1][0] - q[0][0], q[1][1] - q[0][1], q[1][2] - q[0][2]];
                        let ac = [q[2][0] - q[0][0], q[2][1] - q[0][1], q[2][2] - q[0][2]];
                        [
                            ab[1] * ac[2] - ab[2] * ac[1],
                            ab[2] * ac[0] - ab[0] * ac[2],
                            ab[0] * ac[1] - ab[1] * ac[0],
                        ]
                    };
                    let (no, nn) = (crs(&po), crs(&pn));
                    let lo = (no[0] * no[0] + no[1] * no[1] + no[2] * no[2]).sqrt();
                    let ln = (nn[0] * nn[0] + nn[1] * nn[1] + nn[2] * nn[2]).sqrt();
                    if lo < 1e-18 {
                        continue; // 原本就退化:无朝向可比,交给薄片清理
                    }
                    if ln < 1e-18 || no[0] * nn[0] + no[1] * nn[1] + no[2] * nn[2] < 0.5 * lo * ln {
                        return false;
                    }
                }
                true
            }
            let mut v2t: Vec<Vec<u32>> = vec![Vec::new(); n];
            for (ti, t) in self.tris.iter().enumerate() {
                for &vi in t.iter() {
                    v2t[vi as usize].push(ti as u32);
                }
            }
            let live_tol = 0.5 * adf.size / (1u64 << adf.max_depth) as Real;
            let mut n_snap = 0usize;
            for i in 0..n {
                if !sharp[i] {
                    continue;
                }
                let q = [
                    self.verts[i][0] as f64,
                    self.verts[i][1] as f64,
                    self.verts[i][2] as f64,
                ];
                let (ki, kj, kk) = key_of(q);
                let (mut best_d2, mut best_pt) = (f64::INFINITY, [0.0f64; 3]);
                for da in -1i64..=1 {
                    for db in -1i64..=1 {
                        for dc in -1i64..=1 {
                            if let Some(list) = grid.get(&(ki + da, kj + db, kk + dc)) {
                                for &si in list {
                                    let cpt = seg_closest(q, rim_segs[si as usize]);
                                    let d2 = (cpt[0] - q[0]).powi(2)
                                        + (cpt[1] - q[1]).powi(2)
                                        + (cpt[2] - q[2]).powi(2);
                                    if d2 < best_d2 {
                                        best_d2 = d2;
                                        best_pt = cpt;
                                    }
                                }
                            }
                        }
                    }
                }
                if !(best_d2.is_finite() && best_d2 <= lim * lim) {
                    continue;
                }
                if (adf.eval(best_pt) as Real).abs() > live_tol {
                    continue; // 该段缘线已被切掉
                }
                let mut frac = strength;
                for _ in 0..4 {
                    let cand = [
                        (q[0] + frac * (best_pt[0] - q[0])) as f32 as f64,
                        (q[1] + frac * (best_pt[1] - q[1])) as f32 as f64,
                        (q[2] + frac * (best_pt[2] - q[2])) as f32 as f64,
                    ];
                    if snap_keeps_orientation(&self.verts, &self.tris, &v2t[i], i, cand) {
                        self.verts[i] = [cand[0] as f32, cand[1] as f32, cand[2] as f32];
                        n_snap += 1;
                        break;
                    }
                    frac *= 0.5;
                }
            }
            n_snap
        }

        /// **折叠与薄片修复**:叶轮边缘的"破损/缺口"指纹是两类退化结构——
        /// ①真折叠:相邻两面法线夹角 >60° 且一方顶点穿到对方平面背后
        ///   (叶轮实测 10,423 对,穿透深度 p50 0.054mm;提取在脊边处的
        ///   三线性插值零面自身振荡 + 平滑/吸附的顶点移动越过缘顶);
        /// ②薄片:最小高度 = 2·面积/最长边 ≈ 0 的针状三角(法线随机,
        ///   渲染成边缘黑点,f32 导出后可能变孔洞)。
        /// 两者都通过**塌缩其最短边**消除(合并两点→两面退化删除,无洞)。
        ///
        /// 折叠判定用**场梯度仲裁**:折叠面的法线与局部场梯度(外向)反向,
        /// 而合法凹角/凸角两面的法线都与梯度同向——纯几何判据(顶点异侧)
        /// 会把凹角误当折叠(凹角顶点也在邻面平面背后)。只对二面角 >60°
        /// 的边做场查询(约 2 万条),代价 ~1s。
        ///
        /// 塌缩安全:链接条件(两顶点公共邻点恰为被删两面的对顶点,否则
        /// 制造非流形)+ 每 pass 候选顶点不相交 + 边长 < max_edge 守卫。
        /// 返回(塌缩次数, 删除三角数)。
        pub fn repair_folds(
            &mut self,
            adf: &Adf,
            fold_cos: f64,
            _pen_margin: f64,
            sliver_h: f64,
            max_edge: f64,
            passes: usize,
        ) -> (usize, usize) {
            let mut total_collapsed = 0usize;
            let mut total_removed = 0usize;
            let hstep = (1e-3 * adf.size) as f64;
            for _p in 0..passes {
                let n_t = self.tris.len();
                let nv = self.verts.len();
                // 面法线 / 2·面积 / 质心 / 最小高度
                let mut tn = vec![[0.0f64; 3]; n_t];
                let mut ta2 = vec![0.0f64; n_t];
                let mut hc = vec![[0.0f64; 3]; n_t];
                let mut hmin = vec![f64::INFINITY; n_t];
                for (ti, t) in self.tris.iter().enumerate() {
                    let (a, b, c) = (
                        self.verts[t[0] as usize],
                        self.verts[t[1] as usize],
                        self.verts[t[2] as usize],
                    );
                    let (a, b, c) = (
                        [a[0] as f64, a[1] as f64, a[2] as f64],
                        [b[0] as f64, b[1] as f64, b[2] as f64],
                        [c[0] as f64, c[1] as f64, c[2] as f64],
                    );
                    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                    let bc = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
                    let cr = [
                        ab[1] * ac[2] - ab[2] * ac[1],
                        ab[2] * ac[0] - ab[0] * ac[2],
                        ab[0] * ac[1] - ab[1] * ac[0],
                    ];
                    let l = (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt();
                    ta2[ti] = l;
                    tn[ti] = if l > 1e-20 {
                        [cr[0] / l, cr[1] / l, cr[2] / l]
                    } else {
                        [0.0; 3]
                    };
                    hc[ti] = [(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0, (a[2] + b[2] + c[2]) / 3.0];
                    let e1 = (ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2]).sqrt();
                    let e2 = (ac[0] * ac[0] + ac[1] * ac[1] + ac[2] * ac[2]).sqrt();
                    let e3 = (bc[0] * bc[0] + bc[1] * bc[1] + bc[2] * bc[2]).sqrt();
                    let emax = e1.max(e2).max(e3);
                    hmin[ti] = if emax > 1e-12 { l / emax } else { 0.0 };
                }
                // 顶点 → 三角
                let mut v2t: Vec<Vec<u32>> = vec![Vec::new(); nv];
                for (ti, t) in self.tris.iter().enumerate() {
                    for k in 0..3 {
                        v2t[t[k] as usize].push(ti as u32);
                    }
                }
                // 边 → [tri0, tri1, 计数](计数 != 2 = 开放边或非流形边,候选跳过)
                let mut e2t: std::collections::HashMap<u64, [u32; 3], FxBuild> =
                    std::collections::HashMap::with_capacity_and_hasher(n_t * 3, FxBuild);
                for (ti, t) in self.tris.iter().enumerate() {
                    for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                        let key = if a < b {
                            ((a as u64) << 32) | b as u64
                        } else {
                            ((b as u64) << 32) | a as u64
                        };
                        let e = e2t.entry(key).or_insert([u32::MAX, u32::MAX, 0u32]);
                        if e[0] == u32::MAX {
                            e[0] = ti as u32;
                        } else if e[1] == u32::MAX {
                            e[1] = ti as u32;
                        }
                        e[2] += 1;
                    }
                }
                // 场梯度(惰性:只对候选面查询)
                let grad = |q: [f64; 3]| -> [f64; 3] {
                    let q = [q[0] as Real, q[1] as Real, q[2] as Real];
                    let d0 = adf.eval(q) as f64;
                    let mut g = [0.0f64; 3];
                    for k in 0..3 {
                        let mut qq = q;
                        qq[k] += hstep;
                        g[k] = (adf.eval(qq) as f64 - d0) / hstep;
                    }
                    g
                };
                // bad 标记:薄片(无条件)+ 折叠(法线与场梯度反向)
                let mut bad = vec![false; n_t];
                let mut n_sliver = 0usize;
                for ti in 0..n_t {
                    if hmin[ti] < sliver_h {
                        bad[ti] = true;
                        n_sliver += 1;
                    }
                }
                let mut n_fold = 0usize;
                for (&ekey, ts) in e2t.iter() {
                    if ts[2] != 2 {
                        continue;
                    }
                    let (t0, t1) = (ts[0] as usize, ts[1] as usize);
                    let d = tn[t0][0] * tn[t1][0] + tn[t0][1] * tn[t1][1] + tn[t0][2] * tn[t1][2];
                    if d >= fold_cos {
                        continue;
                    }
                    // 梯度在**共享边中点**求值:脊线处梯度 = 两面法线的角平分方向,
                    // 折叠面(法线翻过脊顶)与它必呈钝角;合法凸/凹角两面都在
                    // ±90° 锥内。质心求值会被脊顶另一侧的梯度翻转干扰(实测漏检)。
                    let (ea, eb) = ((ekey >> 32) as u32, ekey as u32);
                    let pa = self.verts[ea as usize];
                    let pb = self.verts[eb as usize];
                    let mid = [
                        (pa[0] as f64 + pb[0] as f64) * 0.5,
                        (pa[1] as f64 + pb[1] as f64) * 0.5,
                        (pa[2] as f64 + pb[2] as f64) * 0.5,
                    ];
                    let g = grad(mid);
                    for &t in &[t0, t1] {
                        if bad[t] {
                            continue;
                        }
                        if tn[t][0] * g[0] + tn[t][1] * g[1] + tn[t][2] * g[2] < 0.3 {
                            bad[t] = true;
                            n_fold += 1;
                        }
                    }
                }
                if n_fold == 0 && n_sliver == 0 {
                    break;
                }
                eprintln!(
                    "[repair] pass {}: 薄片 {n_sliver} 折叠 {n_fold}",
                    _p + 1
                );
                // 候选塌缩:bad 三角的最短边(u,v),死两面 = 共享 (u,v) 的两面
                let mut used_v = vec![false; nv];
                let mut used_t = vec![false; n_t];
                let mut uf: Vec<u32> = (0..nv as u32).collect();
                fn ufind(p: &mut [u32], mut x: u32) -> u32 {
                    while p[x as usize] != x {
                        p[x as usize] = p[p[x as usize] as usize];
                        x = p[x as usize];
                    }
                    x
                }
                let mut pos: Vec<[f64; 3]> = self
                    .verts
                    .iter()
                    .map(|v| [v[0] as f64, v[1] as f64, v[2] as f64])
                    .collect();
                let mut n_collapsed = 0usize;
                let mut merged_roots: Vec<u32> = Vec::new();
                for (&_key, &ts) in e2t.iter() {
                    if ts[2] != 2 {
                        continue;
                    }
                    let (t0, t1) = (ts[0] as usize, ts[1] as usize);
                    // 选 bad 三角(两面都 bad 时选面积小的)
                    let tb = if bad[t0] && bad[t1] {
                        if ta2[t0] <= ta2[t1] { t0 } else { t1 }
                    } else if bad[t0] {
                        t0
                    } else if bad[t1] {
                        t1
                    } else {
                        continue;
                    };
                    if used_t[tb] {
                        continue;
                    }
                    // bad 三角三边,取"两端点都未用且最短"的边(key 归一化与 e2t 一致)
                    let tri = self.tris[tb];
                    let mut best = (u64::MAX, 0u32, 0u32, f64::INFINITY);
                    for k in 0..3 {
                        let (x, y) = (tri[k], tri[(k + 1) % 3]);
                        if used_v[x as usize] || used_v[y as usize] {
                            continue;
                        }
                        let d = {
                            let dx = pos[x as usize][0] - pos[y as usize][0];
                            let dy = pos[x as usize][1] - pos[y as usize][1];
                            let dz = pos[x as usize][2] - pos[y as usize][2];
                            (dx * dx + dy * dy + dz * dz).sqrt()
                        };
                        if d < best.3 {
                            let ekey = if x < y {
                                ((x as u64) << 32) | y as u64
                            } else {
                                ((y as u64) << 32) | x as u64
                            };
                            best = (ekey, x, y, d);
                        }
                    }
                    if best.0 == u64::MAX || best.3 > max_edge {
                        continue;
                    }
                    // 共享该边的两面(即死面)
                    let (da, db) = match e2t.get(&best.0) {
                        Some(s) if s[2] == 2 => (s[0] as usize, s[1] as usize),
                        _ => continue,
                    };
                    if used_t[da] || used_t[db] {
                        continue;
                    }
                    let (u, v) = (best.1 as usize, best.2 as usize);
                    // 链接条件:u,v 公共邻点恰为死两面的对顶点
                    let apex_of = |t: usize| -> u32 {
                        let tt = self.tris[t];
                        if tt[0] != u as u32 && tt[0] != v as u32 {
                            tt[0]
                        } else if tt[1] != u as u32 && tt[1] != v as u32 {
                            tt[1]
                        } else {
                            tt[2]
                        }
                    };
                    let want = [apex_of(da), apex_of(db)];
                    let mut common: Vec<u32> = Vec::new();
                    for &ti in v2t[u].iter() {
                        let tt = self.tris[ti as usize];
                        for k in 0..3 {
                            let w = tt[k];
                            if w as usize == u || w as usize == v {
                                continue;
                            }
                            if v2t[v].iter().any(|&tj| {
                                let t2 = self.tris[tj as usize];
                                t2[0] == w || t2[1] == w || t2[2] == w
                            }) && !common.contains(&w)
                            {
                                common.push(w);
                            }
                        }
                    }
                    if common.len() != 2
                        || !common.contains(&want[0])
                        || !common.contains(&want[1])
                    {
                        continue;
                    }
                    // 1-环保留:塌缩影响 u/v 的全部邻接三角;若环内有其他候选
                    // 已用的顶点,两个塌缩会在同一 pass 内相互作用(候选校验
                    // 基于 pass 开始快照,相互作用会制造非流形边,实测 +93)。
                    // 保留整个 1-环后,本 pass 塌缩彼此完全隔离。
                    let mut ring_conflict = false;
                    for &ti in v2t[u].iter().chain(v2t[v].iter()) {
                        let tt = self.tris[ti as usize];
                        for k in 0..3 {
                            if used_v[tt[k] as usize] {
                                ring_conflict = true;
                                break;
                            }
                        }
                        if ring_conflict {
                            break;
                        }
                    }
                    if ring_conflict {
                        continue;
                    }
                    // 执行塌缩:u 并入 v(中点),死两面
                    let mid = [
                        (pos[u][0] + pos[v][0]) * 0.5,
                        (pos[u][1] + pos[v][1]) * 0.5,
                        (pos[u][2] + pos[v][2]) * 0.5,
                    ];
                    pos[v] = mid;
                    merged_roots.push(v as u32);
                    let (ru, rv) = (ufind(&mut uf, u as u32), ufind(&mut uf, v as u32));
                    uf[ru as usize] = rv;
                    for &ti in v2t[u].iter().chain(v2t[v].iter()) {
                        let tt = self.tris[ti as usize];
                        for k in 0..3 {
                            used_v[tt[k] as usize] = true;
                        }
                    }
                    used_t[da] = true;
                    used_t[db] = true;
                    n_collapsed += 1;
                }
                if n_collapsed == 0 {
                    break;
                }
                // 合并点投影回零等值面:弦中点在凸缘处必落在材料内/外,
                // 邻接三角会翘起成针刺(实测修复后出现尖齿)。Newton 投影
                // 把它拉回零面,消除塌缩自身的几何副作用。
                for &r in merged_roots.iter() {
                    let mut q = pos[r as usize];
                    for _ in 0..3 {
                        let q_r = [q[0] as Real, q[1] as Real, q[2] as Real];
                        let d0 = adf.eval(q_r) as f64;
                        if d0.abs() > max_edge {
                            break;
                        }
                        let mut g = [0.0f64; 3];
                        for k in 0..3 {
                            let mut qq = q_r;
                            qq[k] += hstep;
                            g[k] = (adf.eval(qq) as f64 - d0) / hstep;
                        }
                        let g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
                        if g2 < 0.25 {
                            break;
                        }
                        let mut st = [-d0 * g[0] / g2, -d0 * g[1] / g2, -d0 * g[2] / g2];
                        let sl = (st[0] * st[0] + st[1] * st[1] + st[2] * st[2]).sqrt();
                        if sl > max_edge {
                            let f = max_edge / sl;
                            for k in 0..3 {
                                st[k] *= f;
                            }
                        }
                        q = [q[0] + st[0], q[1] + st[1], q[2] + st[2]];
                    }
                    pos[r as usize] = q;
                }
                // 重建:跳过死面;顶点按并查集根合并,未引用顶点压缩掉
                let root = |x: u32| -> u32 {
                    let mut r = x;
                    loop {
                        let nr = uf[r as usize];
                        if nr == r {
                            break;
                        }
                        r = nr;
                    }
                    r
                };
                let mut remap: Vec<u32> = vec![u32::MAX; nv];
                let mut newv: Vec<[f32; 3]> = Vec::new();
                let mut new_tris: Vec<[u32; 3]> = Vec::with_capacity(n_t);
                let mut n_drop = 0usize;
                for t in self.tris.iter() {
                    let r = [root(t[0]), root(t[1]), root(t[2])];
                    if r[0] == r[1] || r[1] == r[2] || r[0] == r[2] {
                        n_drop += 1;
                        continue;
                    }
                    let mut nt = [0u32; 3];
                    for k in 0..3 {
                        if remap[r[k] as usize] == u32::MAX {
                            remap[r[k] as usize] = newv.len() as u32;
                            let pr = pos[r[k] as usize];
                            newv.push([pr[0] as f32, pr[1] as f32, pr[2] as f32]);
                        }
                        nt[k] = remap[r[k] as usize];
                    }
                    new_tris.push(nt);
                }
                self.tris = new_tris;
                self.verts = newv;
                total_collapsed += n_collapsed;
                total_removed += n_drop;
            }
            (total_collapsed, total_removed)
        }

        /// **QEM 自适应简化**(Garland–Heckbert 二次型误差 + 特征边平面对约束)。
        ///
        /// 平坦区域二次型误差增长慢 → 大步塌缩成大三角;叶缘/锐边处误差陡增
        /// → 三角保持细密 —— "按需调整面片大小",体积缩小而形状误差有界。
        /// 塌缩安全性:链接条件(防非流形)+ 邻面法线翻转检查(防折叠)+
        /// 误差阈值 `err2_max`(几何误差的硬上限,单位 mm²)。
        /// 特征保持:二面角 > `feature_deg` 的边,向两端点额外添加相邻两面
        /// 平面约束(标准 planar-pair),缘线顶点几乎不动。
        ///
        /// `err2_max`: 允许的平方误差上限(mm²),如 0.0004 ≈ 0.02mm 弦差;
        /// `feature_deg`: 特征边二面角阈值;`target_tris`: 0 = 纯误差驱动。
        /// 返回(塌缩次数, 简化后三角数)。
        pub fn decimate(&mut self, err2_max: f64, feature_deg: f64, target_tris: usize) -> usize {
            self.decimate_with_creases(err2_max, feature_deg, target_tris, None)
        }

        pub fn decimate_with_creases(&mut self, err2_max: f64, feature_deg: f64, target_tris: usize, creases: Option<&ReferenceCreases>) -> usize {
            self.decimate_impl(err2_max, feature_deg, target_tris, creases, None, None, true, true)
        }

        /// P37:`decimate_with_creases` 的分段并行版(大网格)。按面质心沿最长轴切成 32 段(`ADF_DECIMATE_BLOCKS`),
        /// 各段并行做 QEM(跨段顶点锁定、不清薄片、不压缩顶点);合并后只在接缝 2 环内再做一遍(其余顶点锁定),
        /// 最后统一清薄片、压缩。段数固定 ⟹ 结果与线程数无关。< 40 万三角或设了 `ADF_DECIMATE_SERIAL` 时退回串行。
        /// 与串行的差别只在接缝附近(接缝 2 环外各段内的塌缩次序与串行不同,但判据相同)。
        pub fn decimate_with_creases_parallel(&mut self, err2_max: f64, feature_deg: f64, creases: Option<&ReferenceCreases>) -> usize {
            let k: usize = std::env::var("ADF_DECIMATE_BLOCKS").ok().and_then(|v| v.parse().ok()).unwrap_or(32);
            if k < 2 || self.tris.len() < 400_000 || std::env::var_os("ADF_DECIMATE_SERIAL").is_some() {
                return self.decimate_with_creases(err2_max, feature_deg, 0, creases);
            }
            self.decimate_with_creases_blocks(err2_max, feature_deg, creases, k)
        }

        /// 分段并行 QEM 本体(无规模门槛,测试直接调用)。
        pub fn decimate_with_creases_blocks(&mut self, err2_max: f64, feature_deg: f64, creases: Option<&ReferenceCreases>, k: usize) -> usize {
            let (n, cross) = self.run_in_slabs(k.max(1), |m, lock| {
                m.decimate_impl(err2_max, feature_deg, 0, creases, Some(lock), None, false, false)
            });
            let near = self.ring_expand(&cross, 2);
            let lock: Vec<bool> = near.iter().map(|&b| !b).collect();
            n + self.decimate_impl(err2_max, feature_deg, 0, creases, Some(lock.as_slice()), Some(near.as_slice()), true, true)
        }

        /// P37:按面质心沿最长轴切成 `k` 段(各段面数相近)。返回每段 (子网格, 子网格顶点 → 全局顶点号),
        /// 以及"跨段"顶点标记(相邻面分属不止一段)。段内面保持全局升序,顶点按首次出现编号 ⟹ 确定性。
        fn split_slabs(&self, k: usize) -> (Vec<(Mesh, Vec<u32>)>, Vec<bool>) {
            let nt = self.tris.len();
            let nv = self.verts.len();
            let cen: Vec<[f64; 3]> = self
                .tris
                .iter()
                .map(|t| {
                    let p = t.map(|i| g64(self.verts[i as usize]));
                    scale(add(add(p[0], p[1]), p[2]), 1.0 / 3.0)
                })
                .collect();
            let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
            for c in &cen {
                for a in 0..3 {
                    lo[a] = lo[a].min(c[a]);
                    hi[a] = hi[a].max(c[a]);
                }
            }
            let ax = (0..3).max_by(|&x, &y| (hi[x] - lo[x]).total_cmp(&(hi[y] - lo[y]))).unwrap_or(0);
            let mut order: Vec<u32> = (0..nt as u32).collect();
            order.sort_by(|&a, &b| cen[a as usize][ax].total_cmp(&cen[b as usize][ax]).then(a.cmp(&b)));
            let mut block = vec![0u32; nt];
            for (r, &f) in order.iter().enumerate() {
                block[f as usize] = ((r as u64 * k as u64) / nt.max(1) as u64) as u32;
            }
            drop(order);
            drop(cen);
            let mut owner = vec![u32::MAX; nv];
            let mut cross = vec![false; nv];
            let mut faces: Vec<Vec<u32>> = vec![Vec::new(); k];
            for (f, t) in self.tris.iter().enumerate() {
                let b = block[f];
                faces[b as usize].push(f as u32);
                for &v in t {
                    let o = &mut owner[v as usize];
                    if *o == u32::MAX {
                        *o = b;
                    } else if *o != b {
                        cross[v as usize] = true;
                    }
                }
            }
            let mut local = vec![u32::MAX; nv];
            let mut parts = Vec::with_capacity(k);
            for fs in faces {
                let mut verts: Vec<[f32; 3]> = Vec::new();
                let mut gid: Vec<u32> = Vec::new();
                let mut tris: Vec<[u32; 3]> = Vec::with_capacity(fs.len());
                for &f in &fs {
                    let t = self.tris[f as usize];
                    let mut lt = [0u32; 3];
                    for (k2, &v) in t.iter().enumerate() {
                        if local[v as usize] == u32::MAX {
                            local[v as usize] = verts.len() as u32;
                            verts.push(self.verts[v as usize]);
                            gid.push(v);
                        }
                        lt[k2] = local[v as usize];
                    }
                    tris.push(lt);
                }
                for &g in &gid {
                    local[g as usize] = u32::MAX;
                }
                parts.push((Mesh { verts, tris }, gid));
            }
            (parts, cross)
        }

        /// P37:切成 `k` 段,在每段上并行执行 `f(子网格, 跨段顶点锁)`,再按段序合并。
        /// 约定:`f` 不得压缩、增加顶点,不得移动加锁顶点(合并按子网格顶点号映射回全局号)。
        /// 合并后被塌缩掉的顶点留作孤点,由后续带压缩的一遍清除。返回 (各段返回值之和, 跨段标记)。
        fn run_in_slabs<F>(&mut self, k: usize, f: F) -> (usize, Vec<bool>)
        where
            F: Fn(&mut Mesh, &[bool]) -> usize + Sync,
        {
            let (parts, cross) = self.split_slabs(k);
            let slots: Vec<std::sync::Mutex<Option<(Mesh, Vec<u32>, usize)>>> =
                parts.into_iter().map(|(m, g)| std::sync::Mutex::new(Some((m, g, 0)))).collect();
            let next = std::sync::atomic::AtomicUsize::new(0);
            let workers = nthreads().min(slots.len()).max(1);
            {
                let (slots, next, cross, f) = (&slots, &next, &cross, &f);
                std::thread::scope(|sc| {
                    for _ in 0..workers {
                        sc.spawn(move || loop {
                            let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if i >= slots.len() {
                                break;
                            }
                            let mut slot = slots[i].lock().expect("段锁中毒");
                            if let Some((m, g, n)) = slot.as_mut() {
                                let lock: Vec<bool> = g.iter().map(|&v| cross[v as usize]).collect();
                                *n = f(m, &lock);
                            }
                        });
                    }
                });
            }
            let mut total = 0usize;
            let mut tris = Vec::with_capacity(self.tris.len());
            for slot in slots {
                let (m, g, n) = slot.into_inner().expect("段锁中毒").expect("段结果缺失");
                assert_eq!(m.verts.len(), g.len(), "分段处理不得增删顶点");
                total += n;
                for (i, &v) in g.iter().enumerate() {
                    if !cross[v as usize] {
                        self.verts[v as usize] = m.verts[i];
                    }
                }
                tris.extend(m.tris.iter().map(|t| t.map(|i| g[i as usize])));
            }
            self.tris = tris;
            (total, cross)
        }

        /// 标记 `mark` 及其 `rings` 环邻域内的顶点。
        fn ring_expand(&self, mark: &[bool], rings: usize) -> Vec<bool> {
            let mut cur = mark.to_vec();
            for _ in 0..rings {
                let prev = cur.clone();
                for t in &self.tris {
                    if t.iter().any(|&v| prev[v as usize]) {
                        for &v in t {
                            cur[v as usize] = true;
                        }
                    }
                }
            }
            cur
        }

        /// QEM 本体。`locked`:端点被锁的边不塌缩;`seed`:初始堆只放至少一端被标记的边(塌缩后邻边照常入堆);
        /// `slivers`:末尾清薄片;`compact`:压缩掉孤立顶点(为假时顶点号不变,只删死面)。
        #[allow(clippy::too_many_arguments)]
        fn decimate_impl(
            &mut self,
            err2_max: f64,
            feature_deg: f64,
            target_tris: usize,
            creases: Option<&ReferenceCreases>,
            locked: Option<&[bool]>,
            seed: Option<&[bool]>,
            slivers: bool,
            compact: bool,
        ) -> usize {
            let mut keep = creases.map(|g| self.crease_targets(g, err2_max.sqrt().max(0.04 * g.radius)));
            // 脊线(二面角>25°)的合并上限:**与 err2 解耦**。阶梯合并代价由
            // 采样步长决定(~2.5e-3 mm² @ h≈0.1),把它绑在 err2 上意味着
            // "用户收紧弦差预算"会顺带让直角缘阶梯退不回去(实测 50×2.5e-5
            // = 1.25e-3 < 阶梯代价 → 满屏短黑线/翻折片)。默认取二者较大者。
            let ridge_thr: f64 = std::env::var("ADF_RIDGE_ERR2")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or((err2_max * 50.0).max(4.0e-3));
            let n_t0 = self.tris.len();
            let nv = self.verts.len();
            // ── 1) 面平面 + 顶点二次型 ──
            // 对称 4×4 存 10 元:[q00 q01 q02 q03 q11 q12 q13 q22 q23 q33]
            let zero_q = [0.0f64; 10];
            let mut q: Vec<[f64; 10]> = vec![zero_q; nv];
            let add_plane = |qv: &mut [f64; 10], n: [f64; 3], d: f64| {
                let p = [n[0], n[1], n[2], d];
                let mut k = 0usize;
                for r in 0..4 {
                    for c in r..4 {
                        qv[k] += p[r] * p[c];
                        k += 1;
                    }
                }
            };
            // 面法线缓存(翻转检查用)
            let mut tri_n: Vec<[f64; 3]> = vec![[0.0; 3]; n_t0];
            for (ti, t) in self.tris.iter().enumerate() {
                let (a, b, c) = (
                    self.verts[t[0] as usize],
                    self.verts[t[1] as usize],
                    self.verts[t[2] as usize],
                );
                let (a, b, c) = (
                    [a[0] as f64, a[1] as f64, a[2] as f64],
                    [b[0] as f64, b[1] as f64, b[2] as f64],
                    [c[0] as f64, c[1] as f64, c[2] as f64],
                );
                let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                let cr = [
                    ab[1] * ac[2] - ab[2] * ac[1],
                    ab[2] * ac[0] - ab[0] * ac[2],
                    ab[0] * ac[1] - ab[1] * ac[0],
                ];
                let l = (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt();
                if l < 1e-12 {
                    continue;
                }
                let n = [cr[0] / l, cr[1] / l, cr[2] / l];
                tri_n[ti] = n;
                let d = -(n[0] * a[0] + n[1] * a[1] + n[2] * a[2]);
                for &vi in t.iter() {
                    add_plane(&mut q[vi as usize], n, d);
                }
            }
            // ── 2) 特征边平面对约束 ──
            {
                let mut emap: std::collections::HashMap<u64, [u32; 3], FxBuild> =
                    std::collections::HashMap::with_capacity_and_hasher(n_t0 * 3, FxBuild);
                for (ti, t) in self.tris.iter().enumerate() {
                    for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                        let key = if a < b {
                            ((a as u64) << 32) | b as u64
                        } else {
                            ((b as u64) << 32) | a as u64
                        };
                        let e = emap.entry(key).or_insert([u32::MAX, u32::MAX, 0u32]);
                        if e[0] == u32::MAX {
                            e[0] = ti as u32;
                        } else if e[1] == u32::MAX {
                            e[1] = ti as u32;
                        }
                        e[2] += 1;
                    }
                }
                let cos_feat = (feature_deg as f64).to_radians().cos();
                for (key, ts) in emap.iter() {
                    if ts[2] != 2 {
                        continue;
                    }
                    let (t0, t1) = (ts[0] as usize, ts[1] as usize);
                    let d = tri_n[t0][0] * tri_n[t1][0]
                        + tri_n[t0][1] * tri_n[t1][1]
                        + tri_n[t0][2] * tri_n[t1][2];
                    if d >= cos_feat {
                        continue;
                    }
                    // 特征约束:相邻两面平面对(面约束重复计入)→ 两端点
                    let feat_planes = |out: &mut Vec<([f64; 3], f64)>, ti: usize| {
                        let n = tri_n[ti];
                        let t = self.tris[ti];
                        let p0 = self.verts[t[0] as usize];
                        let dd = -(n[0] * p0[0] as f64 + n[1] * p0[1] as f64 + n[2] * p0[2] as f64);
                        out.push((n, dd));
                    };
                    let mut planes = Vec::with_capacity(2);
                    feat_planes(&mut planes, t0);
                    feat_planes(&mut planes, t1);
                    let a = (key >> 32) as u32;
                    let b = (key & 0xffff_ffff) as u32;
                    for (n, dd) in planes.iter() {
                        add_plane(&mut q[a as usize], *n, *dd);
                        add_plane(&mut q[b as usize], *n, *dd);
                    }
                }
            }
            // ── 3) 顶点 → 三角邻接(CSR 不行——塌缩要动态增删,用 Vec<Vec>)──
            let mut v2t: Vec<Vec<u32>> = vec![Vec::new(); nv];
            for (ti, t) in self.tris.iter().enumerate() {
                for k in 0..3 {
                    v2t[t[k] as usize].push(ti as u32);
                }
            }
            // ── 4) 边堆(懒失效:顶点版本号)──
            let mut ver = vec![0u32; nv];
            let mut dead_tri = vec![false; n_t0];
            let quad_eval = |qv: &[f64; 10], v: [f64; 3]| -> f64 {
                let x = [v[0], v[1], v[2], 1.0];
                let mut s = 0.0;
                let mut k = 0usize;
                for r in 0..4 {
                    for c in r..4 {
                        s += qv[k] * x[r] * x[c] * if r == c { 1.0 } else { 2.0 };
                        k += 1;
                    }
                }
                s
            };
            // 4×4 线性求解(Q 对称,底行 [0 0 0 1];列主元高斯)
            let solve_pos = |qq: &[f64; 10]| -> Option<[f64; 3]> {
                // 完整 4×4
                let idx = [0usize, 1, 2, 3, 4, 5, 6, 7, 8];
                let _ = idx;
                let mut m = [
                    [qq[0], qq[1], qq[2], qq[3], 0.0],
                    [qq[1], qq[4], qq[5], qq[6], 0.0],
                    [qq[2], qq[5], qq[7], qq[8], 0.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0],
                ];
                // 高斯消元(部分主元)
                for col in 0..4 {
                    let mut piv = col;
                    for r in (col + 1)..4 {
                        if m[r][col].abs() > m[piv][col].abs() {
                            piv = r;
                        }
                    }
                    if m[piv][col].abs() < 1e-12 {
                        return None;
                    }
                    m.swap(col, piv);
                    for r in 0..4 {
                        if r != col && m[r][col].abs() > 1e-15 {
                            let f = m[r][col] / m[col][col];
                            for c in col..5 {
                                m[r][c] -= f * m[col][c];
                            }
                        }
                    }
                }
                let v = [
                    m[0][4] / m[0][0],
                    m[1][4] / m[1][1],
                    m[2][4] / m[2][2],
                ];
                if v.iter().all(|x| x.is_finite()) {
                    Some(v)
                } else {
                    None
                }
            };
            let edge_cost = |qa: &[f64; 10], qb: &[f64; 10], pa: [f32; 3], pb: [f32; 3]| -> (f64, [f64; 3]) {
                let mut qq = [0.0f64; 10];
                for k in 0..10 {
                    qq[k] = qa[k] + qb[k];
                }
                let af = [pa[0] as f64, pa[1] as f64, pa[2] as f64];
                let bf = [pb[0] as f64, pb[1] as f64, pb[2] as f64];
                let mid = [(af[0] + bf[0]) * 0.5, (af[1] + bf[1]) * 0.5, (af[2] + bf[2]) * 0.5];
                let ab = [bf[0] - af[0], bf[1] - af[1], bf[2] - af[2]];
                let lab = (ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2]).sqrt();
                // 滑动钳制:平坦方向二次型几乎无成本,求解位置会被甩出表面
                // (桌腿实测甩出 5.5mm)——钳到中点邻域 1.5×|ab|(2.5 时顶点
                // 链式迁移能漂移几十 mm,配纵横比守卫双保险),钳后重估成本
                let clamp_mid = |v: [f64; 3]| -> [f64; 3] {
                    let d = [v[0] - mid[0], v[1] - mid[1], v[2] - mid[2]];
                    let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                    let lim = 1.5 * lab;
                    if l <= lim || l < 1e-12 {
                        v
                    } else {
                        let f = lim / l;
                        [mid[0] + d[0] * f, mid[1] + d[1] * f, mid[2] + d[2] * f]
                    }
                };
                let cands = [
                    solve_pos(&qq).map(clamp_mid),
                    Some(af),
                    Some(bf),
                    Some(mid),
                ];
                let mut best = (f64::INFINITY, mid);
                for cv in cands.into_iter().flatten() {
                    let c = quad_eval(&qq, cv);
                    if c < best.0 {
                        best = (c, cv);
                    }
                }
                best
            };
            // 初始边集
            let ord = |c: f64| -> u64 { if c >= 0.0 { c.to_bits() } else { !c.to_bits() } };
            let mut heap: std::collections::BinaryHeap<std::cmp::Reverse<(u64, u32, u32, u32, u32)>> =
                std::collections::BinaryHeap::with_capacity(n_t0 * 2);
            let mut banned: std::collections::HashSet<u64, FxBuild> =
                std::collections::HashSet::with_capacity_and_hasher(1024, FxBuild);
            {
                let mut seen: std::collections::HashSet<u64, FxBuild> =
                    std::collections::HashSet::with_capacity_and_hasher(n_t0 * 3, FxBuild);
                for t in self.tris.iter() {
                    for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                        let key = if a < b {
                            ((a as u64) << 32) | b as u64
                        } else {
                            ((b as u64) << 32) | a as u64
                        };
                        if !seen.insert(key) {
                            continue;
                        }
                        if seed.is_some_and(|sd| !(sd[a as usize] || sd[b as usize])) {
                            continue;
                        }
                        let (c, _) = edge_cost(
                            &q[a as usize],
                            &q[b as usize],
                            self.verts[a as usize],
                            self.verts[b as usize],
                        );
                        heap.push(std::cmp::Reverse((ord(c), a, b, ver[a as usize], ver[b as usize])));
                    }
                }
            }
            // ── 5) 塌缩主循环 ──
            let mut n_collapsed = 0usize;
            let mut n_live = n_t0;
            while let Some(std::cmp::Reverse((cbits, a, b, va, vb))) = heap.pop() {
                let cost = f64::from_bits(if (cbits >> 63) == 1 { !cbits } else { cbits });
                // 全局 pop 上限 = 50×err2(锐边放宽档);普通边在下方按自身
                // 阈值过滤。锯齿阶梯合并代价 = 垂直抖动²(~2.5e-3)≫err2,
                // 不放宽则阶梯永不合并 → 满屏短黑线(实测均匀网格 5.7 万条)。
                if cost > ridge_thr {
                    break;
                }
                if banned.contains(&(if a < b {
                    ((a as u64) << 32) | b as u64
                } else {
                    ((b as u64) << 32) | a as u64
                })) {
                    continue;
                }
                if ver[a as usize] != va || ver[b as usize] != vb {
                    continue; // 懒失效
                }
                if locked.is_some_and(|l| l[a as usize] || l[b as usize]) {
                    continue; // P37:跨段/接缝外顶点锁定
                }
                if keep.as_ref().is_some_and(|v| v[a as usize].is_some() && v[b as usize].is_some()) { continue; }
                if n_live <= target_tris && target_tris > 0 {
                    break;
                }
                // 共享边 (a,b) 的活三角必须恰 2
                let shared: Vec<u32> = {
                    let sa = &v2t[a as usize];
                    let sb = &v2t[b as usize];
                    let (long, short) = if sa.len() < sb.len() { (sb, sa) } else { (sa, sb) };
                    short
                        .iter()
                        .filter(|ti| long.contains(ti) && !dead_tri[**ti as usize])
                        .copied()
                        .collect()
                };
                if shared.len() != 2 {
                    continue;
                }
                // ── 锐边放宽:脊线边(两面二面角 > 25°)用 50×err2 上限。
                // 合并后的残余偏移由调用方的 snap_sharp_to_rim 拉回原始缘线。
                let thr = {                    let tn = |ti: u32| -> [f64; 3] {
                        let t = self.tris[ti as usize];
                        let (p0, p1, p2) = (
                            self.verts[t[0] as usize],
                            self.verts[t[1] as usize],
                            self.verts[t[2] as usize],
                        );
                        let (p0, p1, p2) = (
                            [p0[0] as f64, p0[1] as f64, p0[2] as f64],
                            [p1[0] as f64, p1[1] as f64, p1[2] as f64],
                            [p2[0] as f64, p2[1] as f64, p2[2] as f64],
                        );
                        let ab = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
                        let ac = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
                        let cr = [
                            ab[1] * ac[2] - ab[2] * ac[1],
                            ab[2] * ac[0] - ab[0] * ac[2],
                            ab[0] * ac[1] - ab[1] * ac[0],
                        ];
                        let l = (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt();
                        if l > 1e-20 {
                            [cr[0] / l, cr[1] / l, cr[2] / l]
                        } else {
                            [0.0; 3]
                        }
                    };
                    let n0 = tn(shared[0]);
                    let n1 = tn(shared[1]);
                    let d = n0[0] * n1[0] + n0[1] * n1[1] + n0[2] * n1[2];
                    if d < (-25.0f64).to_radians().cos() {
                        ridge_thr
                    } else {
                        err2_max
                    }
                };
                if cost > thr {
                    continue;
                }
                // 链接条件:公共邻点恰为两面相对顶点
                fn nbrs_of(v2t: &[Vec<u32>], tris: &[[u32; 3]], dead: &[bool], vi: usize) -> Vec<u32> {
                    let mut s: Vec<u32> = Vec::with_capacity(8);
                    for &ti in v2t[vi].iter() {
                        if dead[ti as usize] {
                            continue;
                        }
                        let t = tris[ti as usize];
                        for k in 0..3 {
                            if t[k] as usize != vi && !s.contains(&t[k]) {
                                s.push(t[k]);
                            }
                        }
                    }
                    s
                }
                let (na, nb) = (
                    nbrs_of(&v2t, &self.tris, &dead_tri, a as usize),
                    nbrs_of(&v2t, &self.tris, &dead_tri, b as usize),
                );
                fn opp_of(tris: &[[u32; 3]], ti: u32, a: u32, b: u32) -> u32 {
                    let t = tris[ti as usize];
                    if t[0] != a && t[0] != b {
                        t[0]
                    } else if t[1] != a && t[1] != b {
                        t[1]
                    } else {
                        t[2]
                    }
                }
                let want = [opp_of(&self.tris, shared[0], a, b), opp_of(&self.tris, shared[1], a, b)];
                let common: Vec<u32> = na.iter().filter(|x| nb.contains(x)).copied().collect();
                if common.len() != 2
                    || !common.contains(&want[0])
                    || !common.contains(&want[1])
                {
                    continue;
                }
                // 最优位置
                let (_, pos) = edge_cost(
                    &q[a as usize],
                    &q[b as usize],
                    self.verts[a as usize],
                    self.verts[b as usize],
                );
                // 两端都移动到同一 f32 导出坐标。用该坐标验证两端的存留面。
                let mut pos = pos.map(|v| v as f32 as f64);
                if let Some(fixed) = keep.as_ref().and_then(|v| v[a as usize].or(v[b as usize])) {
                    pos = fixed.map(|v| v as f32 as f64);
                    if quad_eval(&q[a as usize], pos) + quad_eval(&q[b as usize], pos) > thr { continue; }
                }
                // 翻转检查 + 纵横比守卫:环绕三角(含 b→pos 代换)新法线
                // 不与旧法线反向;非脊线塌缩还要求新三角不过分细长。
                let mut flip = false;
                let mut ugly = false;
                for &ti in v2t[a as usize].iter().chain(v2t[b as usize].iter()) {
                    if dead_tri[ti as usize] || shared.contains(&ti) {
                        continue;
                    }
                    let t = self.tris[ti as usize];
                    let mut pv = [[0.0f64; 3]; 3];
                    for k in 0..3 {
                        let vi = t[k] as usize;
                        if vi == a as usize || vi == b as usize {
                            pv[k] = pos;
                        } else {
                            let p = self.verts[vi];
                            pv[k] = [p[0] as f64, p[1] as f64, p[2] as f64];
                        }
                    }
                    let ab = [pv[1][0] - pv[0][0], pv[1][1] - pv[0][1], pv[1][2] - pv[0][2]];
                    let ac = [pv[2][0] - pv[0][0], pv[2][1] - pv[0][1], pv[2][2] - pv[0][2]];
                    let cr = [
                        ab[1] * ac[2] - ab[2] * ac[1],
                        ab[2] * ac[0] - ab[0] * ac[2],
                        ab[0] * ac[1] - ab[1] * ac[0],
                    ];
                    let l = (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt();
                    if l < 1e-12 {
                        flip = true;
                        break;
                    }
                    let nw = [cr[0] / l, cr[1] / l, cr[2] / l];
                    let on = tri_n[ti as usize];
                    if nw[0] * on[0] + nw[1] * on[1] + nw[2] * on[2] < 0.2 {
                        flip = true;
                        break;
                    }
                    // 纵横比守卫:平坦区二次型简并,顶点沿面链式滑移,环三角
                    // 被拉成细长穿越三角(实测铺满整个盘面)。归一化纵横比 =
                    // 最小高度/最长边 = |叉积|/最长边²(等边 0.866);< 0.08
                    // 拒绝。注意是除以 emax²——除以 emax 得到的是绝对高度,
                    // 会把 h=0.09 网格的健康三角(高度≈0.05)全部误杀
                    // (实测:叶轮塌缩 384 万→5.3 万)。脊线边(thr 放宽档)
                    // 豁免——缘带长三角合法。
                    if thr <= err2_max {
                        let e0 = (ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2]).sqrt();
                        let e1 = (ac[0] * ac[0] + ac[1] * ac[1] + ac[2] * ac[2]).sqrt();
                        let bc = [pv[2][0] - pv[1][0], pv[2][1] - pv[1][1], pv[2][2] - pv[1][2]];
                        let e2 = (bc[0] * bc[0] + bc[1] * bc[1] + bc[2] * bc[2]).sqrt();
                        let emax = e0.max(e1).max(e2);
                        if emax > 1e-12 && l / (emax * emax) < 0.08 {
                            ugly = true;
                            break;
                        }
                    }
                }
                if flip || ugly {
                    // 拒绝塌缩:永久封禁该边(保守;邻域塌缩后仍封禁,
                    // 避免同条目反复弹出)。封禁不 bump 版本——邻边条目仍有效。
                    banned.insert(if a < b { ((a as u64) << 32) | b as u64 } else { ((b as u64) << 32) | a as u64 });
                    continue;
                }
                // ── 执行塌缩:b → a ──
                if let Some(v) = keep.as_mut() { v[a as usize] = v[a as usize].or(v[b as usize]); }
                self.verts[a as usize] = [pos[0] as f32, pos[1] as f32, pos[2] as f32];
                for k in 0..10 {
                    q[a as usize][k] += q[b as usize][k];
                }
                ver[a as usize] += 1;
                ver[b as usize] += 1;
                let mut b_tris = std::mem::take(&mut v2t[b as usize]);
                for ti in b_tris.iter() {
                    let t = &mut self.tris[*ti as usize];
                    for k in 0..3 {
                        if t[k] == b {
                            t[k] = a;
                        }
                    }
                }
                for ti in b_tris.iter() {
                    if self.tris[*ti as usize][0] == self.tris[*ti as usize][1]
                        || self.tris[*ti as usize][1] == self.tris[*ti as usize][2]
                        || self.tris[*ti as usize][0] == self.tris[*ti as usize][2]
                    {
                        dead_tri[*ti as usize] = true;
                        n_live -= 1;
                    }
                }
                // 邻接更新:b 的活三角移入 a;死三角从所有端点移除
                let mut moved: Vec<u32> = Vec::with_capacity(b_tris.len());
                for ti in b_tris.iter() {
                    if !dead_tri[*ti as usize] {
                        moved.push(*ti);
                    }
                }
                let dead_of_a: Vec<u32> = {
                    let lst = &v2t[a as usize];
                    lst.iter().filter(|&ti| dead_tri[*ti as usize]).copied().collect()
                };
                for ti in dead_of_a.iter() {
                    // 从其余两端点的邻接中移除
                    let t = self.tris[*ti as usize];
                    for k in 0..3 {
                        let vi = t[k] as usize;
                        if vi != a as usize {
                            v2t[vi].retain(|&x| x != *ti);
                        }
                    }
                }
                v2t[a as usize].retain(|&x| !dead_tri[x as usize]);
                for ti in moved.iter() {
                    let t = self.tris[*ti as usize];
                    for k in 0..3 {
                        let vi = t[k] as usize;
                        if vi != a as usize && !v2t[vi].contains(ti) {
                            v2t[vi].push(*ti);
                        }
                    }
                    if !v2t[a as usize].contains(ti) {
                        v2t[a as usize].push(*ti);
                    }
                }
                b_tris.clear();
                n_collapsed += 1;
                // 周边边重新入堆
                for &w in nbrs_of(&v2t, &self.tris, &dead_tri, a as usize).iter() {
                    if w == a {
                        continue;
                    }
                    let (c, _) = edge_cost(
                        &q[a as usize],
                        &q[w as usize],
                        self.verts[a as usize],
                        self.verts[w as usize],
                    );
                    heap.push(std::cmp::Reverse((
                        ord(c),
                        a,
                        w,
                        ver[a as usize],
                        ver[w as usize],
                    )));
                }
            }
            // ── 6) 压缩 ──
            if !compact {
                // P37 分段模式:顶点号不变,只删死面与退化面
                let mut newt: Vec<[u32; 3]> = Vec::with_capacity(n_live);
                for (ti, t) in self.tris.iter().enumerate() {
                    if dead_tri[ti] || t[0] == t[1] || t[1] == t[2] || t[0] == t[2] {
                        continue;
                    }
                    newt.push(*t);
                }
                self.tris = newt;
                return n_collapsed;
            }
            let mut remap: Vec<u32> = vec![u32::MAX; nv];
            let mut newv: Vec<[f32; 3]> = Vec::new();
            let mut newt: Vec<[u32; 3]> = Vec::with_capacity(n_live);
            for (ti, t) in self.tris.iter().enumerate() {
                if !dead_tri.is_empty() && dead_tri[ti] {
                    continue;
                }
                let mut nt = [0u32; 3];
                let mut ok = true;
                for k in 0..3 {
                    let vi = t[k] as usize;
                    if remap[vi] == u32::MAX {
                        remap[vi] = newv.len() as u32;
                        newv.push(self.verts[vi]);
                    }
                    nt[k] = remap[vi];
                }
                if nt[0] == nt[1] || nt[1] == nt[2] || nt[0] == nt[2] {
                    ok = false;
                }
                if ok {
                    newt.push(nt);
                }
            }
            self.tris = newt;
            self.verts = newv;
            // ── 7) 薄片收尾清理:塌缩链会挤出新薄片(高度≈0,法线随机),
            // f32 导出后部分退化为纯线段——查看器把它们画成表面短黑线
            // (实测 9 个 f32 零面积 + 2733 个高度<0.005mm)。按最短边塌缩。
            if slivers {
                n_collapsed += self.collapse_slivers(0.02, 6);
            }
            n_collapsed
        }

        /// **薄片清理**:最小高度 = 2·面积/最长边 < `sliver_h` 的针状三角,
        /// 塌缩其最短边消除(合并两点→两面退化删除,无洞)。带链接条件、
        /// 翻转拒绝与 1-环保留(与 `repair_folds` 同一套安全机制,但不依赖场)。
        /// 返回塌缩次数。
        pub fn collapse_slivers(&mut self, sliver_h: f64, passes: usize) -> usize {
            self.collapse_slivers_ex(sliver_h, passes, 0.0, 0.0)
        }

        /// 扩展版:额外按**归一化纵横比**(最小高度/最长边 = 2·面积/最长边²)
        /// 塌缩"大而扁"的三角(repair2 的折叠塌缩无形状守卫,会在平面挤出
        /// 4mm×0.07mm 细长条——绝对高度判据 hmin<0.02 抓不到它们)。
        /// aspect_lim<=0 或 min_area2<=0 时关闭该判据。
        pub fn collapse_slivers_ex(
            &mut self,
            sliver_h: f64,
            passes: usize,
            aspect_lim: f64,
            min_area2: f64,
        ) -> usize {
            let mut total = 0usize;
            for _p in 0..passes {
                let n_t = self.tris.len();
                let nv = self.verts.len();
                // 高度 + 最短边(黑线 = 退化三角的短边)
                let mut hmin = vec![f64::INFINITY; n_t];
                let mut emin = vec![f64::INFINITY; n_t];
                let mut emax_v = vec![0.0f64; n_t];
                let mut area2_v = vec![0.0f64; n_t];
                for (ti, t) in self.tris.iter().enumerate() {
                    let (a, b, c) = (
                        self.verts[t[0] as usize],
                        self.verts[t[1] as usize],
                        self.verts[t[2] as usize],
                    );
                    let (a, b, c) = (
                        [a[0] as f64, a[1] as f64, a[2] as f64],
                        [b[0] as f64, b[1] as f64, b[2] as f64],
                        [c[0] as f64, c[1] as f64, c[2] as f64],
                    );
                    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                    let bc = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
                    let cr = [
                        ab[1] * ac[2] - ab[2] * ac[1],
                        ab[2] * ac[0] - ab[0] * ac[2],
                        ab[0] * ac[1] - ab[1] * ac[0],
                    ];
                    let l = (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt();
                    let e1 = (ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2]).sqrt();
                    let e2 = (ac[0] * ac[0] + ac[1] * ac[1] + ac[2] * ac[2]).sqrt();
                    let e3 = (bc[0] * bc[0] + bc[1] * bc[1] + bc[2] * bc[2]).sqrt();
                    let emax = e1.max(e2).max(e3);
                    hmin[ti] = if emax > 1e-12 { l / emax } else { 0.0 };
                    emin[ti] = e1.min(e2).min(e3);
                    emax_v[ti] = emax;
                    area2_v[ti] = l;
                }
                // 邻接与边表(带计数)
                let mut v2t: Vec<Vec<u32>> = vec![Vec::new(); nv];
                for (ti, t) in self.tris.iter().enumerate() {
                    for k in 0..3 {
                        v2t[t[k] as usize].push(ti as u32);
                    }
                }
                let mut e2t: std::collections::HashMap<u64, [u32; 3], FxBuild> =
                    std::collections::HashMap::with_capacity_and_hasher(n_t * 3, FxBuild);
                for (ti, t) in self.tris.iter().enumerate() {
                    for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                        let key = if a < b {
                            ((a as u64) << 32) | b as u64
                        } else {
                            ((b as u64) << 32) | a as u64
                        };
                        let e = e2t.entry(key).or_insert([u32::MAX, u32::MAX, 0u32]);
                        if e[0] == u32::MAX {
                            e[0] = ti as u32;
                        } else if e[1] == u32::MAX {
                            e[1] = ti as u32;
                        }
                        e[2] += 1;
                    }
                }
                // 面法线(翻转检查用)
                let mut tn = vec![[0.0f64; 3]; n_t];
                for (ti, t) in self.tris.iter().enumerate() {
                    let (a, b, c) = (
                        self.verts[t[0] as usize],
                        self.verts[t[1] as usize],
                        self.verts[t[2] as usize],
                    );
                    let (a, b, c) = (
                        [a[0] as f64, a[1] as f64, a[2] as f64],
                        [b[0] as f64, b[1] as f64, b[2] as f64],
                        [c[0] as f64, c[1] as f64, c[2] as f64],
                    );
                    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                    let cr = [
                        ab[1] * ac[2] - ab[2] * ac[1],
                        ab[2] * ac[0] - ab[0] * ac[2],
                        ab[0] * ac[1] - ab[1] * ac[0],
                    ];
                    let l = (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt();
                    tn[ti] = if l > 1e-20 {
                        [cr[0] / l, cr[1] / l, cr[2] / l]
                    } else {
                        [0.0; 3]
                    };
                }
                // 候选:薄片三角的最短边塌缩(快照非冲突)
                let mut used_v = vec![false; nv];
                let mut used_t = vec![false; n_t];
                let mut uf: Vec<u32> = (0..nv as u32).collect();
                fn ufind2(p: &mut [u32], mut x: u32) -> u32 {
                    while p[x as usize] != x {
                        p[x as usize] = p[p[x as usize] as usize];
                        x = p[x as usize];
                    }
                    x
                }
                let mut pos: Vec<[f64; 3]> = self
                    .verts
                    .iter()
                    .map(|v| [v[0] as f64, v[1] as f64, v[2] as f64])
                    .collect();
                let mut n_collapsed = 0usize;
                for (ti, _) in self.tris.iter().enumerate() {
                    let fat = aspect_lim > 0.0
                        && hmin[ti] / emax_v[ti].max(1e-12) < aspect_lim
                        && area2_v[ti] > min_area2;
                    if used_t[ti] || (!fat && hmin[ti] >= sliver_h && emin[ti] >= 0.02) {
                        continue;
                    }
                    let tri = self.tris[ti];
                    // 最短边(两端点未用)
                    let mut best = (u64::MAX, 0u32, 0u32, f64::INFINITY);
                    for k in 0..3 {
                        let (x, y) = (tri[k], tri[(k + 1) % 3]);
                        if used_v[x as usize] || used_v[y as usize] {
                            continue;
                        }
                        let d = {
                            let dx = pos[x as usize][0] - pos[y as usize][0];
                            let dy = pos[x as usize][1] - pos[y as usize][1];
                            let dz = pos[x as usize][2] - pos[y as usize][2];
                            (dx * dx + dy * dy + dz * dz).sqrt()
                        };
                        if d < best.3 {
                            let ekey = if x < y {
                                ((x as u64) << 32) | y as u64
                            } else {
                                ((y as u64) << 32) | x as u64
                            };
                            best = (ekey, x, y, d);
                        }
                    }
                    if best.0 == u64::MAX {
                        continue;
                    }
                    let (da, db) = match e2t.get(&best.0) {
                        Some(s) if s[2] == 2 => (s[0] as usize, s[1] as usize),
                        _ => continue,
                    };
                    if used_t[da] || used_t[db] {
                        continue;
                    }
                    let (u, v) = (best.1 as usize, best.2 as usize);
                    // 链接条件
                    fn nbrs2(v2t: &[Vec<u32>], tris: &[[u32; 3]], dead: &[bool], vi: usize) -> Vec<u32> {
                        let mut s: Vec<u32> = Vec::with_capacity(8);
                        for &tj in v2t[vi].iter() {
                            if dead[tj as usize] {
                                continue;
                            }
                            let t = tris[tj as usize];
                            for k in 0..3 {
                                if t[k] as usize != vi && !s.contains(&t[k]) {
                                    s.push(t[k]);
                                }
                            }
                        }
                        s
                    }
                    let (na, nb) = (
                        nbrs2(&v2t, &self.tris, &used_t, u),
                        nbrs2(&v2t, &self.tris, &used_t, v),
                    );
                    fn opp2(tris: &[[u32; 3]], ti: usize, u: usize, v: usize) -> u32 {
                        let t = tris[ti];
                        if t[0] as usize != u && t[0] as usize != v {
                            t[0]
                        } else if t[1] as usize != u && t[1] as usize != v {
                            t[1]
                        } else {
                            t[2]
                        }
                    }
                    let want = [opp2(&self.tris, da, u, v), opp2(&self.tris, db, u, v)];
                    let common: Vec<u32> = na.iter().filter(|x| nb.contains(x)).copied().collect();
                    if common.len() != 2
                        || !common.contains(&want[0])
                        || !common.contains(&want[1])
                    {
                        continue;
                    }
                    // 翻转检查(存留三角)
                    let mid = [
                        (pos[u][0] + pos[v][0]) * 0.5,
                        (pos[u][1] + pos[v][1]) * 0.5,
                        (pos[u][2] + pos[v][2]) * 0.5,
                    ];
                    let mut flip = false;
                    for &tj in v2t[u].iter().chain(v2t[v].iter()) {
                        if used_t[tj as usize] || tj as usize == da || tj as usize == db {
                            continue;
                        }
                        let t = self.tris[tj as usize];
                        let mut pv = [[0.0f64; 3]; 3];
                        for k in 0..3 {
                            let vi2 = t[k] as usize;
                            pv[k] = if vi2 == u || vi2 == v {
                                mid
                            } else {
                                pos[vi2]
                            };
                        }
                        let ab = [pv[1][0] - pv[0][0], pv[1][1] - pv[0][1], pv[1][2] - pv[0][2]];
                        let ac = [pv[2][0] - pv[0][0], pv[2][1] - pv[0][1], pv[2][2] - pv[0][2]];
                        let cr = [
                            ab[1] * ac[2] - ab[2] * ac[1],
                            ab[2] * ac[0] - ab[0] * ac[2],
                            ab[0] * ac[1] - ab[1] * ac[0],
                        ];
                        let l = (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt();
                        if l < 1e-12 {
                            flip = true;
                            break;
                        }
                        let nn = [cr[0] / l, cr[1] / l, cr[2] / l];
                        let on = tn[tj as usize];
                        if nn[0] * on[0] + nn[1] * on[1] + nn[2] * on[2] < 0.2 {
                            flip = true;
                            break;
                        }
                    }
                    if flip {
                        used_t[ti] = true; // 本 pass 放弃该薄片
                        continue;
                    }
                    // 1-环保留
                    let mut ring_conflict = false;
                    for &tj in v2t[u].iter().chain(v2t[v].iter()) {
                        let tt = self.tris[tj as usize];
                        for k in 0..3 {
                            if used_v[tt[k] as usize] {
                                ring_conflict = true;
                                break;
                            }
                        }
                        if ring_conflict {
                            break;
                        }
                    }
                    if ring_conflict {
                        continue;
                    }
                    // 执行
                    pos[v] = mid;
                    let (ru, rv) = (ufind2(&mut uf, u as u32), ufind2(&mut uf, v as u32));
                    uf[ru as usize] = rv;
                    for &tj in v2t[u].iter().chain(v2t[v].iter()) {
                        let tt = self.tris[tj as usize];
                        for k in 0..3 {
                            used_v[tt[k] as usize] = true;
                        }
                    }
                    used_t[da] = true;
                    used_t[db] = true;
                    n_collapsed += 1;
                }
                if n_collapsed == 0 {
                    break;
                }
                // 重建压缩
                let root = |mut r: u32| -> u32 {
                    while uf[r as usize] != r {
                        r = uf[r as usize];
                    }
                    r
                };
                let mut remap: Vec<u32> = vec![u32::MAX; nv];
                let mut newv: Vec<[f32; 3]> = Vec::new();
                let mut newt: Vec<[u32; 3]> = Vec::with_capacity(n_t);
                for t in self.tris.iter() {
                    let r = [root(t[0]), root(t[1]), root(t[2])];
                    if r[0] == r[1] || r[1] == r[2] || r[0] == r[2] {
                        continue;
                    }
                    let mut nt = [0u32; 3];
                    for k in 0..3 {
                        if remap[r[k] as usize] == u32::MAX {
                            remap[r[k] as usize] = newv.len() as u32;
                            let pr = pos[r[k] as usize];
                            newv.push([pr[0] as f32, pr[1] as f32, pr[2] as f32]);
                        }
                        nt[k] = remap[r[k] as usize];
                    }
                    newt.push(nt);
                }
                self.tris = newt;
                self.verts = newv;
                total += n_collapsed;
            }
            total
        }

        /// 丢弃**碎屑分量**:三角数 < `min_tris`,**或**体积 < `min_vol_frac` × 总体积。
        /// 这类分量是场里孤立的小符号岛,在查看器里表现为浮在表面的斑点;切片软件
        /// 通常也直接忽略。用"相对体积"而不只看三角数,是因为三角数与网格密度挂钩
        /// (本工程 2×h_min 提取时,1mm³ 只有几十个三角)。
        /// 返回(丢弃的分量数, 丢弃的三角数)。
        pub fn drop_small_components(&mut self, min_tris: usize, min_vol_frac: f64) -> (usize, usize) {
            let n = self.tris.len();
            // P40:CSR 边表 + 数组计数代替 HashMap。旧版:每条边首次出现记下面,后续出现与首面合并 ——
            // 这里对每条边的出现序列同样"全部并入首面",连通分量划分相同;体积按面序累加(与旧版同序),
            // 判定与返回值逐位一致。
            let rows = edge_rows(&self.tris, self.verts.len());
            let mut par: Vec<u32> = (0..n as u32).collect();
            fn find(p: &mut [u32], mut x: u32) -> u32 {
                while p[x as usize] != x {
                    p[x as usize] = p[p[x as usize] as usize];
                    x = p[x as usize];
                }
                x
            }
            for r in 0..self.verts.len() {
                let row = &rows.ent[rows.start[r] as usize..rows.start[r + 1] as usize];
                let mut a = 0usize;
                while a < row.len() {
                    let mut b = a + 1;
                    while b < row.len() && row[b].0 == row[a].0 {
                        let x = find(&mut par, row[b].1 / 3);
                        let y = find(&mut par, row[a].1 / 3);
                        if x != y {
                            par[y as usize] = x;
                        }
                        b += 1;
                    }
                    a = b;
                }
            }
            drop(rows);
            let face_vol: Vec<f64> = {
                let (tris, verts) = (&self.tris, &self.verts);
                let chunk = n.div_ceil(nthreads()).max(1);
                std::thread::scope(|scope| {
                    let hs: Vec<_> = tris
                        .chunks(chunk)
                        .map(|part| {
                            scope.spawn(move || {
                                part.iter()
                                    .map(|t| {
                                        let (a, b, c) = (verts[t[0] as usize], verts[t[1] as usize], verts[t[2] as usize]);
                                        let (ax, ay, az) = (a[0] as f64, a[1] as f64, a[2] as f64);
                                        let (bx, by, bz) = (b[0] as f64, b[1] as f64, b[2] as f64);
                                        let (cx, cy, cz) = (c[0] as f64, c[1] as f64, c[2] as f64);
                                        (ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)) / 6.0
                                    })
                                    .collect::<Vec<f64>>()
                            })
                        })
                        .collect();
                    hs.into_iter().flat_map(|h| h.join().expect("体积线程 panic")).collect()
                })
            };
            let root: Vec<u32> = (0..n as u32).map(|f| find(&mut par, f)).collect();
            drop(par);
            let mut count = vec![0u32; n];
            let mut vols = vec![0.0f64; n];
            let mut total_vol = 0.0f64;
            for f in 0..n {
                let r = root[f] as usize;
                count[r] += 1;
                vols[r] += face_vol[f];
                total_vol += face_vol[f];
            }
            let total_vol = total_vol.abs();
            let mut drop_face = vec![false; n];
            let mut dropped_root = vec![false; n];
            let (mut nd, mut nt) = (0usize, 0usize);
            for f in 0..n {
                let r = root[f] as usize;
                if (count[r] as usize) < min_tris || vols[r].abs() < min_vol_frac * total_vol {
                    drop_face[f] = true;
                    nt += 1;
                    if !dropped_root[r] {
                        dropped_root[r] = true;
                        nd += 1;
                    }
                }
            }
            if nt > 0 {
                let mut i = 0usize;
                self.tris.retain(|_| {
                    let keep = !drop_face[i];
                    i += 1;
                    keep
                });
            } else {
                nd = 0;
            }
            (nd, nt)
        }

        /// 拓扑体检:返回(开放边数, 非流形边数, 连通分量数)。
        /// **开放边** = 只被 1 个三角使用的边,即"洞"的边界 —— STL 打印/网格修复
        /// 关注的正是这个数:闭合实体应为 0。非流形边 >2 个三角同边,同样不可打印。
        pub fn edge_stats(&self) -> (u64, u64, u32) {
            let mut cnt: std::collections::HashMap<(u32, u32), u32> = std::collections::HashMap::new();
            for t in &self.tris {
                for e in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                    let k = if e.0 < e.1 { e } else { (e.1, e.0) };
                    *cnt.entry(k).or_insert(0) += 1;
                }
            }
            let (mut open, mut nonman) = (0u64, 0u64);
            for (_, c) in cnt.iter() {
                if *c == 1 {
                    open += 1;
                } else if *c > 2 {
                    nonman += 1;
                }
            }
            // 并查集数连通分量(顶点共享即连通)。只统计**被三角引用**的顶点 ——
            // 清理碎屑后残留的孤立顶点不是分量(否则 928k 顶点里未用到的每个都算一个,
            // 实测把 10 个分量报成 541 个)。
            let n = self.verts.len();
            let mut par: Vec<u32> = (0..n as u32).collect();
            fn find(p: &mut [u32], mut x: u32) -> u32 {
                while p[x as usize] != x {
                    p[x as usize] = p[p[x as usize] as usize];
                    x = p[x as usize];
                }
                x
            }
            for t in &self.tris {
                let a = find(&mut par, t[0]);
                for &b in &t[1..] {
                    let rb = find(&mut par, b);
                    if a != rb {
                        par[rb as usize] = a;
                    }
                }
            }
            let mut roots = std::collections::HashSet::new();
            for t in &self.tris {
                for &v in t.iter() {
                    roots.insert(find(&mut par, v));
                }
            }
            (open, nonman, roots.len() as u32)
        }

        /// 有符号体积(闭合网格 + 一致绕向时 ≈ 材料体积;用于自检绕向)。
        pub fn signed_volume(&self) -> f64 {
            let mut v = 0.0f64;
            for t in &self.tris {
                let (a, b, c) =
                    (g64(self.verts[t[0] as usize]), g64(self.verts[t[1] as usize]), g64(self.verts[t[2] as usize]));
                v += dot3(a, cross3(b, c)) / 6.0;
            }
            v
        }
    }

    /// 立方体 12 条棱(角点索引对,约定同 Node 角点位)。
    const EDGES: [(usize, usize); 12] = [
        (0, 1), (2, 3), (4, 5), (6, 7), // x 向
        (0, 2), (1, 3), (4, 6), (5, 7), // y 向
        (0, 4), (1, 5), (2, 6), (3, 7), // z 向
    ];

    /// 立方体 6 个面:**环序角点** + 对应棱(`e[i]` 连接 `c[i]` 与 `c[i+1]`)。
    /// 用于"一片表面"的面内连通判定:面的 4 个角点值只有共享它的两格完全一致,
    /// 因此在此面内做的连通决策跨格天然一致(见 `extract_uniform` 步骤 2)。
    const FACES: [([usize; 4], [usize; 4]); 6] = [
        ([0, 2, 6, 4], [4, 10, 6, 8]),  // x = 0
        ([1, 3, 7, 5], [5, 11, 7, 9]),  // x = 1
        ([0, 1, 5, 4], [0, 9, 2, 8]),   // y = 0
        ([2, 3, 7, 6], [1, 11, 3, 10]), // y = 1
        ([0, 1, 3, 2], [0, 5, 1, 4]),   // z = 0
        ([4, 5, 7, 6], [2, 7, 3, 6]),   // z = 1
    ];

    /// quad 按需翻转:使三角形法向朝 want 方向(消除绕向猜测)。
    fn push_quad(verts: &[[f32; 3]], q: [u32; 4], want: [f64; 3], tris: &mut Vec<[u32; 3]>) {
        let (v0, v1, v2) = (g64(verts[q[0] as usize]), g64(verts[q[1] as usize]), g64(verts[q[2] as usize]));
        let nq = cross3(sub(v1, v0), sub(v2, v0));
        if dot3(nq, want) >= 0.0 {
            tris.push([q[0], q[1], q[2]]);
            tris.push([q[0], q[2], q[3]]);
        } else {
            tris.push([q[0], q[2], q[1]]);
            tris.push([q[0], q[3], q[2]]);
        }
    }

    /// 单向棱连接的一个 k 切片(k ∈ [klo,khi)),产出 quad 到 out。
    /// 与串行版逐字对齐:同一坐标、同一 quad 顶点序、同一 want 方向。
    /// 三个方向共用,区别仅在索引表达式(见 axis 分支)。
    #[allow(clippy::too_many_arguments)]
    fn quads_slab(
        axis: usize,
        klo: usize,
        khi: usize,
        shape: [usize; 3],
        vals: &[f32],
        cell_vert: &[u32],
        cell_pack: &[u64],
        verts: &[[f32; 3]],
        out: &mut Vec<[u32; 3]>,
    ) {
        let [nx, ny, _] = shape;
        let (dx, dy) = (nx + 1, ny + 1);
        // 取 (cell, 该 cell 内的棱号) 对应的顶点:4 bit 槽号 → 槽内顶点号。
        // 棱号直接取自"这条网格棱在该格内是哪条棱",与端点符号无关 ⟹ 四邻格
        // 各自查到的都是"包含该交叉点的那片表面",不会再跨片错接。
        let vid = |i: usize, j: usize, k: usize, e: usize| -> u32 {
            let c = (k * ny + j) * nx + i;
            let s = (cell_pack[c] >> (4 * e)) & 0xF;
            if s == 0 {
                u32::MAX
            } else {
                cell_vert[c] + (s as u32 - 1)
            }
        };
        // 一条网格棱的 4 个邻 cell,以及该棱在各自格内的**局部棱号**(轴固定):
        // 局部棱号 = 4·轴 + 该轴上的位置(0..3),与端点谁负无关。
        for k in klo..khi {
            match axis {
                0 => {
                    for j in 1..ny {
                        for i in 0..nx {
                            let a = vals[(k * dy + j) * dx + i];
                            let b = vals[(k * dy + j) * dx + i + 1];
                            if (a < 0.0) == (b < 0.0) {
                                continue;
                            }
                            let le: [usize; 4] = [0, 1, 3, 2];
                            let q = [
                                vid(i, j, k, le[0]),
                                vid(i, j - 1, k, le[1]),
                                vid(i, j - 1, k - 1, le[2]),
                                vid(i, j, k - 1, le[3]),
                            ];
                            if q.iter().any(|v| *v == u32::MAX) {
                                continue;
                            }
                            let want = if a < 0.0 { [1.0, 0.0, 0.0] } else { [-1.0, 0.0, 0.0] };
                            push_quad(verts, q, want, out);
                        }
                    }
                }
                1 => {
                    for j in 0..ny {
                        for i in 1..nx {
                            let a = vals[(k * dy + j) * dx + i];
                            let b = vals[(k * dy + j + 1) * dx + i];
                            if (a < 0.0) == (b < 0.0) {
                                continue;
                            }
                            let le: [usize; 4] = [4, 5, 7, 6];
                            let q = [
                                vid(i, j, k, le[0]),
                                vid(i - 1, j, k, le[1]),
                                vid(i - 1, j, k - 1, le[2]),
                                vid(i, j, k - 1, le[3]),
                            ];
                            if q.iter().any(|v| *v == u32::MAX) {
                                continue;
                            }
                            let want = if a < 0.0 { [0.0, 1.0, 0.0] } else { [0.0, -1.0, 0.0] };
                            push_quad(verts, q, want, out);
                        }
                    }
                }
                _ => {
                    for j in 1..ny {
                        for i in 1..nx {
                            let a = vals[(k * dy + j) * dx + i];
                            let b = vals[((k + 1) * dy + j) * dx + i];
                            if (a < 0.0) == (b < 0.0) {
                                continue;
                            }
                            let le: [usize; 4] = [8, 9, 11, 10];
                            let q = [
                                vid(i, j, k, le[0]),
                                vid(i - 1, j, k, le[1]),
                                vid(i - 1, j - 1, k, le[2]),
                                vid(i, j - 1, k, le[3]),
                            ];
                            if q.iter().any(|v| *v == u32::MAX) {
                                continue;
                            }
                            let want = if a < 0.0 { [0.0, 0.0, 1.0] } else { [0.0, 0.0, -1.0] };
                            push_quad(verts, q, want, out);
                        }
                    }
                }
            }
        }
    }

    // 三线性插值在叶内是角值的凸组合，全非负叶不可能产生负样本。
    // 在提取格坐标中包住可能含材料的叶，并留一个完整正值网格层供四边形装配。
    // 小于四格的内部节点直接整体纳入，免去遍历数百万细叶。
    fn material_region(adf: &Adf, n: usize) -> Option<([usize; 3], [usize; 3])> {
        let mut lower = [n; 3];
        let mut upper = [0; 3];
        let mut stack = vec![(adf.root, [0.0; 3], n as Real)];
        while let Some((id, origin, width)) = stack.pop() {
            match &adf.nodes[id as usize] {
                Node::Leaf(corners) if corners.iter().all(|&v| v >= 0.0) => continue,
                Node::Internal(children) if width > 4.0 => {
                    for (i, &child) in children.iter().enumerate() {
                        stack.push((child, child_origin(origin, width, i), width * 0.5));
                    }
                    continue;
                }
                _ => {}
            }
            for a in 0..3 {
                lower[a] = lower[a].min((origin[a].floor() as usize).saturating_sub(1));
                upper[a] = upper[a].max(((origin[a] + width).ceil() as usize).saturating_add(1).min(n));
            }
        }
        if lower[0] >= upper[0] { return None; }
        Some((lower, std::array::from_fn(|a| upper[a] - lower[a])))
    }

    /// 均匀重采样 + Surface Nets 提取。h:重采样格宽(建议 h_min 或其 2 倍)。
    pub fn extract_uniform(adf: &Adf, h: Real) -> Mesh {
        extract_with_reference(adf, h, None)
    }

    /// 未切削 STL 重建时，用源三角面与变号格边的交点校准切平面。
    /// 对切削场或解析毛坯照常提取 ADF，避免把新加工面贴回毛坯。
    pub fn extract_mesh_roundtrip(adf: &Adf, h: Real) -> Mesh {
        let reference = match &adf.field.base {
            BaseField::Mesh(mesh) if adf.field.cuts.is_empty() && mesh.has_closed_surface() => Some(mesh),
            _ => None,
        };
        extract_with_reference(adf, h, reference)
    }

    fn extract_with_reference(adf: &Adf, h: Real, reference: Option<&meshfield::MeshField>) -> Mesh {
        let n = ((adf.size / h).round() as usize).max(1);
        let Some((offset, shape)) = material_region(adf, n) else {
            return Mesh { verts: Vec::new(), tris: Vec::new() };
        };
        eprintln!("[nets.grid] 全域 {n}³ 格，提取 {} × {} × {} 格", shape[0], shape[1], shape[2]);
        extract_region(adf, n, offset, shape, reference)
    }

    // 正定 QEF 在交点重心到无约束解的线段上单调下降。取与格壁相交的
    // 可行线段，微量回退使相邻格的顶点仍分开。
    fn limit_qef_to_cell(mass: [Real; 3], optimum: [Real; 3], origin: [Real; 3], step: Real) -> [Real; 3] {
        let delta = sub(optimum, mass);
        let mut limit = 1.0_f64;
        for k in 0..3 {
            if delta[k] > 0.0 {
                limit = limit.min((origin[k] + step - mass[k]) / delta[k]);
            } else if delta[k] < 0.0 {
                limit = limit.min((origin[k] - mass[k]) / delta[k]);
            }
        }
        add(mass, scale(delta, limit.max(0.0) * 0.999))
    }

    /// P39:把 k ∈ [k0, k1) 层格点的 ADF 值按叶写入 `out`(布局 (k−k0)·plane + j·dx + i)。
    /// `qs[a]` 是 a 轴格点坐标(已按 eval 的方式钳位,单调不减)。节点 = 格点集合
    /// {q : lo ≤ q < hi}(逐轴),lo/hi 是祖先的分割阈值 o+s/2 —— 与 eval 下探时的判据相同,
    /// 因此每个格点恰好落在 eval 会走到的那片叶里。
    fn raster_leaves(adf: &Adf, qs: &[Vec<Real>; 3], k0: usize, k1: usize, dx: usize, plane: usize, out: &mut [f32]) {
        if k0 >= k1 {
            return;
        }
        struct Frame {
            idx: u32,
            o: [Real; 3],
            s: Real,
            lo: [Real; 3],
            hi: [Real; 3],
        }
        let range = |a: usize, lo: Real, hi: Real| -> (usize, usize) {
            let q = &qs[a];
            (q.partition_point(|&v| v < lo), q.partition_point(|&v| v < hi))
        };
        let mut stack = vec![Frame {
            idx: adf.root,
            o: adf.min,
            s: adf.size,
            lo: [Real::NEG_INFINITY; 3],
            hi: [Real::INFINITY; 3],
        }];
        while let Some(f) = stack.pop() {
            let (xs, xe) = range(0, f.lo[0], f.hi[0]);
            let (ys, ye) = range(1, f.lo[1], f.hi[1]);
            let (zs, ze) = range(2, f.lo[2], f.hi[2]);
            let (zs, ze) = (zs.max(k0), ze.min(k1));
            if xs >= xe || ys >= ye || zs >= ze {
                continue;
            }
            match &adf.nodes[f.idx as usize] {
                Node::Leaf(c) => {
                    for k in zs..ze {
                        let uz = (qs[2][k] - f.o[2]) / f.s;
                        for j in ys..ye {
                            let uy = (qs[1][j] - f.o[1]) / f.s;
                            let row = (k - k0) * plane + j * dx;
                            for i in xs..xe {
                                let ux = (qs[0][i] - f.o[0]) / f.s;
                                out[row + i] = trilinear([ux, uy, uz], c);
                            }
                        }
                    }
                }
                Node::Internal(ch) => {
                    let h = f.s * 0.5;
                    for (ci, &child) in ch.iter().enumerate() {
                        let (mut lo, mut hi) = (f.lo, f.hi);
                        for a in 0..3 {
                            let t = f.o[a] + h;
                            if (ci >> a) & 1 == 1 {
                                lo[a] = lo[a].max(t);
                            } else {
                                hi[a] = hi[a].min(t);
                            }
                        }
                        stack.push(Frame { idx: child, o: child_origin(f.o, f.s, ci), s: h, lo, hi });
                    }
                }
            }
        }
    }

    fn extract_region(
        adf: &Adf,
        n: usize,
        offset: [usize; 3],
        shape: [usize; 3],
        reference: Option<&meshfield::MeshField>,
    ) -> Mesh {
        let [nx, ny, nz] = shape;
        let [dx, dy, dz] = shape.map(|v| v + 1);
        let step = adf.size / n as Real;
        let edge_splits = (1usize << adf.max_depth).div_ceil(n).next_power_of_two().ilog2();

        // 1) 角点均匀重采样(走 ADF eval——自适应结构在这里兑现价值)
        //    逐点纯函数、与访问顺序无关 → 按 k 平面切带并行,结果逐位可复现。
        //    dim³ 次 eval 是本函数的主要成本(桌腿 513³≈1.35 亿次,占提取耗时 ~85%);
        //    串行时 4.0s,串行也是它把 extract 顶成全管线最大单项。
        let t_all = std::time::Instant::now();
        let mut vals = vec![0f32; dx * dy * dz];
        {
            let plane = dx * dy;
            let ppw = dz.div_ceil(nthreads()).max(1);
            let band = ppw * plane;
            let chunks: Vec<&mut [f32]> = vals.chunks_mut(band).collect();
            // P39:按叶光栅化代替逐点 `adf.eval`。逐点 eval 每次从根下探 ~11 层(指针追逐、缓存不命中),
            // 翅膀 depth 11 要做 25.8 亿次。这里每个线程对自己的 k 层段做一次八叉树 DFS,叶内格点直接三线性。
            // 坐标表达式、钳位(clamp_to_domain)、子格归属(child_index 的同一阈值 o+s/2)、叶内坐标
            // u=(q−o)/s 与 eval 完全相同 ⟹ 每个格点的值与 eval **逐位一致**。
            let (min, size) = (adf.min, adf.size);
            let axis_q = |a: usize, n_pts: usize| -> Vec<Real> {
                (0..n_pts)
                    .map(|i| clamp(min[a] + (offset[a] + i) as Real * step, min[a], min[a] + size - 1e-9))
                    .collect()
            };
            let qs = [axis_q(0, dx), axis_q(1, dy), axis_q(2, dz)];
            let qs = &qs;
            std::thread::scope(|sc| {
                for (w, chunk) in chunks.into_iter().enumerate() {
                    let k0 = w * ppw;
                    let k1 = (k0 + ppw).min(dz);
                    sc.spawn(move || raster_leaves(adf, qs, k0, k1, dx, plane, chunk));
                }
            });
        }
        let t_sample = t_all.elapsed();

        // 2) 变号 cell 每片表面一个顶点 = 该片 12 棱交叉点的 QEF 最小二乘
        //    按 ck 切 k 平面并行:每片产出**局部**顶点号写入本片 cell_vert,
        //    之后按片序合并(片内顶点先于后片)⟹ 与串行顶点编号逐位一致。
        // **每格多顶点**:一片表面 = 该格 12 条交叉棱经**共享面**连通后的分量。
        //    判据只看面(相邻两格对同一面的 4 个角点值完全相同,面内 4 交叉的鞍点
        //    用同一个 decider 消歧)⟹ 分量归属跨格天然一致,quad 装配不会再
        //    把两片表面错接到一起 —— P19 遗留破损/翻折襟翼的治本点。
        //    每格至多 4 个分量。它们在顶点数组中连续，每格只存首个索引。
        //    cell_pack:每条棱 4 bit,存"该棱所属分量落在第几个顶点槽"(0 = 无)。
        let mut cell_vert = vec![u32::MAX; nx * ny * nz];
        let mut cell_pack = vec![0u64; nx * ny * nz];
        let mut verts: Vec<[f32; 3]> = Vec::new();
        {
            let nslab = nz;
            let mut slabs: Vec<&mut [u32]> = cell_vert.chunks_mut(nx * ny).collect();
            let mut packs: Vec<&mut [u64]> = cell_pack.chunks_mut(nx * ny).collect();
            debug_assert_eq!(slabs.len(), nslab);
            let locals: Vec<std::sync::Mutex<Vec<[f32; 3]>>> =
                (0..nslab).map(|_| std::sync::Mutex::new(Vec::new())).collect();
            let mut jobs: Vec<(usize, &mut [u32], &mut [u64])> = slabs
                .drain(..)
                .zip(packs.drain(..))
                .enumerate()
                .map(|(s, (a, b))| (s, a, b))
                .collect();
            let nt = nthreads().min(nslab).max(1);
            let mut per: Vec<Vec<(usize, &mut [u32], &mut [u64])>> =
                (0..nt).map(|_| Vec::new()).collect();
            for (s, job) in jobs.drain(..).enumerate() {
                per[s % nt].push(job);
            }
            let vmin = adf.min;
            let vals_r: &[f32] = &vals;
            std::thread::scope(|sc| {
                for mine in per.into_iter() {
                    let locals = &locals;
                    sc.spawn(move || {
                        fn uf_find(p: &mut [u8; 12], mut x: u8) -> u8 {
                            while p[x as usize] != x {
                                p[x as usize] = p[p[x as usize] as usize];
                                x = p[x as usize];
                            }
                            x
                        }
                        fn uf_union(p: &mut [u8; 12], a: u8, b: u8) {
                            let (ra, rb) = (uf_find(p, a), uf_find(p, b));
                            if ra != rb {
                                p[rb as usize] = ra;
                            }
                        }
                        for (ck, slab, pk_slab) in mine {
                            let mut out: Vec<[f32; 3]> = Vec::new();
                            for cj in 0..ny {
                                for ci in 0..nx {
                                    let mut cv = [0f32; 8];
                                    let (mut any_neg, mut any_pos) = (false, false);
                                    for c in 0..8 {
                                        let (cx, cy, cz) = (c & 1, (c >> 1) & 1, (c >> 2) & 1);
                                        let v = vals_r[((ck + cz) * dy + (cj + cy)) * dx + (ci + cx)];
                                        cv[c] = v;
                                        any_neg |= v < 0.0;
                                        any_pos |= v >= 0.0;
                                    }
                                    if !(any_neg && any_pos) {
                                        continue;
                                    }
                                    // 12 棱交叉位
                                    let mut xr = [false; 12];
                                    for (ei, &(a, b)) in EDGES.iter().enumerate() {
                                        xr[ei] = (cv[a] < 0.0) != (cv[b] < 0.0);
                                    }
                                    // 交叉棱经 6 个面连通:面内 2 交叉直接连;4 交叉
                                    // (鞍点)按面心符号消歧 —— 中心为负 ⟹ 负区连成一体
                                    // ⟹ 正角是两个孤岛,交叉绕孤岛成对相连。
                                    let mut ep = [0u8; 12];
                                    for i in 0..12 {
                                        ep[i] = i as u8;
                                    }
                                    for (fc, fe) in FACES {
                                        let mut cr = [false; 4];
                                        let mut m = 0;
                                        for i in 0..4 {
                                            cr[i] = xr[fe[i]];
                                            if cr[i] {
                                                m += 1;
                                            }
                                        }
                                        if m == 2 {
                                            // 两个交叉可能相邻(面上只有一个角异号)或对位
                                            // (两个角相邻同号)——都存在一条面内弧把两者连起来。
                                            let mut ix = [0usize; 2];
                                            let mut k2 = 0;
                                            for (i, &c) in cr.iter().enumerate() {
                                                if c {
                                                    ix[k2] = fe[i];
                                                    k2 += 1;
                                                }
                                            }
                                            uf_union(&mut ep, ix[0] as u8, ix[1] as u8);
                                        } else if m == 4 {
                                            // 面内鞍点消歧:用**渐近 decider**(Nielson–Hamann)
                                            // 而不是 4 角均值 —— 双线性曲面的鞍点值为
                                            //   s = (v0·v2 − v1·v3) / (v0+v2−v1−v3)
                                            // 薄壁面上 4 角值几乎相等,均值会把符号判反。
                                            // 退化(分母≈0)时才退回均值。
                                            let (v0, v1, v2, v3) = (
                                                cv[fc[0]] as f64,
                                                cv[fc[1]] as f64,
                                                cv[fc[2]] as f64,
                                                cv[fc[3]] as f64,
                                            );
                                            let den = v0 + v2 - v1 - v3;
                                            let sv = if den.abs() > 1e-12 {
                                                (v0 * v2 - v1 * v3) / den
                                            } else {
                                                0.25 * (v0 + v1 + v2 + v3)
                                            };
                                            let island_neg = sv >= 0.0;
                                            if (cv[fc[0]] < 0.0) == island_neg {
                                                // 孤岛是角 0/2 → 绕它们的交叉对为 (e3,e0)、(e1,e2)
                                                uf_union(&mut ep, fe[3] as u8, fe[0] as u8);
                                                uf_union(&mut ep, fe[1] as u8, fe[2] as u8);
                                            } else {
                                                // 孤岛是角 1/3 → (e0,e1)、(e2,e3)
                                                uf_union(&mut ep, fe[0] as u8, fe[1] as u8);
                                                uf_union(&mut ep, fe[2] as u8, fe[3] as u8);
                                            }
                                        }
                                    }
                                    // 分量号按"根首次出现"编号(确定性,与线程调度无关)
                                    let mut rid = [u8::MAX; 12];
                                    let mut pid = [u8::MAX; 12];
                                    let mut ncomp = 0u8;
                                    for e in 0..12 {
                                        if !xr[e] {
                                            continue;
                                        }
                                        let r = uf_find(&mut ep, e as u8) as usize;
                                        if rid[r] == u8::MAX {
                                            rid[r] = ncomp;
                                            ncomp += 1;
                                        }
                                        pid[e] = rid[r];
                                    }
                                    let o = [
                                        vmin[0] + (offset[0] + ci) as Real * step,
                                        vmin[1] + (offset[1] + cj) as Real * step,
                                        vmin[2] + (offset[2] + ck) as Real * step,
                                    ];
                                    let od = [o[0] as f64, o[1] as f64, o[2] as f64];
                                    let sd = step as f64;
                                    // 每个分量各自做一次 QEF,产出自己的顶点
                                    let mut pk = 0u64;
                                    let mut nslot = 0usize;
                                    for comp in 0..ncomp {
                                        if nslot >= 4 {
                                            break;
                                        }
                                        let mut cp = [[0f64; 3]; 12];
                                        let mut cn = [[0f64; 3]; 12];
                                        let mut cnt = 0usize;
                                        for (ei, &(a, b)) in EDGES.iter().enumerate() {
                                            let (va, vb) = (cv[a] as f64, cv[b] as f64);
                                            if (va < 0.0) == (vb < 0.0) {
                                                continue;
                                            }
                                            if pid[ei] != comp {
                                                continue;
                                            }
                                            let pa = corner_pos(o, step, a);
                                            let pb = corner_pos(o, step, b);
                                            let axis = ei / 4;
                                            let mut lo = pa;
                                            let mut hi = pb;
                                            let (mut vlo, mut vhi) = (va, vb);
                                            // 粗棱可跨越薄壁的中轴，两端距离间不再线性。
                                            // 先按细叶尺度缩短变号区间，再用该叶的梯度拟合切平面。
                                            for _ in 0..edge_splits {
                                                let mut mid = lo;
                                                mid[axis] = 0.5 * (lo[axis] + hi[axis]);
                                                let vm = adf.eval(mid) as f64;
                                                if (vm < 0.0) == (vlo < 0.0) {
                                                    lo = mid;
                                                    vlo = vm;
                                                } else {
                                                    hi = mid;
                                                    vhi = vm;
                                                }
                                            }
                                            let mut pt = lo;
                                            pt[axis] += (hi[axis] - lo[axis]) * vlo / (vlo - vhi);
                                            if let Some((hit, normal)) = reference.and_then(|m| {
                                                m.edge_plane(pt, axis, pa[axis], pb[axis])
                                            }) {
                                                cp[cnt] = hit;
                                                cn[cnt] = normal;
                                                cnt += 1;
                                                continue;
                                            }
                                            let (corners, local, width) = adf.leaf_coordinates(adf.clamp_to_domain(pt));
                                            let g = tri_grad(corners, local, 1.0 / width);
                                            let gl = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
                                            cp[cnt] = pt;
                                            cn[cnt] = if gl > 1e-9 {
                                                [g[0] / gl, g[1] / gl, g[2] / gl]
                                            } else {
                                                // 异号已确定交叉。跨叶跳变时局部斜率可能为零，
                                                // 用格边有向差分作退路，不能丢交叉或整片网格。
                                                let mut direction = [0.0; 3];
                                                direction[axis] = (vb - va).signum();
                                                direction
                                            };
                                            cnt += 1;
                                        }
                                        if cnt == 0 {
                                            continue;
                                        }
                                        let mut mass = [0f64; 3];
                                        for i in 0..cnt {
                                            for k in 0..3 {
                                                mass[k] += cp[i][k];
                                            }
                                        }
                                        for k in 0..3 {
                                            mass[k] /= cnt as f64;
                                        }
                                        let mut am = [[0f64; 3]; 3];
                                        let mut bv = [0f64; 3];
                                        for i in 0..cnt {
                                            let nn = cn[i];
                                            let dd = nn[0] * cp[i][0]
                                                + nn[1] * cp[i][1]
                                                + nn[2] * cp[i][2];
                                            for r in 0..3 {
                                                for q in 0..3 {
                                                    am[r][q] += nn[r] * nn[q];
                                                }
                                                bv[r] += nn[r] * dd;
                                            }
                                        }
                                        let lam =
                                            0.01 * (am[0][0] + am[1][1] + am[2][2]) / 3.0 + 1e-9;
                                        for k in 0..3 {
                                            am[k][k] += lam;
                                            bv[k] += lam * mass[k];
                                        }
                                        let det = am[0][0]
                                            * (am[1][1] * am[2][2] - am[1][2] * am[2][1])
                                            - am[0][1]
                                                * (am[1][0] * am[2][2] - am[1][2] * am[2][0])
                                            + am[0][2]
                                                * (am[1][0] * am[2][1] - am[1][1] * am[2][0]);
                                        let mut pos = mass;
                                        if det.abs() > 1e-12 {
                                            for k in 0..3 {
                                                let mut m = am;
                                                for r in 0..3 {
                                                    m[r][k] = bv[r];
                                                }
                                                let dk = m[0][0]
                                                    * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                                                    - m[0][1]
                                                        * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                                                    + m[0][2]
                                                        * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
                                                pos[k] = dk / det;
                                            }
                                        }
                                        if (0..3).any(|k| !pos[k].is_finite()) {
                                            pos = mass;
                                        } else if (0..3).any(|k| pos[k] < od[k] - 1e-9 || pos[k] > od[k] + sd + 1e-9) {
                                            pos = if reference.is_some() { limit_qef_to_cell(mass, pos, od, sd) } else { mass };
                                        }
                                        let vidx = out.len() as u32;
                                        for e in 0..12 {
                                            if xr[e] && pid[e] == comp {
                                                pk |= ((nslot + 1) as u64) << (4 * e);
                                            }
                                        }
                                        if nslot == 0 {
                                            slab[cj * nx + ci] = vidx;
                                        }
                                        out.push([pos[0] as f32, pos[1] as f32, pos[2] as f32]);
                                        nslot += 1;
                                    }
                                    pk_slab[cj * nx + ci] = pk;
                                }
                            }
                            *locals[ck].lock().unwrap() = out;
                        }
                    });
                }
            });
            // 按片序合并:局部号 + 偏移 → 全局号
            for s in 0..nslab {
                let off = verts.len() as u32;
                let mut loc = locals[s].lock().unwrap();
                let base = s * nx * ny;
                for e in cell_vert[base..base + nx * ny].iter_mut() {
                    if *e != u32::MAX {
                        *e += off;
                    }
                }
                verts.append(&mut loc);
            }
        }
        let t_verts = t_all.elapsed();
        // 3) 网格棱变号 → 环绕该棱的 4 个 cell 顶点连 quad(push_quad 自定向)
        //    串行顺序 = 全 x 向 → 全 y 向 → 全 z 向;每向内部 k 递增。
        //    因此每向按 k 切片并行、再按片序合并,三角顺序与串行**逐位一致**。
        //    棱四邻 cell 只在 j/k ∈ [1, n) 内完整;域边界层跳过(表面不该触边,
        //    远场噪声变号在此组装会读到越界 cell_vert——实测 Impeller panic)。
        let mut tris: Vec<[u32; 3]> = Vec::new();
        for axis in 0..3usize {
            let k_lo = if axis == 2 { 0 } else { 1 };
            let nslab = nz - k_lo;
            if nslab == 0 {
                continue;
            }
            let locals: Vec<std::sync::Mutex<Vec<[u32; 3]>>> =
                (0..nslab).map(|_| std::sync::Mutex::new(Vec::new())).collect();
            let nt = nthreads().min(nslab).max(1);
            std::thread::scope(|sc| {
                for w in 0..nt {
                    let locals = &locals;
                    let vals = &vals;
                    let cell_vert = &cell_vert;
                    let cell_pack = &cell_pack;
                    let verts = &verts;
                    sc.spawn(move || {
                        let mut s = w;
                        while s < nslab {
                            let mut out: Vec<[u32; 3]> = Vec::new();
                            quads_slab(
                                axis,
                                k_lo + s,
                                k_lo + s + 1,
                                shape,
                                vals,
                                cell_vert,
                                cell_pack,
                                verts,
                                &mut out,
                            );
                            *locals[s].lock().unwrap() = out;
                            s += nt;
                        }
                    });
                }
            });
            for mut slot in locals {
                tris.append(slot.get_mut().unwrap());
            }
        }

        let t_quads = t_all.elapsed();
        let mut mesh = Mesh { verts, tris };
        // 绕向统一:个别 quad 会因场值不稳而定向相反,背面剔除的渲染器会显示成
        // 斑点状的"洞"。见 Mesh::orient_consistently 的说明。
        let flipped = mesh.orient_consistently();
        eprintln!("[nets.winding] 翻正 {flipped} 个三角(绕向一致化)");
        let t_end = t_all.elapsed();
        eprintln!(
            "[nets.time] 采样 {:.1}s | 顶点 {:.1}s | 组装 {:.1}s | 绕向 {:.1}s",
            t_sample.as_secs_f64(),
            (t_verts - t_sample).as_secs_f64(),
            (t_quads - t_verts).as_secs_f64(),
            (t_end - t_quads).as_secs_f64()
        );
        mesh
    }

    #[test]
    fn cropped_extraction_matches_the_full_lattice() {
        // 格数与八叉树深度非整除，材料分别位于域内和域边。
        for center in [[5.7, 3.13, -1.23], [2.95, 3.5, -1.2]] {
            let adf = Adf::build(
                Stock::RoundBox { c: center, half: [0.48, 0.25, 0.7], r: 0.07 },
                [3.0, 2.0, -5.0], 8.0, 0.003, 0.12, 6,
            );
            let n = 59;
            let (offset, shape) = material_region(&adf, n).unwrap();
            assert!(shape.iter().product::<usize>() < n.pow(3) / 4, "{offset:?} {shape:?}");
            let cropped = extract_region(&adf, n, offset, shape, None);
            let full = extract_region(&adf, n, [0; 3], [n; 3], None);
            assert!(!full.tris.is_empty());
            assert_eq!(cropped.verts, full.verts);
            assert_eq!(cropped.tris, full.tris);
        }
    }

    #[test]
    fn source_planes_recover_an_off_grid_box_corner() {
        let min = [-0.59, -0.63, -0.51];
        let max = [0.73, 0.87, 0.69];
        let vertex = |v| std::array::from_fn(|a: usize| if (v & (1usize << a)) == 0 { min[a] } else { max[a] });
        let mut triangles = Vec::new();
        for [a, b, c, d] in [[0,4,6,2], [1,3,7,5], [0,1,5,4], [2,6,7,3], [0,2,3,1], [4,5,7,6]] {
            triangles.push([vertex(a), vertex(b), vertex(c)]);
            triangles.push([vertex(a), vertex(c), vertex(d)]);
        }
        let adf = Adf::build_with(
            Composite { base: BaseField::Mesh(meshfield::MeshField::new(triangles)), cuts: Vec::new() },
            [-2.0; 3], 4.0, 0.001, 0.2, 5,
        );
        let mesh = extract_mesh_roundtrip(&adf, 0.5);
        let error = mesh.verts.iter().map(|v| len(sub(g64(*v), max))).fold(f64::INFINITY, f64::min);
        assert!(error < 0.005, "upper corner at {max:?} missed by {error} mm");
        assert_eq!(mesh.edge_stats(), (0, 0, 1));
        let mut mesh = extract_mesh_roundtrip(&adf, 0.08);
        let tris_before = mesh.tris.len();
        mesh.decimate(1e-4, 25.0, 0);
        assert!(mesh.tris.len() < tris_before / 2);
        let flat = mesh.tris.iter().filter(|t| {
            let [a, b, c] = t.map(|id| g64(mesh.verts[id as usize]));
            cross3(sub(b, a), sub(c, a)) == [0.0; 3]
        }).count();
        assert_eq!(flat, 0, "simplified box contains collinear STL facets");
    }

    #[test]
    fn outside_qef_minimum_retains_the_feasible_part_of_the_step() {
        let center = [-0.5, 0.5, 1.5];
        let target = [-2.0, 0.8, 1.2];
        let actual = limit_qef_to_cell(center, target, [-1.0, 0.0, 1.0], 1.0);
        assert!(actual[0] > -1.0 && actual[0] < -0.999);
        assert!(actual[1] > center[1] && actual[2] < center[2]);
        assert!(len(sub(actual, target)) < len(sub(center, target)));
    }

    /// 二进制 STL 导出。
    pub fn write_stl(path: &str, m: &Mesh) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(&[0u8; 80])?;
        f.write_all(&(m.tris.len() as u32).to_le_bytes())?;
        for t in &m.tris {
            let (a, b, c) =
                (g64(m.verts[t[0] as usize]), g64(m.verts[t[1] as usize]), g64(m.verts[t[2] as usize]));
            let n = cross3(sub(b, a), sub(c, a));
            let l = dot3(n, n).sqrt();
            let nn = if l > 0.0 { scale(n, 1.0 / l) } else { [0.0; 3] };
            for v in [nn, a, b, c] {
                for x in v {
                    f.write_all(&(x as f32).to_le_bytes())?;
                }
            }
            f.write_all(&0u16.to_le_bytes())?;
        }
        Ok(())
    }
}

// ─────────────────────────── 测试 ───────────────────────────

// ─────────────────── STL 读取(binary + ASCII)───────────────────

pub mod stlio {
    use super::*;
    /// 三角网格(顶点按三角形展开——STL 本就如此)。
    pub struct TriMesh {
        pub tris: Vec<[[Real; 3]; 3]>,
    }

    /// 二进制判定:84 + 50n == len;否则尝试 ASCII(以 solid 开头)。
    /// 法线被忽略:符号由射线奇偶决定,绕向无关紧要。
    pub fn load_stl(path: &str) -> std::io::Result<TriMesh> {
        let data = std::fs::read(path)?;
        if data.len() >= 84 {
            let n = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
            if data.len() == 84 + n * 50 && n > 0 {
                let mut tris = Vec::with_capacity(n);
                let mut off = 84usize;
                for _ in 0..n {
                    off += 12; // 法线
                    let mut v = [[0.0; 3]; 3];
                    for vv in &mut v {
                        for k in 0..3 {
                            vv[k] = f32::from_le_bytes(data[off..off + 4].try_into().unwrap()) as Real;
                            off += 4;
                        }
                    }
                    off += 2; // 属性字节数
                    tris.push(v);
                }
                return Ok(TriMesh { tris });
            }
        }
        if data.starts_with(b"solid") {
            let text = String::from_utf8_lossy(&data);
            let mut verts: Vec<[Real; 3]> = Vec::new();
            for line in text.lines() {
                let t = line.trim();
                if let Some(rest) = t.strip_prefix("vertex") {
                    let nums: Vec<Real> =
                        rest.split_whitespace().filter_map(|w| w.parse().ok()).collect();
                    if nums.len() >= 3 {
                        verts.push([nums[0], nums[1], nums[2]]);
                    }
                }
            }
            if !verts.is_empty() && verts.len() % 3 == 0 {
                let tris = verts.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
                return Ok(TriMesh { tris });
            }
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "无法识别的 STL(二进制长度不符且 ASCII 解析失败)",
        ))
    }

    /// 顶点按 0.1 µm 合并后，每条非零边应恰有两次反向关联。
    pub fn watertight_report(tris: &[[[Real; 3]; 3]]) -> (usize, bool) {
        use std::collections::HashMap;
        let q = |v: Real| (v * 10000.0).round() as i64; // 0.1µm 量化
        let mut vid: HashMap<[i64; 3], u32> = HashMap::new();
        let mut edges: HashMap<(u32, u32), u8> = HashMap::new();
        let mut paired = !tris.is_empty();
        let mut nv = 0u32;
        for t in tris {
            let mut ids = [0u32; 3];
            for (k, v) in t.iter().enumerate() {
                let key = [q(v[0]), q(v[1]), q(v[2])];
                let id = match vid.get(&key) {
                    Some(&i) => i,
                    None => {
                        let i = nv;
                        vid.insert(key, i);
                        nv += 1;
                        i
                    }
                };
                ids[k] = id;
            }
            for k in 0..3 {
                let (a, b) = (ids[k], ids[(k + 1) % 3]);
                let key = if a < b { (a, b) } else { (b, a) };
                paired &= super::record_edge(edges.entry(key).or_default(), a, b);
            }
        }
        (tris.len(), paired && edges.values().all(|&mask| mask == 3))
    }
}

// ─────────────────── 网格场源:BVH + 最近点 + 射线奇偶符号 ───────────────────

fn record_edge(mask: &mut u8, a: u32, b: u32) -> bool {
    let bit = if a < b { 1 } else { 2 };
    let first = a != b && *mask & bit == 0;
    *mask |= bit;
    first
}

pub mod meshfield {
    use super::*;

    fn cross3(a: [Real; 3], b: [Real; 3]) -> [Real; 3] {
        [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    }
    fn dot3(a: [Real; 3], b: [Real; 3]) -> Real {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }
    fn sub3(a: [Real; 3], b: [Real; 3]) -> [Real; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    /// 网格 BVH 节点:cnt > 0 为叶(引用 order[start..start+cnt] 的三角形)。
    struct Bvn {
        mn: [Real; 3],
        mx: [Real; 3],
        left: u32,
        right: u32,
        start: u32,
        cnt: u32,
    }

    /// 点到三角形最近点(Ericson《Real-Time Collision Detection》§5.1.5 区域法;
    /// 与报告 §9.1 的 Eberly 算法同族)。返回最近点坐标。
    /// 最近点 + 特征码(Ericson 区域法顺带判定):0/1/2 = 顶点,3/4/5 = 边
    /// AB/AC/BC,6 = 面。伪法线符号法(Bærentzen & Aanæs 2005)用特征码选
    /// 对应的伪法线 → 点积符号即内外符号,零射线。
    pub fn closest_point_feature(p: [Real; 3], t: &[[Real; 3]; 3]) -> ([Real; 3], u8) {
        let ab = sub3(t[1], t[0]);
        let ac = sub3(t[2], t[0]);
        let ap = sub3(p, t[0]);
        let d1 = dot3(ab, ap);
        let d2 = dot3(ac, ap);
        if d1 <= 0.0 && d2 <= 0.0 {
            return (t[0], 0);
        }
        let bp = sub3(p, t[1]);
        let d3 = dot3(ab, bp);
        let d4 = dot3(ac, bp);
        if d3 >= 0.0 && d4 <= d3 {
            return (t[1], 1);
        }
        let vc = d1 * d4 - d3 * d2;
        if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
            let v = d1 / (d1 - d3);
            return (add3(t[0], scale3(ab, v)), 3);
        }
        let cp = sub3(p, t[2]);
        let d5 = dot3(ab, cp);
        let d6 = dot3(ac, cp);
        if d6 >= 0.0 && d5 <= d6 {
            return (t[2], 2);
        }
        let vb = d5 * d2 - d1 * d6;
        if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
            let w = d2 / (d2 - d6);
            return (add3(t[0], scale3(ac, w)), 4);
        }
        let va = d3 * d6 - d5 * d4;
        if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
            let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            return (add3(t[1], scale3(sub3(t[2], t[1]), w)), 5);
        }
        let denom = 1.0 / (va + vb + vc);
        let v = vb * denom;
        let w = vc * denom;
        (add3(add3(t[0], scale3(ab, v)), scale3(ac, w)), 6)
    }
    pub fn closest_point_triangle(p: [Real; 3], t: &[[Real; 3]; 3]) -> [Real; 3] {
        let ab = sub3(t[1], t[0]);
        let ac = sub3(t[2], t[0]);
        let ap = sub3(p, t[0]);
        let d1 = dot3(ab, ap);
        let d2 = dot3(ac, ap);
        if d1 <= 0.0 && d2 <= 0.0 {
            return t[0];
        }
        let bp = sub3(p, t[1]);
        let d3 = dot3(ab, bp);
        let d4 = dot3(ac, bp);
        if d3 >= 0.0 && d4 <= d3 {
            return t[1];
        }
        let vc = d1 * d4 - d3 * d2;
        if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
            let v = d1 / (d1 - d3);
            return add3(t[0], scale3(ab, v));
        }
        let cp = sub3(p, t[2]);
        let d5 = dot3(ab, cp);
        let d6 = dot3(ac, cp);
        if d6 >= 0.0 && d5 <= d6 {
            return t[2];
        }
        let vb = d5 * d2 - d1 * d6;
        if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
            let w = d2 / (d2 - d6);
            return add3(t[0], scale3(ac, w));
        }
        let va = d3 * d6 - d5 * d4;
        if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
            let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            return add3(t[1], scale3(sub3(t[2], t[1]), w));
        }
        let denom = 1.0 / (va + vb + vc);
        let v = vb * denom;
        let w = vc * denom;
        add3(add3(t[0], scale3(ab, v)), scale3(ac, w))
    }
    fn add3(a: [Real; 3], b: [Real; 3]) -> [Real; 3] {
        [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
    }
    fn scale3(a: [Real; 3], k: Real) -> [Real; 3] {
        [a[0] * k, a[1] * k, a[2] * k]
    }

    fn vertex_key(v: [Real; 3]) -> [u64; 3] {
        v.map(|x| if x == 0.0 { 0 } else { x.to_bits() })
    }

    // +X 射线。YZ 投影用半开边规则：公共对角线上的交点只由一面计数。
    fn crosses_x(p: [Real; 3], t: &[[Real; 3]; 3]) -> bool {
        let v = t.map(|q| sub3(q, p));
        let edge = |a: [Real; 3], b: [Real; 3]| a[1] * b[2] - a[2] * b[1];
        let w = [edge(v[1], v[2]), edge(v[2], v[0]), edge(v[0], v[1])];
        let area = w[0] + w[1] + w[2];
        if area == 0.0 { return false; }
        for i in 0..3 {
            if w[i] == 0.0 {
                let a = v[(i + 1) % 3];
                let b = v[(i + 2) % 3];
                let upper = a[2] < b[2] || (a[2] == b[2] && a[1] > b[1]);
                if upper != (area > 0.0) { return false; }
            } else if (w[i] > 0.0) != (area > 0.0) {
                return false;
            }
        }
        let x = w[0] * v[0][0] + w[1] * v[1][0] + w[2] * v[2][0];
        if area > 0.0 { x > 0.0 } else { x < 0.0 }
    }

    /// 固定对角射线方向(非轴对齐 → 避开典型网格顶点/棱,奇偶可靠)。
    pub const PARITY_DIR: [Real; 3] = [0.408_248_3, 0.816_496_6, 0.408_248_3]; // normalize(1,2,1)

    /// Möller–Trumbore:一般方向射线与三角形相交(t > 0,含 u/v 判定)。
    fn ray_hits_tri(p: [Real; 3], dir: [Real; 3], t: &[[Real; 3]; 3]) -> bool {
        let e1 = sub3(t[1], t[0]);
        let e2 = sub3(t[2], t[0]);
        let h = cross3(dir, e2);
        let a = dot3(e1, h);
        if a.abs() < 1e-12 {
            return false;
        }
        let f = 1.0 / a;
        let s = sub3(p, t[0]);
        let u = f * dot3(s, h);
        if !(0.0..=1.0).contains(&u) {
            return false;
        }
        let q = cross3(s, e1);
        let v = f * dot3(dir, q);
        if v < 0.0 || u + v > 1.0 {
            return false;
        }
        let tt = f * dot3(e2, q);
        tt > 1e-9
    }

    /// 网格场:精确点到网格距离(BVH)+ 射线奇偶符号(+ 位置位模式缓存)。
    /// 已知限制:非水密网格的符号在破洞附近不可靠(§9.2)。
    pub struct MeshField {
        pub(super) has_paired_edges: bool,
        exact_vertex_ids: bool,
        use_even_odd: bool,
        pub tris: Vec<[[Real; 3]; 3]>,
        bvh: Vec<Bvn>,
        order: Vec<u32>,
        root: u32,
        /// 每三角形 AABB(查询时免 18 次 min/max 重算)
        tri_aabb: Vec<([Real; 3], [Real; 3])>,
        /// 伪法线数据(Bærentzen & Aanæs 2005;要求水密+一致绕向):
        /// 面法线(单位)/ 角度加权顶点法线 / 边伪法线(相邻两面法线之和)
        face_n: Vec<[f32; 3]>,
        vert_n: Vec<[f32; 3]>,
        edge_n: std::collections::HashMap<(u32, u32), [f32; 3]>,
        /// 每三角形的量化顶点 id(伪法线索引用)
        tri_vid: Vec<[u32; 3]>,
        pub aabb_min: [Real; 3],
        pub aabb_max: [Real; 3],
        cache: std::sync::Mutex<std::collections::HashMap<[u64; 3], f64>>,
    }

    fn centroid(t: &[[Real; 3]; 3]) -> [Real; 3] {
        scale3(add3(add3(t[0], t[1]), t[2]), 1.0 / 3.0)
    }

    impl MeshField {
        /// 源网格场的常驻堆占用(字节,容量口径)。
        ///
        /// 用 `capacity()` 而不是 `len()`:Vec 的扩容按 2 倍走,活着占用的是
        /// 容量;HashMap 另按 hashbrown 的 8/7 装载因子折算桶开销。
        pub(super) fn mem_bytes(&self) -> usize {
            use std::mem::size_of;
            macro_rules! cap {
                ($v:expr, $t:ty) => {
                    $v.capacity() * size_of::<$t>()
                };
            }
            let mut b = 0;
            b += cap!(self.tris, [[Real; 3]; 3]);
            b += cap!(self.bvh, Bvn);
            b += cap!(self.order, u32);
            b += cap!(self.tri_aabb, ([Real; 3], [Real; 3]));
            b += cap!(self.face_n, [f32; 3]);
            b += cap!(self.vert_n, [f32; 3]);
            b += cap!(self.tri_vid, [u32; 3]);
            // 边法线表:桶数 ≈ capacity·8/7,每桶 = (K,V) + 1 B 控制字节
            let kv = size_of::<((u32, u32), [f32; 3])>() + 1;
            b += self.edge_n.capacity() * kv * 8 / 7;
            let cache = self.cache.lock().map(|c| c.capacity()).unwrap_or(0);
            b += cache * (size_of::<([u64; 3], f64)>() + 1) * 8 / 7;
            b + size_of::<Self>()
        }

        /// Two consistently oriented, nondegenerate facets meeting at a sharp edge.
        /// Discard open and multiply incident edges, including coincident sheets.
        pub(super) fn sharp_segments(&self, angle: Real) -> Vec<crate::creases::Segment> {
            let mut incidence = std::collections::HashMap::<(u32,u32), (u32,u8,u8)>::new();
            let cos_angle = angle.to_radians().cos();
            let mut segments = Vec::new();
            for (i, ids) in self.tri_vid.iter().enumerate() {
                for k in 0..3 {
                    let (a,b) = (ids[k], ids[(k+1)%3]);
                    let e = incidence.entry((a.min(b), a.max(b))).or_insert((i as u32,k as u8,0));
                    e.2 = e.2.saturating_add(1);
                }
            }
            for (i, ids) in self.tri_vid.iter().enumerate() {
                for k in 0..3 {
                    let (a,b) = (ids[k], ids[(k+1)%3]);
                    let (j,l,count) = incidence[&(a.min(b), a.max(b))];
                    let (j,l) = (j as usize,l as usize);
                    if count != 2 || i == j { continue; }
                    let edge = [self.tris[i][k], self.tris[i][(k+1)%3]];
                    if edge != [self.tris[j][(l+1)%3],self.tris[j][l]] { continue; }
                    let normals = [self.face_n[i],self.face_n[j]].map(|n| n.map(Real::from));
                    if normals.iter().all(|n| dot3(*n,*n) > 0.9) && dot3(normals[0],normals[1]) < cos_angle {
                        segments.push((edge,normals));
                    }
                }
            }
            segments
        }

        pub fn new(tris: Vec<[[Real; 3]; 3]>) -> MeshField {
            // 空网格:根会是 cnt=0 的"叶",查询按 cnt>0 判叶会一路下降到 (0,0) 自环。
            // 距离场对空网格无定义,早失败优于死循环。
            assert!(!tris.is_empty(), "MeshField::new: 需要非空三角形列表");
            let mut order: Vec<u32> = (0..tris.len() as u32).collect();
            let mut bvh: Vec<Bvn> = Vec::new();
            let (mut aabb_min, mut aabb_max) = ([f64::MAX; 3], [f64::MIN; 3]);
            fn rec(
                tris: &[[[Real; 3]; 3]],
                order: &mut [u32],
                bvh: &mut Vec<Bvn>,
                lo: usize,
                hi: usize,
                aabb_min: &mut [Real; 3],
                aabb_max: &mut [Real; 3],
            ) -> u32 {
                let (mut mn, mut mx) = ([f64::MAX; 3], [f64::MIN; 3]);
                let mut cmn = [f64::MAX; 3];
                let mut cmx = [f64::MIN; 3];
                for &t in &order[lo..hi] {
                    for v in &tris[t as usize] {
                        for k in 0..3 {
                            mn[k] = mn[k].min(v[k]);
                            mx[k] = mx[k].max(v[k]);
                        }
                    }
                    let c = centroid(&tris[t as usize]);
                    for k in 0..3 {
                        cmn[k] = cmn[k].min(c[k]);
                        cmx[k] = cmx[k].max(c[k]);
                    }
                }
                for k in 0..3 {
                    aabb_min[k] = aabb_min[k].min(mn[k]);
                    aabb_max[k] = aabb_max[k].max(mx[k]);
                }
                let cnt = hi - lo;
                let self_idx = bvh.len() as u32;
                bvh.push(Bvn { mn, mx, left: 0, right: 0, start: lo as u32, cnt: cnt as u32 });
                if cnt <= 8 {
                    return self_idx;
                }
                let mut axis = 0;
                let mut best = -1.0;
                for k in 0..3 {
                    if cmx[k] - cmn[k] > best {
                        best = cmx[k] - cmn[k];
                        axis = k;
                    }
                }
                order[lo..hi].sort_by(|&a, &b| {
                    centroid(&tris[a as usize])[axis].partial_cmp(&centroid(&tris[b as usize])[axis]).unwrap()
                });
                let mid = lo + cnt / 2;
                let l = rec(tris, order, bvh, lo, mid, aabb_min, aabb_max);
                let rr = rec(tris, order, bvh, mid, hi, aabb_min, aabb_max);
                // cnt = 0 是内部节点的**判别标志**:两条查询路径(closest_features_seeded /
                // ray_parity_inside)都用 `cnt > 0 ⇒ 叶` 分流。此处置 0 之外任何值都会让
                // 内部节点被当成叶,查询退化为扫描整段 order[lo..hi](= 全子树三角形),
                // 树查询变线性扫描(实测 15.5 万三角 270µs/查,且与种子无关)。
                bvh[self_idx as usize] = Bvn { mn, mx, left: l, right: rr, start: lo as u32, cnt: 0 };
                self_idx
            }
            let mut amin = [f64::MAX; 3];
            let mut amax = [f64::MIN; 3];
            let root = rec(&tris, &mut order, &mut bvh, 0, tris.len(), &mut amin, &mut amax);
            for k in 0..3 {
                aabb_min[k] = amin[k];
                aabb_max[k] = amax[k];
            }
            // 三角形按 BVH 叶序物理重排:叶内 ≤8 tri 连续 ~576B(8 cache line),
            // 免随机跨 11MB 数组——实测 231 tri 测试 1.9ms(8µs/次)→ 预期 ~50µs/查
            let mut perm_tris: Vec<[[Real; 3]; 3]> = Vec::with_capacity(tris.len());
            for &t in order.iter() {
                perm_tris.push(tris[t as usize]);
            }
            let tris = perm_tris;
            for (i, o) in order.iter_mut().enumerate() {
                *o = i as u32;
            }
            let tri_aabb: Vec<([Real; 3], [Real; 3])> = tris
                .iter()
                .map(|tri| {
                    let (mut tmn, mut tmx) = ([f64::MAX; 3], [f64::MIN; 3]);
                    for v in tri {
                        for k in 0..3 {
                            tmn[k] = tmn[k].min(v[k]);
                            tmx[k] = tmx[k].max(v[k]);
                        }
                    }
                    (tmn, tmx)
                })
                .collect();
            // ── 伪法线(Bærentzen & Aanæs 2005)──
            // 顶点量化合并(与 watertight_report 同口径 0.1µm)
            let q = |v: Real| (v * 10000.0).round() as i64;
            let mut vid_map: std::collections::HashMap<[i64; 3], u32> = std::collections::HashMap::new();
            let mut tri_vid: Vec<[u32; 3]> = Vec::with_capacity(tris.len());
            let mut representatives: Vec<[Real; 3]> = Vec::new();
            let roundoff = (0..3).map(|a| aabb_max[a] - aabb_min[a]).fold(0.0, Real::max)
                * (32.0 * Real::EPSILON);
            let mut welded = false;
            let mut exact_vertex_ids = true;
            let mut nv = 0u32;
            for t in &tris {
                let mut ids = [0u32; 3];
                for (k, v) in t.iter().enumerate() {
                    let key = [q(v[0]), q(v[1]), q(v[2])];
                    let id = match vid_map.get(&key) {
                        Some(&i) => {
                            exact_vertex_ids &= *v == representatives[i as usize];
                            welded |= (0..3).any(|a| (v[a] - representatives[i as usize][a]).abs() > roundoff);
                            i
                        }
                        None => {
                            vid_map.insert(key, nv);
                            representatives.push(*v);
                            nv += 1;
                            nv - 1
                        }
                    };
                    ids[k] = id;
                }
                tri_vid.push(ids);
            }
            let mut vert_n = vec![[0f64; 3]; nv as usize];
            let mut edge_n: std::collections::HashMap<(u32, u32), ([f64; 3], u8)> =
                std::collections::HashMap::new();
            let mut has_paired_edges = !tris.is_empty();
            let mut face_n = Vec::with_capacity(tris.len());
            for (ti, t) in tris.iter().enumerate() {
                let n = cross3(sub3(t[1], t[0]), sub3(t[2], t[0])); // 模长 = 2×面积
                let nl = len(n);
                let fn_unit = if nl > 0.0 { scale3(n, 1.0 / nl) } else { [0.0; 3] };
                face_n.push([fn_unit[0] as f32, fn_unit[1] as f32, fn_unit[2] as f32]);
                // 角度加权顶点法线 = 相邻面叉积之和(cross 模长含 sin ≈ 角度加权)
                for k in 0..3 {
                    let vi = tri_vid[ti][k] as usize;
                    for c in 0..3 {
                        vert_n[vi][c] += n[c];
                    }
                }
                // 边伪法线 = 相邻两面法线之和(未归一)
                for k in 0..3 {
                    let (a, b) = (tri_vid[ti][k], tri_vid[ti][(k + 1) % 3]);
                    let key = if a < b { (a, b) } else { (b, a) };
                    let (e, mask) = edge_n.entry(key).or_default();
                    has_paired_edges &= record_edge(mask, a, b);
                    for c in 0..3 {
                        e[c] += n[c];
                    }
                }
            }
            let vert_n: Vec<[f32; 3]> = vert_n
                .iter()
                .map(|v| {
                    let l = len(*v);
                    let u = if l > 0.0 { scale3(*v, 1.0 / l) } else { [0.0; 3] };
                    [u[0] as f32, u[1] as f32, u[2] as f32]
                })
                .collect();
            let edge_n: std::collections::HashMap<(u32, u32), [f32; 3]> = edge_n
                .into_iter()
                .map(|(k, (v, mask))| {
                    has_paired_edges &= mask == 3;
                    let l = len(v);
                    let u = if l > 0.0 { scale3(v, 1.0 / l) } else { [0.0; 3] };
                    (k, [u[0] as f32, u[1] as f32, u[2] as f32])
                })
                .collect();
            // 非流形或绕向混杂的闭合三角链仍有全局奇偶性。验证原坐标的边关联，
            // 避免法线索引的合并容差跨过真实开缝。
            let closed = has_paired_edges && !welded || face_n.iter().any(|n| *n != [0.0; 3]) && {
                let mut odd = std::collections::HashSet::new();
                for tri in &tris {
                    let keys = tri.map(vertex_key);
                    for i in 0..3 {
                        let (a, b) = (keys[i], keys[(i + 1) % 3]);
                        if a == b { continue; }
                        let edge = (a.min(b), a.max(b));
                        if !odd.remove(&edge) { odd.insert(edge); }
                    }
                }
                odd.is_empty()
            };
            has_paired_edges &= closed;
            let use_even_odd = closed && !has_paired_edges;
            MeshField {
                has_paired_edges,
                exact_vertex_ids,
                use_even_odd,
                tris,
                bvh,
                order,
                root,
                tri_aabb,
                face_n,
                vert_n,
                edge_n,
                tri_vid,
                aabb_min,
                aabb_max,
                cache: std::sync::Mutex::new(std::collections::HashMap::new()),
            }
        }

        pub fn triangle_count(&self) -> usize {
            self.tris.len()
        }

        pub fn has_closed_surface(&self) -> bool {
            self.has_paired_edges || self.use_even_odd
        }

        /// 未切削重建中，独立实体若相互穿入或相距不足一个格宽，单一符号场
        /// 会吞掉相邻表面。将它们分组后在同一原域提取。切削仍使用整体材料场。
        pub fn into_roundtrip_layers(self, h: Real) -> Vec<Self> {
            if !self.has_closed_surface() { return vec![self]; }
            let mut parents: Vec<usize> = (0..self.tris.len()).collect();
            fn root(parents: &mut [usize], mut i: usize) -> usize {
                while parents[i] != i {
                    parents[i] = parents[parents[i]];
                    i = parents[i];
                }
                i
            }
            // 只通过整条原始边连接面。两个实体即使共用一个顶点，也可能
            // 在同一格内有各自的表面。多面共边则保留在同一个三角链中。
            let mut vertices = std::collections::HashMap::new();
            let mut owners = std::collections::HashMap::new();
            let paired = self.has_paired_edges && self.exact_vertex_ids;
            for (i, tri) in self.tris.iter().enumerate() {
                // 普通流形直接复用已经建立的顶点索引；出现量化合并才重建精确索引。
                let ids = if self.exact_vertex_ids {
                    self.tri_vid[i]
                } else {
                    tri.map(|v| {
                        let next = u32::try_from(vertices.len()).expect("顶点数超过 u32 索引范围");
                        *vertices.entry(vertex_key(v)).or_insert(next)
                    })
                };
                for (a, b) in [(ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])] {
                    if a == b { continue; }
                    use std::collections::hash_map::Entry;
                    let other = match owners.entry((a.min(b), a.max(b))) {
                        Entry::Vacant(slot) => { slot.insert(i); continue; }
                        // 成对边只需保存到第二个面；多面共边则保留首面供后续关联。
                        Entry::Occupied(slot) if paired => slot.remove(),
                        Entry::Occupied(slot) => *slot.get(),
                    };
                    let (left, right) = (root(&mut parents, i), root(&mut parents, other));
                    parents[left] = right;
                }
            }
            drop(vertices);
            drop(owners);
            let mut ids = std::collections::HashMap::new();
            let mut shells: Vec<Vec<usize>> = Vec::new();
            for i in 0..parents.len() {
                let key = root(&mut parents, i);
                let id = *ids.entry(key).or_insert_with(|| {
                    shells.push(Vec::new());
                    shells.len() - 1
                });
                shells[id].push(i);
            }
            if shells.len() < 2 { return vec![self]; }

            // 大壳先放；扩展两格让可能共用一格的表面分开。反向壳可能表示空腔，
            // 必须与外壳共同定号。体积计算以壳上顶点为原点，避免远离原点时消减。
            shells.sort_by_key(|s| std::cmp::Reverse(s.len()));
            let mut boxes: Vec<Vec<([Real; 3], [Real; 3])>> = Vec::new();
            let mut layers: Vec<Vec<usize>> = Vec::new();
            for shell in shells {
                let origin = self.tris[shell[0]][0];
                let (mut lo, mut hi) = (origin, origin);
                let mut volume6 = 0.0;
                for &index in &shell {
                    let t = &self.tris[index];
                    volume6 += dot3(sub3(t[0], origin), cross3(sub3(t[1], origin), sub3(t[2], origin)));
                    for v in t {
                        for axis in 0..3 {
                            lo[axis] = lo[axis].min(v[axis]);
                            hi[axis] = hi[axis].max(v[axis]);
                        }
                    }
                }
                if volume6 <= 0.0 { return vec![self]; }
                let separated = |(mn, mx): &([Real; 3], [Real; 3])| (0..3).any(|a|
                    lo[a] > mx[a] + 2.0 * h || hi[a] < mn[a] - 2.0 * h
                );
                let layer = boxes.iter().position(|group| group.iter().all(separated)).unwrap_or(boxes.len());
                if layer == boxes.len() {
                    boxes.push(Vec::new());
                    layers.push(Vec::new());
                }
                boxes[layer].push((lo, hi));
                layers[layer].extend(shell);
            }
            if layers.len() < 2 { return vec![self]; }
            let tris = self.tris;
            layers.into_iter().map(|indices| {
                let mut layer = Self::new(indices.into_iter().map(|i| tris[i]).collect());
                layer.use_even_odd = layer.has_closed_surface();
                layer
            }).collect()
        }

        /// 参考平面只在其与有限格边的交点落在原三角内时才有效。
        pub(super) fn edge_plane(
            &self, p: [Real; 3], axis: usize, low: Real, high: Real,
        ) -> Option<([Real; 3], [Real; 3])> {
            let tolerance = (high - low) * 1e-6;
            let intersect = |id: u32| {
                let tri = &self.tris[id as usize];
                let normal = self.face_n[id as usize].map(Real::from);
                if normal[axis].abs() <= 1e-8 { return None; }
                let mut hit = p;
                hit[axis] += dot3(normal, sub3(tri[0], p)) / normal[axis];
                if hit[axis] < low || hit[axis] > high { return None; }
                let delta = sub3(closest_point_triangle(hit, tri), hit);
                (dot3(delta, delta) <= tolerance * tolerance).then_some((hit, normal))
            };
            let (_, nearest, _, _) = self.closest_features(p);
            if let Some(hit) = intersect(nearest) { return Some(hit); }

            // 薄壁或转角处最近面可能平行于格边，真实交点在相邻面上。
            // 检查与这段有限格边相交的 BVH 叶；多次穿越时取最接近场估计的位置。
            let j = (axis + 1) % 3;
            let k = (axis + 2) % 3;
            let mut best = None;
            let mut best_delta = Real::INFINITY;
            let mut stack = [0u32; 64];
            stack[0] = self.root;
            let mut count = 1;
            while count > 0 {
                count -= 1;
                let node = &self.bvh[stack[count] as usize];
                if node.mx[axis] < low || node.mn[axis] > high ||
                    p[j] < node.mn[j] || p[j] > node.mx[j] ||
                    p[k] < node.mn[k] || p[k] > node.mx[k] { continue; }
                if node.cnt == 0 {
                    stack[count] = node.left;
                    stack[count + 1] = node.right;
                    count += 2;
                } else {
                    for &id in &self.order[node.start as usize..(node.start + node.cnt) as usize] {
                        if let Some((hit, normal)) = intersect(id) {
                            let delta = (hit[axis] - p[axis]).abs();
                            if delta < best_delta {
                                best_delta = delta;
                                best = Some((hit, normal));
                            }
                        }
                    }
                }
            }
            best
        }

        /// 精确点到网格距离(非负)。
        pub fn unsigned_distance(&self, p: [Real; 3]) -> Real {
            self.closest_features(p).0
        }

        /// 候选列表上的精确最近(带特征码;烘焙网格的局部初始化用)。
        pub fn closest_among(&self, p: [Real; 3], cands: &[u32]) -> (Real, u32, u8, [Real; 3]) {
            let mut best = f64::INFINITY;
            let mut best_t = 0u32;
            let mut best_feat = 6u8;
            let mut best_cp = [0.0; 3];
            for &t in cands {
                let (cp, feat) = closest_point_feature(p, &self.tris[t as usize]);
                let dd = dot3(sub3(cp, p), sub3(cp, p));
                if dd < best {
                    best = dd;
                    best_t = t;
                    best_feat = feat;
                    best_cp = cp;
                }
            }
            (best.sqrt(), best_t, best_feat, best_cp)
        }

        /// 由 (三角形, 特征码, 最近点) 给出有符号距离。
        ///
        /// 面特征(feat=6):cp 在三角内部,p−cp ⊥ 面伪法线判号**精确**。
        /// 边/顶点特征(feat≤5):edge_n/vert_n(邻面法线和)的 ±90° 锥与
        /// 面的 Voronoi 区**不对齐**——薄叶片边缘(材料楔角 θ 小)周围约
        /// 180°−θ 的空气方位落在锥外被误判"内"(叶轮实测:缝隙区一片格点
        /// bake=−0.609 vs 真值 +0.609,零等值面外扩 1.15mm,提取出 203 个
        /// 碎屑分量)。故边/顶点一律**随机方向射线投票**仲裁:全局穿越计数
        /// 不依赖局部方位;方向按位置哈希取 + 原点微抖,避免固定方向与
        /// 规则三角化网格系统性掠射(桌腿固定 7 向实测:模型底面规则网格
        /// 擦边 → 2134 个碎屑分量)。
        pub fn signed_from_feature(&self, p: [Real; 3], t: u32, feat: u8, cp: [Real; 3]) -> Real {
            let d = len(sub3(p, cp));
            if self.use_even_odd {
                return if self.inside_closed(p) { -d } else { d };
            }
            let sgn = if feat == 6 {
                let n: [Real; 3] = [
                    self.face_n[t as usize][0] as Real,
                    self.face_n[t as usize][1] as Real,
                    self.face_n[t as usize][2] as Real,
                ];
                if dot3(sub3(p, cp), n) >= 0.0 { 1.0 } else { -1.0 }
            } else if self.parity_inside_vote(p) {
                -1.0
            } else {
                1.0
            };
            sgn * d
        }

        /// [诊断] 特征伪法线与三角顶点(signed_from_feature 的内部视角)
        pub fn debug_feature_normal(&self, t: u32, feat: u8) -> ([Real; 3], [u32; 3]) {
            let n: [Real; 3] = match feat {
                0..=2 => {
                    let v = self.vert_n[self.tri_vid[t as usize][feat as usize] as usize];
                    [v[0] as Real, v[1] as Real, v[2] as Real]
                }
                3 => {
                    let (a, b) = (self.tri_vid[t as usize][0], self.tri_vid[t as usize][1]);
                    let v = self.edge_n[&(a.min(b), a.max(b))];
                    [v[0] as Real, v[1] as Real, v[2] as Real]
                }
                4 => {
                    let (a, b) = (self.tri_vid[t as usize][0], self.tri_vid[t as usize][2]);
                    let v = self.edge_n[&(a.min(b), a.max(b))];
                    [v[0] as Real, v[1] as Real, v[2] as Real]
                }
                5 => {
                    let (a, b) = (self.tri_vid[t as usize][1], self.tri_vid[t as usize][2]);
                    let v = self.edge_n[&(a.min(b), a.max(b))];
                    [v[0] as Real, v[1] as Real, v[2] as Real]
                }
                _ => {
                    let v = self.face_n[t as usize];
                    [v[0] as Real, v[1] as Real, v[2] as Real]
                }
            };
            (n, self.tri_vid[t as usize])
        }

        /// best-first 最近点核心:(距离² 开方, 三角形索引, 特征码, 最近点)。
        /// best-first 遍历(min-heap 按 AABB 距离²):堆顶距离 ≥ 当前 best 即全局早退;
        /// 子节点按距离近者先入堆;三角形先做自身 AABB 早退再跑精确区域算法。
        /// (VTune 实证:LIFO 无序遍历 + 无三角形早退 → unsigned_distance 272s/90%)
        /// 第 `t` 个源三角形的单位面法线(`closest_features` 返回的三角形索引)。
        pub fn face_normal(&self, t: u32) -> [Real; 3] {
            self.face_n[t as usize].map(Real::from)
        }

        pub fn closest_features(&self, p: [Real; 3]) -> (Real, u32, u8, [Real; 3]) {
            self.closest_features_seeded(p, f64::INFINITY)
        }

        /// 带种子版:seed_d2 = 真实距离的**上界²**(调用方须保证 ≥ d²)——
        /// 堆顶 ≥ best 的剪枝从第一次比较就生效,远点查询的堆展开从数千节点
        /// 砍到最近邻域。实测:15.5 万网格薄叶片几何 30–100µs/键 → 个位数 µs。
        /// 种子来源:BakeGrid 值 + 2.5·h_c(EDT 界 d−h ≲ grid ≲ d+1.8h ⇒ seed ≥ d)。
        pub fn closest_features_seeded(&self, p: [Real; 3], seed_d2: f64) -> (Real, u32, u8, [Real; 3]) {
            let mut best = seed_d2;
            let mut best_t = 0u32;
            let mut best_feat = 6u8;
            let mut best_cp = [0.0; 3];
            // 播种:网格种子(粗距 + 2.5h_c 上界)直接作堆剪枝初值。
            // (旧"贪心下降"实测 1.24ms/调用 = 80% 耗时,且与网格种子功能重复,已删;
            //  网格 EDT 界 d−h ≲ grid ≲ d+1.8h ⇒ seed = grid+2.5h ≥ d,上界成立)
            // 有界栈 **DFS + 剪枝**(替代 best-first 堆)。
            // 动机(VTune):每次查询新建 BinaryHeap 的分配/扩容/释放占构建 CPU ~22%,
            // pop 的 sift 另占 ~34%。种子已紧(粗距 + 2.5h_c 上界),`d2 >= best` 剪枝
            // 与 best-first 排除的是同一批子树,故**遍历结果相同**;DFS 前沿只有 O(树深),
            // 固定数组即可,免分配、免 sift。近子后入栈 ⟹ 近子先展开(先收紧 best)。
            const DFS_CAP: usize = 64; // ≥ 2× 最大树深
            let mut stack = [(0.0f64, 0u32); DFS_CAP];
            let mut sp = 1usize;
            stack[0] = (0.0, self.root);
            while sp > 0 {
                sp -= 1;
                let (d2_node, ni) = stack[sp];
                if d2_node >= best {
                    continue; // 该子树不可能有更近点
                }
                    let n = &self.bvh[ni as usize];
                    if n.cnt > 0 {
                        for &t in &self.order[n.start as usize..(n.start + n.cnt) as usize] {
                            let tri = &self.tris[t as usize];
                            // 三角形自身 AABB 早退(预计算,免每次 18 次 min/max)
                            let (tmn, tmx) = &self.tri_aabb[t as usize];
                            let mut dd2 = 0.0;
                            for k in 0..3 {
                                let dv = if p[k] < tmn[k] {
                                    tmn[k] - p[k]
                                } else if p[k] > tmx[k] {
                                    p[k] - tmx[k]
                                } else {
                                    0.0
                                };
                                dd2 += dv * dv;
                            }
                            if dd2 >= best {
                                continue;
                            }
                            let (cp, feat) = closest_point_feature(p, tri);
                            let dd = dot3(sub3(cp, p), sub3(cp, p));
                            if dd < best {
                                best = dd;
                                best_t = t;
                                best_feat = feat;
                                best_cp = cp;
                            }
                        }
                    } else {
                        let mut cs = [(0.0f64, 0u32); 2];
                        for (idx, c) in [n.left, n.right].iter().enumerate() {
                            let cn = &self.bvh[*c as usize];
                            let mut d2 = 0.0;
                            for k in 0..3 {
                                let dv = if p[k] < cn.mn[k] {
                                    cn.mn[k] - p[k]
                                } else if p[k] > cn.mx[k] {
                                    p[k] - cn.mx[k]
                                } else {
                                    0.0
                                };
                                d2 += dv * dv;
                            }
                            cs[idx] = (d2, *c);
                        }
                        // 远者先入、近者后入 → 近者先弹出
                        let order = if cs[0].0 <= cs[1].0 { [1usize, 0] } else { [0, 1] };
                        for idx in order {
                            let (d2, c) = cs[idx];
                            if d2 < best && sp < DFS_CAP {
                                stack[sp] = (d2, c);
                                sp += 1;
                            }
                        }
                    }
                }
            (best.sqrt(), best_t, best_feat, best_cp)
        }

        /// 伪法线有符号距离(Bærentzen & Aanæs 2005):最近特征点积伪法线定号,
        /// **零射线**——水密流形网格上符号判定精确(剪影轮廓为测度零例外)。
        /// 动机(实测):15.5 万三角网格上一条奇偶射线 20–100µs,数千万键的
        /// 射线成本是数量级级瓶颈;最近点查询本来就找到了特征,符号免费。
        /// 非水密/绕向不一致网格:破洞附近符号局部不可靠(与射线法同口径)。
        pub fn signed_distance_pn(&self, p: [Real; 3]) -> Real {
            let (d, t, feat, cp) = self.closest_features(p);
            let _ = d;
            self.signed_from_feature(p, t, feat, cp)
        }

        // 闭合输入使用有限三角面交点的奇偶性。固定方向便于对共享棱执行
        // 一致的半开规则，无需抖动采样位置或以多次射线投票。
        fn inside_closed(&self, p: [Real; 3]) -> bool {
            let mut odd = false;
            let mut stack = [0u32; 64];
            stack[0] = self.root;
            let mut count = 1;
            while count != 0 {
                count -= 1;
                let node = &self.bvh[stack[count] as usize];
                if node.mx[0] <= p[0] || p[1] < node.mn[1] || p[1] > node.mx[1]
                    || p[2] < node.mn[2] || p[2] > node.mx[2] { continue; }
                if node.cnt == 0 {
                    stack[count] = node.left;
                    stack[count + 1] = node.right;
                    count += 2;
                } else {
                    for &id in &self.order[node.start as usize..(node.start + node.cnt) as usize] {
                        odd ^= crosses_x(p, &self.tris[id as usize]);
                    }
                }
            }
            odd
        }

        /// 射线奇偶核心:给定方向的穿越计数(偶=外,奇=内)。要求水密网格。
        fn parity_hits(&self, p: [Real; 3], dir: [Real; 3]) -> u32 {
            let inv = [1.0 / dir[0], 1.0 / dir[1], 1.0 / dir[2]];
            let mut hits = 0u32;
            let mut stack = [0u32; 128];
            let mut sp = 0usize;
            stack[sp] = self.root;
            sp += 1;
            while sp > 0 {
                sp -= 1;
                let ni = stack[sp];
                let n = &self.bvh[ni as usize];
                // 一般化 slab 测试(t 范围与 [0, ∞) 相交)
                let (mut tmin, mut tmax) = (0.0f64, f64::INFINITY);
                let mut miss = false;
                for k in 0..3 {
                    let t1 = (n.mn[k] - p[k]) * inv[k];
                    let t2 = (n.mx[k] - p[k]) * inv[k];
                    let (tn, tf) = if t1 <= t2 { (t1, t2) } else { (t2, t1) };
                    if tn > tmin { tmin = tn; }
                    if tf < tmax { tmax = tf; }
                    if tmin > tmax {
                        miss = true;
                        break;
                    }
                }
                if miss || tmax < 0.0 {
                    continue;
                }
                if n.cnt > 0 {
                    for &t in &self.order[n.start as usize..(n.start + n.cnt) as usize] {
                        if ray_hits_tri(p, dir, &self.tris[t as usize]) {
                            hits += 1;
                        }
                    }
                } else {
                    stack[sp] = n.left;
                    sp += 1;
                    stack[sp] = n.right;
                    sp += 1;
                }
            }
            hits
        }

        /// 射线奇偶:偶数=外,奇数=内(要求水密网格;固定对角射线避开顶点)。
        /// 栈数组替代 per-call Vec 堆分配。现主要作伪法线符号的交叉验证与
        /// 水密性诊断;热路径符号已由 signed_distance_pn 承担。
        pub fn ray_parity_inside(&self, p: [Real; 3]) -> bool {
            if self.use_even_odd { return self.inside_closed(p); }
            self.parity_hits(p, PARITY_DIR) % 2 == 1
        }

        /// 多方向射线奇偶投票:叶片缝隙/锐边处伪法线判号不可靠(见
        /// signed_from_feature),用全局穿越计数仲裁。
        ///
        /// 方向按**位置哈希**伪随机取、原点微抖:固定方向集会与规则三角化
        /// 网格(平面阵列)系统性掠射——桌腿实测固定 7 向在模型底面成片
        /// 擦边错号,提取出 2134 个碎屑分量;随机方向逐点不同,擦边残余
        /// 退化为孤立错号格点,不连片、可被碎屑清理兜住。
        pub fn parity_inside_vote(&self, p: [Real; 3]) -> bool {
            // splitmix64:零依赖、分布均匀、逐点不同
            let mut s = p[0].to_bits()
                ^ p[1].to_bits().rotate_left(17)
                ^ p[2].to_bits().rotate_left(34);
            let mut next = || {
                s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^ (z >> 31)
            };
            // 原点抖动:避开与表面顶点/边精确共线(1e-4·尺寸 ≪ 任何几何特征)
            const JITTER: f64 = 1e-4;
            const VOTES: usize = 5;
            let mut inside = 0usize;
            for _ in 0..VOTES {
                // 均匀 [0,1):指数位固定 1.0、尾数取随机位,减 1 得 [0,1)
                let mut u = || f64::from_bits(0x3FF0_0000_0000_0000 | (next() >> 12)) - 1.0;
                // 均匀球面方向:z 均匀取 [-1,1],φ 均匀取 [0,2π)
                let z = u() * 2.0 - 1.0;
                let phi = u() * 2.0 * std::f64::consts::PI;
                let r = (1.0 - z * z).sqrt();
                let dir = [r * phi.cos(), r * phi.sin(), z];
                let q = [
                    p[0] + (u() * 2.0 - 1.0) * JITTER,
                    p[1] + (u() * 2.0 - 1.0) * JITTER,
                    p[2] + (u() * 2.0 - 1.0) * JITTER,
                ];
                inside += (self.parity_hits(q, dir) % 2 == 1) as usize;
            }
            inside * 2 > VOTES
        }

        pub fn signed_distance(&self, p: [Real; 3]) -> Real {
            let d = self.unsigned_distance(p);
            if self.ray_parity_inside(p) {
                -d
            } else {
                d
            }
        }

        pub fn sd_cached(&self, p: [Real; 3]) -> Real {
            let key = [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()];
            let mut c = self.cache.lock().unwrap();
            if let Some(v) = c.get(&key) {
                return *v;
            }
            let v = self.signed_distance(p);
            c.insert(key, v);
            v
        }
    }

    impl super::Sdf for MeshField {
        fn sd(&self, p: [Real; 3]) -> Real {
            self.sd_cached(p)
        }
    }

    /// UV 球网格(测试/标定用)。绕向一致外向(伪法线符号法要求一致绕向;
    /// 生成时按"面法线 · 质心径向 > 0"定向,水密)。
    pub fn uv_sphere(c: [Real; 3], r: Real, slices: usize, stacks: usize) -> Vec<[[Real; 3]; 3]> {
        let mut tris = Vec::new();
        let pt = |i: usize, j: usize| -> [Real; 3] {
            let phi = (i as Real) / (stacks as Real) * std::f64::consts::PI;
            let theta = (j as Real) / (slices as Real) * 2.0 * std::f64::consts::PI;
            let sp = phi.sin();
            [c[0] + r * sp * theta.cos(), c[1] + r * sp * theta.sin(), c[2] + r * phi.cos()]
        };
        // 面法线指向外侧(与质心径向同侧)
        let outward = |mut t: [[Real; 3]; 3]| -> [[Real; 3]; 3] {
            let n = cross3(sub3(t[1], t[0]), sub3(t[2], t[0]));
            let ctr = scale3(add3(add3(t[0], t[1]), t[2]), 1.0 / 3.0);
            let radial = sub3(ctr, c);
            if dot3(n, radial) < 0.0 {
                t.swap(1, 2);
            }
            t
        };
        for i in 0..stacks {
            for j in 0..slices {
                let a = pt(i, j);
                let b = pt(i + 1, j);
                let d = pt(i, j + 1);
                let e = pt(i + 1, j + 1);
                if i > 0 {
                    tris.push(outward([a, b, d]));
                }
                if i + 1 < stacks {
                    tris.push(outward([d, b, e]));
                }
            }
        }
        tris
    }
}

impl Sdf for Stock {
    fn sd(&self, p: [Real; 3]) -> Real {
        Stock::sd(self, p)
    }
}

pub trait Sdf {
    fn sd(&self, p: [Real; 3]) -> Real;
}

#[cfg(test)]
mod stl_tests {
    use super::*;
    use meshfield::MeshField;

    #[test]
    fn stl_loader_binary_roundtrip() {
        // 手工构造 2 三角二进制 STL → 加载 → 断言
        let mut data = vec![0u8; 80];
        data.extend_from_slice(&2u32.to_le_bytes());
        for tri in 0..2u32 {
            for f in [
                0.0f32,
                1.0,
                0.0,
                tri as f32, // 平移区分两个三角形
                0.0,
                0.0,
                1.0f32,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ] {
                data.extend_from_slice(&f.to_le_bytes());
            }
            data.extend_from_slice(&0u16.to_le_bytes());
        }
        std::fs::create_dir_all("target").unwrap();
        std::fs::write("target/_test_bin.stl", &data).unwrap();
        let m = stlio::load_stl("target/_test_bin.stl").unwrap();
        assert_eq!(m.tris.len(), 2);
        assert!((m.tris[1][0][0] - 1.0).abs() < 1e-6, "平移分量应被读出");
    }

    #[test]
    fn stl_loader_rejects_garbage() {
        let bad = b"this is not an stl file at all......";
        std::fs::write("target/_test_bad.stl", bad).unwrap();
        assert!(stlio::load_stl("target/_test_bad.stl").is_err());
    }

    #[test]
    fn mesh_field_uv_sphere_sign_and_distance() {
        let tris = meshfield::uv_sphere([0.0; 3], 1.0, 24, 12);
        let (nt, wt) = stlio::watertight_report(&tris);
        assert!(nt > 400 && wt, "UV 球应水密:{nt} 三角, 水密={wt}");
        let f = MeshField::new(tris);
        // 中心为深材料
        assert!(f.signed_distance([0.0; 3]) < -0.8, "中心应为材料");
        // 外部为空气
        assert!(f.signed_distance([2.0, 0.0, 0.0]) > 0.8, "球外 1mm 应为空气");
        // 表面附近 ≈ 0(网格弦差 + 网格量化以内)
        assert!(f.signed_distance([1.0, 0.0, 0.0]).abs() < 0.1, "表面点应 ≈0");
        // 与解析球对拍(随机点)
        let mut rng = Rng::new(99);
        for _ in 0..200 {
            let p = [
                rng.f01() * 4.0 - 2.0,
                rng.f01() * 4.0 - 2.0,
                rng.f01() * 4.0 - 2.0,
            ];
            let analytic = len(sub(p, [0.0; 3])) - 1.0;
            let got = f.signed_distance(p) as Real;
            assert!(
                (got - analytic).abs() < 0.12,
                "p={p:?} 解析={analytic:.4} 网格场={got:.4}(24×12 UV 球弦差内)"
            );
        }
    }

    #[test]
    fn mesh_field_adf_end_to_end() {
        // UV 球 → MeshField → ADF → 切削 → eval 语义验证
        let tris = meshfield::uv_sphere([0.0; 3], 1.0, 24, 12);
        let mut a = Adf::build_with(
            Composite { base: BaseField::Mesh(MeshField::new(tris)), cuts: Vec::new() },
            [-2.0, -2.0, -2.0],
            4.0,
            0.005,
            0.2,
            6,
        );
        // 中心仍为材料(带限结构:深内部只是符号为负的粗格近似)
        assert!(a.eval([0.0; 3]) < 0.0, "中心应为材料(符号)");
        // 切掉上半球(大胶囊沿 x 轴穿过 y=0, z=0.5, r=0.5)
        a.add_cut(Capsule { a: [-2.0, 0.0, 0.5], b: [2.0, 0.0, 0.5], r: 0.5 });
        // 切削轴心处:原为材料,现应为空气(eval > 0)
        let v = a.eval([0.0, 0.0, 0.5]) as Real;
        assert!(v > 0.0, "切削轴心应为空气:{v}");
        // 底极点表面附近:符号与幅值都应正确(带限内、精化区)
        let pole = a.eval([0.0, 0.0, -1.05]) as Real;   // 球底外 0.05 → 空气
        assert!(pole > 0.0, "球底外 0.05 应为空气:{pole}");
        let inner = a.eval([0.0, 0.0, -0.95]) as Real;  // 球内 0.05 → 材料
        assert!(inner < 0.0, "球内 0.05 应为材料:{inner}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIN: [Real; 3] = [-4.0, -4.0, -4.0];
    const SIZE: Real = 8.0;

    /// P23 回归:刀路穿过**深材料粗叶**(需细分 ≥2 层)时,refresh 局部子树的内层
    /// 索引必须打标。旧版把它们当全局索引,孔壁外整片材料被接成远场空气叶;
    /// 子索引不越界,tree_index_validator 抓不到 —— 只能对照真值。
    #[test]
    fn refresh_deep_subdivision_matches_truth_at_cut_wall() {
        let mut a = Adf::build(
            Stock::RoundBox { c: [0.0; 3], half: [8.0; 3], r: 2.0 },
            [-12.0; 3],
            24.0,
            0.005,
            0.5,
            8,
        );
        a.add_cut(Capsule { a: [-14.0, 0.0, 0.0], b: [14.0, 0.0, 0.0], r: 3.0 });
        let mut rng = Rng::new(2323);
        let (mut max_e, mut n_sign) = (0.0f64, 0u32);
        for _ in 0..4000 {
            let x = -6.0 + 12.0 * rng.f01();
            let th = std::f64::consts::TAU * rng.f01();
            let rr = 3.0 + (rng.f01() - 0.5) * 0.6;
            let p = [x, rr * th.cos(), rr * th.sin()];
            let truth = a.true_sd(p);
            let got = a.eval(p) as Real;
            max_e = max_e.max((got - truth).abs());
            if truth.abs() > 0.02 && (got < 0.0) != (truth < 0.0) {
                n_sign += 1;
            }
        }
        assert_eq!(n_sign, 0, "孔壁邻域出现错号点(refresh 局部索引错接?)max_e={max_e}");
        assert!(max_e < 0.05, "孔壁邻域 ADF 偏离复合场真值 {max_e} mm");
        for n in a.nodes.iter() {
            if let Node::Internal(ch) = n {
                assert!(ch.iter().all(|&c| (c as usize) < a.nodes.len() && !is_local(c)));
            }
        }
    }

    #[test]
    fn certified_far_leaf_cannot_cross_zero_from_a_coarse_bake_sample() {
        let stock = BaseField::Stock(Stock::RoundBox { c: [5.0; 3], half: [1.0; 3], r: 0.0 });
        let grid = BakeGrid { min: [0.0; 3], n: 2, h: 1.0, data: vec![-0.8, 0.9, 1.4, 1.2, 0.7, 1.1, 1.1, 1.2] };
        let mut nodes = Vec::new();
        let id = far_leaf(&mut nodes, &ExactSrc::Base(&stock), Some(&grid), 1.0, 0.4, [0.0; 3], 1.0,
                          4, &mut BuildStats::default());
        let Node::Leaf(values) = &nodes[id as usize] else { panic!("expected leaf"); };
        assert!(values.iter().all(|&d| d > 0.0), "certified exterior has signs {values:?}");
    }

    #[test]
    fn paired_edges_reject_even_nonmanifold_incidence() {
        let p = [[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.]];
        let shell: Vec<_> = [[0,2,1], [0,1,3], [0,3,2], [1,2,3]].map(|t| t.map(|id| p[id])).to_vec();
        assert!(stlio::watertight_report(&shell).1);
        assert!(meshfield::MeshField::new(shell.clone()).has_paired_edges);
        let doubled = shell.repeat(2);
        assert!(!stlio::watertight_report(&doubled).1);
        assert!(!meshfield::MeshField::new(doubled).has_paired_edges);
    }

    #[test]
    fn closed_shell_with_reversed_face_has_consistent_sign_on_both_sides() {
        let p = [[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.]];
        // The z=0 face has reversed winding, although every edge still has two uses.
        let shell = [[0,1,2], [0,1,3], [0,3,2], [1,2,3]].map(|t| t.map(|id| p[id]));
        let mesh = meshfield::MeshField::new(shell.to_vec());
        assert!(mesh.signed_distance_pn([0.2, 0.2, -0.05]) > 0.0);
        assert!(mesh.signed_distance_pn([0.2, 0.2, 0.05]) < 0.0);
    }

    #[test]
    fn roundtrip_separates_close_solids_and_preserves_a_cavity() {
        let p = [[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.]];
        let outer = [[0,2,1], [0,1,3], [0,3,2], [1,2,3]].map(|t| t.map(|id| p[id]));
        let inner = outer.map(|t| t.map(|v| v.map(|x| x * 0.2 + 0.1)));
        let mut solids = outer.to_vec();
        solids.extend(inner);
        let layers = meshfield::MeshField::new(solids).into_roundtrip_layers(0.0625);
        assert_eq!(layers.len(), 2);
        // Both solids contain this lattice node; global parity would erase one surface.
        assert!(layers.iter().all(|mesh| mesh.signed_distance_pn([0.125; 3]) < 0.0));

        let mut hollow = outer.to_vec();
        hollow.extend(inner.map(|[a,b,c]| [a,c,b]));
        let layers = meshfield::MeshField::new(hollow).into_roundtrip_layers(0.0625);
        assert_eq!(layers.len(), 1);
        assert!(layers[0].signed_distance_pn([0.125; 3]) > 0.0);
    }

    #[test]
    fn roundtrip_separates_solids_touching_only_at_one_vertex() {
        let p = [[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.]];
        let outer = [[0,2,1], [0,1,3], [0,3,2], [1,2,3]].map(|t| t.map(|id| p[id]));
        let inner = outer.map(|t| t.map(|v| v.map(|x| x * 0.5)));
        let mut input = outer.to_vec();
        input.extend(inner);
        let layers = meshfield::MeshField::new(input).into_roundtrip_layers(0.0625);
        assert_eq!(layers.len(), 2, "point contact must not cancel the two interiors");
        assert!(layers.iter().all(|m| m.signed_distance_pn([0.125; 3]) < 0.0));
    }

    #[test]
    fn weld_tolerance_does_not_certify_an_open_crack() {
        let p = [[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.]];
        let mut shell = [[0,2,1], [0,1,3], [0,3,2], [1,2,3]].map(|t| t.map(|id| p[id]));
        shell[0][0][2] += 1e-6;
        assert!(stlio::watertight_report(&shell).1, "weld-only report");
        assert!(!meshfield::MeshField::new(shell.to_vec()).has_closed_surface());
    }

    #[test]
    fn parity_ray_through_face_diagonal_counts_one_crossing() {
        let corners: [[Real; 3]; 8] = std::array::from_fn(|i|
            [(i & 1) as Real, ((i >> 1) & 1) as Real, ((i >> 2) & 1) as Real]
        );
        let mut cube = Vec::new();
        for [a,b,c,d] in [[0,4,6,2], [1,3,7,5], [0,1,5,4], [2,6,7,3], [0,2,3,1], [4,5,7,6]] {
            cube.push([a,b,c].map(|i| corners[i]));
            cube.push([a,c,d].map(|i| corners[i]));
        }
        cube[0].swap(1, 2);
        let mesh = meshfield::MeshField::new(cube);
        assert!(mesh.signed_distance_pn([0.5, 0.25, 0.25]) < 0.0);
        assert!(mesh.signed_distance_pn([-0.5, 0.25, 0.25]) > 0.0);
    }

    fn small_adf() -> Adf {
        // 圆角盒 half=2 r=0.5;h_min = 8/2^7 = 0.0625mm;壳层带宽 0.3mm
        Adf::build(
            Stock::RoundBox { c: [0.0; 3], half: [2.0; 3], r: 0.5 },
            MIN,
            SIZE,
            0.005,
            0.3,
            7,
        )
    }

    #[test]
    fn eval_center_is_material() {
        let a = small_adf();
        let v = a.eval([0.0, 0.0, 0.0]);
        assert!(v < -1.5, "中心应为深材料,得到 {v}");
    }

    #[test]
    fn surface_points_near_zero() {
        let a = small_adf();
        let mut rng = Rng::new(42);
        let mut max_abs = 0.0f32;
        for _ in 0..500 {
            // 采样圆角盒表面:简化为顶面 z = +2.5(r=0.5 → 外沿 2.5)
            let x = (rng.f01() * 2.0 - 1.0) * 1.8;
            let y = (rng.f01() * 2.0 - 1.0) * 1.8;
            let v = a.eval([x, y, 2.5]);
            max_abs = max_abs.max(v.abs());
        }
        assert!(max_abs < 0.03, "顶面(平坦区)读数偏差应≈0,得到 {max_abs}");
    }

    #[test]
    fn cut_makes_air_and_keeps_untouched() {
        let mut a = small_adf();
        let probe = [0.0, 1.5, 0.0]; // 未触区域(壳层外,粗格)
        let before = a.eval(probe);
        a.add_cut(Capsule { a: [-1.5, 0.0, 2.0], b: [1.5, 0.0, 2.0], r: 0.4 });
        let after_untouched = a.eval(probe);
        assert_eq!(before.to_bits(), after_untouched.to_bits(), "未触区域必须逐位不变");

        let air = a.eval([0.0, 0.0, 2.0]); // 胶囊轴心 → 已切为空气(true = +0.4)
        assert!(air > 0.3, "刀轴心应为空气,得到 {air}");
        let deep = a.eval([0.0, 0.0, 0.0]); // 切深以下仍是材料(true = max(-2.5,-1.6) = -1.6)
        assert!(deep < -1.0, "切深下方应仍为材料,得到 {deep}");
    }

    #[test]
    fn determinism() {
        let a = small_adf();
        let b = small_adf();
        assert_eq!(a.nodes.len(), b.nodes.len());
        for (x, y) in a.nodes.iter().zip(b.nodes.iter()) {
            match (x, y) {
                (Node::Leaf(p), Node::Leaf(q)) => {
                    for i in 0..8 {
                        assert_eq!(p[i].to_bits(), q[i].to_bits());
                    }
                }
                (Node::Internal(p), Node::Internal(q)) => assert_eq!(p, q),
                _ => panic!("节点类型不一致"),
            }
        }
    }

    #[test]
    fn max_depth_respected() {
        let a = Adf::build(
            Stock::Sphere { c: [0.0; 3], r: 1.0 },
            MIN,
            SIZE,
            1e-6,   // 极小 eps → 逼到 max_depth
            0.3,    // band
            5,
        );
        let (_, _, hist) = a.leaf_stats_bytes();
        println!("DBG hist={hist:?}");
        for (d, n) in hist.iter().enumerate() {
            if *n > 0 {
                assert!((d as u8) <= 5, "出现超过 max_depth 的叶");
            }
        }
    }

    #[test]
    fn redistance_fixes_superseded_underestimate() {
        let mut a = Adf::build(
            Stock::RoundBox { c: [0.0; 3], half: [2.0; 3], r: 0.5 },
            MIN,
            SIZE,
            0.005,
            0.3,
            7,
        );
        // cut1:窄通道(floor z=1.5)→ cut2:更宽更深的口袋(r=0.6)超越 cut1 的 floor
        a.add_cut(Capsule { a: [-1.5, 0.0, 1.8], b: [1.5, 0.0, 1.8], r: 0.3 });
        a.add_cut(Capsule { a: [-1.5, 0.0, 1.2], b: [1.5, 0.0, 1.2], r: 0.6 });
        let probe = [0.0, 0.0, 1.7]; // cut1 的旧 floor 上方空气(已被 cut2 超越)
        let before = a.eval(probe) as Real;
        // 真值:cut2 腔在 z=1.7 的半径 = √(0.6²−0.5²) ≈ 0.3317 → 壁距 ≈ 0.3317
        let st = a.redistance(8);
        let after = a.eval(probe) as Real;
        assert!(st.corrected > 0, "应有角点被修正");
        assert!(after > before + 0.03, "重距离化应抬高被超越的下界:{before} → {after}");
        assert!(
            (after - 0.3317).abs() < 0.08,
            "重距离化后应接近真值 0.3317(实际 {after},含 ±h/2=±0.031 量化)"
        );
        // 平坦面零点偏置保持(写回守卫:界面邻域不动)
        let bias = a.eval([0.0, 1.5, 2.5]) as Real;
        assert!(bias.abs() < 0.01, "平坦面偏置应保持 ≈0:{bias}");
    }

    #[test]
    fn coarsen_reduces_leaves_and_keeps_field() {
        // 场景:两道宽深切削(r=1.2)挖出大口袋 → 旧顶面壳层细格深藏袋中成为
        // 过度细化 → 粗化回收;壳层守卫保证口袋壁附近不合并
        let mut a = Adf::build(
            Stock::RoundBox { c: [0.0; 3], half: [2.0; 3], r: 0.5 },
            MIN,
            SIZE,
            0.005,
            0.3,
            7,
        );
        for zc in [1.6f64, 0.4] {
            a.add_cut(Capsule { a: [-1.5, 0.0, zc], b: [1.5, 0.0, zc], r: 1.2 });
        }
        let probes = [[0.0, 0.0, 1.5], [0.0, 0.0, 0.5], [0.0, 1.5, 0.0], [1.9, 1.9, 1.9]];
        let before: Vec<f32> = probes.iter().map(|p| a.eval(*p)).collect();
        let (n1, _, _) = a.leaf_stats_bytes();
        let merged = a.coarsen_with(0.02);
        let (n2, _, _) = a.leaf_stats_bytes();
        assert!(merged > 0, "大口袋空气区应有过度细化格可回收");
        assert!(n2 < n1, "粗化应减少叶数:{n1} → {n2}(合并 {merged})");
        // 口袋壁附近保持细格(壁在 |y| ≈ 1.16, z=0.7)
        assert!(a.depth_at([0.0, 1.0, 0.7]) >= 5, "口袋壁邻域被粗化");
        for (p, b) in probes.iter().zip(before.iter()) {
            let aft = a.eval(*p);
            assert!(
                (aft - b).abs() <= 0.05,
                "粗化后场值漂移超限:p={p:?} before={b} after={aft}"
            );
        }
    }

    #[test]
    fn tree_index_validator() {
        // 构建后校验全部子索引;切削(并行 refresh)后再校验,定位损坏阶段
        let mut adf = Adf::build(
            Stock::RoundBox { c: [0.0; 3], half: [8.0; 3], r: 2.0 },
            [-12.0; 3],
            24.0,
            0.005,
            0.5,
            8,
        );
        let check = |nodes: &[Node]| {
            for (i, n) in nodes.iter().enumerate() {
                if let Node::Internal(ch) = n {
                    for c in ch {
                        if (*c as usize) >= nodes.len() {
                            eprintln!("[dbg] 损坏:节点 {i} 子 {c} 越界 len={}", nodes.len());
                            for (di, n2) in nodes[i.saturating_sub(3)..(i + 4).min(nodes.len())].iter().enumerate() {
                                let ni2 = i.saturating_sub(3) + di;
                                match n2 {
                                    Node::Internal(ch) => eprintln!(
                                        "[dbg]   节点 {ni2} Internal 子 {:?}",
                                        ch.iter().map(|x| *x as usize).collect::<Vec<_>>()
                                    ),
                                    Node::Leaf(_c2) => eprintln!("[dbg]   节点 {ni2} Leaf"),
                                }
                            }
                            panic!("越界");
                        }
                    }
                }
            }
        };
        check(&adf.nodes);
        eprintln!("[dbg] 构建后校验通过 len={}", adf.nodes.len());
        adf.add_cut(Capsule { a: [-5.0, 0.0, 9.0], b: [5.0, 0.0, 9.0], r: 1.5 });
        eprintln!("[dbg] 切削完成 len={}", adf.nodes.len());
        check(&adf.nodes);
        eprintln!("[dbg] 切削后校验通过");
    }


    /// 绕向统一:一个立方体里故意翻反一个三角,必须被纠正回来,
    /// 且纠正后每条边都被两个三角**反向**各走一次(无同向边)。
    #[test]
    fn orient_consistently_fixes_flipped_triangle() {
        let v: Vec<[f32; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        // 外法线朝外的立方体
        let t: Vec<[u32; 3]> = vec![
            [0, 2, 1], [0, 3, 2], // z=0
            [4, 5, 6], [4, 6, 7], // z=1
            [0, 1, 5], [0, 5, 4], // y=0
            [3, 7, 6], [3, 6, 2], // y=1
            [0, 4, 7], [0, 7, 3], // x=0
            [1, 2, 6], [1, 6, 5], // x=1
        ];
        let mut m = nets::Mesh { verts: v, tris: t };
        let vol0 = m.signed_volume();
        assert!(vol0 > 0.0, "初始绕向应为外向 vol={vol0}");
        // 故意翻反 3 个
        for i in [0usize, 5, 11] {
            m.tris[i].swap(1, 2);
        }
        let flipped = m.orient_consistently();
        // 计数不必等于 3:洪泛可能先把整块调成"统一但朝内",再由整组体积检查翻回来,
        // 于是同一条边会被翻两次。真正要守住的是下面的朝向与逐边一致性。
        assert!(flipped >= 3, "至少应改动 3 个三角,实测 {flipped}");
        assert!((m.signed_volume() - vol0).abs() < 1e-9, "体积应保持");
        // 逐边检查:必须一条同向边都没有
        let mut dir: std::collections::HashMap<(u32, u32), [u32; 2]> =
            std::collections::HashMap::new();
        for t in &m.tris {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                let k = if a < b { (a, b) } else { (b, a) };
                dir.entry(k).or_insert([0, 0])[if a < b { 0 } else { 1 }] += 1;
            }
        }
        for (k, c) in dir {
            assert_eq!(c, [1, 1], "边 {k:?} 两向不应各一次,实测 {c:?}");
        }
    }

    /// 带外定号必须**区分材料内部与封闭空腔**:前者不可达且邻带为负,后者不可达
    /// 但邻带为正。合成:外层球 r=13 是"实体/空气"界面(内负外正),内层球 r=5
    /// 是封闭空腔(内正外负)。期望:r>14.5 正、6.5<r<11.5 负、r<3.5 正。
    /// 只按"不可达即内部"实现时,空腔会被判成材料 —— 这条测试正是那个回归点。
    #[test]
    fn flood_sign_distinguishes_cavity_from_material() {
        const N: usize = 41;
        const R1: f32 = 13.0;
        const R2: f32 = 5.0;
        let c0 = (N / 2) as f32;
        let mut phi = vec![f32::INFINITY; N * N * N];
        for k in 0..N {
            for j in 0..N {
                for i in 0..N {
                    let r = (((i as f32 - c0).powi(2)
                        + (j as f32 - c0).powi(2)
                        + (k as f32 - c0).powi(2)) as f32)
                        .sqrt();
                    let v = if (r - R1).abs() <= 1.5 {
                        Some(r - R1) // 实体外表面:外正内负
                    } else if (r - R2).abs() <= 1.5 {
                        Some(R2 - r) // 封闭空腔:内正外负
                    } else {
                        None
                    };
                    if let Some(v) = v {
                        phi[(k * N + j) * N + i] = v;
                    }
                }
            }
        }
        let sign = flood_outside_sign(&phi, N);
        let at = |x: usize, y: usize, z: usize| sign[(z * N + y) * N + x];
        assert_eq!(at(0, 0, 0), 1, "域边界必须是外部");
        assert_eq!(at(N / 2, N / 2, N / 2), 1, "封闭空腔内部应为空腔(+1)");
        assert_eq!(at(N / 2 + 9, N / 2, N / 2), -1, "实体层(r=9)应为材料(-1)");
        // 带内:必须沿用精确 phi 符号
        assert_eq!(at(N / 2 + 13, N / 2, N / 2), 1, "外表面带外侧应为正");
        assert_eq!(at(N / 2 + 12, N / 2, N / 2), -1, "外表面带内侧应为负");
    }

    /// 轴对齐平面:EDT 必须**精确**复现(种子为 v=0 的整层格点,不存在离散化鼓包)。
    /// 这是对三轴合成的硬校验:任何一轴的下抛物线包络写错都会破坏它。
    #[test]
    fn edt_redist_exact_for_axis_aligned_plane() {
        let gn = [24usize, 20, 18];
        let (nx, ny, nz) = (gn[0], gn[1], gn[2]);
        let hf = 0.1f32;
        let c0 = 11.0f32;
        let mut d = vec![f32::INFINITY; nx * ny * nz];
        for k in 0..nz {
            for j in 0..ny {
                d[(k * ny + j) * nx + 11] = 0.0;
            }
        }
        edt_redist(&mut d, gn, hf, 8);
        let mut maxdev = 0.0f32;
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let truth = (i as f32 - c0).abs() * hf;
                    let v = d[(k * ny + j) * nx + i];
                    maxdev = maxdev.max((v - truth).abs());
                }
            }
        }
        assert!(maxdev <= 1e-5, "轴对齐平面应被精确重建,实测最大偏差 {maxdev}mm");
    }

    /// 斜面:种子只存在于 |d|<1 格的格点层,离散种子覆盖造成系统性高估(鼓包),
    /// 实测 ~1 格。此测试把该量级钉住,防止 EDT 实现退化而不被察觉。
    #[test]
    fn edt_redist_tilted_plane_within_one_cell() {
        let gn = [40usize, 33, 27];
        let (nx, ny, nz) = (gn[0], gn[1], gn[2]);
        let hf = 0.1f32;
        let nrm = [0.6f64, 0.8, 0.0];
        let p0 = [20.0f64, 16.5, 13.0];
        let mut d = vec![f32::INFINITY; nx * ny * nz];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let q = [i as f64, j as f64, k as f64];
                    let dist = (nrm[0] * (q[0] - p0[0]) + nrm[1] * (q[1] - p0[1])).abs();
                    if dist < 1.0 {
                        d[(k * ny + j) * nx + i] = (dist * hf as f64) as f32;
                    }
                }
            }
        }
        let mut got = d.clone();
        edt_redist(&mut got, gn, hf, 8);
        let (mut maxdev, mut sum, mut cnt) = (0.0f64, 0.0f64, 0u64);
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let q = [i as f64, j as f64, k as f64];
                    let truth = (nrm[0] * (q[0] - p0[0]) + nrm[1] * (q[1] - p0[1])).abs()
                        * hf as f64;
                    let v = got[(k * ny + j) * nx + i];
                    assert!(v.is_finite(), "EDT 后仍为 INF,@({i},{j},{k})");
                    let e = (v as f64 - truth).abs();
                    maxdev = maxdev.max(e);
                    sum += e;
                    cnt += 1;
                }
            }
        }
        eprintln!(
            "[edt-redist] 斜面重建 {cnt} 格 | max {:.2} 格 | mean {:.2} 格",
            maxdev / hf as f64,
            sum / cnt as f64 / hf as f64
        );
        assert!(
            maxdev <= 1.2 * hf as f64,
            "斜面距离重建误差 {:.2} 格超过 1.2 格",
            maxdev / hf as f64
        );
    }

    /// EDT 重距离化必须逐位可复现(按线并行的确定性):不同工人数结果一致。
    #[test]
    fn edt_redist_reproducible_across_threads() {
        let gn = [37usize, 29, 31];
        let hf = 0.1f32;
        let mut base = vec![f32::INFINITY; gn[0] * gn[1] * gn[2]];
        let (nx, ny) = (gn[0], gn[1]);
        for k in 0..gn[2] {
            for j in 0..ny {
                for i in 0..nx {
                    if i == nx / 2 && (j + k) % 7 == 0 {
                        base[(k * ny + j) * nx + i] = 0.05;
                    }
                }
            }
        }
        for &nt in &[1usize, 3, 8, 16, 32] {
            let mut got = base.clone();
            edt_redist(&mut got, gn, hf, nt);
            // 与 1 工人(纯串行)对照:逐位一致
            let mut reference = base.clone();
            edt_redist(&mut reference, gn, hf, 1);
            assert_eq!(reference, got, "{nt} 工人的 EDT 结果与串行不一致");
        }
    }

    #[test]
    fn edt_1d_basic() {
        // 8 格线,站点在 3(值 0)与 6(值 0):最近站平方距离
        let mut sq = vec![f64::INFINITY; 8];
        sq[3] = 0.0;
        sq[6] = 0.0;
        let st = [9u32, 9, 9, 3, 9, 9, 6, 9];
        let (d, arg) = edt_1d(&sq, &st);
        let want = [9.0, 4.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        for q in 0..8 {
            assert!((d[q] - want[q]).abs() < 1e-9, "d[{q}]={} want {}", d[q], want[q]);
        }
        assert_eq!(arg[0], 3);
        assert_eq!(arg[5], 6); // 5 与 3 差 2、与 6 差 1 → 站 6
        // 站点带值(f = 平方偏移):站 3 值 10、站 6 值 0
        let mut sq2 = vec![f64::INFINITY; 8];
        sq2[3] = 10.0;
        sq2[6] = 0.0;
        let (d2, a2) = edt_1d(&sq2, &st);
        // q=0: 站3 距 9 + 10 = 19 vs 站6 距 36 → 19(站 3)
        assert!((d2[0] - 19.0).abs() < 1e-9, "d2[0]={}", d2[0]);
        assert_eq!(a2[0], 3);
        // q=5: 站3 4+10=14 vs 站6 1 → 1(站 6)
        assert!((d2[5] - 1.0).abs() < 1e-9);
        assert_eq!(a2[5], 6);
    }

    #[test]
    fn edt_3d_composition() {
        // 4³ 网格,单站 (1,1,1):三轴合成后 d2 = 各轴偏移平方和
        let n = 4usize;
        let mut d2 = vec![f64::INFINITY; 64];
        let mut arg = vec![u32::MAX; 64];
        let site = (1 * n + 1) * n + 1;
        d2[site] = 0.0;
        arg[site] = site as u32;
        edt_axis(&mut d2, &mut arg, n, 0);
        edt_axis(&mut d2, &mut arg, n, 1);
        edt_axis(&mut d2, &mut arg, n, 2);
        for k in 0..4 {
            for j in 0..4 {
                for i in 0..4 {
                    let want = (i as f64 - 1.0).powi(2) + (j as f64 - 1.0).powi(2) + (k as f64 - 1.0).powi(2);
                    let got = d2[(k * n + j) * n + i];
                    assert!(
                        (got - want).abs() < 1e-9,
                        "({i},{j},{k}) got {got} want {want}"
                    );
                    assert_eq!(arg[(k * n + j) * n + i], site as u32);
                }
            }
        }
    }

    #[test]
    fn bake_grid_accuracy() {
        // 烘焙网格 vs 解析球:h = 4/128,量化+扫描总偏差应 ≤ ~2h
        let tris = meshfield::uv_sphere([0.0; 3], 1.0, 32, 16);
        let mf = meshfield::MeshField::new(tris);
        let (g, _) = bake_field(&BaseField::Mesh(mf), [-2.0; 3], 4.0);
        let mut rng = Rng::new(1);
        let mut worst = 0.0f64;
        let mut argp = [0.0; 3];
        for _ in 0..3000 {
            let p = [-2.0 + rng.f01() * 4.0, -2.0 + rng.f01() * 4.0, -2.0 + rng.f01() * 4.0];
            let gv = g.sample_nearest(p) as f64;
            let truth = len(p) - 1.0;
            if (gv - truth).abs() > worst {
                worst = (gv - truth).abs();
                argp = p;
            }
        }
        // 对角格线:角格 (0,0,0) → 球心方向,看波前停滞位置
        let mut prof = String::new();
        for t in (0..128).step_by(12) {
            let v = g.data[(t * 128 + t) * 128 + t];
            let pos = [-2.0 + t as f64 * (4.0 / 128.0); 3];
            prof.push_str(&format!("{:.2}/{:.2} ", v, len(pos) - 1.0));
        }
        println!("[prof diag] 网格/真值: {prof}");
        assert!(worst < 0.07, "烘焙网格最大偏差 {worst:.4} @ {argp:?}(h = 0.031)");
    }

    #[test]
    fn pn_sign_matches_parity_and_analytic() {
        // ① 解析对拍:一致绕向 UV 球的伪法线有符号距离 ≈ 解析球(200 点)
        let tris = meshfield::uv_sphere([0.0; 3], 1.0, 24, 12);
        let mf = meshfield::MeshField::new(tris);
        let mut rng = Rng::new(777);
        for _ in 0..200 {
            let p = [
                (rng.f01() * 2.0 - 1.0) * 1.6,
                (rng.f01() * 2.0 - 1.0) * 1.6,
                (rng.f01() * 2.0 - 1.0) * 1.6,
            ];
            let pn = mf.signed_distance_pn(p);
            let truth = len(p) - 1.0;
            assert!(
                (pn - truth).abs() < 0.03,
                "球伪法线偏差 {pn:.4} vs 真值 {truth:.4} @ {p:?}"
            );
        }
        // ② 交叉验证:真实 STL 壳层带内,伪法线符号 == 射线奇偶符号
        // (避开启音轮廓测度零点:|dot(p−cp, n)| > 0.01 才比较)
        let path = "stls/桌腿固定器.stl";
        let Ok(tm) = stlio::load_stl(path) else {
            eprintln!("跳过:{path} 不存在(测试资产)");
            return;
        };
        let mf2 = meshfield::MeshField::new(tm.tris);
        let mut checked = 0u64;
        for _ in 0..800 {
            let p = [
                mf2.aabb_min[0] + rng.f01() * (mf2.aabb_max[0] - mf2.aabb_min[0]),
                mf2.aabb_min[1] + rng.f01() * (mf2.aabb_max[1] - mf2.aabb_min[1]),
                mf2.aabb_min[2] + rng.f01() * (mf2.aabb_max[2] - mf2.aabb_min[2]),
            ];
            let d = mf2.unsigned_distance(p);
            if d > 0.5 {
                continue; // 只看壳层带(符号才有意义)
            }
            let pn = mf2.signed_distance_pn(p);
            let parity = if mf2.ray_parity_inside(p) { -d } else { d };
            // 伪法线在剪影处可能邻零——只有明显分歧才断言失败
            assert!(
                (pn - parity).abs() < 1e-6 || pn.signum() == parity.signum(),
                "符号分歧 @ {p:?}: pn={pn:.4} parity={parity:.4}"
            );
            checked += 1;
        }
        assert!(checked > 50, "有效样本过少:{checked}");
    }

    /// 域**逐轴**钳位的回归测试:`min` 各轴不同时,`min[0]+size` 绝不能用作
    /// 其他轴的上界。旧实现如此,导致 `eval` 把高坐标钳到边界、落到错误的叶里
    /// (叶轮远场 mean 8.8mm / max 82mm 的真实根因)。
    #[test]
    fn eval_clamps_per_axis_not_axis0() {
        // 域 y∈[-1,3],而 min[0]+size = 2.0 —— 旧代码会把 y=2.9 钳成 2.0
        let a = Adf::build(
            Stock::Sphere { c: [0.0, 2.6, 0.0], r: 0.5 },
            [-2.0, -1.0, -2.0],
            4.0,
            0.005,
            0.2,
            6,
        );
        let p = [0.0, 2.9, 0.0];
        let got = a.eval(p) as Real;
        let truth = a.true_sd(p);
        assert!(truth < 0.0, "该点在球内,真值应为负(实际 {truth})");
        assert!(
            got < 0.0 && (got - truth).abs() < 0.15,
            "y=2.9 被误钳(旧实现钳到 y=2.0 → 正值):got={got} truth={truth}"
        );
    }

    #[test]
    fn coarse_extraction_resolves_two_faces_of_a_thin_plate() {
        // 粗网格只有 z=0 一层位于板内，另一侧面控制该层的距离。
        // 仅线性插值 z=0 与 0.5 的值会把上表面从 0.32 错置到 0.18。
        let adf = Adf::build(
            Stock::RoundBox { c: [0.0, 0.0, 0.11], half: [1.2, 1.2, 0.21], r: 0.0 },
            [-2.0; 3], 4.0, 0.001, 0.2, 6,
        );
        let mesh = nets::extract_uniform(&adf, 0.5);
        let interior: Vec<_> = mesh.verts.iter()
            .filter(|v| v[0].abs() < 0.7 && v[1].abs() < 0.7).collect();
        assert!(interior.iter().any(|v| v[2] > 0.0) && interior.iter().any(|v| v[2] < 0.0));
        for v in interior {
            let expected = if v[2] > 0.0 { 0.32 } else { -0.10 };
            assert!((v[2] - expected).abs() < 0.01, "薄板面位置 {v:?}");
        }
        assert_eq!(mesh.edge_stats(), (0, 0, 1));
    }

    #[test]
    fn flat_leaf_crossings_keep_the_surface_closed() {
        let mut adf = Adf::build(
            Stock::Sphere { c: [0.0; 3], r: 1.1 },
            [-2.0; 3], 4.0, 0.001, 0.3, 5,
        );
        // 相邻叶分别为常量仍可在公共面处异号，局部叶内梯度不能决定连通性。
        for node in &mut adf.nodes {
            if let Node::Leaf(values) = node {
                let center = trilinear([0.5; 3], values);
                *values = [if center < 0.0 { -1.0 } else { 1.0 }; 8];
            }
        }
        let m = nets::extract_uniform(&adf, 0.25);
        assert!(!m.tris.is_empty(), "丢失了全部变号棱");
        assert_eq!(m.edge_stats().0, 0, "平坦叶中的变号棱不能造成开口");
    }

    #[test]
    fn reference_projection_limits_travel_and_preserves_incident_face_orientation() {
        let reference = meshfield::MeshField::new(vec![[
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        ]]);
        let mut mesh = nets::Mesh {
            verts: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 0.4, 0.5]],
            tris: vec![[0, 1, 2]],
        };
        mesh.project_to_reference(&reference, 0.1);
        assert!((mesh.verts[2][2] - 0.4).abs() < 1e-6);
        // 两个固定基点和一个只在 z 轴上错开的顶点，直投影会挤扁该三角。
        mesh.verts[2] = [0.2, 0.0, 0.5];
        mesh.project_to_reference(&reference, 1.0);
        assert!(mesh.verts[2][2] > 0.125 && mesh.verts[2][2] < 0.5);
    }

    #[test]
    fn face_fit_reduces_chord_error_at_the_same_triangle_count() {
        let p = [[0., 0., 0.], [1., 0., 1.], [0., 1., 1.]];
        let q = [[0.5, 0., 0.25], [0.5, 0.5, 0.5], [0., 0.5, 0.25]];
        let source = meshfield::MeshField::new(vec![
            [p[0], q[0], q[2]], [q[0], p[1], q[1]],
            [q[2], q[1], p[2]], [q[0], q[1], q[2]],
        ]);
        let mut mesh = nets::Mesh { verts: p.map(|v| v.map(|x| x as f32)).to_vec(), tris: vec![[0, 1, 2]] };
        let center_distance = |m: &nets::Mesh| {
            let v: Vec<_> = m.verts.iter().map(|p| p.map(Real::from)).collect();
            source.unsigned_distance(scale(add(add(v[0], v[1]), v[2]), 1.0 / 3.0))
        };
        let before = center_distance(&mesh);
        mesh.project_to_reference(&source, 0.5);
        assert_eq!(center_distance(&mesh), before);
        mesh.project_with_face_samples(&source, 0.5, true);
        assert!(center_distance(&mesh) < before * 0.6);
        assert_eq!(mesh.tris, [[0, 1, 2]]);
    }

    #[test]
    fn reference_edge_recovers_a_crossing_hidden_by_a_nearer_parallel_facet() {
        let mesh = meshfield::MeshField::new(vec![
            [[0.1, -1.0, 0.01], [0.9, -1.0, 0.01], [0.5, 1.0, 0.01]],
            [[0.8, -1.0, -1.0], [0.8, 1.0, -1.0], [0.8, 0.0, 1.0]],
        ]);
        let (hit, _) = mesh.edge_plane([0.5, 0.0, 0.0], 0, 0.0, 1.0).expect("finite edge intersects x=0.8");
        assert!((hit[0] - 0.8).abs() < 1e-10);
        assert!(mesh.edge_plane([0.5, 0.0, 0.0], 0, 0.0, 0.7).is_none());
    }

    /// P40:CSR 边表版的绕向统一 / 碎屑清理:细分立方体随机翻转部分面后应恢复一致绕向且体积为正;
    /// 外加一个远处的小四面体,碎屑清理应只丢掉它。
    #[test]
    fn csr_orient_and_debris_cleanup() {
        let n = 6i32;
        let mut idx = std::collections::HashMap::<[i32; 3], u32>::new();
        let mut verts: Vec<[f32; 3]> = Vec::new();
        let mut tris: Vec<[u32; 3]> = Vec::new();
        for ax in 0..3 {
            let (u, v) = ((ax + 1) % 3, (ax + 2) % 3);
            for s in [0, n] {
                for i in 0..n {
                    for j in 0..n {
                        let mut ids = [0u32; 4];
                        for (c, (di, dj)) in [(0, 0), (1, 0), (1, 1), (0, 1)].into_iter().enumerate() {
                            let mut q = [0i32; 3];
                            q[ax] = s;
                            q[u] = i + di;
                            q[v] = j + dj;
                            ids[c] = *idx.entry(q).or_insert_with(|| {
                                verts.push(q.map(|x| x as f32));
                                (verts.len() - 1) as u32
                            });
                        }
                        if s == n {
                            tris.push([ids[0], ids[1], ids[2]]);
                            tris.push([ids[0], ids[2], ids[3]]);
                        } else {
                            tris.push([ids[0], ids[2], ids[1]]);
                            tris.push([ids[0], ids[3], ids[2]]);
                        }
                    }
                }
            }
        }
        let n_cube = tris.len();
        for (f, t) in tris.iter_mut().enumerate() {
            if f % 7 == 3 {
                t.swap(1, 2);
            }
        }
        let b = verts.len() as u32;
        verts.extend([[100.0, 0.0, 0.0], [100.1, 0.0, 0.0], [100.0, 0.1, 0.0], [100.0, 0.0, 0.1]]);
        tris.extend([[b, b + 2, b + 1], [b, b + 1, b + 3], [b, b + 3, b + 2], [b + 1, b + 2, b + 3]]);
        let mut m = nets::Mesh { verts, tris };
        let flipped = m.orient_consistently();
        assert!(flipped > 0);
        let mut dir = std::collections::HashMap::<(u32, u32), i32>::new();
        for t in &m.tris {
            for k in 0..3 {
                *dir.entry((t[k], t[(k + 1) % 3])).or_insert(0) += 1;
            }
        }
        for (&(a, c), &cnt) in &dir {
            assert_eq!(cnt, 1, "有向边重复 {a}->{c}:绕向未统一");
            assert!(dir.contains_key(&(c, a)), "缺反向边 {c}->{a}");
        }
        assert!(m.signed_volume() > 0.0, "整体朝向应向外");
        let (nd, nt) = m.drop_small_components(50, 1e-4);
        assert_eq!((nd, nt), (1, 4), "应只丢掉小四面体");
        assert_eq!(m.tris.len(), n_cube);
    }

    /// P37:分段并行 QEM / 源网格简化在细分立方体(闭合流形、90° 折痕)上:体积不变、各面仍轴对齐、确实简化。
    #[test]
    fn slab_parallel_decimation_keeps_a_subdivided_cube() {
        fn cube(n: i32) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
            let mut idx = std::collections::HashMap::<[i32; 3], u32>::new();
            let mut verts: Vec<[f32; 3]> = Vec::new();
            let mut tris = Vec::new();
            for ax in 0..3 {
                let (u, v) = ((ax + 1) % 3, (ax + 2) % 3);
                for s in [0, n] {
                    for i in 0..n {
                        for j in 0..n {
                            let mut ids = [0u32; 4];
                            for (c, (di, dj)) in [(0, 0), (1, 0), (1, 1), (0, 1)].into_iter().enumerate() {
                                let mut q = [0i32; 3];
                                q[ax] = s;
                                q[u] = i + di;
                                q[v] = j + dj;
                                ids[c] = *idx.entry(q).or_insert_with(|| {
                                    verts.push(q.map(|x| x as f32 / n as f32));
                                    (verts.len() - 1) as u32
                                });
                            }
                            if s == n {
                                tris.push([ids[0], ids[1], ids[2]]);
                                tris.push([ids[0], ids[2], ids[3]]);
                            } else {
                                tris.push([ids[0], ids[2], ids[1]]);
                                tris.push([ids[0], ids[3], ids[2]]);
                            }
                        }
                    }
                }
            }
            (verts, tris)
        }
        let check = |m: &nets::Mesh, n0: usize, what: &str| {
            let mut vol = 0.0;
            for t in &m.tris {
                let [a, b, c] = t.map(|i| m.verts[i as usize].map(|x| x as Real));
                let (u, v) = (sub(b, a), sub(c, a));
                let nn = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
                let l = (nn[0] * nn[0] + nn[1] * nn[1] + nn[2] * nn[2]).sqrt();
                assert!(l > 0.0 && nn.iter().any(|x| x.abs() / l > 0.999), "{what}: 面不再轴对齐 {nn:?}");
                vol += a[0] * nn[0] + a[1] * nn[1] + a[2] * nn[2];
            }
            assert!((vol / 6.0 - 1.0).abs() < 1e-4, "{what}: 体积 {}", vol / 6.0);
            assert!(m.tris.len() * 2 < n0, "{what}: 简化不足 {} / {n0}", m.tris.len());
            let (open, nonman, _) = m.edge_stats();
            assert_eq!((open, nonman), (0, 0), "{what}: 拓扑损坏");
        };
        let (verts, tris) = cube(12);
        let n0 = tris.len();
        let mut a = nets::Mesh { verts: verts.clone(), tris: tris.clone() };
        a.decimate_with_creases_blocks(1e-4, 25.0, None, 4);
        check(&a, n0, "QEM 分段");
        let (sv, st) = cube(1);
        let source = meshfield::MeshField::new(
            st.iter().map(|t| t.map(|i| sv[i as usize].map(|x| x as Real))).collect(),
        );
        let creases = nets::ReferenceCreases::new(&source, 0.4, 35.0);
        let mut b = nets::Mesh { verts, tris };
        b.decimate_to_reference_blocks(&source, &creases, 0.01, 0.005, 4);
        check(&b, n0, "源网格简化 分段");
        // P38:分段交替收尾 / 分段视觉清理不得破坏闭合性与贴面
        b.alternate_crease_cleanup_blocks(&source, &creases, 0.01, 1.0 / 12.0, 4);
        check(&b, n0, "交替收尾 分段");
        b.clean_visual_spurs_blocks(&source, 2, 4);
        check(&b, n0, "视觉清理 分段");
    }

    /// P33:平面上一个微斜面(顶点抬起 0.003mm、离邻顶点 0.01mm)造成的 > 20° 多余线应被塌缩掉。
    #[test]
    fn visual_spur_cleanup_removes_a_micro_tilted_face() {
        let source = meshfield::MeshField::new(vec![
            [[0.,0.,0.], [1.,0.,0.], [1.,1.,0.]],
            [[0.,0.,0.], [1.,1.,0.], [0.,1.,0.]],
        ]);
        // A B C D 方角,O 中心,w 在 O 旁 0.01 处抬起 0.003
        let mut mesh = nets::Mesh {
            verts: vec![[0.,0.,0.], [1.,0.,0.], [1.,1.,0.], [0.,1.,0.], [0.5,0.5,0.], [0.5,0.51,0.003]],
            tris: vec![[0,1,4], [1,2,4], [3,0,4], [2,3,5], [3,4,5], [4,2,5]],
        };
        let n = mesh.clean_visual_spurs(&source, 3);
        assert!(n >= 1, "微斜面应被塌缩");
        assert_eq!(mesh.tris.len(), 4);
        for t in &mesh.tris {
            let [a, b, c] = t.map(|i| mesh.verts[i as usize].map(|x| x as Real));
            let (u, v) = (sub(b, a), sub(c, a));
            let nz = u[0] * v[1] - u[1] * v[0];
            let l = (nz * nz + (u[1] * v[2] - u[2] * v[1]).powi(2) + (u[2] * v[0] - u[0] * v[2]).powi(2)).sqrt();
            assert!(nz / l > 0.999, "面未贴平:{t:?}");
        }
    }

    /// P30:跨 90° 折痕的斜面应被切开,出现沿折痕的边,所有面落到两侧源面上。
    #[test]
    fn crease_embedding_cuts_a_ramp_back_into_two_planes() {
        let source = meshfield::MeshField::new(vec![
            [[0.,0.,0.], [2.,0.,0.], [0.,1.,0.]],
            [[0.,0.,0.], [0.,0.,1.], [2.,0.,0.]],
        ]);
        let creases = nets::ReferenceCreases::new(&source, 0.4, 35.0);
        let mut mesh = nets::Mesh {
            verts: vec![[0.,0.5,0.], [1.,0.5,0.], [0.,0.,0.5], [1.,0.,0.5]],
            tris: vec![[0,2,1], [1,2,3]],
        };
        let (corners, cuts) = mesh.embed_reference_creases(&creases, 0.005, 0.5);
        assert_eq!(corners, 1, "(0,0,0) 角点应插入,(2,0,0) 太远应跳过");
        assert!(cuts >= 1, "斜面应被切开");
        let on_crease = |i: u32| { let p = mesh.verts[i as usize]; p[1].abs() < 1e-6 && p[2].abs() < 1e-6 };
        assert!(mesh.tris.iter().any(|t| (0..3).any(|k| on_crease(t[k]) && on_crease(t[(k + 1) % 3]))), "应出现沿折痕的边");
        for t in &mesh.tris {
            let [a, b, c] = t.map(|i| mesh.verts[i as usize].map(|x| x as Real));
            let (u, v) = (sub(b, a), sub(c, a));
            let n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
            let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!(n[1] / l > 0.99 || n[2] / l > 0.99, "面未落到源面上或翻转:{t:?} n={n:?}");
        }
    }

    /// P29:平面片上的细分网格应被合并,且不移动、不翻转、面积不变。
    #[test]
    fn reference_decimation_merges_a_planar_patch_without_moving_it() {
        let source = meshfield::MeshField::new(vec![
            [[0.,0.,0.], [1.,0.,0.], [1.,1.,0.]],
            [[0.,0.,0.], [1.,1.,0.], [0.,1.,0.]],
        ]);
        let creases = nets::ReferenceCreases::new(&source, 0.4, 35.0);
        let mut verts = Vec::new();
        for j in 0..5 {
            for i in 0..5 {
                verts.push([i as f32 * 0.25, j as f32 * 0.25, 0.0]);
            }
        }
        let id = |a: u32, b: u32| b * 5 + a;
        let mut tris = Vec::new();
        for j in 0..4u32 {
            for i in 0..4u32 {
                tris.push([id(i, j), id(i + 1, j), id(i + 1, j + 1)]);
                tris.push([id(i, j), id(i + 1, j + 1), id(i, j + 1)]);
            }
        }
        let before: Vec<[f32; 3]> = verts.clone();
        let mut mesh = nets::Mesh { verts, tris };
        let n = mesh.decimate_to_reference(&source, &creases, 0.01, 0.005);
        assert!(n >= 5, "内部顶点应被塌缩,实际 {n}");
        assert!(mesh.tris.len() <= 24, "三角数 {}", mesh.tris.len());
        let mut area = 0.0;
        for t in &mesh.tris {
            let [a, b, c] = t.map(|i| mesh.verts[i as usize].map(|x| x as Real));
            let (u, v) = (sub(b, a), sub(c, a));
            let nz = u[0] * v[1] - u[1] * v[0];
            let (nx, ny) = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2]);
            assert!(nz > 0.0 && nx.abs() < 1e-9 && ny.abs() < 1e-9, "面翻转或离开平面");
            area += nz / 2.0;
        }
        assert!((area - 1.0).abs() < 1e-6, "覆盖面积变化:{area}");
        for p in &mesh.verts {
            assert!(before.contains(p), "顶点被移动:{p:?}");
        }
    }

    /// P28:顶点都在源面上、却斜跨折痕的两面(对角线跨折痕)应被翻成沿折痕的边。
    #[test]
    fn reference_repair_flips_a_diagonal_that_straddles_a_crease() {
        let source = meshfield::MeshField::new(vec![
            [[0.,0.,0.], [2.,0.,0.], [0.,1.,0.]],
            [[0.,0.,0.], [0.,0.,1.], [2.,0.,0.]],
        ]);
        let creases = nets::ReferenceCreases::new(&source, 0.4, 35.0);
        // c0, c1 在折痕上;p 在 z=0 面,s 在 y=0 面;对角线 p–s 斜跨折痕
        let mut mesh = nets::Mesh {
            verts: vec![[0.,0.,0.], [1.,0.,0.], [0.5,0.5,0.], [0.5,0.,0.5]],
            tris: vec![[3,2,0], [2,3,1]],
        };
        let (before, flips, _, after) = mesh.repair_reference_defects(&source, &creases, 30.0, 2);
        assert_eq!(before, 2, "两面都应判为缺陷");
        assert_eq!(flips, 1);
        assert_eq!(after, 0);
        let has_crease_edge = mesh.tris.iter().any(|t| t.contains(&0) && t.contains(&1));
        assert!(has_crease_edge, "应出现沿折痕的 c0–c1 边:{:?}", mesh.tris);
        for t in &mesh.tris {
            let [a, b, c] = t.map(|i| mesh.verts[i as usize].map(|x| x as Real));
            let (u, v) = (sub(b, a), sub(c, a));
            let n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
            let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!(n[1] / l > 0.99 || n[2] / l > 0.99, "面未贴合源面:{n:?}");
        }
    }

    /// P26:折痕旁 0.03mm 的"半吸附"顶点(二面角 90° 的短边 = 查看器里的毛刺)
    /// 应被塌缩到折痕上的邻点;其余顶点、面朝向不变。
    #[test]
    fn crease_spur_vertex_collapses_onto_the_crease() {
        let source = meshfield::MeshField::new(vec![
            [[0.,0.,0.], [2.,0.,0.], [0.,1.,0.]],
            [[0.,0.,0.], [0.,0.,1.], [2.,0.,0.]],
        ]);
        let creases = nets::ReferenceCreases::new(&source, 0.4, 35.0);
        // c0, c1 在折痕(x 轴)上;w 在 z=0 面上离折痕 0.03;t 在 z=0 面,s 在 y=0 面
        let mut mesh = nets::Mesh {
            verts: vec![[0.2,0.,0.], [0.5,0.,0.], [0.35,0.03,0.], [0.35,0.5,0.], [0.35,0.,0.5]],
            tris: vec![[0,2,3], [2,1,3], [0,4,2], [2,4,1]],
        };
        let before: Vec<[f32; 3]> = mesh.verts.clone();
        let (flips, collapses, _) = mesh.merge_crease_spurs(&creases, 30.0, 0.005, 0.12, 0.3, 0.02, 2);
        // 两条候选对角线共用对边 c0–c1,但只有一条合法:
        //   (v2,v3) → 新面 [c0,v2,c1] 落在 z=0 面上却是**反向**(绕向为 −z),
        //             由 spur_agree 拦下(实测该面 agree = −1,不翻);
        //   (v2,v4) → 新面 [c0,v4,c1] 在 y=0 面、[c1,v2,c0] 在 z=0 面,两面都顺源法线,
        //             且先把 v2 的 4 个邻面降到 2 个,随后的塌缩才合法(shared==2)。
        // 所以这里恰好 1 次翻边是设计内的必要步骤,不是漏网的反向面。
        assert_eq!(flips, 1, "只允许 (v2,v4) 这条合法对角线,(v2,v3) 应被源法线一致度拦下");
        assert_eq!(collapses, 1, "w 应被塌缩");
        assert_eq!(mesh.tris.len(), 2);
        assert_eq!(mesh.verts.len(), 4);
        assert!(!mesh.verts.contains(&before[2]), "半吸附顶点应消失");
        for p in &mesh.verts {
            assert!(before.contains(p), "不应移动任何保留顶点:{p:?}");
        }
        // 两个存留面仍分别在 z=0 与 y=0 平面上,朝向与原来一致
        for t in &mesh.tris {
            let [a, b, c] = t.map(|i| mesh.verts[i as usize].map(|x| x as Real));
            let (u, v) = (sub(b, a), sub(c, a));
            let n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
            assert!(n[0].abs() < 1e-9 && (n[1].abs() < 1e-9 || n[2].abs() < 1e-9), "面离开源平面:{n:?}");
            if n[1].abs() < 1e-9 {
                assert!(n[2] > 0.0, "z=0 面翻转");
            } else {
                assert!(n[1] > 0.0, "y=0 面翻转");
            }
        }
    }

    #[test]
    fn reconstruction_keeps_a_sharp_ridge_on_both_supporting_planes() {
        let source = meshfield::MeshField::new(vec![
            [[0.,0.,0.], [2.,0.,0.], [0.,1.,0.]],
            [[0.,0.,0.], [0.,0.,1.], [2.,0.,0.]],
        ]);
        let mut mesh = nets::Mesh {
            verts: vec![[0.2,0.06,0.04],[1.4,0.06,0.04],[0.2,0.8,0.],[0.2,0.,0.8]],
            tris: vec![[0,1,2],[0,3,1]],
        };
        let creases = nets::ReferenceCreases::new(&source, 0.4, 35.0);
        mesh.fit_reference(&source, 0.2, true, Some(&creases));
        for p in &mesh.verts[..2] {
            assert!(p[1].abs().max(p[2].abs()) < 0.005, "ridge drift: {p:?}");
        }
    }

    #[test]
    fn exported_coincidence_checks_used_indices_and_canonicalizes_signed_zero() {
        let mut m = nets::Mesh {
            verts: vec![[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [-0.,0.,0.], [0.,0.,1.]],
            tris: vec![[0,1,2]],
        };
        assert!(!m.has_coincident_export_vertices());
        m.tris.push([3,2,4]);
        assert!(m.has_coincident_export_vertices());
    }

    #[test]
    fn face_samples_resolve_curvature_away_from_the_centroid() {
        let vertex = |i, j| [i as Real / 6.0, j as Real / 6.0, if (i,j) == (1,1) { 0.06 } else { 0.0 }];
        let mut fine = Vec::new();
        for i in 0..6 {
            for j in 0..6-i {
                fine.push([vertex(i,j), vertex(i+1,j), vertex(i,j+1)]);
                if i+j < 5 { fine.push([vertex(i+1,j), vertex(i+1,j+1), vertex(i,j+1)]); }
            }
        }
        let source = meshfield::MeshField::new(fine);
        let mut coarse = nets::Mesh { verts: vec![[0.,0.,0.], [1.,0.,0.], [0.,1.,0.]], tris: vec![[0,1,2]] };
        coarse.project_with_face_samples(&source, 0.2, true);
        assert!(coarse.verts[0][2] > 0.01);
        assert!(coarse.verts[0][2] > 2.0 * coarse.verts[1][2]);
    }

    #[test]
    fn zero_projection_respects_clamped_axes() {
        let mut adf = Adf::build(
            Stock::Sphere { c: [0.0; 3], r: 1.0 },
            [-2.0; 3], 4.0, 0.005, 0.2, 4,
        );
        // 单一三线性叶精确表示 x+y。x 在域外时，组合了钳位的场只沿 y 变化。
        let corners = std::array::from_fn(|i| {
            let p = corner_pos(adf.min, adf.size, i);
            (p[0] + p[1]) as f32
        });
        adf.nodes = vec![Node::Leaf(corners)];
        adf.root = 0;
        let mut m = nets::Mesh {
            verts: vec![[10.0, -1.0, 0.0], [-10.0, 1.0, 0.0], [10.0, 10.0, 0.0], [1.0, 0.0, 0.0]],
            tris: vec![],
        };
        m.project_to_zero(&adf, 1, 2.0);
        assert_eq!(m.verts[0], [10.0, -2.0, 0.0]);
        assert_eq!(m.verts[1], [-10.0, 2.0, 0.0]);
        assert_eq!(m.verts[2], [10.0, 10.0, 0.0]);
        assert_eq!(m.verts[3], [0.5, -0.5, 0.0]);
    }

    #[test]
    fn adjacent_projection_steps_preserve_the_starting_face_normal() {
        let reference = meshfield::MeshField::new(vec![[
            [-9.0, -9.0, 0.0], [9.0, -9.0, 0.0], [0.0, 9.0, 0.0],
        ]]);
        let mut mesh = nets::Mesh {
            verts: vec![[0.0, -0.51, -0.06], [-0.04, -0.10, 1.0], [0.02, -0.93, -1.09]],
            tris: vec![[0, 1, 2]],
        };
        let normal = |v: &[[f32; 3]]| {
            let ab = sub(v[1].map(f64::from), v[0].map(f64::from));
            let ac = sub(v[2].map(f64::from), v[0].map(f64::from));
            [ab[1]*ac[2]-ab[2]*ac[1], ab[2]*ac[0]-ab[0]*ac[2], ab[0]*ac[1]-ab[1]*ac[0]]
        };
        let before = normal(&mesh.verts);
        mesh.project_to_reference(&reference, 0.35);
        assert!(dot(before, normal(&mesh.verts)) >= 0.25 * dot(before, before));
    }

    #[test]
    fn nets_sphere_volume_and_stl() {
        let a = Adf::build(
            Stock::Sphere { c: [0.0; 3], r: 1.0 },
            [-2.0, -2.0, -2.0],
            4.0,
            0.005,
            0.2,
            6,
        );
        let m = nets::extract_uniform(&a, 0.125);
        assert!(m.tris.len() > 100, "球面三角形数过少:{}", m.tris.len());
        let v = m.signed_volume();
        assert!(
            (v - 4.18879).abs() < 0.8,
            "球体积应 ≈4.19(实际 {v:.3})——绕向或提取有误"
        );
        nets::write_stl("target/test_sphere.stl", &m).expect("STL 写出失败");
    }

    /// 提取的并行化不得引入顺序相关:顶点号由"k 片序 + 片内 cj/ci 序"决定,
    /// 三角由"方向序 × k 片序 × 片内 j/i 序"决定,故任何工人数下都应逐位一致。
    /// 本测试连跑两次比对(竞争或非确定合并会在此暴露),并把拓扑自洽性
    /// (三角索引不越界、有符号体积为正)一起守住。
    #[test]
    fn nets_extract_deterministic_and_wound() {
        let a = Adf::build(
            Stock::RoundBox { c: [0.0; 3], half: [1.5; 3], r: 0.4 },
            [-2.0; 3],
            4.0,
            0.005,
            0.2,
            6,
        );
        let m1 = nets::extract_uniform(&a, 0.1);
        let m2 = nets::extract_uniform(&a, 0.1);
        assert_eq!(m1.verts, m2.verts, "两次提取的顶点不一致(并行不确定)");
        assert_eq!(m1.tris, m2.tris, "两次提取的三角不一致(并行不确定)");
        let nv = m1.verts.len() as u32;
        for t in &m1.tris {
            for &i in t {
                assert!(i < nv, "三角索引 {i} 越界(顶点数 {nv})——片间偏移合并有误");
            }
        }
        let v = m1.signed_volume();
        assert!(v > 0.0, "有符号体积应为正(绕向一致),实际 {v}");
    }
}
