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
    // ── ① 候选光栅化(按 k-板并行,线程本地表,无锁;三角形 AABB 外扩 1 格)──
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
                    let i0 = clampi(((cmn[0] - min[0]) / h).floor() - 1.0);
                    let i1 = clampi(((cmx[0] - min[0]) / h).ceil() + 1.0);
                    let j0 = clampi(((cmn[1] - min[1]) / h).floor() - 1.0);
                    let j1 = clampi(((cmx[1] - min[1]) / h).ceil() + 1.0);
                    let kk0 = clampi(((cmn[2] - min[2]) / h).floor() - 1.0).max(k0);
                    let kk1 = (clampi(((cmx[2] - min[2]) / h).ceil() + 1.0) + 1).min(k1);
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

#[derive(Default, Clone, Copy, Debug)]
pub struct BuildStats {
    pub subdivided: u64,
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
    o: [Real; 3],
    s: Real,
    depth: u8,
    stats: &mut BuildStats,
) -> u32 {
    let mut corners = [0f32; 8];
    for k in 0..8 {
        let p = corner_pos(o, s, k);
        let mut v = match grid {
            Some(g) => g.sample_nearest(p) as f32,
            None => src.signed(p) as f32,
        };
        stats.sd_evals += 1;
        if v.abs() < band_floor() {
            v = sign as f32 * band_floor();
        }
        corners[k] = v;
    }
    stats.leaves += 1;
    stats.leaf_depth[depth as usize] += 1;
    nodes.push(Node::Leaf(corners));
    (nodes.len() - 1) as u32
}

/// 远叶角点幅值下限 = 壳层带宽(该叶不可能含 |d| ≤ band 的点)
#[inline]
fn band_floor() -> f32 {
    0.5
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
            ch[i] = build_cell_rebuild(nodes, sd, child_origin(o, s, i), hs, depth + 1, eps, band, max_depth, stats);
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
        return far_leaf(nodes, src, grid, c_center.signum() as Real, o, s, depth, stats);
    }
    if lazy_sub && depth < max_depth {
        // 近大格:网格裕量可能误报(EDT 高估侧 ~2.7h)→ 精确中心复核(1 查)
        // 把误报环带格立即打成远叶(网格角点),阻断细分级联;含表面格 |c|≤hd 必通过
        if grid.is_some() {
            let c_ex = src.signed(ctr);
            stats.sd_evals += 1;
            if c_ex.abs() > band + half_diag {
                return far_leaf(nodes, src, grid, c_ex.signum(), o, s, depth, stats);
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
            return far_leaf(nodes, src, grid, c_ex.signum(), o, s, depth, stats);
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
        let q = self.clamp_to_domain(p);
        let mut o = self.min;
        let mut s = self.size;
        let mut idx = self.root;
        loop {
            match self.nodes[idx as usize] {
                Node::Leaf(c) => {
                    // 远场粗叶的插值误差**不能**靠"改读烘焙网格"解决(P6 已否决):
                    // 实测叶轮 77% 体积由 d4 叶覆盖(阈值内),改用网格三线性后
                    // [error] 仅 8.77 → 8.88mm;把网格提到 257³ 也只到 8.42mm,
                    // 而 bake 涨 4×。说明误差源是**网格自身的符号**(薄壁件上 EDT 的
                    // 最近站符号会取反,max ≈ 2×距离),不是插值或分辨率。见 P5。
                    let u = [(q[0] - o[0]) / s, (q[1] - o[1]) / s, (q[2] - o[2]) / s];
                    return trilinear(u, &c);
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
        Self::writeback_walk(&mut self.nodes, self.root, self.min, self.size, gmin, gmax, h, gn, &d, &mut corrected);
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
                    Self::writeback_walk(nodes, ch[i], child_origin(o, s, i), s * 0.5, gmin, gmax, h, gn, d, corrected);
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
    pub fn leaf_stats_bytes(&self) -> (u64, u64, [u64; 17]) {
        let mut n = 0u64;
        let mut bytes = 0u64;
        let mut hist = [0u64; 17];
        let mut stack = vec![(self.root, 0u8)];
        while let Some((idx, d)) = stack.pop() {
            match self.nodes[idx as usize] {
                Node::Leaf(_) => {
                    n += 1;
                    bytes += 40; // [f32;8] + enum 标签/对齐
                    hist[d as usize] += 1;
                }
                Node::Internal(ch) => {
                    bytes += 40;
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

    pub struct Mesh {
        pub verts: Vec<[f32; 3]>,
        pub tris: Vec<[u32; 3]>,
    }

    fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    }
    fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }
    fn g64(v: [f32; 3]) -> [f64; 3] {
        [v[0] as f64, v[1] as f64, v[2] as f64]
    }

    impl Mesh {
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
            // 边表:(tri<<1|方向) 打包成 u32,每条边两个槽 —— 避免每个边一次 Vec 分配
            // (实测 Vec 版 482k 三角要 260ms,打包版降到 ~60ms)
            const EMPTY: u32 = u32::MAX;
            let mut map: std::collections::HashMap<u64, [u32; 2], FxBuild> =
                std::collections::HashMap::with_capacity_and_hasher(n * 3, FxBuild);
            for (ti, t) in self.tris.iter().enumerate() {
                for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                    let key = if a < b {
                        ((a as u64) << 32) | b as u64
                    } else {
                        ((b as u64) << 32) | a as u64
                    };
                    let packed = ((ti as u32) << 1) | (a < b) as u32;
                    let slot = map.entry(key).or_insert([EMPTY, EMPTY]);
                    if slot[0] == EMPTY {
                        slot[0] = packed;
                    } else if slot[1] == EMPTY {
                        slot[1] = packed;
                    } else {
                        // 非流形边:标记为"不可用",定向时不跨越它
                        slot[1] = EMPTY;
                        slot[0] = EMPTY;
                    }
                }
            }
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
                // ① 沿**流形边**(恰好 2 个三角)洪泛,把组内绕向调成一致。
                //    非流形边(>2)不定向 —— 在那里做决定会把错误沿整片传播,
                //    实测不设此限时 10% 的三角被误翻、体积掉到 −53%。
                while let Some(ti) = stack.pop() {
                    group.push(ti);
                    let t = self.tris[ti as usize];
                    for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                        let key = if a < b {
                            ((a as u64) << 32) | b as u64
                        } else {
                            ((b as u64) << 32) | a as u64
                        };
                        let fwd = a < b;
                        let Some(slot) = map.get(&key) else { continue };
                        if slot[0] == EMPTY || slot[1] == EMPTY {
                            continue;
                        }
                        for &packed in slot.iter() {
                            let nt = packed >> 1;
                            let nf = (packed & 1) == 1;
                            if nt == ti || seen[nt as usize] {
                                continue;
                            }
                            seen[nt as usize] = true;
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

        /// 丢弃**碎屑分量**:三角数 < `min_tris`,**或**体积 < `min_vol_frac` × 总体积。
        /// 这类分量是场里孤立的小符号岛,在查看器里表现为浮在表面的斑点;切片软件
        /// 通常也直接忽略。用"相对体积"而不只看三角数,是因为三角数与网格密度挂钩
        /// (本工程 2×h_min 提取时,1mm³ 只有几十个三角)。
        /// 返回(丢弃的分量数, 丢弃的三角数)。
        pub fn drop_small_components(&mut self, min_tris: usize, min_vol_frac: f64) -> (usize, usize) {
            let n = self.tris.len();
            let mut par: Vec<u32> = (0..n as u32).collect();
            fn find(p: &mut [u32], mut x: u32) -> u32 {
                while p[x as usize] != x {
                    p[x as usize] = p[p[x as usize] as usize];
                    x = p[x as usize];
                }
                x
            }
            let mut edge: std::collections::HashMap<u64, u32, FxBuild> =
                std::collections::HashMap::with_capacity_and_hasher(n * 3, FxBuild);
            for (ti, t) in self.tris.iter().enumerate() {
                for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                    let key = if a < b {
                        ((a as u64) << 32) | b as u64
                    } else {
                        ((b as u64) << 32) | a as u64
                    };
                    if let Some(&other) = edge.get(&key) {
                        let x = find(&mut par, ti as u32);
                        let y = find(&mut par, other);
                        if x != y {
                            par[y as usize] = x;
                        }
                    } else {
                        edge.insert(key, ti as u32);
                    }
                }
            }
            let mut count: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
            let mut vols: std::collections::HashMap<u32, f64> = std::collections::HashMap::new();
            let mut total_vol = 0.0f64;
            for ti in 0..n {
                let r = find(&mut par, ti as u32);
                *count.entry(r).or_insert(0) += 1;
                let t = self.tris[ti];
                let (a, b, c) = (
                    self.verts[t[0] as usize],
                    self.verts[t[1] as usize],
                    self.verts[t[2] as usize],
                );
                let (ax, ay, az) = (a[0] as f64, a[1] as f64, a[2] as f64);
                let (bx, by, bz) = (b[0] as f64, b[1] as f64, b[2] as f64);
                let (cx, cy, cz) = (c[0] as f64, c[1] as f64, c[2] as f64);
                let v = (ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx)
                    + az * (bx * cy - by * cx))
                    / 6.0;
                *vols.entry(r).or_insert(0.0) += v;
                total_vol += v;
            }
            let total_vol = total_vol.abs();
            let mut drop = vec![false; n];
            let (mut nd, mut nt) = (0usize, 0usize);
            for ti in 0..n {
                let r = find(&mut par, ti as u32);
                let small = count[&r] < min_tris
                    || vols[&r].abs() < min_vol_frac * total_vol;
                if small {
                    drop[ti] = true;
                    nt += 1;
                }
            }
            if nt > 0 {
                let mut i = 0usize;
                self.tris.retain(|_| {
                    let keep = !drop[i];
                    i += 1;
                    keep
                });
                // 统计丢了几个分量
                let mut dropped_roots = std::collections::HashSet::new();
                for ti in 0..n {
                    if drop[ti] {
                        dropped_roots.insert(find(&mut par, ti as u32));
                    }
                }
                nd = dropped_roots.len();
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
        n: usize,
        dim: usize,
        vals: &[f32],
        cell_vert: &[u32],
        verts: &[[f32; 3]],
        out: &mut Vec<[u32; 3]>,
    ) {
        let cv = |i: usize, j: usize, k: usize| -> u32 { cell_vert[(k * n + j) * n + i] };
        for k in klo..khi {
            match axis {
                0 => {
                    for j in 1..n {
                        for i in 0..n {
                            let a = vals[(k * dim + j) * dim + i];
                            let b = vals[(k * dim + j) * dim + i + 1];
                            if (a < 0.0) == (b < 0.0) {
                                continue;
                            }
                            let q = [cv(i, j, k), cv(i, j - 1, k), cv(i, j - 1, k - 1), cv(i, j, k - 1)];
                            if q.iter().any(|v| *v == u32::MAX) {
                                continue;
                            }
                            let want = if a < 0.0 { [1.0, 0.0, 0.0] } else { [-1.0, 0.0, 0.0] };
                            push_quad(verts, q, want, out);
                        }
                    }
                }
                1 => {
                    for j in 0..n {
                        for i in 1..n {
                            let a = vals[(k * dim + j) * dim + i];
                            let b = vals[(k * dim + (j + 1)) * dim + i];
                            if (a < 0.0) == (b < 0.0) {
                                continue;
                            }
                            let q = [cv(i, j, k), cv(i - 1, j, k), cv(i - 1, j, k - 1), cv(i, j, k - 1)];
                            if q.iter().any(|v| *v == u32::MAX) {
                                continue;
                            }
                            let want = if a < 0.0 { [0.0, 1.0, 0.0] } else { [0.0, -1.0, 0.0] };
                            push_quad(verts, q, want, out);
                        }
                    }
                }
                _ => {
                    for j in 1..n {
                        for i in 1..n {
                            let a = vals[(k * dim + j) * dim + i];
                            let b = vals[((k + 1) * dim + j) * dim + i];
                            if (a < 0.0) == (b < 0.0) {
                                continue;
                            }
                            let q = [cv(i, j, k), cv(i - 1, j, k), cv(i - 1, j - 1, k), cv(i, j - 1, k)];
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

    /// 均匀重采样 + Surface Nets 提取。h:重采样格宽(建议 h_min 或其 2 倍)。
    pub fn extract_uniform(adf: &Adf, h: Real) -> Mesh {
        let n = ((adf.size / h).round() as usize).max(1);
        let dim = n + 1;
        let step = adf.size / n as Real;

        // 1) 角点均匀重采样(走 ADF eval——自适应结构在这里兑现价值)
        //    逐点纯函数、与访问顺序无关 → 按 k 平面切带并行,结果逐位可复现。
        //    dim³ 次 eval 是本函数的主要成本(桌腿 513³≈1.35 亿次,占提取耗时 ~85%);
        //    串行时 4.0s,串行也是它把 extract 顶成全管线最大单项。
        let mut vals = vec![0f32; dim * dim * dim];
        {
            let plane = dim * dim;
            let ppw = dim.div_ceil(nthreads()).max(1);
            let band = ppw * plane;
            let chunks: Vec<&mut [f32]> = vals.chunks_mut(band).collect();
            let min = adf.min;
            std::thread::scope(|sc| {
                for (w, chunk) in chunks.into_iter().enumerate() {
                    let k0 = w * ppw;
                    sc.spawn(move || {
                        for (idx, slot) in chunk.iter_mut().enumerate() {
                            let k = k0 + idx / plane;
                            let rem = idx % plane;
                            let (j, i) = (rem / dim, rem % dim);
                            *slot = adf.eval([
                                min[0] + i as Real * step,
                                min[1] + j as Real * step,
                                min[2] + k as Real * step,
                            ]);
                        }
                    });
                }
            });
        }

        // 2) 变号 cell 一个顶点 = 12 棱交叉点平均(经典 Surface Nets)
        //    按 ck 切 k 平面并行:每片产出**局部**顶点号写入本片 cell_vert,
        //    之后按片序合并(片内顶点先于后片)⟹ 与串行顶点编号逐位一致。
        let mut cell_vert = vec![u32::MAX; n * n * n];
        let mut verts: Vec<[f32; 3]> = Vec::new();
        {
            let nslab = n;
            let mut slabs: Vec<&mut [u32]> = cell_vert.chunks_mut(n * n).collect();
            debug_assert_eq!(slabs.len(), nslab);
            let locals: Vec<std::sync::Mutex<Vec<[f32; 3]>>> =
                (0..nslab).map(|_| std::sync::Mutex::new(Vec::new())).collect();
            let mut jobs: Vec<(usize, &mut [u32])> = slabs.drain(..).enumerate().collect();
            let nt = nthreads().min(nslab).max(1);
            let mut per: Vec<Vec<(usize, &mut [u32])>> =
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
                        for (ck, slab) in mine {
                            let mut out: Vec<[f32; 3]> = Vec::new();
                            for cj in 0..n {
                                for ci in 0..n {
                                    let mut cv = [0f32; 8];
                                    let (mut any_neg, mut any_pos) = (false, false);
                                    for c in 0..8 {
                                        let (dx, dy, dz) = (c & 1, (c >> 1) & 1, (c >> 2) & 1);
                                        let v = vals_r[((ck + dz) * dim + (cj + dy)) * dim + (ci + dx)];
                                        cv[c] = v;
                                        any_neg |= v < 0.0;
                                        any_pos |= v >= 0.0;
                                    }
                                    if !(any_neg && any_pos) {
                                        continue;
                                    }
                                    let o = [
                                        vmin[0] + ci as Real * step,
                                        vmin[1] + cj as Real * step,
                                        vmin[2] + ck as Real * step,
                                    ];
                                    let (mut vx, mut vy, mut vz) = (0.0, 0.0, 0.0);
                                    let mut cnt = 0u32;
                                    for (a, b) in EDGES {
                                        let (va, vb) = (cv[a] as Real, cv[b] as Real);
                                        if (va < 0.0) != (vb < 0.0) {
                                            let t = va / (va - vb);
                                            let pa = corner_pos(o, step, a);
                                            let pb = corner_pos(o, step, b);
                                            vx += pa[0] + (pb[0] - pa[0]) * t;
                                            vy += pa[1] + (pb[1] - pa[1]) * t;
                                            vz += pa[2] + (pb[2] - pa[2]) * t;
                                            cnt += 1;
                                        }
                                    }
                                    if cnt == 0 {
                                        continue;
                                    }
                                    let c = cnt as Real;
                                    slab[cj * n + ci] = out.len() as u32; // 局部号
                                    out.push([(vx / c) as f32, (vy / c) as f32, (vz / c) as f32]);
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
                let base = s * n * n;
                for e in cell_vert[base..base + n * n].iter_mut() {
                    if *e != u32::MAX {
                        *e += off;
                    }
                }
                verts.append(&mut loc);
            }
        }
        // 3) 网格棱变号 → 环绕该棱的 4 个 cell 顶点连 quad(push_quad 自定向)
        //    串行顺序 = 全 x 向 → 全 y 向 → 全 z 向;每向内部 k 递增。
        //    因此每向按 k 切片并行、再按片序合并,三角顺序与串行**逐位一致**。
        //    棱四邻 cell 只在 j/k ∈ [1, n) 内完整;域边界层跳过(表面不该触边,
        //    远场噪声变号在此组装会读到越界 cell_vert——实测 Impeller panic)。
        let mut tris: Vec<[u32; 3]> = Vec::new();
        for axis in 0..3usize {
            let k_lo = if axis == 2 { 0 } else { 1 };
            let nslab = n - k_lo;
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
                    let verts = &verts;
                    sc.spawn(move || {
                        let mut s = w;
                        while s < nslab {
                            let mut out: Vec<[u32; 3]> = Vec::new();
                            quads_slab(
                                axis,
                                k_lo + s,
                                k_lo + s + 1,
                                n,
                                dim,
                                vals,
                                cell_vert,
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

        let mut mesh = Mesh { verts, tris };
        // 绕向统一:个别 quad 会因场值不稳而定向相反,背面剔除的渲染器会显示成
        // 斑点状的"洞"。见 Mesh::orient_consistently 的说明。
        let flipped = mesh.orient_consistently();
        eprintln!("[nets.winding] 翻正 {flipped} 个三角(绕向一致化)");
        mesh
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

    /// 水密性粗检:顶点去重(0.1µm 格)后统计出现奇数次的边。
    /// 返回 (三角形数, 是否水密)。
    pub fn watertight_report(tris: &[[[Real; 3]; 3]]) -> (usize, bool) {
        use std::collections::HashMap;
        let q = |v: Real| (v * 10000.0).round() as i64; // 0.1µm 量化
        let mut vid: HashMap<[i64; 3], u32> = HashMap::new();
        let mut edges: HashMap<(u32, u32), u32> = HashMap::new();
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
                *edges.entry(key).or_insert(0) += 1;
            }
        }
        let odd = edges.values().filter(|&c| c % 2 == 1).count();
        (tris.len(), odd == 0)
    }
}

// ─────────────────── 网格场源:BVH + 最近点 + 射线奇偶符号 ───────────────────

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
            let mut nv = 0u32;
            for t in &tris {
                let mut ids = [0u32; 3];
                for (k, v) in t.iter().enumerate() {
                    let key = [q(v[0]), q(v[1]), q(v[2])];
                    let id = match vid_map.get(&key) {
                        Some(&i) => i,
                        None => {
                            vid_map.insert(key, nv);
                            nv += 1;
                            nv - 1
                        }
                    };
                    ids[k] = id;
                }
                tri_vid.push(ids);
            }
            let mut vert_n = vec![[0f64; 3]; nv as usize];
            let mut edge_n: std::collections::HashMap<(u32, u32), [f64; 3]> =
                std::collections::HashMap::new();
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
                    let e = edge_n.entry(key).or_insert([0.0; 3]);
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
                .map(|(k, v)| {
                    let l = len(v);
                    let u = if l > 0.0 { scale3(v, 1.0 / l) } else { [0.0; 3] };
                    (k, [u[0] as f32, u[1] as f32, u[2] as f32])
                })
                .collect();
            MeshField {
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

        /// 由 (三角形, 特征码, 最近点) 给出伪法线有符号距离。
        pub fn signed_from_feature(&self, p: [Real; 3], t: u32, feat: u8, cp: [Real; 3]) -> Real {
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
            if dot3(sub3(p, cp), n) >= 0.0 {
                len(sub3(p, cp))
            } else {
                -len(sub3(p, cp))
            }
        }

        /// best-first 最近点核心:(距离² 开方, 三角形索引, 特征码, 最近点)。
        /// best-first 遍历(min-heap 按 AABB 距离²):堆顶距离 ≥ 当前 best 即全局早退;
        /// 子节点按距离近者先入堆;三角形先做自身 AABB 早退再跑精确区域算法。
        /// (VTune 实证:LIFO 无序遍历 + 无三角形早退 → unsigned_distance 272s/90%)
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

        /// 射线奇偶:偶数=外,奇数=内(要求水密网格;固定对角射线避开顶点)。
        /// 栈数组替代 per-call Vec 堆分配。现主要作伪法线符号的交叉验证与
        /// 水密性诊断;热路径符号已由 signed_distance_pn 承担。
        pub fn ray_parity_inside(&self, p: [Real; 3]) -> bool {
            let dir = PARITY_DIR;
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
            hits % 2 == 1
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
                                    Node::Leaf(c2) => eprintln!("[dbg]   节点 {ni2} Leaf"),
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
                    let want = ((i as f64 - 1.0).powi(2) + (j as f64 - 1.0).powi(2) + (k as f64 - 1.0).powi(2));
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
