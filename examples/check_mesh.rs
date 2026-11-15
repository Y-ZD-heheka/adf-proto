//! 网格体检:按连通分量拆解 STL,报告各分量规模/包围盒/体积,
//! 并统计非流形边与"碎片"占比。
//!
//! 用法:cargo run --release --example check_mesh -- stl_out.stl

use adf_proto::stlio;
use std::collections::HashMap;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "stl_out.stl".to_string());
    let tm = stlio::load_stl(&path).expect("STL 加载失败");
    let nt = tm.tris.len();
    println!("[mesh] {path} | 三角 {nt}");

    // 顶点焊接(按坐标量化到 1e-4mm)后才能做连通分量
    let mut id: HashMap<[i64; 3], u32> = HashMap::new();
    let mut pos: Vec<[f64; 3]> = Vec::new();
    let mut tris: Vec<[u32; 3]> = Vec::with_capacity(nt);
    for t in &tm.tris {
        let mut tri = [0u32; 3];
        for (k, v) in t.iter().enumerate() {
            let key = [
                (v[0] * 1e4).round() as i64,
                (v[1] * 1e4).round() as i64,
                (v[2] * 1e4).round() as i64,
            ];
            let nxt = pos.len() as u32;
            let idx = *id.entry(key).or_insert(nxt);
            if idx == nxt {
                pos.push(*v);
            }
            tri[k] = idx;
        }
        if tri[0] != tri[1] && tri[1] != tri[2] && tri[2] != tri[0] {
            tris.push(tri);
        }
    }
    println!("[mesh] 焊接后 顶点 {} | 有效三角 {}", pos.len(), tris.len());

    // 边统计
    let mut ec: HashMap<(u32, u32), u32> = HashMap::new();
    for t in &tris {
        for e in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let k = if e.0 < e.1 { e } else { (e.1, e.0) };
            *ec.entry(k).or_insert(0) += 1;
        }
    }
    let open = ec.values().filter(|c| **c == 1).count();
    let nonman = ec.values().filter(|c| **c > 2).count();
    println!("[edge] 开放边 {open} | 非流形边 {nonman} | 总边 {}", ec.len());

    // 绕向一致性:闭合流形网格里每条边应被两个三角**反向**各走一次。
    // 若某条边的两个半边走同方向 → 相邻三角绕向相反(背面朝外,渲染成"洞")。
    let mut dir: HashMap<(u32, u32, u8), u32> = HashMap::new();
    for t in &tris {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let (k, f) = if a < b { ((a, b), 0u8) } else { ((b, a), 1u8) };
            *dir.entry((k.0, k.1, f)).or_insert(0) += 1;
        }
    }
    let mut flipped_edges = 0usize;
    for ((a, b, f), c) in dir.iter() {
        let other = dir.get(&(*a, *b, 1 - *f)).copied().unwrap_or(0);
        if *c > 0 && other == 0 && *c >= 2 {
            flipped_edges += 1;
        }
    }
    println!(
        "[winding] 同向边(相邻三角绕向相反) {flipped_edges} / {} = {:.3}%",
        ec.len(),
        flipped_edges as f64 / ec.len() as f64 * 100.0
    );
    // 同向边把"绕向相反"的三角连成组:若成片,渲染器按背面剔除就会显示成大洞
    {
        let mut edge_tris: HashMap<(u32, u32), Vec<(usize, u8)>> = HashMap::new();
        for (ti, t) in tris.iter().enumerate() {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                let (k, f) = if a < b { ((a, b), 0u8) } else { ((b, a), 1u8) };
                edge_tris.entry(k).or_default().push((ti, f));
            }
        }
        let mut tp: Vec<u32> = (0..tris.len() as u32).collect();
        fn tf(p: &mut [u32], mut x: u32) -> u32 {
            while p[x as usize] != x {
                p[x as usize] = p[p[x as usize] as usize];
                x = p[x as usize];
            }
            x
        }
        for list in edge_tris.values() {
            if list.len() != 2 {
                continue;
            }
            if list[0].1 == list[1].1 {
                let a = tf(&mut tp, list[0].0 as u32);
                let b = tf(&mut tp, list[1].0 as u32);
                if a != b {
                    tp[b as usize] = a;
                }
            }
        }
        let mut groups: HashMap<u32, usize> = HashMap::new();
        let mut in_bad = vec![false; tris.len()];
        for list in edge_tris.values() {
            if list.len() == 2 && list[0].1 == list[1].1 {
                in_bad[list[0].0] = true;
                in_bad[list[1].0] = true;
            }
        }
        for ti in 0..tris.len() {
            if in_bad[ti] {
                *groups.entry(tf(&mut tp, ti as u32)).or_insert(0) += 1;
            }
        }
        let mut g: Vec<usize> = groups.values().copied().collect();
        g.sort_unstable_by(|a, b| b.cmp(a));
        let total: usize = g.iter().sum();
        println!(
            "[winding] 反向三角 {} 个(占 {:.3}%),分 {} 组;最大 {:.0?}",
            total,
            total as f64 / tris.len() as f64 * 100.0,
            g.len(),
            &g[..g.len().min(8)]
        );
    }

    // 并查集
    let n = pos.len();
    let mut par: Vec<u32> = (0..n as u32).collect();
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            p[x as usize] = p[p[x as usize] as usize];
            x = p[x as usize];
        }
        x
    }
    for t in &tris {
        let a = find(&mut par, t[0]);
        for &b in &t[1..] {
            let rb = find(&mut par, b);
            if a != rb {
                par[rb as usize] = a;
            }
        }
    }
    // 分量:三角数、体积、bbox
    #[derive(Default, Clone)]
    struct Comp {
        ntri: usize,
        vol: f64,
        mn: [f64; 3],
        mx: [f64; 3],
    }
    let mut comps: HashMap<u32, Comp> = HashMap::new();
    for t in &tris {
        let r = find(&mut par, t[0]);
        let e = comps.entry(r).or_insert(Comp {
            mn: [f64::MAX; 3],
            mx: [f64::MIN; 3],
            ..Default::default()
        });
        e.ntri += 1;
        let p: Vec<[f64; 3]> = t.iter().map(|&i| pos[i as usize]).collect();
        e.vol += p[0][0] * (p[1][1] * p[2][2] - p[2][1] * p[1][2])
            - p[0][1] * (p[1][0] * p[2][2] - p[2][0] * p[1][2])
            + p[0][2] * (p[1][0] * p[2][1] - p[2][0] * p[1][1])
            / 1.0;
        for v in &p {
            for k in 0..3 {
                e.mn[k] = e.mn[k].min(v[k]);
                e.mx[k] = e.mx[k].max(v[k]);
            }
        }
    }
    let mut list: Vec<(u32, Comp)> = comps.into_iter().collect();
    list.sort_by_key(|(_, c)| std::cmp::Reverse(c.ntri));
    println!("[comp] 共 {} 个分量,前 10 大:", list.len());
    let mut total_vol = 0.0f64;
    for (i, (_, c)) in list.iter().enumerate() {
        total_vol += c.vol / 6.0;
        if i < 10 {
            println!(
                "  #{i:<2} 三角 {:>7} | 体积 {:>9.2} | bbox [{:.1},{:.1},{:.1}]–[{:.1},{:.1},{:.1}]",
                c.ntri,
                (c.vol / 6.0).abs(),
                c.mn[0], c.mn[1], c.mn[2], c.mx[0], c.mx[1], c.mx[2]
            );
        }
    }
    let small: usize = list.iter().filter(|(_, c)| c.ntri < 50).count();
    let small_tri: usize = list.iter().filter(|(_, c)| c.ntri < 50).map(|(_, c)| c.ntri).sum();
    println!(
        "[comp] 碎片(<50 三角) {small} 个,占三角 {small_tri}({:.2}%)| 总净体积 {:.1}",
        small_tri as f64 / tris.len() as f64 * 100.0,
        total_vol.abs()
    );
}
