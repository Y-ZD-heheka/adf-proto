# adf-proto — 自适应采样距离场(ADF)CNC 仿真原型

对应《rust-cnc-cam-research.md》§8.10/§22 的工程验证:STL → 自适应 ADF → 胶囊切削(布尔减)
→ 重距离化 → Surface Nets 提取 → STL/体积对账。零第三方依赖,edition 2024。

## 构建 / 测试 / 演示

```bash
cargo test --release          # 20/20
cargo run --release --bin stl-demo                                    # 桌腿固定器
cargo run --release --bin stl-demo -- stls/Turbo_Impeller.stl 0.2     # 涡轮叶轮压测

# 性能定位工具
cargo run --release --example bench_bvh      # 最近点查询:深度/种子/线程扩展性
cargo run --release --example bench_redist   # 重距离化:EDT vs 八序扫掠
```

## 性能基线(16 物理核 / 24 逻辑核,Raptor Lake)

| 阶段 | 桌腿固定器(794 tri, depth10) | Turbo Impeller(154870 tri, depth9) |
|---|---|---|
| 烘焙 | 94ms | 361ms |
| 构建 | 2.11s | 1.20s |
| 切削 ×2(并行) | 0.07 / 0.16s | 0.22 / 0.10s |
| 重距离化 | 0.50s | 0.29s |
| 提取+STL | 4.86s | 0.62s |
| **全管线** | **~7.8s** | **~2.5s** |

历史对照(同一台机器,2026-11-15 修正前):构建 3.35s / 121.7s,重距离化 4.76s / 2.70s,
全管线 13.3s / 137s。两处主要收益来自:①BVH 叶判别修正(见开发文档 P1),
②重距离化改精确 EDT(见 P2)。

## 文档

- [开发文档.md](开发文档.md) — 后续路线(P1–P8)、已知限制、教训索引、验证纪律
- 调研主报告:`../rust-cnc-cam-research.md`(未入库)

## 测试资产

`stls/` 仅保留 demo/测试引用的两件;其余 STL 资产在 CNCSIM 项目目录,不入库。
