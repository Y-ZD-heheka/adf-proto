# adf-proto — 自适应采样距离场(ADF)CNC 仿真原型

对应《rust-cnc-cam-research.md》§8.10/§22 的工程验证:STL → 自适应 ADF → 胶囊切削(布尔减)
→ 重距离化 → Surface Nets 提取 → STL/体积对账。零第三方依赖,edition 2024。

## 构建 / 测试 / 演示

```bash
cargo test --release          # 17/17
cargo run --release --bin stl-demo                                    # 桌腿固定器
cargo run --release --bin stl-demo -- stls/Turbo_Impeller.stl 0.2     # 涡轮叶轮压测
```

## 性能基线(24 核)

| 阶段 | 桌腿固定器(794 tri) | Turbo Impeller(154870 tri) |
|---|---|---|
| 构建 | 3.35s | 121.7s |
| 切削 ×2(并行) | 0.10 / 0.25s | 8.1 / 3.7s |
| 全管线 | ~13.3s | ~137s |

## 文档

- [开发文档.md](开发文档.md) — 后续路线(P1–P8)、已知限制、教训索引、验证纪律
- 调研主报告:`../rust-cnc-cam-research.md`(未入库)

## 测试资产

`stls/` 仅保留 demo/测试引用的两件;其余 STL 资产在 CNCSIM 项目目录,不入库。
