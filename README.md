# adf-proto — 自适应采样距离场(ADF)CNC 仿真原型

对应《rust-cnc-cam-research.md》§8.10/§22 的工程验证:STL → 自适应 ADF → 胶囊切削(布尔减)
→ 重距离化 → Surface Nets 提取 → STL/体积对账。零第三方依赖,edition 2024。

## 构建 / 测试 / 演示

```bash
cargo test --release          # 24/24
cargo run --release --bin stl-demo                                    # 桌腿固定器
cargo run --release --bin stl-demo -- stls/Turbo_Impeller.stl 0.2     # 涡轮叶轮压测

# 性能定位工具
cargo run --release --example bench_bvh      # 最近点查询:深度/种子/线程扩展性
cargo run --release --example bench_redist   # 重距离化:EDT vs 八序扫掠
cargo run --release --example check_mesh      # 导出网格体检:分量/碎屑/绕向
cargo run --release --example probe_field     # 场体检:体积/表面邻域/顶点偏离
```

## 性能基线(16 物理核 / 24 逻辑核,Raptor Lake)

| 阶段 | 桌腿固定器(794 tri, depth10) | Turbo Impeller(154870 tri, depth9) |
|---|---|---|
| 烘焙 | 122ms | 214ms |
| 构建 | 0.95s | 0.77s |
| 切削 ×2(并行) | 0.06 / 0.12s | 0.13 / 0.07s |
| 重距离化 | 0.49s | 0.18s |
| 提取+STL(含绕向统一) | 0.90s | 0.16s |
| **全管线** | **~2.7s** | **~1.6s** |

历史对照(同一台机器):初版 13.3s / 137s;BVH 叶判别修正 7.8s / 2.5s;提取并行 + 热路径去开销
2.4s / 1.8s;去掉容器与 libm 开销后 2.2s / 1.5s。七处收益分别来自:①BVH 叶判别修正(P1),②重距离化改精确 EDT(P2),
③表面提取三阶段并行 + 移除构建热路径的诊断计时器(P9/P11),④BVH 遍历去堆改有界栈 DFS(P13),
⑤去掉热路径上的 libm `round` 调用(P14),⑥修复域钳位用错轴(P6),⑦符号改洪泛定号 +
种子保留亚格距离(P5/P5b)。

## 精度基线(对照 `BaseField::sd` 精确值)

| 指标 | 桌腿固定器 | Turbo Impeller |
|---|---|---|
| `[error]` 随机 2000 点 max / mean | 1.75 / 0.55 mm | 2.65 / 0.72 mm |
| `[error.split]` 符号相反 | 0 点 | 0 点 |
| `[error.cut]` 刀路区邻域(**加工相关**) | **0.19 mm** | **0.19 mm** |

历史:`[error.cut]` 曾为 0.294 / 0.282mm(种子丢亚格距离),`[error]` 叶轮曾为
max 81.98 / mean 8.77mm(域钳位错轴 + 符号取自最近站)。

常驻内存(树 + 格点缓存):桌腿 ~227MB(树 174 + 缓存 53),叶轮 ~122MB(树 97 + 缓存 25);
页粒度 4³×4B 的取舍见开发文档 P8。

## 文档

- [开发文档.md](开发文档.md) — 后续路线(P1–P8)、已知限制、教训索引、验证纪律
- 调研主报告:`../rust-cnc-cam-research.md`(未入库)

## 测试资产

`stls/` 仅保留 demo/测试引用的两件;其余 STL 资产在 CNCSIM 项目目录,不入库。
