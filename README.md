# adf-proto — 自适应采样距离场(ADF)CNC 仿真原型

对应《rust-cnc-cam-research.md》§8.10/§22 的工程验证:STL → 自适应 ADF → 胶囊切削(布尔减)
→ 重距离化 → Surface Nets 提取 → STL/体积对账。零第三方依赖,edition 2024。

## 构建 / 测试 / 演示

```bash
cargo test --release
cargo run --release --bin stl-demo                                    # 桌腿固定器
cargo run --release --bin stl-demo -- stls/Turbo_Impeller.stl         # 涡轮叶轮切削演示

# 无切削重建：输入文件、目标格宽、输出文件、提取倍率
cargo run --release --bin stl_roundtrip -- stls/Turbo_Impeller.stl 0.1 rt_imp.stl 1

# 切削区探针(P23 新增):验证孔壁邻域场值正确
cargo run --release --example diag_cut

# 多角度对比渲染(P23 新增):原始 vs 输出,缺陷着色
python qa_render.py stls/Turbo_Impeller.stl stl_out.stl _qa_render --up y --tag stl_cut
python qa_render.py stls/Turbo_Impeller.stl rt_imp.stl _qa_render --up y --tag rt_imp
python qa_render.py stls/桌腿固定器.stl rt_leg.stl _qa_render --up z --tag rt_leg

# 性能定位工具
cargo run --release --example bench_bvh      # 最近点查询:深度/种子/线程扩展性
cargo run --release --example bench_redist   # 重距离化:EDT vs 八序扫掠
cargo run --release --example check_mesh      # 导出网格体检:分量/碎屑/绕向
cargo run --release --example probe_field     # 场体检:体积/表面邻域/顶点偏离
```

无切削的 `stl_roundtrip` 从 ADF 判断表面连接。如果输入的每条边都恰有两个反向相邻面，程序在变号格边上用输入三角形校准交点和法线，再用 QEM 简化并有限度地向源面投影。输入有开放边、多重关联或绕向冲突时，程序采用距离场的交点与后处理。提取器会裁去可证为空气的外围格，日志 `[nets.grid]` 给出实际格数。

设置 `ADF_NO_DEC=1` 可跳过简化及其后处理，`ADF_NO_PROJ=1` 可跳过简化后的投影。`ADF_FIELD_POST=1` 可启用距离场的平滑、吸附及折叠修复。指定 `ADF_REPAIR`、`ADF_SNAP_LIM` 或 `ADF_SNAP_STR` 也会启用这条流程。`ADF_NO_SMOOTH`、`ADF_NO_SNAP2` 与 `ADF_NO_REPAIR2` 分别关闭该流程中的相应步骤。切削演示沿加工后的 ADF 提取和后处理。

## 性能基线(16 物理核 / 24 逻辑核,Raptor Lake)

| 阶段 | 桌腿固定器(794 tri, depth10) | Turbo Impeller(154870 tri, depth10) |
|---|---|---|
| 烘焙 | 157ms | 592ms |
| 构建 | 1.21s | 4.52s |
| 切削 ×2(并行) | 0.10 / 0.25s | 0.90 / 0.53s |
| 重距离化 | 1.33s | 1.00s |
| 提取+简化+后处理 | ~34s | ~38s |
| **全管线(含提取)** | **~37s** | **~45s** |

> 注:提取阶段含 QEM 简化(主要耗时),场阶段(构建+切削+重距离)桌腿 2.6s / 叶轮 7.0s。

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
| `[error.cut]` 刀路区邻域 mean (**加工面精度**) | **0.188 mm** | **0.170 mm** |
| `[error.cut]` max | 0.84 mm | 4.32 mm† |
| 输出体积偏差(roundtrip) | +0.00% | −0.00% |
| 输出三角数(roundtrip,桌腿 h=0.08 / 叶轮 h=0.1,mult=1,err2=1e-4) | 345,750 | 373,956 |
| 输出非流形边 | 0(闭合流形) | 22 |

† `[error.cut]` max 是**指标口径**问题,不影响输出:最差点 (11.40,11.36,−1.83) 在材料内、离刀路壁
  1.33mm,所在 depth=4 粗叶在 refresh 触碰范围(胶囊 AABB ± band 0.5mm)之外,保留切削前的 base 值;
  而 `[error.cut]` 统计 |真值|≤2mm,比刷新带宽宽。|d|>band 的点不在零面上。孔壁邻域由 diag_cut
  6 个探针确认正确;切削演示刀路附近顶点离真表面 >0.5mm 的比例 19.5% → 0(P23)。

历史:`[error.cut]` mean 曾为 0.294 / 0.282mm(种子丢亚格距离,P5b 修复);`[error]` 叶轮曾为
max 81.98 / mean 8.77mm(域钳位错轴 + 符号取自最近站,P5/P6 修复)。
P23 新增修复:① `build_cell_rebuild` 内层子索引打 `LOCAL_TAG`(消除刀路穿过粗叶时接进远场空气叶的 bug);
② `writeback_walk` 精确性判据(刀路 AABB 厚板内原始表面空气角点不被 EDT 覆盖,消除"橘皮");
③ `snap_sharp_to_rim` 翻转守卫(消除叶轮底盘 88% 整面翻转三角)。

常驻内存(树 + 格点缓存):桌腿 ~227MB(树 174 + 缓存 53),叶轮 ~122MB(树 97 + 缓存 25);
页粒度 4³×4B 的取舍见开发文档 P8。

## 文档

- [开发文档.md](开发文档.md) — 后续路线(P1–P8)、已知限制、教训索引、验证纪律
- 调研主报告:`../rust-cnc-cam-research.md`(未入库)

## 测试资产

`stls/` 仅保留 demo/测试引用的两件;其余 STL 资产在 CNCSIM 项目目录,不入库。
