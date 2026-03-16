# Get-Zero 机器人资产生成调研总结

> 调研时间：2026-03-11
> 关联文件：`investigate.ipynb`、`example/` 目录

## 核心结论

1. **Get-Zero 通过 Python 脚本自动生成 633 个 LeapHand 变体**，核心脚本为 `get_zero/rl/scripts/gen_leap_assets.py`。
2. **关节拓扑变化（001-236）完全由代码自动生成**，不需要任何外部建模工具。通过组合式枚举各手指的关节链配置实现。
3. **连杆长度变化（237-633）的 URDF 拼装也是自动化的**，但 Get-Zero 作者用 Blender 手工制作了 5 个加长版 STL mesh（用于 3D 打印实物部件）。
4. **总共只用到 13 个 STL mesh 文件**（8 个原始 + 5 个加长版），通过引用复用给 633 个变体。
5. **代码精确控制 URDF 缩放完全可行**，已通过 2R 机器人示例验证。trimesh / URDF mesh scale 属性都可以代码化操作 mesh，保证 visual/collision 对齐。
6. **建议使用 URDF 作为变体生成格式**：有 Get-Zero 完整参考、特征提取友好、Isaac Lab 自动转 USD。

## 对 Allegro / ShadowHand 的可迁移性

| 维度 | LeapHand → Allegro | LeapHand → ShadowHand |
|------|--------------------|-----------------------|
| 拓扑枚举脚本 | 需重写（不同命名规则、拓扑） | 需重写 + 处理耦合关节 |
| mesh 复用 | 移除关节时可复用原始 mesh | 同上 |
| 长度变体 | 代码可做，不需 Blender | 同上 |
| 工程量 | 中等（1-2 天） | 较高（耦合关节复杂） |
| cross-embodiment 训练 | 可行，结构相似 | 困难，DOF/驱动方式差异大 |

## 推荐技术路线

```
Phase A（最快）→ Phase B → Phase C
同拓扑参数化       关节拓扑变体    mesh 变化
0 新 mesh          0 新 mesh      代码可做
~1 天              ~1-2 天/手型    ~1 天
```

- **Phase A**：只改 joint origin offset / limits / mass / inertia，不改 mesh，不改拓扑。纯 Python XML 操作。
- **Phase B**：参考 Get-Zero 方法，枚举合法关节子集，拼装新 URDF。复用原始 mesh。
- **Phase C**：用代码缩放 mesh（trimesh 或 URDF scale 属性），Get-Zero 用 Blender 是因为需要 3D 打印实物。

## 关键技术验证

已在 `example/` 目录生成完整示例：
- `original_2r.urdf` — 原始 2R 机器人
- `scaled_2r.urdf` — link1 沿 Z 轴 ×1.5 后的 URDF
- `gen_scaled_urdf.py` — 生成脚本，自动更新 geometry、origin、mass、inertia、子关节偏移

验证结果：缩放前后所有 link 的 visual/collision 保持精确对齐。

## 与 plan.ipynb 方法设计的关系

plan.ipynb 中的 static token $x_j^{stat}$ 所需的所有字段（joint limits, axis, rest-pose transform, link geometry, topological features）都可以从 URDF 中纯代码提取。工作流：

```
URDF 变体生成 → 特征提取（$x_j^{stat}$, $e_{ij}$）→ Isaac Lab 仿真 → RL / Distillation 训练
```
