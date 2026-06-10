r"""几何 / mesh 特征编码器子包。

本子包承载**可学习或可替换的几何 adapter**，处理“原始几何特征 → token 可消费
表示”的网络侧问题。它不负责从 URDF / HandCfg / YAML 中解析几何字段。

== 与 `distill/rl/geo_obs.py` 的边界 ==

- `rl/geo_obs.py`：提取静态几何特征，如 l/w/h、mesh offset、tip 形状参数，
  以及 palm→root mount、相邻 joint→joint rest pose、joint→tip offset 等
  **static embodiment edge**；这些量在 episode 内不随 $q_t$ 变化，可缓存。
- 当前姿态下的 joint/link all-pairs SE(3) 属于 **dynamic kinematic edge**，
  它由 $q_t$ 经 FK 计算得到，随时间变化；不应和上述静态几何特征混为一类。
  teacher 当前阶段可以使用它作为 edge feature，student/sim2sim 再另行处理
  真实 URDF frame 语义对齐。
- `left/right` 不作为默认几何输入。它更像从 palm frame 下 finger mount layout
  派生出来的人类标签，而不是独立物理量。
- `models/geometry/`：把上述特征编码为 joint/tip/palm token 的一部分，
  例如低维 MLP、BPS adapter、offline embedding adapter、sim2sim 对齐 adapter。

这个子目录提前拆出，是因为 mesh/几何特征最可能膨胀：generated assets 的 primitive
几何、真实 Leap/Allegro 的不规则 mesh、未来可能的 BPS / point cloud / SDF，都不应
塞进 `tokenizer.py` 主流程里。
"""

__all__: list[str] = []
