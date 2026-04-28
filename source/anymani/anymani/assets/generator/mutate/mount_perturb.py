r"""挂载点扰动变异：在已有 HandCfg 上对 finger 挂载位姿做小范围局部微调。

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ...asset_base import AssetCfgBase, HandCfg
from ...asset_schema_core import PoseCfg, Vector2, Vector6
from ._base import MutatorBase


# ============================================================================
#  配置类
# ============================================================================


# TODO: 实现，并配有很多不同模式。
# NOTE:
# 挂载点扰动不是 link 几何缩放，而是 finger root 相对 palm 的刚体位姿微调。
# 从第一性原理看，$x/y/z$ 不应该天然同尺度：平面内横向/纵向扰动可以略大，
# 离掌面的法向扰动通常应更小。因此 general mode 中位置用椭球、
# 姿态用 $so(3)$ 小扰动球 / 椭球，会比独立矩形更少采到不自然的“盒子角落”。
@dataclass
class MountPerturbCfg(AssetCfgBase):
    r"""挂载点扰动工具配置。

    用于在已构建好的 HandCfg 上做 finger 级挂载位姿的小范围微调。

    这里的扰动语义是**绝对增量式扰动**，而不是相对比例缩放：
    - 位置扰动 $\delta \mathbf{p}$ 的单位为 meter，表示 finger root 的真实平移增量；
    - 姿态扰动 $\delta\boldsymbol{\omega}$ 或 $\delta\psi$ 的单位由 `disturb_unit` 指定，
      表示 finger root 的真实小旋转增量；
    - 不使用 `pos *= (1 + \epsilon)` 这类相对比例形式。

    坐标系语义：
    `finger.mount` 本身是 palm frame 下的刚体位姿锚点，记为
    $T_{P M}=(R_{P M},\mathbf{p}_{P M})$。本配置中 `pos_range` 和 `rot_range`
    默认描述的是当前 mount frame $M$ 中的局部扰动；真正写回 palm frame 时，
    平移需要经过 $R_{P M}$ 旋到 palm frame，姿态需要与 $R_{P M}$ 做右乘复合：
    $$
    \mathbf{p}'_{P M}=\mathbf{p}_{P M}+R_{P M}\delta\mathbf{p}_M,\quad
    R'_{P M}=R_{P M}\exp(\widehat{\delta\boldsymbol{\omega}_M}).
    $$
    这样做的原因是挂载点的物理含义是“手指根部相对手掌在哪里、朝哪里”，
    而不是某个可按比例缩放的长度、宽度或几何尺寸。
    """

    class_type: type["MountPerturbMutator"] | None = None
    """关联的运行时类。"""

    disturb_unit: Literal["deg", "rad"] = "deg"
    """微调范围的单位，默认为度。"""

    sample_space: dict[Literal["pos", "rot"], Literal["cube", "ellipsoid"]] = field(
        default_factory=lambda: {"pos": "ellipsoid", "rot": "ellipsoid"}
    )
    r"""挂载点扰动采样空间配置。

    该字段只规定“扰动向量的合法几何区域”，不直接规定扰动幅度；
    实际尺度由 `pos_range` / `rot_range` 给出的平移或姿态边界控制。
    它与 `distrib` 解耦：`distrib` 决定合法区域内部的概率密度，
    `sample_space` 决定哪些扰动向量被视为合法。

    - ``"cube"``：逐轴独立的轴对齐区域。位置扰动对应当前 mount frame 下的长方体；
      姿态扰动对应 $so(3)$ 局部旋转向量三个分量的轴对齐矩形。
      如果 `pos_range` 或 `rot_range` 使用 `Vector2`，则该模式退化为局部 $z$ 轴上的一维区间。
    - ``"ellipsoid"``：先在单位球 $\|\mathbf{u}\|_2\le1$ 内采样，
      再按各轴半径缩放到椭球：
      $$
      \delta\mathbf{x}=\operatorname{diag}(r_x,r_y,r_z)\mathbf{u}.
      $$
      注意这不是逐轴独立采样；逐轴独立采样对应的是 ``"cube"``。
      位置扰动对应当前 mount frame 下的平移椭球，姿态扰动对应 $so(3)$
      切空间中的小旋转椭球。该模式不会过度采到矩形角点，更适合作为
      general mode 默认值。如果 `pos_range` 或 `rot_range` 使用 `Vector2`，
      则三维椭球同样退化为局部 $z$ 轴上的一维区间。

    默认采用 ``{"pos": "ellipsoid", "rot": "ellipsoid"}``，表示位置和姿态都按几何上更自然的椭球小扰动采样。
    """

    self_mode: Literal["general", "index_ring_yaw_rot", "index_ring_x_pos", "index_ring"] | dict[str, float] | None = "general"
    r"""挂载点扰动的高层形态模式配置。

    该字段描述一次 post-mutate 中“先进入哪一种挂载点扰动假设”，
    不直接等价于某个具体的 $\delta\mathbf{p}$ 或 $\delta\boldsymbol{\omega}$。
    运行时应先根据 `self_mode` 选择高层模式，再由该模式决定哪些 finger 参与扰动、
    是否成对共享采样量、以及扰动方向如何解释；最后才使用 `pos_range` / `rot_range` /
    `sample_space` / `distrib` / `boundary_policy` 等低层字段生成具体数值。

    支持三种输入语义：
    - `None`：不显式指定高层模式，由运行时使用默认模式，通常等价于 ``"general"``。
    - `str`：固定使用某一种模式，例如始终使用 ``"general"``。
    - `dict[str, float]`：混合模式采样。键为模式名，值为模式采样概率；
      所有概率应非负，且概率和应为 1。后续并行化工程时应该注意到这一点

    预设模式：
    - ``"general"``：完全通用的小范围扰动模式，目标是增加资产多元性，
      不引入强 hand-family 先验。通常对所有检测到的 finger mount 使用同一套
      `pos_range` / `rot_range` 规则独立采样，默认包括 thumb / index /
      middle / ring 等所有存在的手指，适合产生“泛化噪声”。
    - ``"index_ring_yaw_rot"``：index / ring 的镜像式根部 yaw 变异。
      该模式来自 Allegro / single-palm 图示中“index 与 ring 围绕 middle 近似对称”的观察：
      middle 作为中心参考指保持不动，index 与 ring 的 mount 绕各自局部 $z$ 轴
      做成对、反向或镜像一致的小旋转。thumb 默认仍参与挂载点扰动，
      但它不参与 index / ring 的镜像耦合关系，而是按该模式的低层默认规则独立采样。
      物理上该模式描述非拇指边界指的展开 / 收拢角变化，同时保留 thumb 的常规形态多样性，
      属于 actual hand family variation，而不是无结构随机噪声。
    - ``"index_ring_x_pos"``：index / ring 的镜像式横向挂载位置变异。
      middle 仍作为中心参考指保持不动，仅让 index 与 ring 沿 palm 平面中的横向 $x$
      方向做成对平移，例如同时远离 middle 或同时靠近 middle；thumb 默认仍按低层规则
      独立扰动。该模式改变的是非拇指三指组的横向间距，同时保留拇指根部位姿的多样性，
      模拟不同机器人手 family 在 finger base spacing 上的差异。
    - ``"index_ring"``：组合模式。先按 ``"index_ring_x_pos"`` 调整 index / ring 的横向间距，
      再按 ``"index_ring_yaw_rot"`` 调整二者的根部 yaw；也可以在实现中等价地理解为
      “位置与姿态的 index/ring 镜像 family variation 同时启用”。thumb 默认仍作为普通
      finger mount 被扰动，但不与 index / ring 共享镜像采样量。

    # NOTE:
    这里的 index/ring 系列模式并不是“只扰动 index 和 ring”。
    更准确地说，它们只把 index / ring 作为一组带镜像先验的耦合对象；
    thumb 仍默认包含在 mount perturb 的作用对象中，只是不参与这组镜像约束。
    middle 更像非拇指组的几何锚点，通常在 index/ring family variation 中保持不动；
    index 与 ring 则承担左右边界手指的 family-level 展开、收拢和间距变化。
    若某个资产缺少 index 或 ring，
    运行时应跳过缺失 finger，而不是强行报错。
    """

    pos_range: Vector2 | Vector6 | None = None
    r"""挂载点位置扰动范围，采用绝对增量语义。

    该字段控制 `finger.mount.pos` 的小范围平移扰动，单位为 meter。
    这里的“绝对增量”指采样得到的是当前 mount frame 中的 $\delta\mathbf{p}_M$，
    后续通过 $R_{P M}\delta\mathbf{p}_M$ 转换到 palm frame 后写回，
    而不是对原始挂载点坐标做比例缩放。
    当 `sample_space="cube"` 时，该字段给出各轴独立采样区间；当
    `sample_space="ellipsoid"` 时，该字段给出椭球的轴向半径 / 边界。

    支持两种语义：
    - `Vector2 = (z_min, z_max)`：一维局部轴向扰动。
      采样标量 $\delta z$，并沿**当前挂载点 frame 的局部 $z$ 轴**移动 finger root。
      这不是 palm frame 的全局 $z$ 平移，而是先由 `finger.mount.rpy` 定义当前 mount frame，
      再取该 frame 的 $z$ 轴方向作为扰动方向。
    - `Vector6 = (x_min, x_max, y_min, y_max, z_min, z_max)`：三维局部平移扰动。
      三个分量均定义在当前挂载点 frame 中，适合 general mode 的 cube / ellipsoid 采样。

    当该字段为 `None` 时，不对挂载点位置施加扰动。
    """

    rot_range: Vector2 | Vector6 | None = None
    r"""挂载点姿态扰动范围，采用绝对增量语义。

    该字段控制 `finger.mount.rpy` 或等价 $SO(3)$ 姿态的小范围旋转扰动，
    单位由 `disturb_unit` 决定，可为 degree 或 radian。
    当 `sample_space="cube"` 时，该字段给出各旋转分量的独立采样区间；
    当 `sample_space="ellipsoid"` 时，该字段给出 $so(3)$ 切空间中旋转椭球的轴向半径 / 边界。

    支持两种语义：
    - `Vector2 = (yaw_min, yaw_max)`：一维局部 yaw 扰动。
      采样标量 $\delta\psi$，并绕**当前挂载点 frame 的局部 $z$ 轴**旋转。
      因此它的物理意义是“以 finger root 自身朝向为参考，轻微改变根部扭转角”，
      而不是简单地对 palm frame 下的全局 yaw 做加法。
    - `Vector6 = (rx_min, rx_max, ry_min, ry_max, rz_min, rz_max)`：三维小旋转扰动。
      三个分量可解释为当前 mount frame 的局部旋转向量
      $\delta\boldsymbol{\omega}_M\in so(3)$，适合 general mode 的 cube / ellipsoid 采样。

    当该字段为 `None` 时，不对挂载点姿态施加扰动。
    """

    distrib: Literal["uniform", "normal"] | dict[str, Any] = "uniform"
    r"""扰动幅度的分布类型。

    该字段描述“采样点在允许扰动区域内部如何分布”，而 `sample_space` 描述
    “允许扰动区域本身的几何形状”。二者是两层概念：

    - `sample_space="cube"` 时，扰动区域是轴对齐矩形盒；
      `uniform` 表示各轴在各自区间内均匀采样，`normal` 表示各轴围绕零增量做独立高斯采样，并交由 `boundary_policy` 处理越界。
    - `sample_space="ellipsoid"` 时，扰动区域是由各轴半径缩放出的椭球；
      `uniform` 表示先在单位球内部按体积均匀采样，再映射到椭球；
      `normal` 表示先在归一化切空间采样零均值高斯向量，再按椭球半径缩放，
      超出单位球的样本交由 `boundary_policy` 处理。

    以位置扰动为例，若椭球半径为 $\mathbf{r}_p=(r_x,r_y,r_z)$，
    则 general mode 的均匀椭球采样可写为：
    $$
    \delta \mathbf{p} = \operatorname{diag}(\mathbf{r}_p)\mathbf{u},
    \quad \mathbf{u}\sim \operatorname{UniformBall}(3).
    $$
    其中 $\operatorname{UniformBall}(3)$ 表示三维单位球内部的体积均匀分布，
    例如可由随机方向 $\mathbf{d}$ 与半径 $\rho=\eta^{1/3},\ \eta\sim U(0,1)$
    构造 $\mathbf{u}=\rho\mathbf{d}$。不能把它实现成逐轴独立均匀采样；
    逐轴独立均匀采样会得到 cube，而不是 ellipsoid。

    姿态扰动同理，只是 $\delta \boldsymbol{\omega}$ 位于 $so(3)$ 切空间，
    后续通过指数映射作用到 mount 姿态上：
    $$
    R_{\text{new}} = R_0\exp(\widehat{\delta\boldsymbol{\omega}}).
    $$
    支持的简写：
    - ``"uniform"``：默认；在 `sample_space` 定义的区域内部均匀采样。
    - ``"normal"``：以零增量为均值，在归一化扰动坐标中采样高斯，
      再由 `boundary_policy` 处理越界样本。

    支持的详细字典：
    - ``{"type": "uniform"}``：等同于 ``"uniform"``。
    - ``{"type": "normal", "sigma_rule": 3}``：把扰动边界视作 $3\sigma$。
    - ``{"type": "normal", "sigma": 1/3}``：直接指定归一化空间中的标准差。
    """

    boundary_policy: Literal["none", "clip", "truncate", "resample"] | None = None
    r"""扰动边界处理策略。

    该字段只规定当分布采样结果超出 `sample_space` 所定义的几何边界时如何处理，
    不改变扰动区域本身，也不改变 `distrib` 所描述的基础分布。

    - ``"none"``：不做额外边界处理，适合理论上不会越界的采样过程。
    - ``"clip"``：把越界扰动投影 / 裁剪回边界内，实现简单稳定，但会在边界处堆积概率质量。
    - ``"truncate"``：直接从截断分布中采样，概率语义最干净，但实现上需要专门的截断采样器。
    - ``"resample"``：拒绝越界样本并重新采样，即 rejection sampling；语义直观，但合法区域很小时效率会下降。

    默认值为 ``None`` 时，可由运行时根据 `sample_space` 与 `distrib` 自动选择：
    均匀分布通常不需要额外处理；正态分布通常应使用 ``"truncate"`` 或 ``"resample"``。
    """

    _target_fingers: list[str] = field(default_factory=list)
    """经解析后的内部统一规范属性。目标 finger 名称列表。"""

    _distribution: Any = field(init=False, repr=False)
    """内部解析 disturb / distrib / boundary_policy 后生成的 scipy.stats 冻结分布对象，供运行时直接调用 .rvs()。

    封装了分布形态和采样策略的“采样器工厂（Sampler Callable），主要是因为不同分布类型和采样方法的差异，需要统一接口。
    """


    def __post_init__(self):
        if self.class_type is None:
            self.class_type = MountPerturbMutator


class MountPerturbMutator(MutatorBase):
    r"""挂载点扰动运行时壳。

    在已构建好的 `HandCfg` 上对目标 finger 的挂载位姿做小范围局部微调，
    不改变拓扑和内部 joint 链。
    """

    cfg: MountPerturbCfg

    def __init__(self, cfg: MountPerturbCfg):
        self.cfg = cfg



__all__ = ["MountPerturbCfg", "MountPerturbMutator"]
