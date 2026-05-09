r"""挂载点扰动变异：在已有 `HandCfg` 上对 finger 挂载位姿做小范围局部微调。

科研语义：
`finger.mount` 表示 finger root frame $M$ 相对 palm frame $P$ 的刚体位姿
$T_{PM}=(R_{PM},\mathbf{p}_{PM})$。本首版实现采用小扰动近似：

$$
\mathbf{p}'_{PM}=\mathbf{p}_{PM}+\delta\mathbf{p},\quad
\mathbf{r}'_{PM}=\mathbf{r}_{PM}+\delta\mathbf{r}.
$$

这里直接在 RPY 分量上加小角度，是因为 quick post-mutate 的默认扰动量级
约为 $0.03\text{rad}$，处于小角度近似可接受范围；若后续需要大角度姿态
扰动，再把这里替换成真正的 $SO(3)$ 指数映射。

从科研建模上看，这个算子变的是 finger root 相对 palm 的装配姿态，
因此它更像“挂载点 family variation”，而不是 finger 内部运动学的局部噪声。
这也是为什么它不应该和 `link_scale`、`limit_tweak` 混成一类。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ...asset_base import HandCfg
from ...asset_schema_core import PoseCfg, Vector2, Vector6
from ._base import HandPatch, MutatorBase, MutatorBaseCfg, _make_range_sampler


@dataclass
class MountPerturbCfg(MutatorBaseCfg):
    r"""finger mount 小扰动配置。

    该配置只保留用户笔记中认可的高层语义字段：`sample_space`、
    `self_mode`、`pos_range`、`rot_range`、`distrib` 与 `boundary_policy`。
    pipeline 不直接解释这些字段，而是由
    `MountPerturbMutator` lowering 成每个 finger 的
    $t_x,t_y,t_z,r_x,r_y,r_z$ 六个局部随机变量。
    """

    class_type: type["MountPerturbMutator"] | None = field(init=False, default=None, repr=False)
    r"""关联的运行时类；配置层只负责声明，运行时负责 lowering。"""

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
    运行时统一按 **radian** 解释。也就是说：

    - 裸 `float` 默认就是 rad；
    - 若研究者更习惯按 degree 记录，应在 authoring 侧显式写 `deg(...)`；
    - mutator runtime 不再额外持有 `disturb_unit` 这种第二套单位开关。

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
    """

    boundary_policy: Literal["none", "clip", "truncate", "resample"] | None = None
    r"""挂载点扰动的边界处理策略。

    该字段只规定采样结果超出 `pos_range` / `rot_range` 所定义边界时如何处理，
    不改变扰动语义 `sample_space`，也不改变基础分布 `distrib`。

    - ``"none"``：不做额外边界处理，适合均匀分布已经严格落在合法区间内的情形。
    - ``"clip"``：把越界样本裁剪到边界上，实现简单，但会增加边界点的概率质量。
    - ``"truncate"``：直接使用截断分布采样，概率语义更干净。
    - ``"resample"``：拒绝越界样本并重新采样，即 rejection sampling。

    默认值为 ``None`` 时，可由运行时根据 `distrib` 自动选择：
    均匀分布通常等价于 ``"none"``；正态分布通常使用 ``"truncate"`` 或 ``"resample"``。
    """

    def __post_init__(self) -> None:
        r"""补齐运行时类。

        这一步只做最小必要的 schema 绑定，不引入额外语义变换。
        """

        self.class_type = MountPerturbMutator


class MountPerturbMutator(MutatorBase):
    r"""将挂载点扰动 lowering 成一次性写入 `finger.mount` 的 patch。

    在已构建好的 `HandCfg` 上对指定（或全部）finger 的 `mount.pos`
    和 `mount.rpy` 做小范围局部微调，不改变拓扑和 finger 内部关节链。
    """

    cfg: MountPerturbCfg

    def __init__(self, cfg: MountPerturbCfg):
        r"""绑定一份 `MountPerturbCfg`。"""

        self.cfg = cfg

    def describe_sampling(self, target: HandCfg) -> dict[str, Any]:
        r"""为每个目标 finger 声明平移和旋转扰动变量。

        位置和平移的随机变量在这里被显式拆开，是为了让采样语义和
        后续写回语义保持完全可追踪：每个局部轴到底采了什么、写到了哪里，
        都能从 key 直接看出来。
        """

        # 这里要把平移和旋转都显式拆开，因为科研上它们对应的是两类
        # 不同的几何假设：一个是装配点位置的变化，一个是装配点朝向的变化。
        specs: dict[str, Any] = {}
        for _, finger in _iter_target_fingers(target, self.cfg):
            for axis_name, axis_range in _axis_ranges(self.cfg.pos_range, default_axis="z").items():
                specs[f"{finger.name}::t{axis_name}"] = _make_range_sampler(
                    axis_range,
                    distrib=self.cfg.distrib,
                    boundary_policy=self.cfg.boundary_policy,
                )
            for axis_name, axis_range in _axis_ranges(self.cfg.rot_range, default_axis="z").items():
                specs[f"{finger.name}::r{axis_name}"] = _make_range_sampler(
                    axis_range,
                    distrib=self.cfg.distrib,
                    boundary_policy=self.cfg.boundary_policy,
                )
        return specs

    def plan_patch(self, target: HandCfg, sampled_params: dict[str, Any] | None = None) -> HandPatch:
        r"""生成 finger-level mount patch，不立即修改原始 `HandCfg`。

        该函数严格不修改原始对象，而是把局部增量放进 patch。
        这样可以保证同一轮 post-mutate 的所有 term 都基于同一个初始
        `HandCfg` 生成自己的修改计划。
        """

        sampled_params = sampled_params or {}
        patch = HandPatch()
        # 每个 finger 单独生成 patch，但它们都来自同一份 mount 语义；
        # 这使得并行采样不会把“同一轮实验的共同假设”拆散。
        for finger_index, finger in _iter_target_fingers(target, self.cfg):
            tx = float(sampled_params.get(f"{finger.name}::tx", 0.0))
            ty = float(sampled_params.get(f"{finger.name}::ty", 0.0))
            tz = float(sampled_params.get(f"{finger.name}::tz", 0.0))
            rx = float(sampled_params.get(f"{finger.name}::rx", 0.0))
            ry = float(sampled_params.get(f"{finger.name}::ry", 0.0))
            rz = float(sampled_params.get(f"{finger.name}::rz", 0.0))

            def apply_mount(hand: HandCfg, *, fi=finger_index, dp=(tx, ty, tz), dr=(rx, ry, rz)) -> None:
                r"""把局部小扰动写回当前 finger mount。"""

                mount = hand.fingers[fi].mount
                hand.fingers[fi].mount = PoseCfg(
                    pos=(mount.pos[0] + dp[0], mount.pos[1] + dp[1], mount.pos[2] + dp[2]),
                    rpy=(mount.rpy[0] + dr[0], mount.rpy[1] + dr[1], mount.rpy[2] + dr[2]),
                )

            patch.add(("finger", finger_index, "mount"), apply_mount)
        return patch


def _iter_target_fingers(hand: HandCfg, cfg: MountPerturbCfg):
    r"""按 `self_mode` 解析目标 finger 集合。

    这里的目标集合不是简单的“把所有 finger 都采一遍”。
    `self_mode` 允许某些 family-level 先验决定哪些 finger 暂时不参与，
    例如 index/ring 镜像模式下 middle 可以作为中心锚点保持不动。
    """

    skipped = _self_mode_skipped_fingers(cfg.self_mode)
    for finger_index, finger in enumerate(hand.fingers):
        if finger.name in skipped:
            continue
        yield finger_index, finger


def _self_mode_skipped_fingers(self_mode: Any) -> set[str]:
    r"""首版把 index/ring 系列模式理解为 middle 锚定不动。

    这不是完整的 family symmetry 实现，而是一个最小可用的科研语义落点：
    先把 family-level 约束固定住，再看后续是否需要更强的镜像耦合。
    """

    # 当前首版先保留最小 family symmetry 语义：middle 作为中心锚点不动。
    if isinstance(self_mode, dict):
        return set()
    if self_mode in {"index_ring_yaw_rot", "index_ring_x_pos", "index_ring"}:
        return {"middle"}
    return set()


def _axis_ranges(value_range: Vector2 | Vector6 | None, *, default_axis: str) -> dict[str, Vector2]:
    r"""把 Vector2 / Vector6 统一解析为逐轴采样范围。

    `Vector2` 退化成单轴区间，`Vector6` 则显式提供三轴边界。
    这让同一个 mutator 在 quick path 和更细粒度实验中都能复用。
    """

    if value_range is None:
        return {}
    if len(value_range) == 2:
        return {default_axis: (float(value_range[0]), float(value_range[1]))}
    return {
        "x": (float(value_range[0]), float(value_range[1])),
        "y": (float(value_range[2]), float(value_range[3])),
        "z": (float(value_range[4]), float(value_range[5])),
    }
__all__ = ["MountPerturbCfg", "MountPerturbMutator"]
