r"""统一资产生成配置。

本文件只承担“声明式配置模块”的职责，不承担执行逻辑。
它的定位刻意对齐 Isaac Lab 中 `tasks/.../config/*.py` 的写法：

1. 研究者只需要在这里看和改配置，不需要翻到 CLI runner 里；
2. `HandGeneratorCfg` 仍是资产生产的最高 façade，不再额外包新的 run cfg；
3. pre-made 与 post-mutate 的分离工作流，在这里被表示成两份正式
   `HandGeneratorCfg` 常量，而不是两份过程式脚本。

当前科研工作流以 leap/allegro 联合实验为主，同时保留两种典型运行面：

- 完整 pre-made 枚举：
  对离散 topology × connectivity 空间做系统性展开；
- 独立 post-mutate 调试：
  从某个已有 pre-made topology 根出发，反复做后变异实验。

# NOTE:
执行逻辑由 `assets/scripts/generate.py` 和 `_asset_generate_runner.py`
负责；这里仅声明配置常量与少量 runner 级占位策略。
"""

from __future__ import annotations

from pathlib import Path

from ..asset_physics import AssetPhysicsCfg, DensityProfileCfg
from ..generator.hand_generator import HandGeneratorCfg
from ..generator.mutate import (
    HandMutatorCfg,
    LimitTweakCfg,
    LinkProximalOverlapCfg,
    LinkScaleCfg,
    MountPerturbCfg,
    TipReplaceCfg,
)
from ..units import cm, deg, g_cm3, mm
from ..validator.hand_rules import HandValidatorCfg
from . import AssetRunStrategyCfg

ConnectivityFacade = dict[str, dict[str, list[str]]] | None  # hand_preset -> finger_slot -> allowed connectivity recipes
EditablePath = str | Path  # 允许研究者直接写相对路径或绝对路径


# ============================================================================
#  统一物理资产配置；设为 None 可显式关闭 physics closure
# ============================================================================

ASSET_PHYSICS_CFG: AssetPhysicsCfg | None = AssetPhysicsCfg(
    density=DensityProfileCfg(
        # default=g_cm3(0.65),  # 全局默认密度 $\rho$ [$\mathrm{kg}/\mathrm{m}^3$]
        palm=g_cm3(0.55),  # palm 专用密度；
        finger_link=g_cm3(1.2),  # 普通 finger link 专用密度；
        fingertip=g_cm3(0.72),  # primitive fingertip 专用密度；TPU 材料约 1.2 $$\mathrm{g}/\mathrm{cm}^3$$，这里填充率设 60 %
        custom_tip=g_cm3(0.72),  # 同上
    )
)


# ============================================================================
#  pre-made 配置
# ============================================================================

HAND_PRESETS: list[str] = ["single_palm_allegro", "single_palm_leap"]  # pre-made 默认覆盖 Allegro + LEAP

CONNECTIVITY_PRESETS: ConnectivityFacade = None  # None 表示使用 registry 中全部合法 connectivity recipe

# # 部分已注册示例
# CONNECTIVITY_PRESETS: ConnectivityFacade = {
#     "single_palm_allegro": {
#         "thumb": [
#             "allegro_thumb_full",
#             "allegro_thumb_drop_j3",
#         ],
#         "index": [
#             "allegro_non_thumb_full",
#             "allegro_non_thumb_drop_j3",
#         ],
#         "middle": [
#             "allegro_non_thumb_full",
#             "allegro_non_thumb_drop_j3",
#         ],
#         "ring": [
#             "allegro_non_thumb_full",
#             "allegro_non_thumb_drop_j3",
#         ],
#     },
#     "single_palm_leap": {
#         "thumb": ["leap_thumb_full"],
#         "index": ["leap_non_thumb_full"],
#         "middle": ["leap_non_thumb_full"],
#         "ring": ["leap_non_thumb_full"],
#     },
# }

PRE_MADE_OUTPUT_DIR: Path = Path(__file__).resolve().parents[1] / "generated"  # 默认写回 assets/generated/

# 这几个阈值是后续 manipulation 任务的最小机械合理性锚点，不是单纯工程过滤。
PRE_MADE_VALIDATOR_CFG: HandValidatorCfg | None = HandValidatorCfg(
    pre_made=HandValidatorCfg.PreMadeCfg(
        finger_count_min=3,  # 至少保留 3 根手指，避免退化成非灵巧手拓扑
        require_non_thumb_with_min_revolute_dof=3,  # 非拇指手指至少保留 3 个 revolute DOF
        check_palm_thumb_binding=True,  # 拇指仍需与 palm 保持合法绑定关系
    )
)

PRE_MADE_SHOW_REGISTRY = True  # CLI 是否打印当前 finger-level connectivity registry

PRE_MADE_PRINT_RESULT_LIMIT: int | None = 40  # 终端 preview 上限，避免全量枚举时刷屏

PRE_MADE_CFG = HandGeneratorCfg(
    mode="made",  # 只做离散 pre-made，不进入 post-mutate Monte Carlo
    artifact_level="bundle",  # 默认导出完整 bundle，便于 mutate-only 恢复
    output_dir=PRE_MADE_OUTPUT_DIR,  # 产物根目录保持在 `assets/generated/`
    handedness="all",  # 默认同时枚举左右手
    hand_presets=list(HAND_PRESETS),  # canonical base hand 候选集合
    connectivity_presets=CONNECTIVITY_PRESETS,  # 每个 base hand 允许搭配的 connectivity recipe
    mixed=True,  # 只允许 non-thumb 跨 family；thumb 始终绑定 base palm family
    missing=True,  # 允许 pre-made 离散空间包含缺指 topology
    Validate=PRE_MADE_VALIDATOR_CFG,  # pre-made hand-level validator
    Physics=ASSET_PHYSICS_CFG,  # pre-made 导出前按统一物理资产配置闭合刚体参数
    recolored="anatomy_soft_v1",  # 导出前的可视 recolor 方案
    max_enumerate=None,  # None 表示不截断 pre-made 笛卡尔展开空间
    premade_parallel=True,  # 默认开启 pre-made 样本级并行
    premade_parallel_workers=None,  # None 表示由 HandGenerator 根据 CPU 数自动推断
    premade_parallel_fallback="serial"  # 并行失败后默认回退串行
)


# ============================================================================
#  post-mutate 配置
# ============================================================================

# 独立 post-mutate 来源必须是 topology root，且该目录应直接持有 pre-made 的 hand.yaml。
POST_MUTATE_SOURCE_TOPOLOGY_PATH: EditablePath = (
    "AnyMani/source/anymani/anymani/assets/generated/2026-08-12_18-16-48/single_palm_leap/right_t4_i4_m4_r4"
)

POST_MUTATE_PRINT_RESULT_LIMIT: int | None = 10  # 终端 preview 上限


class QuickPostMutateCfg(HandMutatorCfg):
    r"""当前独立 post-mutate 调试用 term container。

    这里保留的是“当前最常用的一套后变异组合”，而不是 post-mutate 唯一合法形式。
    其 Declare / Sample / Apply 语义由 `HandMutatorCfg` 与各 mutator 自己负责：
    """

    link_scale = LinkScaleCfg(
        self_mode={"identity": 0.2, "general": 0.4, "only_length": 0.4},  # 每个候选独立抽 mode；only_length 只消费 length range
        scale_type="rel",  # 采用相对缩放语义，而不是绝对长度增量
        # link_scale=(0.9, 1.1),  # 主长度方向允许约 $\pm10\%$ 的轻扰动
        link_scale=(0.75, 1.25, 0.9, 1.1, 0.9, 1.1),
        # clip=(0.8, 1.2),  # 防止极端采样把 link 拉到明显脱离原家族的尺度
        distrib="uniform",  # 首版默认使用均匀分布
        boundary_policy="clip",  # 越界样本直接裁剪回合法区间
    )
    link_proximal_overlap = LinkProximalOverlapCfg(
        self_mode={"identity": 0.2, "disturb": 0.5, "homologous_non_thumb": 0.3},  # 整手先选 no-op、逐 owner 或 non-thumb 同源槽共享模式
        overhang_delta_ratio=(-1, 2),  # signed ratio：最多缩减 $10\%$ child span，或增加 $20\%$ child span 的 proximal overhang
        max_parent_overlap_ratio=0.4,  # 最终 overhang 不得超过变异前 parent 净 span 的一半
        distrib="uniform",  # 在声明区间内均匀采样 $\eta_i$
        boundary_policy="clip",  # ratio 越界时裁回合法区间，最终几何另受 parent-relative cap
    )
    mount_perturb = MountPerturbCfg(
        self_mode={"identity": 0.2, "general": 0.2, "index_ring_x_pos": 0.2, "index_ring_yaw_rot": 0.2, "index_ring": 0.2},  # 保留一部分 pre-made 原姿态权重，再在 index/ring family variation 中采样
        pos_radius=cm(0.8),  # mount 平移扰动半径当前取 $0.8\text{cm}$，配合后续的 mirror_x_range 形成 index/ring family variation
        rot_radius=deg(5),  # mount 局部旋转扰
        mirror_x_range=(cm(-1.0), cm(1.0)),  # index/ring 横向间距在 palm-frame $x$ 上做镜像 cube 采样，量级约 $\pm1\text{cm}$
        mirror_yaw_range=(deg(-5), deg(5)),  # index/ring 根部 yaw 在局部 frame 上做镜像 cube 采样，量级约 $\pm5^\circ$
        thumb_pos_radius=cm(1.0),  # index/ring 模式下，thumb 仍保留局部椭球平移扰动，作为独立 family variation
        thumb_rot_radius=deg(5.0),  # thumb 局部旋转椭球半径当前取 $5^\circ$
        distrib="uniform",  # 默认在合法区域内均匀采样
        boundary_policy="clip",  # 首版仍使用简单可解释的裁剪策略
    )
    limit_tweak = LimitTweakCfg(
        disturb_object="independent",  # 每个关节的 lower / upper 分别独立采样；如需同关节上下界同移，改用 "shared"
        disturb_type="add",  # 以加性微调方式改 limit，而不是比例缩放
        joint_range=(deg(-10), deg(10)),
        self_mode={"identity":0.2, "disturb":0.5, "homologous_non_thumb":0.3},
        # clip={"abs": 0.12},  # 限制微调绝对幅值，避免 limit 被推得过大
        distrib={"type": "uniform"},  # 在配置区间内均匀采样；`deg(...)` 已在 authoring 侧换算为 rad
        boundary_policy="clip",  # 首版仍优先保持行为确定、容易调试
    )
    tip_replace = TipReplaceCfg(
        self_mode={"identity": 0.2, "same": 0.5, "general": 0.3},  # 每个候选独立抽 mode；validator 可自然改变 accepted 分布
        tip_range={"cs": 0.2, "leap_cube": 0.2, "round": 0.2, "wedge": 0.2, "thinner": 0.2},  # tip_type 概率只是 proposal 分布
        scale=(0.9, 1.1),  # tip size 只做约 $\pm2\%$ 的轻微缩放
        cs_ratio={"add": (-0.15, 0.15)},  # `cs` 固定半径 $r$，只微调 $\lambda=h/r$
    )


POST_MUTATE_MUTATOR_CFG = QuickPostMutateCfg()  # 独立 post-mutate 默认 mutator term container

# 后变异不能把 asset 从局部随机化推到明显不合理的机械体。
POST_MUTATE_VALIDATOR_CFG: HandValidatorCfg | None = HandValidatorCfg(
    post_mutate=HandValidatorCfg.PostMutateCfg(
        finger_count_min=3,  # 后变异后仍需保持至少 3 指
        require_non_thumb_with_min_revolute_dof=3,  # 非拇指手指仍需有足够活动自由度
        check_finger_spacing=True,  # 显式检查挂载扰动后是否出现过近手指间距
        min_finger_spacing=mm(5),  # 最小合法间距当前取 $5\text{mm}$；显式写单位避免注释与数值脱节
        check_finger_length=True,  # 显式检查 home pose 下 finger 沿 nominal distal axis 的真实几何长度
        max_thumb_length=cm(18.5),  # thumb 的轴向真实长度上限；比 non-thumb 略放宽，容纳当前拇指基座段几何包络
        max_non_thumb_length=cm(18.0),  # non-thumb 的轴向真实长度上限；先挡掉 link_scale 产生的极端长指
        check_mount_consistency=True,  # mount perturb 后仍需保持 mount 语义一致性
        sdf_device="cuda",  # 正式 dataset validator 固定使用 GPU，CUDA 不可用时 fail-hard
        sdf_mesh_backend="warp",  # mesh SDF 固定 Warp；禁止同一 dataset 内隐式混入 CPU fallback
    )
)

# `mode="mutate"` 目前要求 cfg 上存在 `source_topology_dir`，因此这里放一个静态占位符；
# 真正运行前由统一 runner 根据来源 topology 路径替换成正式 pre-made topology 根目录。
POST_MUTATE_CFG = HandGeneratorCfg(
    mode="mutate",  # 只做后变异，不重新枚举 pre-made 空间
    artifact_level="bundle",  # 默认导出 bundle，便于事后比对 sidecar / URDF / summary
    source_topology_dir=Path("__post_mutate_topology_dir__"),  # 运行前由 runner 动态替换成 pre-made topology 根
    output_dir=Path("__post_mutate_output_dir__"),  # 兼容占位；新目录 contract 实际由 source_topology_dir 驱动
    n_samples=20,  # 本轮 Geometry SSL 试水需要 20 个成功 variants；shortfall run 只作生成诊断
    post_mutate_seed=20260813,  # 固定联合 proposal 随机序列，便于生成 run 完整重放
    post_mutate_attempts_per_variant=10,  # 每个计划 variant 独立拥有十次完整重抽预算
    post_mutate_require_unique_geometry=True,  # 数据集 variant set 拒绝 mother no-op 与 set 内静态几何重复，并在当前槽位补抽
    post_mutate_sdf_execution="central_gpu_batch",  # 单 GPU actor 批量验证；mother workers 不持有 CUDA context
    Mutate=POST_MUTATE_MUTATOR_CFG,  # 当前默认 mutator term container
    Validate=POST_MUTATE_VALIDATOR_CFG,  # 后变异 hand-level validator
    Physics=ASSET_PHYSICS_CFG,  # 后变异 validator 之前按同一套物理资产配置闭合刚体参数
    recolored="anatomy_soft_v1",  # 后变异样本的可视 recolor 方案
)


# ============================================================================
#  runner 策略占位
# ============================================================================

ASSET_RUN_STRATEGY = AssetRunStrategyCfg(
    topology_selection_mode="all",  # 当前只实现“覆盖全部 topology”的保守策略
    topology_selection_count=None,  # 随机子集策略尚未实现，因此这里必须保持空值
)


__all__ = [
    "ASSET_RUN_STRATEGY",
    "ASSET_PHYSICS_CFG",
    "POST_MUTATE_CFG",
    "POST_MUTATE_PRINT_RESULT_LIMIT",
    "POST_MUTATE_SOURCE_TOPOLOGY_PATH",
    "PRE_MADE_CFG",
    "PRE_MADE_PRINT_RESULT_LIMIT",
    "PRE_MADE_SHOW_REGISTRY",
]
