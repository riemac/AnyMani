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
from typing import Literal

from . import AssetRunStrategyCfg
from ..generator.hand_generator import HandGeneratorCfg
from ..generator.mutate import (
    HandMutatorCfg,
    LimitTweakCfg,
    LinkProximalOverlapCfg,
    LinkScaleCfg,
    MountPerturbCfg,
    TipReplaceCfg,
)
from ..validator.hand_rules import HandValidatorCfg


ConnectivityFacade = dict[str, dict[str, list[str]]] | None
r"""pre-made connectivity façade 类型。

语义上它表示：
`hand_preset_name -> finger_slot_name -> allowed_connectivity_recipe_names`。

`None` 表示不在此处手工约束，回退到 registry 中该 hand family 的全部合法 recipe。
"""

RecolorFacade = str | dict[str, tuple[float, float, float, float]] | bool | None
r"""recolor façade 类型。

这里保留了几种科研上常见的输入方式：

- `str`：使用已注册 recolor preset 名；
- `dict`：显式指定 RGBA；
- `bool`：快速开关；
- `None`：沿用默认行为。
"""

ArtifactLevel = Literal["hand_cfg", "urdf", "bundle"]
r"""资产导出粒度。

- `hand_cfg`：只保留内存中的轻量结构；
- `urdf`：强调 URDF 落盘；
- `bundle`：同时保留 sidecar / urdf / tree 等完整产物。
"""

EditablePath = str | Path
r"""允许研究者在配置文件里直接写相对路径或绝对路径。"""


# ============================================================================
#  pre-made 配置
# ============================================================================

HAND_PRESETS: list[str] = ["single_palm_allegro", "single_palm_leap"]
"""当前 pre-made 主实验默认同时覆盖 Allegro 与 LEAP 两个 canonical base hand。"""

CONNECTIVITY_PRESETS: ConnectivityFacade = None
"""为 `None` 时使用 registry 中已注册的全部合法 connectivity recipe。"""

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


HANDEDNESS: Literal["left", "right", "all"] = "all"
"""默认同时枚举左右手，避免只在单侧上做 topology 统计。"""

MIXED = True
"""是否允许 mixed family topology。`True` 只混合 non-thumb；thumb 始终绑定 base palm family。"""

MISSING = True
"""是否允许缺指 topology。`True` 表示 pre-made 离散空间包含 missing finger 变体。"""

PRE_MADE_RECOLORED: RecolorFacade = "anatomy_soft_v1"
"""默认 recolor preset；这属于可视检查时的重要数值锚点。"""

PRE_MADE_MAX_ENUMERATE: int | None = None
"""pre-made 笛卡尔展开上限。`None` 表示不截断全部离散空间。"""

PRE_MADE_ARTIFACT_LEVEL: ArtifactLevel = "bundle"
"""默认直接导出完整 bundle，便于后续 mutate-only 从 sidecar/urdf 恢复。"""

PRE_MADE_OUTPUT_DIR: Path = Path(__file__).resolve().parents[1] / "generated"
"""默认写回 `assets/generated/`，保持子项目内部自包含。"""

PRE_MADE_VALIDATOR_CFG: HandValidatorCfg | None = HandValidatorCfg(
    pre_made=HandValidatorCfg.PreMadeCfg(
        finger_count_min=3,  # 至少保留 3 根手指，避免退化成非灵巧手拓扑
        require_non_thumb_with_min_revolute_dof=3,  # 非拇指手指至少保留 3 个 revolute DOF
        check_palm_thumb_binding=True,  # 拇指仍需与 palm 保持合法绑定关系
    )
)
"""pre-made validator 默认锚点。

# NOTE:
这几个数值直接对应“可用于后续 manipulation 任务的最小机械合理性”，
不是单纯的工程过滤条件。
"""

PREMADE_PARALLEL = True
"""是否默认开启 pre-made 样本级并行。"""

PREMADE_PARALLEL_WORKERS: int | None = None
"""worker 数为 `None` 时由 `HandGenerator` 根据 CPU 数自动推断。"""

PREMADE_PARALLEL_FALLBACK: Literal["serial", "raise"] = "serial"
"""并行失败后默认回退串行，优先保证科研产物能落地。"""

PRE_MADE_SHOW_REGISTRY = True
"""是否在 CLI 中打印当前有效 finger-level connectivity registry。"""

PRE_MADE_PRINT_RESULT_LIMIT: int | None = 40
"""终端 preview 上限，避免全量枚举时刷屏。"""

PRE_MADE_CFG = HandGeneratorCfg(
    mode="made",  # 只做离散 pre-made，不进入 post-mutate Monte Carlo
    artifact_level=PRE_MADE_ARTIFACT_LEVEL,  # 默认导出完整 bundle
    output_dir=PRE_MADE_OUTPUT_DIR,  # 产物根目录保持在 `assets/generated/`
    handedness=HANDEDNESS,  # 左右手枚举策略
    hand_presets=list(HAND_PRESETS),  # canonical base hand 候选集合
    connectivity_presets=CONNECTIVITY_PRESETS,  # 每个 base hand 允许搭配的 connectivity recipe
    mixed=MIXED,  # 只允许 non-thumb 跨 family；thumb 绑定 base palm family
    missing=MISSING,  # 是否允许缺指 topology
    Validate=PRE_MADE_VALIDATOR_CFG,  # pre-made hand-level validator
    recolored=PRE_MADE_RECOLORED,  # 导出前的可视 recolor 方案
    max_enumerate=PRE_MADE_MAX_ENUMERATE,  # 离散笛卡尔空间截断预算
    premade_parallel=PREMADE_PARALLEL,  # 是否并行展开 pre-made 样本
    premade_parallel_workers=PREMADE_PARALLEL_WORKERS,  # worker 数占位
    premade_parallel_fallback=PREMADE_PARALLEL_FALLBACK,  # 并行失败后的回退策略
)
"""正式的 pre-made 主入口。

# NOTE:
这里直接实例化 `HandGeneratorCfg`，而不是再包一层 quick cfg，
是这轮重构的核心设计收敛点之一。
"""


# ============================================================================
#  post-mutate 配置
# ============================================================================

POST_MUTATE_SOURCE_TOPOLOGY_PATH: EditablePath = (
    "AnyMani/source/anymani/anymani/assets/generated/"
    "2026-05-03_09-45-45/single_palm_leap/right_t4_i4_m4_r4"
)
"""独立 post-mutate 的默认来源 topology 根目录。

新 contract 下，这个目录自己就持有 pre-made 的 `hand.yaml`，
runner 与 `HandGenerator` 都不再接受 sample 子目录。
"""

POST_MUTATE_N_SAMPLES: int = 100
"""联合 Monte Carlo 目标样本数 $N=100$。"""

POST_MUTATE_ARTIFACT_LEVEL: ArtifactLevel = "bundle"
"""默认导出 bundle，便于事后比对 sidecar / urdf / summary。"""

POST_MUTATE_RECOLORED: RecolorFacade = "anatomy_soft_v1"
"""后变异样本默认沿用同一套 anatomy recolor preset。"""

POST_MUTATE_PRINT_RESULT_LIMIT: int | None = 10
"""终端里只 preview 前若干个后变异样本。"""


class QuickPostMutateCfg(HandMutatorCfg):
    r"""当前独立 post-mutate 调试用 term container。

    这里保留的是“当前最常用的一套后变异组合”，而不是 post-mutate 唯一合法形式。
    其 Declare / Sample / Apply 语义由 `HandMutatorCfg` 与各 mutator 自己负责：

    1. `link_scale`
       对 finger 链中间 link 的有效长度做轻度相对扰动；
    2. `mount_perturb`
       对 finger root mount 做小范围局部位姿扰动；
    3. `limit_tweak`
       对活动关节 limit 做共享微调；
    4. `tip_replace`
       对末端 tip family / scale 做统一替换。

    数值锚点：
    - `link_scale=(0.9, 1.1)`：长度相对扰动约 $\pm10\%$；
    - `pos_range=(-0.002, 0.002)`：平移扰动量级约 $\pm2\text{mm}$；
    - `rot_range=(-0.03, 0.03)`：小角度旋转扰动约 $\pm1.7^\circ$；
    - `joint_range=(-0.03, 0.03)`：关节限位加性微调约 $\pm0.03\text{rad}$。
    """

    link_scale = LinkScaleCfg(
        scale_type="rel",  # 采用相对缩放语义，而不是绝对长度增量
        link_scale=(0.9, 1.1),  # 主长度方向允许约 $\pm10\%$ 的轻扰动
        clip=(0.8, 1.2),  # 防止极端采样把 link 拉到明显脱离原家族的尺度
        distrib="uniform",  # 首版默认使用均匀分布
        boundary_policy="clip",  # 越界样本直接裁剪回合法区间
    )
    link_proximal_overlap = LinkProximalOverlapCfg(
        self_mode={"identity": 0.2, "disturb": 0.5, "homologous_non_thumb": 0.3},
        overhang_delta_ratio=(-0.1, 0.2),
        max_parent_overlap_ratio=0.5,
        distrib="uniform",
        boundary_policy="clip",
    )
    mount_perturb = MountPerturbCfg(
        disturb_unit="deg",  # 这里保留原配置的角度输入语义
        sample_space={"pos": "ellipsoid", "rot": "ellipsoid"},  # 位置与姿态都按椭球小扰动采样
        self_mode="general",  # 不引入 index/ring 镜像耦合，先做一般性挂载点 family variation
        pos_range=(-0.002, 0.002),  # 平移扰动量级约 $\pm2\text{mm}$
        rot_range=(-0.03, 0.03),  # 小角度姿态扰动区间
        distrib="uniform",  # 默认在合法区域内均匀采样
        boundary_policy="clip",  # 首版仍使用简单可解释的裁剪策略
    )
    limit_tweak = LimitTweakCfg(
        disturb_unit="rad",  # joint limit 内部正式写回就是弧度语义
        disturb_object="shared",  # 每个关节的 lower / upper 共用同一个扰动量
        disturb_type="add",  # 以加性微调方式改 limit，而不是比例缩放
        joint_range=(-0.03, 0.03),  # 共享扰动区间约 $\pm0.03\text{rad}$
        clip={"abs": 0.12},  # 限制微调绝对幅值，避免 limit 被推得过大
        distrib={"type": "normal", "sigma_rule": 3},  # 用约 $3\sigma$ 覆盖配置区间
        boundary_policy="clip",  # 首版仍优先保持行为确定、容易调试
    )
    tip_replace = TipReplaceCfg(
        mode="geometry_swap",  # 首版先做 primitive / preset tip 几何互换
        target_geometry=None,  # `None` 表示由运行时在合法目标几何间自行 resolve
        self_mode="same",  # 全手共享同一套 tip family 假设，保持 morphology coherence
        tip_range=None,  # 候选 tip family 由运行时根据当前 hand family 自动推断
        scale=(0.98, 1.02),  # tip size 只做约 $\pm2\%$ 的轻微缩放
    )


POST_MUTATE_MUTATOR_CFG = QuickPostMutateCfg()
"""独立 post-mutate 默认采用的 mutator term container。"""

POST_MUTATE_VALIDATOR_CFG: HandValidatorCfg | None = HandValidatorCfg(
    post_mutate=HandValidatorCfg.PostMutateCfg(
        finger_count_min=3,  # 后变异后仍需保持至少 3 指
        require_non_thumb_with_min_revolute_dof=3,  # 非拇指手指仍需有足够活动自由度
        check_finger_spacing=True,  # 显式检查挂载扰动后是否出现过近手指间距
        min_finger_spacing=0.01,  # 最小合法间距约 $1\text{cm}$
        check_mount_consistency=True,  # mount perturb 后仍需保持 mount 语义一致性
    )
)
"""post-mutate validator 默认锚点。

这里强调的是“后变异不能把 asset 从局部随机化推到明显不合理的机械体”。
"""

# `mode="mutate"` 目前要求 cfg 上存在 `source_topology_dir`，因此这里放一个静态占位符；
# 真正运行前由统一 runner 根据来源 topology 路径替换成正式 pre-made topology 根目录。
POST_MUTATE_CFG = HandGeneratorCfg(
    mode="mutate",  # 只做后变异，不重新枚举 pre-made 空间
    artifact_level=POST_MUTATE_ARTIFACT_LEVEL,  # 默认导出完整 bundle
    source_topology_dir=Path("__post_mutate_topology_dir__"),  # 运行前由 runner 动态替换成 pre-made topology 根
    output_dir=Path("__post_mutate_output_dir__"),  # 兼容占位；新目录 contract 实际由 source_topology_dir 驱动
    n_samples=POST_MUTATE_N_SAMPLES,  # 目标联合 Monte Carlo 样本数
    Mutate=POST_MUTATE_MUTATOR_CFG,  # 当前默认 mutator term container
    Validate=POST_MUTATE_VALIDATOR_CFG,  # 后变异 hand-level validator
    recolored=POST_MUTATE_RECOLORED,  # 后变异样本的可视 recolor 方案
)
"""正式的独立 post-mutate 主入口。

# NOTE:
这里保留静态占位路径，是因为 `HandGeneratorCfg(mode="mutate")`
目前 contract 上要求 `source_topology_dir` 已存在；而人在调试时更自然填写的是
配置模块里的 topology 根路径，lowering 放到 runner 层做。
"""


# ============================================================================
#  runner 策略占位
# ============================================================================

ASSET_RUN_STRATEGY = AssetRunStrategyCfg(
    topology_selection_mode="all",  # 当前只实现“覆盖全部 topology”的保守策略
    topology_selection_count=None,  # 随机子集策略尚未实现，因此这里必须保持空值
)
"""runner 级未来扩展占位。

用户后续提到的：

- 只对若干随机 topology 做 mutate；
- 或者强制包含 presets full hand 再随机补其它 topology；

都预留在这里，但这轮先不落实现。
"""


__all__ = [
    "ASSET_RUN_STRATEGY",
    "POST_MUTATE_CFG",
    "POST_MUTATE_PRINT_RESULT_LIMIT",
    "POST_MUTATE_SOURCE_TOPOLOGY_PATH",
    "PRE_MADE_CFG",
    "PRE_MADE_PRINT_RESULT_LIMIT",
    "PRE_MADE_SHOW_REGISTRY",
]
