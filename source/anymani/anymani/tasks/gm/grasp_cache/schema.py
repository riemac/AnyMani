r"""Tensor and metadata schema for `gm` grasp cache artifacts.

本文件是契约骨架，不是生成器实现。它把“什么样的数据才算可被 `gm` reset
消费”写成可检查的数据结构，避免后续脚本各自发明 cache 文件格式。

核心建模决策：cache entry 表示 hand semantic frame `{h}` 下的稳定手-物状态，
而不是 world `{w}` 下的任意 spawn pose。给定一个 cache sample：
$$
\xi_i = \left(q_i,\; T^{h}_{o,i},\; \nu_i\right),
$$
其中 $q_i\in\mathbb{R}^{n_q}$ 是 hand joint position，$T^h_{o,i}\in SE(3)$
是 object frame `{o}` 相对 hand semantic frame `{h}` 的位姿，$\nu_i$ 是可选
速度 / 诊断量。在线 reset 再由当前 env 的 hand pose 把 $T^h_o$ 变换到
Isaac Lab 需要的 `{w}` 或 `{e}` 表达。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GraspCacheKey:
    r"""Identify one validated grasp-cache shard.

    该 key 是 cache 的科研语义主键，不是随手拼出的文件名。第一版主线明确
    采用 per-asset cache，因为 post-mutate 后的 mount、link、tip、joint limit
    都会改变接触可行域；同 topology cache 最多作为离线 warm start。

    Args:
        asset_id (str): 单个 hand bundle 的稳定 ID；对应 post-mutate 样本或真实手配置。
        object_id (str): 被操作物体 ID，例如 `dex_cube`；不同物体几何不可共享稳定 cache。
        scale_bucket (str): object scale bucket，例如 `iso_1p20`；scale 改变接触几何，应进入 key。
        pose_distribution (str): 生成时使用的相对 pose 分布标签，例如 `xy_1cm_yaw_uniform_v0`。
    """

    asset_id: str  # hand 资产实例 ID；主线 cache 粒度的第一维
    object_id: str  # object 几何/语义 ID；例如 cube、cylinder 或未来 YCB object
    scale_bucket: str  # object 尺度桶；第一版只承诺 isotropic scale bucket
    pose_distribution: str  # 生成/验证分布标签；把 yaw/position DR 语义写入 key

    def as_posix_path(self) -> str:
        r"""Return a deterministic relative path fragment for this cache key.

        Returns:
            str: 形如 `asset_id/object_id/scale_bucket/pose_distribution` 的相对路径。
        """

        # 路径结构与 key 字段一一对应，便于人工肉眼检查 cache 是否错配。
        return "/".join((self.asset_id, self.object_id, self.scale_bucket, self.pose_distribution))


@dataclass(frozen=True)
class GraspCacheTensorSpec:
    r"""Describe the tensor layout stored inside one cache shard.

    第一版刻意不用裸 `(N, dof + 7)` 作为唯一说明，因为 `7D pose` 往往隐藏
    四元数符号、frame、顺序等歧义。若磁盘为了兼容 Isaac Lab 使用 quaternion，
    metadata 仍必须写清：姿态是 $T^h_o$，四元数顺序是 `wxyz`，且表示 object
    frame `{o}` 相对 hand semantic frame `{h}`。

    Args:
        num_entries (int): cache 样本数 $N$，每条样本应经过 settle / validation。
        dof (int): hand action joint 数 $n_q$；必须与 same-topology joint schema 对齐。
        joint_pos_name (str): joint position tensor 名称，形状 `[N, dof]`，单位 rad。
        object_pose_name (str): object pose tensor 名称，形状 `[N, 7]` 或未来 `[N, 4, 4]`。
        object_pose_frame (str): 位姿表达 frame，第一版固定为 `h_from_o` 语义。
        quat_order (str): 若 object pose 用 quaternion，第一版固定 `wxyz`。
    """

    num_entries: int  # 样本数 $N$；不是 env 数，也不是 rollout step 数
    dof: int  # hand joint 维度 $n_q$；用于 reset 时核对 articulation schema
    joint_pos_name: str = "joint_pos"  # `[N, dof]`，单位 rad，按 action joint order 排列
    object_pose_name: str = "object_pose_h"  # `[N, 7]` 或 `[N, 4, 4]`，语义为 $T^h_o$
    object_pose_frame: str = "h_from_o"  # frame 语义：object 相对 hand semantic frame `{h}`
    quat_order: str = "wxyz"  # Isaac Lab root state 常用 `wxyz`，metadata 必须显式记录


@dataclass(frozen=True)
class GraspCacheMetadata:
    r"""Metadata required to trust and reproduce a grasp cache shard.

    cache 不是普通随机种子文件，而是物理验证后的初始状态分布。因此 metadata
    必须保留生成分布、验证条件和上游资产引用，避免未来训练把不兼容 cache
    当作同一实验条件复用。

    Args:
        key (GraspCacheKey): cache 主键，决定该 shard 对应的 asset/object/scale/分布。
        tensor_spec (GraspCacheTensorSpec): tensor layout 与 frame/quaternion 约定。
        asset_root (Path): 生成该 cache 时使用的 hand bundle 根目录。
        generator_name (str): 离线生成器或候选来源名称，例如 `hora_style_settle_v0`。
        validator_name (str): 验证器名称，例如 `six_axis_force_validation_v0`。
        code_commit (str | None): 生成 cache 的代码 commit；临时实验可为空但不推荐。
    """

    key: GraspCacheKey  # cache 主键；决定训练时如何查找 shard
    tensor_spec: GraspCacheTensorSpec  # tensor 布局；防止 `(N, dof+7)` 语义漂移
    asset_root: Path  # hand bundle 根目录；短期允许指向 `assets/generated/.../<sample_id>`
    generator_name: str  # 候选/settle 生成流程标签；区分 HORA-style、TRO-style 等来源
    validator_name: str  # 稳定性验证协议标签；必须能复现实验语义
    code_commit: str | None = None  # 生成时 git commit；scaffold 阶段允许暂缺
