r"""两族教师离线蒸馏的纯数据合同与 CPU 读取辅助。

本模块把冻结 teacher 的两族 rollout 封装为可审计的 HDF5 artifact；它不 import Isaac、
IsaacLab、网络或训练 runtime，因此 collector 与训练器可以在各自进程中复用同一份合同。
根属性固定为 ``artifact_type=anymani.family_teacher_trajectory`` 与
``schema_version=1.0.0``。动态事实沿时间轴分块追加：每一步都保存当前关节帧、owner
接触和 active mask；teacher mean、实际 behavior action、N040 geometry token 与 FK 只在
``step % sample_stride == 0`` 的稀疏 sample 轴保存。这样总内存只随一个 step 的
``[N, ...]`` batch 与 History30 状态增长，而不会把全 run 装进 Python 列表。

History30 的时间约定是严格的：``initial_history`` 已是 reset 后包含 ``current_0`` 的
完整 ``H_0``，且每一行按 oldest-to-latest 排列。对连续 active 行，

$$
H_t = \operatorname{tail}_{30}\left(H_0 \mathbin{\|}
    [q_1, q_2, \ldots, q_t]\right),
$$

其中 ``q_t`` 代表保存的 ``jnt_current[t]``，``\|`` 是沿时间轴拼接。``append(0)`` 只
核对传入的 H0 与初始历史；之后 writer 逐行重建并只在 active 行比较，inactive 的
reset/padding history 不被拿来跨 episode 拼接。动作、接触、mask、ghost/padding 与 FK
均 fail-closed 检查形状、有限性和适用数值区间；失败轨迹的原始 final 统计始终保留，
质量 mask 只表达是否可进入训练分母，不删除失败事实。
"""

from __future__ import annotations

import json
import operator
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, Final, cast

import h5py
import numpy as np

# 数据格式身份是跨 collector、reader、No-Z 与 Ours 比较的信任根。
ARTIFACT_TYPE: Final[str] = "anymani.family_teacher_trajectory"
SCHEMA_VERSION: Final[str] = "1.0.0"

# 下面的维度对应 actor ABI；它们不是任意 padding 数值，改变会使已保存数据不可训练。
HISTORY_LENGTH: Final[int] = 30
JOINT_COUNT: Final[int] = 16
JOINT_FEATURES: Final[int] = 5
OWNER_COUNT: Final[int] = 21
TIP_COUNT: Final[int] = 4
GEOMETRY_TOKEN_WIDTH: Final[int] = 128
JOINT_KINEMATICS_WIDTH: Final[int] = 15

# 20 Hz 下 30 s 的理论步数；quality gate 的浮点时间容差只吸收 FP32 累加误差。
CONTROL_FREQUENCY_HZ: Final[float] = 20.0
QUALITY_HORIZON_STEPS: Final[int] = 600
DURATION_ATOL_S: Final[float] = 1.0e-3

# 所有静态 geometry evidence 必须按相同、有序资产轴交付。
_REQUIRED_STATIC_SHAPES: Final[dict[str, tuple[int, ...]]] = {
    "actor_jnt_limits": (JOINT_COUNT, 2),
    "jnt_valid": (JOINT_COUNT,),
    "tip_valid": (TIP_COUNT,),
    "owner_valid": (OWNER_COUNT,),
    "shortest_path": (OWNER_COUNT, OWNER_COUNT),
    "parent_direction": (OWNER_COUNT, OWNER_COUNT),
    "child_direction": (OWNER_COUNT, OWNER_COUNT),
    "joint_kinematics": (JOINT_COUNT, JOINT_KINEMATICS_WIDTH),
}

# inactive 行仍保留 evaluator 实际动作；active mask 决定它不能进入 teacher/student 监督样本。
_ACTION_LOW: Final[float] = -1.0
_ACTION_HIGH: Final[float] = 1.0
_MASK_LOW: Final[float] = 0.0
_MASK_HIGH: Final[float] = 1.0
_RANGE_EPS: Final[float] = 1.0e-6

# 这是 collector、dataset、student 与 evaluator 共用的完整 actor ABI；字段值不是可选 hint。
_CANONICAL_ACTOR_ABI: Final[dict[str, object]] = {
    "arm": "direct_token",
    "history_encoder": "tcn",
    "history_length": HISTORY_LENGTH,
    "joint_count": JOINT_COUNT,
    "owner_count": OWNER_COUNT,
    "geometry_width": GEOMETRY_TOKEN_WIDTH,
    "actor_contact": "tip-only-binary",
    "phase_clock_enabled": False,
    "joint_kinematics_width": JOINT_KINEMATICS_WIDTH,
}


class FamilyDatasetError(ValueError):
    r"""family trajectory 合同错误。

    该异常继承 ``ValueError``，使调用者可以按输入合同统一捕获；未完成 artifact 的
    读取则单独使用 ``RuntimeError``，因为那代表生命周期尚未闭合而非数组数值错误。
    """


def _decode_scalar(value: Any) -> Any:
    r"""把 h5py attribute 的 bytes/numpy scalar 还原为可 JSON 化的 Python 标量。"""

    if isinstance(value, bytes):  # HDF5 可把 UTF-8 属性返回为 bytes，先恢复文本身份。
        return value.decode("utf-8")
    if isinstance(value, np.generic):  # np.bool_/np.int64 等属性需要回到 Python 标量。
        return value.item()
    return value  # 普通 str/int/float/bool 无需复制转换。


def _json_default(value: Any) -> Any:
    r"""给 metadata 的 JSON 序列化提供有限、可审计的 NumPy/Path 适配。"""

    if isinstance(value, np.generic):  # seed 或版本号常由 NumPy 产生，保留其标量值。
        return value.item()
    if isinstance(value, np.ndarray):  # asset identity 可能以小型 NumPy 数组交付。
        return value.tolist()
    if isinstance(value, Path):  # 路径只记录为文字，不把文件内容嵌入 artifact。
        return str(value)
    if isinstance(value, bytes):  # 哈希/外部摘要有时由 bytes 交付，按 UTF-8 保存。
        return value.decode("utf-8")
    raise TypeError(f"metadata value {type(value).__name__} is not JSON serializable")


def _json_roundtrip(value: Mapping[str, Any]) -> dict[str, Any]:
    r"""复制并验证 metadata，拒绝 NaN、不可序列化对象与隐式 Python repr。"""

    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, default=_json_default)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise FamilyDatasetError(f"metadata must be strict JSON: {error}") from error
    if not isinstance(decoded, dict):  # Mapping 输入经过 round-trip 后必须仍是 object。
        raise FamilyDatasetError("metadata JSON root must be an object")
    return decoded


def _nonempty_string(value: Any, name: str) -> str:
    r"""读取 provenance 文本，空字符串不能承担可追溯身份。"""

    if not isinstance(value, str) or not value.strip():  # 不把 None、数字或空 hash 当身份。
        raise FamilyDatasetError(f"metadata.{name} must be a non-empty string")
    return value


def _validate_metadata(metadata: Mapping[str, Any], asset_count: int) -> dict[str, Any]:
    r"""验证唯一 canonical metadata schema 与完整 actor ABI。

    必需 key 固定为 ``family``、``teacher_checkpoint_sha256``、``cohort_sha256``、
    ``n040_sha256``、``ordered_assets``、``protocol`` 与 ``actor_abi``。没有历史正式数据
    需要迁移，因此不接受大小写变体、旧别名、嵌套 hash 或“猜一个可能的字段”。额外
    runtime identity 可以继续随 metadata 原样保存，但不能替代这些 mandatory keys。

    ``ordered_assets`` 的顺序与 static 第一轴一一对应；actor ABI 的所有维度/模式都必须
    与三组共享 student 的稳定声明逐项一致，而非只检查新增的 15D kinematics 宽度。
    """

    if not isinstance(metadata, Mapping):  # metadata 不是可追溯 object 就无法写入 HDF5。
        raise TypeError("metadata must be a mapping")
    normalized = _json_roundtrip(metadata)  # 先复制，避免 writer 后续修改 caller 的 dict。

    # canonical key 名字先整体核对；旧 checkpoint_sha/asset_id 等字段不会被猜测替代。
    required = {
        "family",
        "teacher_checkpoint_sha256",
        "cohort_sha256",
        "n040_sha256",
        "ordered_assets",
        "protocol",
        "actor_abi",
    }
    missing = sorted(required.difference(normalized))
    if missing:
        raise FamilyDatasetError(f"metadata is missing canonical fields: {', '.join(missing)}")

    # 三个独立 hash 锚点分别绑定 teacher 权重、cohort 成员和冻结 N040。
    family = _nonempty_string(normalized["family"], "family")
    checkpoint_sha = _nonempty_string(normalized["teacher_checkpoint_sha256"], "teacher_checkpoint_sha256")
    cohort_sha = _nonempty_string(normalized["cohort_sha256"], "cohort_sha256")
    n040_sha = _nonempty_string(normalized["n040_sha256"], "n040_sha256")

    # 资产 identity 是有序序列，不能用 set 或无序 dict 消除重复与顺序事实。
    identities = normalized["ordered_assets"]
    if isinstance(identities, (str, bytes)) or not isinstance(identities, Sequence):
        raise FamilyDatasetError("metadata.ordered_assets must be an ordered sequence")
    if len(identities) != asset_count:
        raise FamilyDatasetError(
            f"metadata.ordered_assets length {len(identities)} disagrees with static asset count {asset_count}"
        )

    # protocol 同时记录动作生成 mode 与 seed；mean 允许显式 null，sample 必须有整数 seed。
    protocol = normalized["protocol"]
    if not isinstance(protocol, Mapping):
        raise FamilyDatasetError("metadata.protocol must be a mapping")
    if "action_mode" not in protocol or "action_seed" not in protocol:
        raise FamilyDatasetError("metadata.protocol must contain action_mode and action_seed")
    action_mode = _nonempty_string(protocol["action_mode"], "protocol.action_mode")
    if action_mode not in {"mean", "sample"}:
        raise FamilyDatasetError("metadata.protocol.action_mode must be 'mean' or 'sample'")
    action_seed = protocol["action_seed"]
    if action_mode == "mean":
        if action_seed is not None:
            raise FamilyDatasetError("metadata.protocol.action_seed must be null for mean mode")
    else:
        if isinstance(action_seed, (bool, np.bool_)):
            raise FamilyDatasetError("metadata.protocol.action_seed must not be bool")
        try:
            operator.index(action_seed)
        except (TypeError, ValueError) as error:
            raise FamilyDatasetError("metadata.protocol.action_seed must be an integer for sample mode") from error

    # actor_abi 的完整维度/模式均是固定 ABI，防止仅改 kinematics_width 就误读其它输入。
    actor_abi = normalized["actor_abi"]
    if not isinstance(actor_abi, Mapping):
        raise FamilyDatasetError("metadata.actor_abi must be a mapping")
    if dict(actor_abi) != _CANONICAL_ACTOR_ABI:
        raise FamilyDatasetError(
            f"metadata.actor_abi disagrees with canonical ABI {_CANONICAL_ACTOR_ABI!r}; got {dict(actor_abi)!r}"
        )

    # 只做 canonical 值的轻量规范化；不添加旧别名，额外 provenance 字段原样保留。
    normalized["family"] = family  # family 族标签，例如 leap/allegro，进入训练 provenance。
    normalized["teacher_checkpoint_sha256"] = checkpoint_sha  # teacher checkpoint 内容身份。
    normalized["cohort_sha256"] = cohort_sha  # canonical cohort 内容身份。
    normalized["n040_sha256"] = n040_sha  # 冻结 N040 artifact 内容身份。
    normalized["ordered_assets"] = list(identities)  # 保持 static 资产轴的确定顺序。
    normalized["protocol"] = dict(protocol)  # action_mode/action_seed 的 canonical 二元组。
    normalized["actor_abi"] = dict(actor_abi)  # 完整稳定维度/模式，禁止只写新增宽度。
    return normalized


def _array(value: Any, name: str) -> np.ndarray:
    r"""转为 ndarray 并拒绝 masked array；mask 缺测不能伪装为物理零。"""

    if np.ma.isMaskedArray(value):  # masked value 的缺测语义与 ghost=0 完全不同。
        raise FamilyDatasetError(f"{name} must be an unmasked numpy array")
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise FamilyDatasetError(f"{name} must be array-like") from error
    return array


def _float_array(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    r"""验证精确 shape 与有限浮点数，并统一 artifact 的 float32 存储。

    输入允许 float64 以方便 collector 累计或 CPU 测试，但保存前统一转换为 FP32；这样
    `frames`、动作、token、history、kinematics 与 FK 在同一个 artifact 内没有隐藏 dtype。
    """

    array = _array(value, name)
    if array.shape != shape:  # 不接受 broadcasting，因为它会复制错误的环境/资产轴。
        raise FamilyDatasetError(f"{name} shape {array.shape} != expected {shape}")
    if array.dtype.kind != "f":  # complex/object/int 不属于声明的连续物理浮点事实。
        raise FamilyDatasetError(f"{name} must be floating with shape {shape}, got dtype {array.dtype}")
    if not bool(np.isfinite(array).all()):  # NaN/Inf 会污染 history 与 quality 归约。
        raise FamilyDatasetError(f"{name} must contain finite values")
    return array.astype(np.float32, copy=False)  # 每步只产生当前 batch 的 FP32 视图/副本。


def _mask_array(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    r"""验证 bool/0-1 mask，输出 float32 以与 frames 的 mask ABI 保持一致。"""

    array = _array(value, name)
    if array.shape != shape:  # mask 轴必须与 N 或 owner/joint 轴逐项对应。
        raise FamilyDatasetError(f"{name} shape {array.shape} != expected {shape}")
    if array.dtype.kind not in "biuf":  # 字符串或 complex 不能作为物理有效性 mask。
        raise FamilyDatasetError(f"{name} must be bool/integer/float mask, got dtype {array.dtype}")
    if array.dtype.kind == "f" and not bool(np.isfinite(array).all()):  # 浮点 mask 也必须有限。
        raise FamilyDatasetError(f"{name} must contain finite values")
    if np.any(array < _MASK_LOW) or np.any(array > _MASK_HIGH):  # 不允许 -1/2 等隐式 mask。
        raise FamilyDatasetError(f"{name} values must lie in [0,1]")
    if not bool(np.isin(array, (0, 1)).all()):  # mask 只允许明确的 false/true，不进行阈值化。
        raise FamilyDatasetError(f"{name} values must be exactly 0 or 1")
    return array.astype(np.float32, copy=False)  # HDF5 active/contact 统一为 FP32。


def _bool_mask(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    r"""读取 static/final 的逻辑 mask，并输出 bool 存储，避免把统计旗标当连续量。"""

    return _mask_array(value, name, shape).astype(bool, copy=False)  # 0/1 检查已在上游完成。


def _index_vector(value: Any, name: str, length: int) -> np.ndarray:
    r"""验证非负整数索引向量并转换为 int64，禁止浮点截断与 bool 冒充索引。"""

    array = _array(value, name)
    if array.shape != (length,):  # env 轴必须是一维、确定顺序的 N 条记录。
        raise FamilyDatasetError(f"{name} shape {array.shape} != expected {(length,)}")
    if array.dtype.kind not in "iu" or array.dtype.kind == "b":  # 不能从 1.0 静默转换索引。
        raise FamilyDatasetError(f"{name} must be an integer vector")
    if np.any(array < 0):  # 资产/replica ID 不定义负值。
        raise FamilyDatasetError(f"{name} must contain nonnegative ids")
    return array.astype(np.int64, copy=False)  # HDF5 中使用明确的有符号 64 位索引。


def _scalar_step(value: Any) -> int:
    r"""读取 Python/NumPy integer step，确保 append 的时间轴可精确比较。"""

    if isinstance(value, (bool, np.bool_)):  # True 不应被解释为 step=1。
        raise FamilyDatasetError("step must be an integer, not bool")
    try:
        step = operator.index(value)
    except (TypeError, ValueError) as error:
        raise FamilyDatasetError("step must be an integer") from error
    if step < 0:
        raise FamilyDatasetError("step must be nonnegative")
    return int(step)


def _validate_range(array: np.ndarray, name: str, low: float, high: float) -> None:
    r"""对已验证 finite 的连续量应用物理/动作闭区间，允许极小 FP32 边界误差。"""

    if np.any(array < low - _RANGE_EPS) or np.any(array > high + _RANGE_EPS):
        raise FamilyDatasetError(f"{name} values must lie in [{low},{high}]")


def _exact_zero(values: np.ndarray, name: str) -> None:
    r"""ghost/padding 必须为精确零，防止无效槽通过学习器泄漏非物理信号。"""

    if values.size and np.any(values != 0.0):  # -0.0 等价于零，任意非零幅值均拒绝。
        raise FamilyDatasetError(f"{name} ghost/padding values must be exactly zero")


def _validate_static(
    static: Mapping[str, Any], asset_count: int
) -> dict[str, np.ndarray]:
    r"""验证按资产存储的 geometry evidence，并保留未知扩展字段的第一轴。

    ``joint_kinematics[a,j,:15]`` 的 15 个数由 caller 按
    ``[home_parent_to_joint(3), R_parent_to_joint(9), axis_local(3)]`` 交付，单位分别是
    米、无量纲旋转矩阵元素和无量纲局部轴；本 writer 不从 URDF 推导它们，只保存并检查
    finite/shape，避免把 collector 的运动学实现偷偷重写在数据层。
    """

    if not isinstance(static, Mapping):  # static 需要具名字段，不接受位置 tuple。
        raise TypeError("static must be a mapping of named arrays")
    missing = sorted(set(_REQUIRED_STATIC_SHAPES).difference(static))
    if missing:  # 新 ABI 的 15D kinematics 缺失时，No-Z/Ours 不能共享同一基础几何。
        raise FamilyDatasetError(f"static is missing required fields: {', '.join(missing)}")

    normalized: dict[str, np.ndarray] = {}
    for name, suffix_shape in _REQUIRED_STATIC_SHAPES.items():  # 保持合同字段的显式顺序。
        expected = (asset_count, *suffix_shape)
        if name in {"jnt_valid", "tip_valid", "owner_valid"}:
            normalized[name] = _bool_mask(static[name], f"static.{name}", expected)
        elif name in {"actor_jnt_limits", "joint_kinematics"}:
            normalized[name] = _float_array(static[name], f"static.{name}", expected)
        else:
            # shortest_path/direction 是离散图证据，整数输入有明确语义；float 输入仍统一 FP32。
            array = _array(static[name], f"static.{name}")
            if array.shape != expected or array.dtype.kind not in "biuf":
                raise FamilyDatasetError(
                    f"static.{name} must be numeric with shape {expected}, got {array.shape}/{array.dtype}"
                )
            if array.dtype.kind == "f":
                if not bool(np.isfinite(array).all()):
                    raise FamilyDatasetError(f"static.{name} must contain finite values")
                normalized[name] = array.astype(np.float32, copy=False)
            else:
                normalized[name] = array.copy()
    limits = normalized["actor_jnt_limits"]  # [A,16,2]，关节物理域的 lower/upper 边界。
    if np.any(limits[..., 0] > limits[..., 1]):  # 反向区间会使 ghost/active 语义不可判定。
        raise FamilyDatasetError("static.actor_jnt_limits lower bound must not exceed upper bound")
    expected_owner_valid = np.concatenate(
        (
            np.ones((asset_count, 1), dtype=bool),
            normalized["jnt_valid"],
            normalized["tip_valid"],
        ),
        axis=1,
    )  # owner 轴固定为 PALM + 16 JOINT + 4 TIP，padding mask 由三类实体 mask 唯一决定。
    if not np.array_equal(normalized["owner_valid"], expected_owner_valid):
        raise FamilyDatasetError("static.owner_valid must equal PALM/JOINT/TIP validity concatenation")

    # 额外 static 数组可以由未来 reader 使用，但仍必须沿同一有序资产轴且数值有限。
    for name, value in static.items():
        if name in normalized:  # 已按强合同处理的字段不重复转换。
            continue
        if not isinstance(name, str) or not name or "/" in name:  # 防止任意 HDF5 路径写入。
            raise FamilyDatasetError(f"static field name {name!r} is not a safe HDF5 dataset name")
        array = _array(value, f"static.{name}")
        if array.ndim < 1 or array.shape[0] != asset_count:  # 扩展也不能改变 asset 轴。
            raise FamilyDatasetError(
                f"static.{name} first dimension {array.shape[:1]} != asset count {asset_count}"
            )
        if array.dtype.kind == "f":  # float 扩展同样统一 FP32 并拒绝 NaN。
            if not bool(np.isfinite(array).all()):
                raise FamilyDatasetError(f"static.{name} must contain finite values")
            normalized[name] = array.astype(np.float32, copy=False)
        elif array.dtype.kind in "biu":  # 离散扩展保留整数/bool，不做隐式物理转换。
            normalized[name] = array.copy()
        else:
            raise FamilyDatasetError(f"static.{name} must be numeric")
    return normalized


def _write_compressed(group: h5py.Group, name: str, data: np.ndarray) -> h5py.Dataset:
    r"""以 lzf+shuffle 优先写一次性静态数据，缺少 lzf 时退回 gzip 而不改变数值。"""

    try:
        return group.create_dataset(
            name,
            data=data,
            dtype=data.dtype,
            compression="lzf",
            shuffle=True,
        )
    except (RuntimeError, ValueError):  # 某些 h5py 构建未编译 lzf；gzip 是等价的无损退路。
        return group.create_dataset(
            name,
            data=data,
            dtype=data.dtype,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )


def _create_stream_dataset(
    group: h5py.Group,
    name: str,
    suffix_shape: tuple[int, ...],
    dtype: np.dtype[Any],
) -> h5py.Dataset:
    r"""创建沿首轴 unlimited 的分块数据集；每个 chunk 对应一个时间/sample step。"""

    shape = (0, *suffix_shape)  # 初始没有写入帧，避免预分配全 run 内存。
    maxshape = (None, *suffix_shape)  # append 只扩展时间/sample 轴。
    chunks = (1, *suffix_shape)  # 单步 chunk 使流写与局部读取都保持可控内存。
    try:
        return group.create_dataset(
            name,
            shape=shape,
            maxshape=maxshape,
            dtype=dtype,
            chunks=chunks,
            compression="lzf",
            shuffle=True,
        )
    except (RuntimeError, ValueError):  # 读取环境若无 lzf，保持 chunk 结构只更换压缩编码。
        return group.create_dataset(
            name,
            shape=shape,
            maxshape=maxshape,
            dtype=dtype,
            chunks=chunks,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )


def _resize_and_write(dataset: h5py.Dataset, index: int, value: Any) -> None:
    r"""将一个 batch 写入 unlimited 首轴，避免任何全 run Python 缓存。"""

    dataset.resize((index + 1, *dataset.shape[1:]))  # 只扩展当前 axis，不改变环境/结构轴。
    dataset[index] = value  # h5py 将本 step 从 FP32 batch 写入压缩 chunk。


def _required_group(parent: h5py.Group | h5py.File, name: str, where: str) -> h5py.Group:
    r"""读取必需 HDF5 group，并把 h5py 的 Group/Dataset/Datatype union 收窄。"""

    value = parent.get(name)  # get 比 ``name in parent`` 更容易同时表达存在与类型合同。
    if not isinstance(value, h5py.Group):
        raise RuntimeError(f"{where} lacks required group {name!r}")
    return cast(h5py.Group, value)


def _required_dataset(parent: h5py.Group | h5py.File, name: str, where: str) -> h5py.Dataset:
    r"""读取必需 HDF5 dataset，并拒绝同名 group/datatype 伪造数组。"""

    value = parent.get(name)  # 训练 reader 需要真实 dataset，而非任意 HDF5 link。
    if not isinstance(value, h5py.Dataset):
        raise RuntimeError(f"{where} lacks required dataset {name!r}")
    return cast(h5py.Dataset, value)


def _store_metadata(file: h5py.File, metadata: Mapping[str, Any]) -> None:
    r"""同时维护 metadata group 与 root JSON attr，兼容轻量 reader 与人工审计。"""

    encoded = json.dumps(metadata, ensure_ascii=False, allow_nan=False, default=_json_default)
    file.attrs["metadata_json"] = encoded  # root attr 让 read_family_metadata 不必扫描动态组。
    group = file.require_group("metadata")  # 具名 group 方便 HDF5 浏览器人工查看。
    group.attrs["json"] = encoded  # 同一份 JSON，避免两处 metadata 发生字段漂移。
    for key, value in metadata.items():  # 标量 provenance 另存 attr，便于 h5dump 快速核对。
        if not isinstance(key, str) or not key or "/" in key:
            continue  # 嵌套/非法 key 仍已在 JSON 中保留，不能破坏整个 artifact 写入。
        try:
            if isinstance(value, (str, int, float, bool)):
                group.attrs[key] = value  # 简单 provenance 直接可见，JSON 仍是完整真源。
                if key not in {"artifact_type", "schema_version"}:
                    file.attrs[key] = value  # root attr 便于轻量 h5dump/旧 reader 快速核对。
            else:
                group.attrs[key] = json.dumps(value, ensure_ascii=False, allow_nan=False, default=_json_default)
        except (TypeError, ValueError):
            continue  # 该字段若无法单独 attr，canonical JSON 仍是唯一读取源。


def _validate_env_mapping(
    env_asset_index: Any,
    env_replica_index: Any,
    static_asset_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    r"""验证 environment→asset/replica 路由；资产顺序不因成绩重排。"""

    asset_array = _array(env_asset_index, "env_asset_index")
    if asset_array.ndim != 1 or asset_array.size < 1:
        raise FamilyDatasetError("env_asset_index must be a nonempty one-dimensional vector")
    env_count = int(asset_array.shape[0])
    assets = _index_vector(asset_array, "env_asset_index", env_count)
    replicas = _index_vector(env_replica_index, "env_replica_index", env_count)
    if np.any(assets >= static_asset_count):  # 显式拒绝越界而不是 h5py 写后才失败。
        raise FamilyDatasetError(
            f"env_asset_index contains id >= static asset count {static_asset_count}"
        )
    return assets, replicas


def _validate_initial_history(
    value: Any,
    env_count: int,
    joint_valid_by_env: np.ndarray,
) -> np.ndarray:
    r"""验证完整 H0，并在初始 reset 状态就封闭 canonical ghost joint。"""

    history = _float_array(
        value,
        "initial_history",
        (env_count, HISTORY_LENGTH, JOINT_COUNT, JOINT_FEATURES),
    )
    invalid_joint = np.broadcast_to(
        (~joint_valid_by_env)[:, None, :, None], history.shape
    )  # [N,30,16,5]，把静态 joint mask 广播到完整时间窗。
    _exact_zero(history[invalid_joint], "initial_history")  # ghost 不可由初始状态携入训练。
    return history


def _validate_summary_array(value: Any, name: str, env_count: int) -> np.ndarray:
    r"""验证 final summary 的逐 env 轴；失败统计也要以原始 finite 事实保存。"""

    array = _array(value, f"trajectory_summary.{name}")
    if array.shape != (env_count,):
        raise FamilyDatasetError(f"trajectory_summary.{name} shape {array.shape} != expected {(env_count,)}")
    if array.dtype.kind == "f":
        if not bool(np.isfinite(array).all()):
            raise FamilyDatasetError(f"trajectory_summary.{name} must contain finite values")
        return array.astype(np.float32, copy=False)
    if array.dtype.kind in "biu":
        return array.copy()
    raise FamilyDatasetError(f"trajectory_summary.{name} must be numeric")


def _validated_terminal_flags(value: Any, name: str, env_count: int) -> np.ndarray:
    r"""读取 final 的 termination/terminated 旗标，保留 bool 语义而非浮点近似。"""

    return _bool_mask(value, f"trajectory_summary.{name}", (env_count,))


class FamilyTrajectoryWriter:
    r"""两族 teacher trajectory 的流式 HDF5 writer。

    每个 writer 绑定一个固定环境轴 ``N``、资产轴 ``A`` 与 History30 初态。根级
    ``frames`` 保存 ``[T,N,...]`` 的 dense 时钟；``samples`` 保存每
    ``sample_stride`` 步的 teacher/student 行与 FK；``initial_history``、``static`` 与
    ``final`` 保存不随时间重复的证据。写入采用 ``h5py.File(..., mode='x')``，因此不会
    覆盖既有文件；未调用 ``finalize`` 的 close/context 结果显式标为 incomplete，reader
    会拒绝误读。

    H0 约定：``initial_history[n]`` 已包含 step0 current，即
    ``initial_history[n,-1] == jnt_current[0,n]``。连续 active 时，writer 用上一行
    history 左移并追加当前帧；inactive 行只做 finite/ghost/padding 检查，不参与 history
    比较。active mask 只能从 true 变 false；重新 reset 的新 episode 应建立新 artifact，
    不在一个固定 trajectory 内 re-activate。
    """

    def __init__(
        self,
        path: str | Path,
        metadata: dict[str, Any],
        *,
        steps: int,
        env_asset_index: np.ndarray,
        env_replica_index: np.ndarray,
        static: dict[str, np.ndarray],
        initial_history: np.ndarray,
        sample_stride: int = 4,
    ) -> None:
        r"""创建不覆盖既有路径的 writer，并写入完整静态合同。

        Args:
            path: 新 HDF5 artifact 路径；文件或目录已存在时抛出 ``FileExistsError``。
            metadata: 必须含 canonical ``family``、``teacher_checkpoint_sha256``、
                ``cohort_sha256``、``n040_sha256``、``ordered_assets``、
                ``protocol.action_mode/action_seed`` 与完整 ``actor_abi``。
            steps: nominal 最大策略步数；通常 600（20 Hz × 30 s），允许 final 在全 env
                已终止且 summary 明确证明时提前封存较短 frames。
            env_asset_index: ``int[N]``，每个环境对应 static 的有序资产行。
            env_replica_index: ``int[N]``，episode split 使用 ``replica_id % 4 == 3`` 验证集。
            static: 每项第一轴为 ``A``，必须含 actor limits、valid masks、三张 owner 图与
                ``joint_kinematics[A,16,15]``。
            initial_history: reset 后完整 H0，形状 ``[N,30,16,5]``，oldest-to-latest。
            sample_stride: 稀疏 teacher/action/geometry/FK 采样间隔，必须为正整数。

        Raises:
            FileExistsError: path 已存在，避免覆盖既有实验事实。
            FamilyDatasetError: 任一静态、身份或 ABI 合同不成立。
        """

        # 先在内存验证输入，避免坏 metadata 先创建一个看似 artifact 的空壳。
        try:
            expected_steps = operator.index(steps)
        except (TypeError, ValueError) as error:
            raise FamilyDatasetError("steps must be an integer") from error
        if isinstance(steps, (bool, np.bool_)) or expected_steps < 1:
            raise FamilyDatasetError("steps must be a positive integer")
        try:
            stride = operator.index(sample_stride)
        except (TypeError, ValueError) as error:
            raise FamilyDatasetError("sample_stride must be an integer") from error
        if isinstance(sample_stride, (bool, np.bool_)) or stride < 1:
            raise FamilyDatasetError("sample_stride must be a positive integer")

        # static 第一轴先由必需 actor limits 推导，其他数组必须精确匹配这个 A。
        raw_static = _array(static.get("actor_jnt_limits"), "static.actor_jnt_limits") if isinstance(static, Mapping) and "actor_jnt_limits" in static else None
        if raw_static is None or raw_static.ndim != 3 or raw_static.shape[1:] != (JOINT_COUNT, 2):
            raise FamilyDatasetError(
                "static.actor_jnt_limits must have shape [A,16,2] to define asset count"
            )
        asset_count = int(raw_static.shape[0])
        if asset_count < 1:
            raise FamilyDatasetError("static asset axis A must be positive")
        assets, replicas = _validate_env_mapping(env_asset_index, env_replica_index, asset_count)
        env_count = int(assets.size)
        normalized_static = _validate_static(static, asset_count)
        joint_valid_by_env = normalized_static["jnt_valid"][assets]  # [N,16] static ghost mask。
        owner_valid_by_env = normalized_static["owner_valid"][assets]  # [N,21] static owner padding mask。
        history = _validate_initial_history(initial_history, env_count, joint_valid_by_env)
        normalized_metadata = _validate_metadata(metadata, asset_count)

        # Path check + HDF5 exclusive create 共同封闭 TOCTOU，保证 writer 不覆盖既有事实。
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(f"family trajectory output already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            handle = h5py.File(destination, mode="x")
        except FileExistsError:
            raise
        except OSError as error:
            raise FamilyDatasetError(f"cannot create family trajectory output {destination}: {error}") from error

        # 以下成员在构造器异常回滚与 append/finalize 生命周期中共同维护。
        self.path = destination  # 外部审计可读取实际 resolve 前的 artifact 路径。
        self._file = handle
        self._closed = False
        self._finalized = False
        self._expected_steps = int(expected_steps)
        self._sample_stride = int(stride)
        self._env_count = env_count
        self._asset_count = asset_count
        self._env_asset_index = assets.copy()
        self._env_replica_index = replicas.copy()
        self._static = {name: np.array(value, copy=True) for name, value in normalized_static.items()}
        self._initial_history = np.array(history, copy=True)
        self._joint_valid_by_env = np.array(joint_valid_by_env, copy=True)
        self._owner_valid_by_env = np.array(owner_valid_by_env, copy=True)
        self._history_state = np.array(history, copy=True)
        self._active_previous = np.ones(env_count, dtype=bool)  # 首行允许任意 true/false 初态。
        self._have_previous_active = False
        self._written_steps = 0
        self._sample_count = 0
        self._metadata = dict(normalized_metadata)
        self._metadata.update(
            {
                "completed": False,
                "incomplete": True,
                "expected_steps": self._expected_steps,
                "nominal_steps": self._expected_steps,
                "steps": 0,
                "recorded_steps": 0,
                "sample_count": 0,
                "sample_stride": self._sample_stride,
            }
        )

        try:
            self._file.attrs["artifact_type"] = ARTIFACT_TYPE  # parser 首先核对格式身份。
            self._file.attrs["schema_version"] = SCHEMA_VERSION  # 版本漂移必须显式失败。
            self._file.attrs["completed"] = False  # 未 finalize 前永远不可被 reader 接受。
            self._file.attrs["incomplete"] = True  # 异常/close 的状态标记，不隐藏半成品。
            self._file.attrs["expected_steps"] = self._expected_steps  # nominal horizon。
            self._file.attrs["steps"] = 0  # 当前已落盘真实步数，finalize 后才等于完整 nominal horizon。
            self._file.attrs["sample_stride"] = self._sample_stride  # 稀疏 sample 的固定步长。
            self._file.attrs["env_count"] = self._env_count  # N environment rows。
            self._file.attrs["asset_count"] = self._asset_count  # A ordered static rows。
            self._file.attrs["history_semantics"] = "H0_includes_current0_oldest_to_latest"  # H0 ABI。
            _store_metadata(self._file, self._metadata)  # metadata JSON 在动态写入前即存在。

            # 路由索引是小型、不可丢失的静态轴，单独写出供 split/reader 使用。
            _write_compressed(self._file, "env_asset_index", self._env_asset_index)
            _write_compressed(self._file, "env_replica_index", self._env_replica_index)
            _write_compressed(self._file, "initial_history", self._initial_history)

            static_group = self._file.create_group("static")  # 统一保存按资产 geometry evidence。
            for name, value in self._static.items():  # 包括未来扩展，但保留同一 A 轴。
                _write_compressed(static_group, name, value)

            frames = self._file.create_group("frames")  # dense 时钟轴 [T,N,...]。
            self._frames_jnt_current = _create_stream_dataset(
                frames, "jnt_current", (self._env_count, JOINT_COUNT, JOINT_FEATURES), np.dtype("float32")
            )
            self._frames_owner_contact = _create_stream_dataset(
                frames, "owner_contact", (self._env_count, OWNER_COUNT, 1), np.dtype("float32")
            )
            self._frames_active = _create_stream_dataset(
                frames, "active", (self._env_count,), np.dtype("float32")
            )

            samples = self._file.create_group("samples")  # 稀疏监督/geometry/FK 轴 [S,N,...]。
            self._samples_step_index = _create_stream_dataset(
                samples, "step_index", (), np.dtype("int64")
            )
            self._samples_teacher_mean = _create_stream_dataset(
                samples, "teacher_mean", (self._env_count, JOINT_COUNT), np.dtype("float32")
            )
            self._samples_behavior_action = _create_stream_dataset(
                samples, "behavior_action", (self._env_count, JOINT_COUNT), np.dtype("float32")
            )
            self._samples_geometry_tokens = _create_stream_dataset(
                samples,
                "geometry_tokens",
                (self._env_count, OWNER_COUNT, GEOMETRY_TOKEN_WIDTH),
                np.dtype("float32"),
            )
            self._samples_joint_origin_fk = _create_stream_dataset(
                samples,
                "joint_origin_fk",
                (self._env_count, JOINT_COUNT, 3),
                np.dtype("float32"),
            )
            self._file.flush()  # constructor完成后外部可以看到明确 incomplete marker。
        except BaseException:
            # 构造中任何 HDF5/字段错误都保留半成品并标 incomplete，随后重新抛出原错误。
            try:
                self._file.attrs["completed"] = False
                self._file.attrs["incomplete"] = True
                self._file.flush()
                self._file.close()
            finally:
                self._closed = True
            raise

    def _ensure_open(self) -> None:
        r"""拒绝在 close/finalize 后追加，防止一个路径承载两个生命周期。"""

        if self._closed:
            raise RuntimeError("family trajectory writer is closed")
        if self._finalized:
            raise RuntimeError("family trajectory writer is already finalized")

    def _validate_step_arrays(
        self,
        *,
        jnt_current: Any,
        owner_contact: Any,
        teacher_mean: Any,
        behavior_action: Any,
        geometry_tokens: Any,
        active: Any,
        history: Any,
        joint_origin_fk: Any,
        sample: bool,
    ) -> tuple[np.ndarray, ...]:
        r"""验证一行动态数组，并执行动作/ghost/padding/mask/FK 的语义边界。"""

        current = _float_array(
            jnt_current,
            "jnt_current",
            (self._env_count, JOINT_COUNT, JOINT_FEATURES),
        )
        owner = _float_array(
            owner_contact,
            "owner_contact",
            (self._env_count, OWNER_COUNT, 1),
        )
        teacher = _float_array(
            teacher_mean,
            "teacher_mean",
            (self._env_count, JOINT_COUNT),
        )
        behavior = _float_array(
            behavior_action,
            "behavior_action",
            (self._env_count, JOINT_COUNT),
        )
        tokens = _float_array(
            geometry_tokens,
            "geometry_tokens",
            (self._env_count, OWNER_COUNT, GEOMETRY_TOKEN_WIDTH),
        )
        active_float = _mask_array(active, "active", (self._env_count,))
        history_array = _float_array(
            history,
            "history",
            (self._env_count, HISTORY_LENGTH, JOINT_COUNT, JOINT_FEATURES),
        )

        # teacher mean 与实际执行 action 都是 canonical wrapper action，物理范围为 [-1,1]。
        _validate_range(teacher, "teacher_mean", _ACTION_LOW, _ACTION_HIGH)
        _validate_range(behavior, "behavior_action", _ACTION_LOW, _ACTION_HIGH)
        _validate_range(owner, "owner_contact", _MASK_LOW, _MASK_HIGH)
        _validate_range(tokens, "geometry_tokens", -np.inf, np.inf)  # finite 已检查，token 无硬幅值。

        active_bool = active_float.astype(bool, copy=False)  # 后续历史状态只使用明确二值 mask。
        if self._have_previous_active and np.any(~self._active_previous & active_bool):
            raise FamilyDatasetError("active mask may only transition true to false; reactivation is forbidden")

        # canonical ghost joint/owner 行必须 exact zero；valid row 可以承载真实状态/接触。
        _exact_zero(current[~self._joint_valid_by_env], "jnt_current")
        invalid_history = np.broadcast_to(
            (~self._joint_valid_by_env)[:, None, :, None], history_array.shape
        )  # [N,30,16,5]，ghost joint 在每个 lag 都必须为零。
        _exact_zero(history_array[invalid_history], "history")
        _exact_zero(teacher[~self._joint_valid_by_env], "teacher_mean")
        _exact_zero(behavior[~self._joint_valid_by_env], "behavior_action")
        _exact_zero(owner[~self._owner_valid_by_env], "owner_contact")
        _exact_zero(tokens[~self._owner_valid_by_env], "geometry_tokens")

        # joint_origin_fk 是 sample-only 的公共米制运动学真值；sample 行缺它即无法复现 FK。
        if sample:
            if joint_origin_fk is None:
                raise FamilyDatasetError("joint_origin_fk is required on every sampled step")
            fk = _float_array(
                joint_origin_fk,
                "joint_origin_fk",
                (self._env_count, JOINT_COUNT, 3),
            )
            _exact_zero(fk[~self._joint_valid_by_env], "joint_origin_fk")
        else:
            if joint_origin_fk is None:
                fk = np.empty((0,), dtype=np.float32)  # 非 sample 行显式省略，避免伪造 FK。
            else:
                fk = _float_array(
                    joint_origin_fk,
                    "joint_origin_fk",
                    (self._env_count, JOINT_COUNT, 3),
                )
                _exact_zero(fk[~self._joint_valid_by_env], "joint_origin_fk")
        return current, owner, teacher, behavior, tokens, active_float, history_array, fk

    def append(
        self,
        step: int,
        *,
        jnt_current: np.ndarray,
        owner_contact: np.ndarray,
        teacher_mean: np.ndarray,
        behavior_action: np.ndarray,
        geometry_tokens: np.ndarray,
        active: np.ndarray,
        history: np.ndarray,
        joint_origin_fk: np.ndarray | None = None,
    ) -> None:
        r"""追加一个连续策略步，并在 sample stride 行保存 teacher/action/geometry/FK。

        ``step`` 必须从 0 开始严格递增 1；重复或遗漏会直接失败，避免训练 reader 将
        两段不连续物理时间误当作一个 episode。保存数组形状固定为
        ``jnt_current[N,16,5]``、``owner_contact[N,21,1]``、动作 ``[N,16]``、
        ``geometry_tokens[N,21,128]``、``active[N]`` 与 ``history[N,30,16,5]``。
        所有浮点值写成 FP32；采样行还要求 ``joint_origin_fk[N,16,3]``（米制 FK）。
        """

        self._ensure_open()  # lifecycle 边界先于数组转换，避免 closed handle 的隐式写入。
        index = _scalar_step(step)  # NumPy scalar step 也必须服从同一整数时钟。
        if index != self._written_steps:  # 0,1,2,... 的严格关系同时排除重复与漏步。
            raise FamilyDatasetError(
                f"step must be contiguous: expected {self._written_steps}, got {index}"
            )
        sample = index % self._sample_stride == 0  # sample 轴不另起时钟，直接来自 dense step。
        current, owner, teacher, behavior, tokens, active_float, history_array, fk = self._validate_step_arrays(
            jnt_current=jnt_current,
            owner_contact=owner_contact,
            teacher_mean=teacher_mean,
            behavior_action=behavior_action,
            geometry_tokens=geometry_tokens,
            active=active,
            history=history,
            joint_origin_fk=joint_origin_fk,
            sample=sample,
        )
        active_bool = active_float.astype(bool, copy=False)  # bool 仅用于 continuity，不写成 bool mask。

        # active 行的 expected history：step0 或 terminal 后首个行必须重新等于 H0；本 contract
        # 禁止 reactivation，因此后者只在初始 inactive rows 的理论路径出现。
        active_indices = np.flatnonzero(active_bool)  # 只为 active 行构造比较，inactive history 不参与。
        if active_indices.size:
            expected = np.empty(
                (active_indices.size, HISTORY_LENGTH, JOINT_COUNT, JOINT_FEATURES),
                dtype=np.float32,
            )
            for local_index, env_index in enumerate(active_indices.tolist()):
                if not self._have_previous_active or not self._active_previous[env_index]:
                    expected[local_index] = self._initial_history[env_index]  # H0 已含 current0。
                    if not np.allclose(
                        current[env_index], self._initial_history[env_index, -1], rtol=1.0e-5, atol=2.0e-6
                    ):
                        raise FamilyDatasetError(
                            f"jnt_current[{env_index}] does not match initial_history H0 latest frame"
                        )
                else:
                    expected[local_index, :-1] = self._history_state[env_index, 1:]  # H_{t-1} 去 oldest。
                    expected[local_index, -1] = current[env_index]  # 追加 q_t，保持 latest=current。
            actual = history_array[active_indices]  # 只核对 active；inactive reset/padding 不做比较。
            difference = np.abs(actual - expected)  # 误差用于清楚报告第一个 shape/数值合同失败。
            if difference.size and not np.allclose(actual, expected, rtol=1.0e-5, atol=2.0e-6):
                raise FamilyDatasetError(
                    "history continuity mismatch on active rows: "
                    f"active_count={active_indices.size}, max_abs_error={float(difference.max()):.6g}"
                )
            self._history_state[active_indices] = expected  # 用理论重建值累积，避免 caller 漂移。

        # inactive row 的状态不再拿来跨 reset 比较；同时清掉 previous-active 标记，后续 reactivation
        # 会在上游 mask 检查处 fail，而不会被误拼为 continuity。
        inactive_indices = np.flatnonzero(~active_bool)  # 每步都允许更多环境完成/提前终止。
        if inactive_indices.size:
            self._history_state[inactive_indices] = self._initial_history[inactive_indices]
        self._active_previous = active_bool.copy()  # true→false 合同在下一行精确核对。
        self._have_previous_active = True

        # dense frames 只扩展一行；没有全 run arrays 或 Python history list。
        _resize_and_write(self._frames_jnt_current, self._written_steps, current)
        _resize_and_write(self._frames_owner_contact, self._written_steps, owner)
        _resize_and_write(self._frames_active, self._written_steps, active_float)

        # sparse samples 保留 teacher/action/token/FK，step_index 明确连接回 dense 时钟。
        if sample:
            sample_index = self._sample_count
            _resize_and_write(self._samples_step_index, sample_index, np.int64(index))
            _resize_and_write(self._samples_teacher_mean, sample_index, teacher)
            _resize_and_write(self._samples_behavior_action, sample_index, behavior)
            _resize_and_write(self._samples_geometry_tokens, sample_index, tokens)
            _resize_and_write(self._samples_joint_origin_fk, sample_index, fk)
            self._sample_count += 1

        self._written_steps += 1
        self._file.attrs["recorded_steps"] = self._written_steps  # crash 后也能审计已写帧数。
        self._file.attrs["steps"] = self._written_steps  # steps 别名只表示真实 dense frames 数。
        self._file.attrs["sample_count"] = self._sample_count  # sparse 轴计数与 step_index 对齐。
        self._metadata["recorded_steps"] = self._written_steps
        self._metadata["steps"] = self._written_steps
        self._metadata["sample_count"] = self._sample_count
        _store_metadata(self._file, self._metadata)
        self._file.flush()  # 每步落盘，保持长采集异常时的 incomplete 可审计性。

    def finalize(self, trajectory_summary: dict[str, np.ndarray]) -> None:
        r"""封存 final summary 与 quality mask，标记 artifact completed。

        必需 summary 字段为 ``net_turns``、``path_turns``、``duration_s``、
        ``termination_drop`` 与 ``termination_axis``，每项形状 ``[N]``；其余逐 env 数组
        原样保留。名义 horizon 未写满时，只接受 summary 中全 env ``terminated=True`` 的
        明确终止证明；这时文件仍可完成封存，但 ``duration_s < 30`` 的轨迹不会通过
        quality gate。若 summary 提供 ``policy_step_count``，quality 还要求精确 600 步，
        以防 FP32 duration 偶然接近 30 s 而实际少采动作。
        """

        self._ensure_open()  # finalize 只能发生一次且必须在 writer open 状态。
        if not isinstance(trajectory_summary, Mapping):
            raise TypeError("trajectory_summary must be a mapping")
        required = ("net_turns", "path_turns", "duration_s", "termination_drop", "termination_axis")
        missing = [name for name in required if name not in trajectory_summary]
        if missing:
            raise FamilyDatasetError(f"trajectory_summary is missing fields: {', '.join(missing)}")

        # 先验证所有 summary，再写 final，确保失败不会留下“半个完成”的根标志。
        summary: dict[str, np.ndarray] = {
            name: _validate_summary_array(value, name, self._env_count)
            for name, value in trajectory_summary.items()
        }
        drop = _validated_terminal_flags(trajectory_summary["termination_drop"], "termination_drop", self._env_count)
        axis = _validated_terminal_flags(trajectory_summary["termination_axis"], "termination_axis", self._env_count)
        if self._written_steps < self._expected_steps:
            terminated_value = trajectory_summary.get("terminated")
            if terminated_value is None:
                raise FamilyDatasetError(
                    "early finalization requires trajectory_summary.terminated proof for every env"
                )
            terminated = _validated_terminal_flags(terminated_value, "terminated", self._env_count)
            if not bool(terminated.all()):
                raise FamilyDatasetError("early finalization requires terminated=True for every env")

        # 默认 quality 是数据准入门，不改变 raw final 分母：失败行仍在 final 中完整保存。
        quality = quality_episode_mask(
            summary["net_turns"],
            summary["path_turns"],
            summary["duration_s"],
            drop,
            axis,
        )
        if self._written_steps < self._expected_steps:
            quality[:] = False  # 物理 frames 未达 nominal horizon，短轨迹即使 duration 被误报也不能合格。
        if "policy_step_count" in summary:
            policy_steps = summary["policy_step_count"]
            if policy_steps.dtype.kind not in "iu" or np.any(policy_steps < 0):
                raise FamilyDatasetError("trajectory_summary.policy_step_count must be nonnegative integer")
            quality &= policy_steps == QUALITY_HORIZON_STEPS  # 20Hz×30s 的显式动作时钟门。
            quality &= self._written_steps >= QUALITY_HORIZON_STEPS  # dense frames 也必须实际覆盖 600 步。

        final_group = self._file.create_group("final")  # 保留原始失败与成功统计，不按 mask 删除。
        for name, value in summary.items():  # dtype 已在 _validate_summary_array 中封闭。
            _write_compressed(final_group, name, value)
        _write_compressed(final_group, "quality_episode_mask", quality.astype(bool))

        # 为每个静态资产报告是否存在合格 episode；零合格只形成可查询事实，不静默补样本。
        qualified_assets: list[int] = []
        zero_assets: list[int] = []
        for asset_index in range(self._asset_count):
            has_qualified = bool(quality[self._env_asset_index == asset_index].any())
            (qualified_assets if has_qualified else zero_assets).append(asset_index)

        # 写入 completion metadata 与 root attrs；先写完 final，再置 completed=True。
        self._metadata.update(
            {
                "completed": True,
                "incomplete": False,
                "steps": self._written_steps,
                "recorded_steps": self._written_steps,
                "sample_count": self._sample_count,
                "qualified_episode_count": int(quality.sum()),
                "qualified_asset_indices": qualified_assets,
                "zero_qualified_asset_indices": zero_assets,
                "has_qualified_episode": bool(quality.any()),
            }
        )
        self._file.attrs["completed"] = True  # reader 的唯一完成信号，最后一步才置 true。
        self._file.attrs["incomplete"] = False  # 与 completed 成对写出，拒绝歧义状态。
        self._file.attrs["recorded_steps"] = self._written_steps
        self._file.attrs["steps"] = self._written_steps
        self._file.attrs["sample_count"] = self._sample_count
        self._file.attrs["qualified_episode_count"] = int(quality.sum())
        self._file.attrs["has_qualified_episode"] = bool(quality.any())
        _store_metadata(self._file, self._metadata)
        self._file.flush()  # finalize 返回后 caller 可 close，reader 看到完整 summary/quality。
        self._finalized = True

    def close(self) -> None:
        r"""关闭 HDF5 handle；未 finalize 的文件显式标为 incomplete 且不可读。"""

        if self._closed:  # context/异常清理可以安全重复调用。
            return
        if not self._finalized:  # 正常 close 也不能把未封存数据伪装成 completed。
            self._metadata.update(
                {
                    "completed": False,
                    "incomplete": True,
                    "recorded_steps": self._written_steps,
                    "steps": self._written_steps,
                    "sample_count": self._sample_count,
                }
            )
            self._file.attrs["completed"] = False
            self._file.attrs["incomplete"] = True
            self._file.attrs["recorded_steps"] = self._written_steps
            self._file.attrs["steps"] = self._written_steps
            self._file.attrs["sample_count"] = self._sample_count
            _store_metadata(self._file, self._metadata)
        self._file.flush()  # flush marker before close，便于外部审计文件状态。
        self._file.close()
        self._closed = True

    def __enter__(self) -> FamilyTrajectoryWriter:
        r"""进入 context manager，返回同一 writer 以保持 append/finalize 链清晰。"""

        self._ensure_open()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: Any) -> bool:
        r"""无论异常与否关闭文件；异常路径保留 incomplete 而不吞掉原异常。"""

        self.close()
        return False

    def __del__(self) -> None:
        r"""在调用者忘记 context/close 时尽力封闭 incomplete 标记，不让 handle 泄漏。"""

        if getattr(self, "_closed", True):  # 构造器中途失败时可能尚未建立完整成员。
            return
        with suppress(Exception):
            self.close()  # GC 路径不抛出异常，磁盘上的 completion 仍保持 fail-closed。


def read_family_metadata(path: str | Path) -> dict[str, Any]:
    r"""读取并验证一个已完成 family trajectory 的 metadata，不加载全量 run。

    返回值保留 canonical metadata，并附加 ``artifact_type``、``schema_version``、完成状态
    与轴计数。任何 ``completed=False``、``incomplete=True``、schema 漂移、动态轴计数不一致
    或缺 final/quality 的文件都抛出 ``RuntimeError``/``ValueError``，这样训练器不会误读
    collector 崩溃留下的半成品。
    """

    destination = Path(path)
    if not destination.is_file():
        raise FileNotFoundError(destination)
    with h5py.File(destination, mode="r") as handle:
        artifact_type = _decode_scalar(handle.attrs.get("artifact_type"))
        schema = _decode_scalar(handle.attrs.get("schema_version"))
        if artifact_type != ARTIFACT_TYPE:
            raise FamilyDatasetError(f"unsupported artifact_type {artifact_type!r}")
        if schema != SCHEMA_VERSION:
            raise FamilyDatasetError(f"unsupported family trajectory schema_version {schema!r}")
        completed = bool(_decode_scalar(handle.attrs.get("completed", False)))
        incomplete = bool(_decode_scalar(handle.attrs.get("incomplete", True)))
        if not completed or incomplete:
            raise RuntimeError(f"family trajectory {destination} is incomplete and cannot be read")

        # complete marker 必须与物理数据组共同存在；缺组的手工伪造文件不可进入训练 reader。
        metadata_group = _required_group(handle, "metadata", "completed family trajectory")
        static_group = _required_group(handle, "static", "completed family trajectory")
        frames = _required_group(handle, "frames", "completed family trajectory")
        samples = _required_group(handle, "samples", "completed family trajectory")
        final_group = _required_group(handle, "final", "completed family trajectory")
        initial_history_dataset = _required_dataset(handle, "initial_history", "completed family trajectory")
        quality_dataset = _required_dataset(final_group, "quality_episode_mask", "completed family trajectory final")

        # 新 ABI 的必需 static 字段也要在 reader 侧存在，避免人工伪造“完成”标志后缺 geometry。
        asset_count = int(_decode_scalar(handle.attrs.get("asset_count", -1)))
        if asset_count < 1:
            raise RuntimeError("completed family trajectory has invalid asset_count")
        for static_name in _REQUIRED_STATIC_SHAPES:
            static_dataset = _required_dataset(static_group, static_name, "completed family trajectory static")
            expected_static_shape = (asset_count, *_REQUIRED_STATIC_SHAPES[static_name])
            if static_dataset.shape != expected_static_shape:
                raise RuntimeError(f"static/{static_name} shape {static_dataset.shape} != {expected_static_shape}")

        raw_json = handle.attrs.get("metadata_json")
        if raw_json is None:
            raw_json = metadata_group.attrs.get("json")
        if raw_json is None:
            raise RuntimeError("completed family trajectory lacks metadata JSON")
        try:
            metadata = json.loads(_decode_scalar(raw_json))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError("family trajectory metadata JSON is invalid") from error
        if not isinstance(metadata, dict) or metadata.get("completed") is not True:
            raise RuntimeError("family trajectory metadata does not prove completed=True")

        recorded_steps = int(_decode_scalar(handle.attrs.get("recorded_steps", -1)))
        sample_count = int(_decode_scalar(handle.attrs.get("sample_count", -1)))
        expected_steps = int(_decode_scalar(handle.attrs.get("expected_steps", -1)))
        sample_stride = int(_decode_scalar(handle.attrs.get("sample_stride", -1)))
        frame_jnt_current = _required_dataset(frames, "jnt_current", "completed family trajectory frames")
        _required_dataset(frames, "owner_contact", "completed family trajectory frames")
        frame_active = _required_dataset(frames, "active", "completed family trajectory frames")
        sample_step_index = _required_dataset(samples, "step_index", "completed family trajectory samples")
        sample_teacher = _required_dataset(samples, "teacher_mean", "completed family trajectory samples")
        sample_behavior = _required_dataset(samples, "behavior_action", "completed family trajectory samples")
        sample_tokens = _required_dataset(samples, "geometry_tokens", "completed family trajectory samples")
        sample_fk = _required_dataset(samples, "joint_origin_fk", "completed family trajectory samples")
        env_assets = _required_dataset(handle, "env_asset_index", "completed family trajectory")
        env_replicas = _required_dataset(handle, "env_replica_index", "completed family trajectory")
        if recorded_steps != int(frame_jnt_current.shape[0]):
            raise RuntimeError("metadata recorded_steps disagrees with frames/jnt_current length")
        if sample_count != int(sample_step_index.shape[0]):
            raise RuntimeError("metadata sample_count disagrees with samples/step_index length")
        if expected_steps < 1 or sample_stride < 1 or recorded_steps < 1 or recorded_steps > expected_steps:
            raise RuntimeError("completed family trajectory has invalid lifecycle counters")
        if env_assets.shape != (frame_active.shape[1],) or env_replicas.shape != env_assets.shape:
            raise RuntimeError("completed family trajectory environment index shape is inconsistent")
        env_count = int(frame_active.shape[1])
        if frame_jnt_current.shape != (recorded_steps, env_count, JOINT_COUNT, JOINT_FEATURES):
            raise RuntimeError("completed family trajectory jnt_current shape is inconsistent")
        if _required_dataset(frames, "owner_contact", "completed family trajectory frames").shape != (
            recorded_steps,
            env_count,
            OWNER_COUNT,
            1,
        ) or frame_active.shape != (recorded_steps, env_count):
            raise RuntimeError("completed family trajectory contact/active shape is inconsistent")
        expected_sample_count = (recorded_steps - 1) // sample_stride + 1
        if sample_count != expected_sample_count:
            raise RuntimeError("metadata sample_count does not match sample_stride and recorded_steps")
        if sample_teacher.shape != (sample_count, env_count, JOINT_COUNT) or sample_behavior.shape != sample_teacher.shape:
            raise RuntimeError("completed family trajectory action sample shape is inconsistent")
        if sample_tokens.shape != (sample_count, env_count, OWNER_COUNT, GEOMETRY_TOKEN_WIDTH):
            raise RuntimeError("completed family trajectory geometry sample shape is inconsistent")
        if sample_fk.shape != (sample_count, env_count, JOINT_COUNT, 3):
            raise RuntimeError("completed family trajectory FK sample shape is inconsistent")
        if initial_history_dataset.shape != (env_count, HISTORY_LENGTH, JOINT_COUNT, JOINT_FEATURES):
            raise RuntimeError("completed family trajectory initial_history shape is inconsistent")
        if initial_history_dataset.dtype != np.dtype("float32"):
            raise RuntimeError("completed family trajectory initial_history must be float32")
        if sample_step_index.dtype.kind not in "iu" or not np.array_equal(
            np.asarray(sample_step_index, dtype=np.int64),
            np.arange(0, recorded_steps, sample_stride, dtype=np.int64),
        ):
            raise RuntimeError("samples/step_index is not the exact stride subsequence of dense steps")
        if quality_dataset.shape != (env_count,):
            raise RuntimeError("final/quality_episode_mask shape is inconsistent")

        # 平铺 canonical metadata 供 reader 直接取 family/hash/protocol/actor ABI。
        result = dict(metadata)
        result.update(
            {
                "artifact_type": artifact_type,
                "schema_version": schema,
                "completed": completed,
                "incomplete": incomplete,
                "expected_steps": expected_steps,
                "nominal_steps": expected_steps,
                "steps": recorded_steps,
                "recorded_steps": recorded_steps,
                "sample_stride": sample_stride,
                "sample_count": sample_count,
                "env_count": int(_decode_scalar(handle.attrs.get("env_count", frame_active.shape[1]))),
                "asset_count": asset_count,
                "path": str(destination),
                "qualified_episode_count": int(
                    _decode_scalar(handle.attrs.get("qualified_episode_count", quality_dataset.shape[0]))
                ),
                "has_qualified_episode": bool(_decode_scalar(handle.attrs.get("has_qualified_episode", False))),
            }
        )
        return result


def reconstruct_history(
    initial_history: np.ndarray,
    frames: np.ndarray,
    step_indices: np.ndarray,
    env_indices: np.ndarray,
) -> np.ndarray:
    r"""从 H0 与 dense ``jnt_current`` 重建 batch History30，严格禁止未来泄漏。

    Args:
        initial_history: ``[N,30,16,5]`` 的 H0，已包含 step0 current。
        frames: ``[T,N,16,5]`` dense current 帧；``frames[0]`` 对应 current0，
            ``frames[t]`` 对应 current_t。
        step_indices: ``[B]`` 目标时间索引，必须满足 ``0 <= t < T``。
        env_indices: ``[B]`` 每个目标对应的环境索引。

    Returns:
        ``np.ndarray``，形状 ``[B,30,16,5]``。对于 ``t<30``，输出由 H0 的后缀和
        ``frames[1:t+1]`` 组成；对于 ``t>=30``，只使用 ``frames[t-29:t+1]``。因此修改
        目标之后的未来帧不会改变该历史，满足 causal actor 的 no-future-leak 合同。
    """

    raw_initial = _array(initial_history, "initial_history")
    if raw_initial.ndim < 1:
        raise FamilyDatasetError("initial_history must have an environment axis")
    init = _float_array(
        raw_initial,
        "initial_history",
        (int(raw_initial.shape[0]), HISTORY_LENGTH, JOINT_COUNT, JOINT_FEATURES),
    )
    frame_array = _array(frames, "frames")
    if frame_array.ndim != 4 or frame_array.shape[1:] != (init.shape[0], JOINT_COUNT, JOINT_FEATURES):
        raise FamilyDatasetError(
            f"frames shape {frame_array.shape} != expected [T,{init.shape[0]},{JOINT_COUNT},{JOINT_FEATURES}]"
        )
    if frame_array.shape[0] < 1:
        raise FamilyDatasetError("frames must contain at least current0")
    if frame_array.dtype.kind != "f" or not bool(np.isfinite(frame_array).all()):
        raise FamilyDatasetError("frames must be finite floating values")
    frame_array = frame_array.astype(np.float32, copy=False)

    step_array = _array(step_indices, "step_indices")
    env_array = _array(env_indices, "env_indices")
    if step_array.ndim == 0:
        step_array = step_array.reshape(1)  # scalar convenience 不改变返回 batch 维度。
    if env_array.ndim == 0:
        env_array = env_array.reshape(1)
    if step_array.ndim != 1 or env_array.ndim != 1 or step_array.shape != env_array.shape:
        raise FamilyDatasetError("step_indices and env_indices must be one-dimensional vectors of equal length")
    if step_array.size < 1:
        raise FamilyDatasetError("history reconstruction batch must be nonempty")
    if step_array.dtype.kind not in "iu" or env_array.dtype.kind not in "iu":
        raise FamilyDatasetError("step_indices and env_indices must be integer vectors")
    if np.any(step_array < 0) or np.any(step_array >= frame_array.shape[0]):
        raise FamilyDatasetError("step_indices contain a value outside dense frame range")
    if np.any(env_array < 0) or np.any(env_array >= init.shape[0]):
        raise FamilyDatasetError("env_indices contain a value outside initial_history environment range")

    output = np.empty((step_array.size, HISTORY_LENGTH, JOINT_COUNT, JOINT_FEATURES), dtype=np.float32)
    for batch_index, (step_value, env_value) in enumerate(zip(step_array.tolist(), env_array.tolist(), strict=True)):
        step = int(step_value)
        env = int(env_value)
        if step == 0:  # H0 已含 current0，不再重复追加 frames[0]。
            output[batch_index] = init[env]
            continue
        first_new = max(1, step - HISTORY_LENGTH + 1)  # 只取 current1..current_t 的最后29/30帧。
        new_frames = frame_array[first_new : step + 1, env]  # 上界排除 step+1，绝不读未来。
        prefix_count = HISTORY_LENGTH - int(new_frames.shape[0])
        if prefix_count > 0:  # t<30 时用 H0 的 latest 后缀补足固定窗口长度。
            prefix = init[env, -prefix_count:]
            output[batch_index] = np.concatenate((prefix, new_frames), axis=0)
        else:
            output[batch_index] = new_frames[-HISTORY_LENGTH:]
    return output


def _quality_float(value: Any, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    r"""验证 quality 输入为 finite float，并返回 float64 以降低 ratio 舍入误差。"""

    array = _array(value, name)
    if shape is not None and array.shape != shape:
        raise FamilyDatasetError(f"{name} shape {array.shape} != expected {shape}")
    if array.ndim != 1 or array.size < 1:
        raise FamilyDatasetError(f"{name} must be a nonempty one-dimensional vector")
    if array.dtype.kind not in "fiu" or array.dtype.kind == "b":
        raise FamilyDatasetError(f"{name} must be finite numeric values")
    if array.dtype.kind == "f" and not bool(np.isfinite(array).all()):
        raise FamilyDatasetError(f"{name} must be finite floating values")
    return array.astype(np.float64, copy=False)


def quality_episode_mask(
    net_turns: np.ndarray,
    path_turns: np.ndarray,
    duration_s: np.ndarray,
    termination_drop: np.ndarray,
    termination_axis: np.ndarray,
    *,
    min_turns: float = 0.5,
    min_direction: float = 0.7,
    horizon_s: float = 30.0,
) -> np.ndarray:
    r"""返回 episode 级质量门，保留原始失败行并只输出 bool mask。

    合格条件是完整安全 30 s、净圈 ``net_turns >= 0.5``，以及有向路径一致性
    ``net_turns / path_turns >= 0.7``；其中 duration 使用 ``1e-3 s`` 的绝对容差吸收
    20 Hz FP32 累加得到的 29.9998。物理 drop 或 axis 任一为真都拒绝。函数不会删行、
    重采样或补造零合格资产；调用者可用返回 mask 与 env_asset_index 做逐资产计数。
    """

    net = _quality_float(net_turns, "net_turns")
    path = _quality_float(path_turns, "path_turns", net.shape)
    duration = _quality_float(duration_s, "duration_s", net.shape)
    drop = _bool_mask(termination_drop, "termination_drop", net.shape)
    axis = _bool_mask(termination_axis, "termination_axis", net.shape)
    try:
        turns_threshold = float(min_turns)
        direction_threshold = float(min_direction)
        horizon = float(horizon_s)
    except (TypeError, ValueError) as error:
        raise FamilyDatasetError("quality thresholds must be finite numbers") from error
    if not np.isfinite(turns_threshold) or turns_threshold < 0.0:
        raise FamilyDatasetError("min_turns must be finite and nonnegative")
    if not np.isfinite(direction_threshold) or not 0.0 <= direction_threshold <= 1.0:
        raise FamilyDatasetError("min_direction must lie in [0,1]")
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise FamilyDatasetError("horizon_s must be finite and positive")
    if np.any(path < 0.0) or np.any(duration < 0.0):
        raise FamilyDatasetError("path_turns and duration_s must be nonnegative")

    # path=0 没有定义方向；显式 require path>0，避免 0/0 被当作合格。
    with np.errstate(divide="ignore", invalid="ignore"):
        direction = np.divide(net, path, out=np.full_like(net, -np.inf), where=path > 0.0)
    return (
        (duration >= horizon - DURATION_ATOL_S)
        & (net >= turns_threshold)
        & (direction >= direction_threshold)
        & ~drop
        & ~axis
    )


def episode_replica_split(
    replica_index: np.ndarray,
    *,
    validation_modulus: int = 4,
    validation_remainder: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    r"""按 replica ID 做 episode 级 train/validation split。

    返回 ``(training_indices, validation_indices)``，元素是输入 vector 的行索引；
    ``replica_id % 4 == 3`` 的完整 episode 进入 validation，其余进入 training。同一
    replica 的所有连续样本必须共享一个分区，避免相邻时间帧泄漏。若 validation 为空，
    函数仍返回空索引并让 caller 显式看到证据缺口，不偷偷复制 train 样本。
    """

    array = _array(replica_index, "replica_index")
    if array.ndim != 1 or array.size < 1 or array.dtype.kind not in "iu" or array.dtype.kind == "b":
        raise FamilyDatasetError("replica_index must be a nonempty integer vector")
    if np.any(array < 0):
        raise FamilyDatasetError("replica_index must contain nonnegative ids")
    try:
        modulus = operator.index(validation_modulus)
        remainder = operator.index(validation_remainder)
    except (TypeError, ValueError) as error:
        raise FamilyDatasetError("replica modulo rule must be integer") from error
    if isinstance(validation_modulus, (bool, np.bool_)) or modulus < 2:
        raise FamilyDatasetError("validation_modulus must be at least 2")
    if isinstance(validation_remainder, (bool, np.bool_)) or not 0 <= remainder < modulus:
        raise FamilyDatasetError("validation_remainder must lie within validation_modulus")
    validation = np.flatnonzero(array % modulus == remainder).astype(np.int64, copy=False)
    training = np.flatnonzero(array % modulus != remainder).astype(np.int64, copy=False)
    return training, validation


# 这两个别名让下游训练器可以使用其更自然的命名，同时共享同一实现与 split 证据。
split_episode_replicas = episode_replica_split
replica_modulo_split = episode_replica_split


__all__ = [
    "ARTIFACT_TYPE",
    "SCHEMA_VERSION",
    "HISTORY_LENGTH",
    "JOINT_COUNT",
    "JOINT_FEATURES",
    "OWNER_COUNT",
    "TIP_COUNT",
    "GEOMETRY_TOKEN_WIDTH",
    "JOINT_KINEMATICS_WIDTH",
    "FamilyDatasetError",
    "FamilyTrajectoryWriter",
    "episode_replica_split",
    "quality_episode_mask",
    "read_family_metadata",
    "reconstruct_history",
    "replica_modulo_split",
    "split_episode_replicas",
]
