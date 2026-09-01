r"""Identity-keyed heterogeneous pregrasp/contact-basin schema-2。

本模块只定义纯Python科研合同，不导入IsaacLab、USD、policy或训练后端。一个pregrasp记录同时回答三件
彼此正交的问题：物理对象是谁、中心candidate达到什么接触等级、该点是否拥有通过扰动认证的局部盆。

等级固定为：

``rejected < support_basin < contact_basin < gravity_robust``。

``coverage=point``只证明中心点，不能满足默认训练provider的basin查询；``coverage=basin``必须携带
scale certificate与局部扰动充分统计量。Palm contact始终合法；contact tier要求至少两个TIP持续参与并
限制finger non-tip，gravity tier再要求至少三个TIP与全部六个hand-frame重力方向。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Literal

PREGRASP_SCHEMA_VERSION = "2.1.0"  # 2.1分别认证reset state与PD preload target，避免丢失接触法向力
PREGRASP_RECORD_ARTIFACT_TYPE = "anymani.pregrasp.record"  # 单个candidate认证记录的稳定artifact类型
PREGRASP_INDEX_ARTIFACT_TYPE = "anymani.pregrasp.index"  # 原子cache index的稳定artifact类型
SCALE_ANCHORS = ("1.1", "1.2", "1.25")  # sealed P0001指定的三个DexCube scale搜索锚点
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")  # 所有物理/cache摘要统一使用小写64-hex SHA-256


class PregraspTier(StrEnum):
    r"""中心candidate达到的接触认证等级。"""

    REJECTED = "rejected"  # 连support稳定性都未满足
    SUPPORT_BASIN = "support_basin"  # palm-supported稳定点/盆，不要求两个TIP
    CONTACT_BASIN = "contact_basin"  # 至少两个TIP且non-tip受限
    GRAVITY_ROBUST = "gravity_robust"  # contact基础上通过六个hand-frame重力方向


class PregraspCoverage(StrEnum):
    r"""认证覆盖的是失败、单点还是局部扰动盆。"""

    REJECTED = "rejected"  # 中心点没有通过最小support gate
    POINT = "point"  # 只验证nominal candidate，不可满足require_basin查询
    BASIN = "basin"  # 通过显式scale与local perturbation certificate


_TIER_RANK = {
    PregraspTier.REJECTED: 0,
    PregraspTier.SUPPORT_BASIN: 1,
    PregraspTier.CONTACT_BASIN: 2,
    PregraspTier.GRAVITY_ROBUST: 3,
}  # 等级序只用于provider minimum-tier比较，不改变每级物理定义


def tier_satisfies(actual: PregraspTier, required: PregraspTier) -> bool:
    r"""判断一个认证等级是否满足minimum-tier查询。"""

    return _TIER_RANK[PregraspTier(actual)] >= _TIER_RANK[PregraspTier(required)]  # 高tier蕴含低tier能力


def _validate_sha256(value: str, field_name: str) -> str:
    r"""严格验证64位小写SHA-256，拒绝伪identity字符串。"""

    parsed = str(value)  # 输入可能来自JSON scalar，但不做大小写或截断修复
    if _SHA256_PATTERN.fullmatch(parsed) is None:
        raise ValueError(f"{field_name} must be a 64-character lowercase SHA-256 digest")
    return parsed


def _freeze_json(value: Any, path: str = "identity") -> Any:
    r"""递归验证finite JSON并冻结容器，避免构造后的mapping mutation改变digest。

    Dict变成只读``MappingProxyType``，list/tuple变成tuple；允许的叶节点只有None/bool/int/finite float/str。
    这份冻结结构只服务in-memory identity，序列化时由 :func:`_thaw_json` 恢复普通JSON容器。
    """

    if value is None or isinstance(value, (bool, str)):  # JSON原生无量纲叶节点
        return value
    if isinstance(value, int) and not isinstance(value, bool):  # bool已在上一分支处理
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain finite JSON numbers")
        return float(value)
    if isinstance(value, Mapping):
        frozen = {str(key): _freeze_json(item, f"{path}.{key}") for key, item in sorted(value.items())}
        return MappingProxyType(frozen)  # 深拷贝后只读，原输入后续修改不影响identity
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, f"{path}[{index}]") for index, item in enumerate(value))
    raise ValueError(f"{path} must contain finite JSON-compatible values, got {type(value).__name__}")


def _thaw_json(value: Any) -> Any:
    r"""把冻结identity恢复为普通JSON-safe dict/list。"""

    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    r"""生成无NaN、字段有序、空白稳定的canonical JSON bytes。"""

    return json.dumps(
        _thaw_json(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")  # digest不受缩进、locale或dict insertion order影响


def stable_digest(payload: Mapping[str, Any]) -> str:
    r"""返回canonical JSON的SHA-256。"""

    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()  # cache/content identity统一摘要边界


def _finite_tuple(values: Sequence[float], *, length: int, field_name: str) -> tuple[float, ...]:
    r"""把固定长度数值序列转成finite float tuple。"""

    parsed = tuple(float(value) for value in values)  # JSON/YAML number统一为Python float
    if len(parsed) != length or not all(math.isfinite(value) for value in parsed):
        raise ValueError(f"{field_name} must contain {length} finite values")
    return parsed


def active_mask_digest(active_joint_mask: Sequence[bool]) -> str:
    r"""计算canonical active-joint routing digest，排除selection-local row。"""

    mask = tuple(bool(value) for value in active_joint_mask)  # canonical v1固定16槽
    if len(mask) != 16 or not any(mask):
        raise ValueError("active_joint_mask must contain 16 entries and at least one active joint")
    return stable_digest({"active_joint_mask": list(mask)})  # 只认证有效动作/关节结构


@dataclass(frozen=True)
class PregraspGate:
    r"""Point、contact、basin与gravity认证使用的显式数值门。

    TIP persistence与finger non-tip上限没有宽松默认值；每次search必须显式选择并把本对象digest写入
    :class:`PregraspLookupKey`。其余阈值同样进入digest，数值校准不会静默复用旧cache。
    """

    min_tip_ge_2_fraction: float  # contact tier中每帧至少2 TIP的最小时间占比
    min_tip_ge_3_fraction: float  # gravity tier中每帧至少3 TIP的最小时间占比
    max_finger_non_tip_fraction: float  # finger non-tip接触的最大时间占比；palm不计入
    max_penetration_depth_m: float  # object-hand最大非法穿透深度，单位m
    max_anchor_distance_m: float  # object相对candidate anchor最大漂移，单位m
    max_linear_velocity_rms_m_s: float  # settle/stress窗口线速度RMS，单位m/s
    max_angular_velocity_rms_rad_s: float  # object角速度RMS，单位rad/s
    max_object_orientation_drift_rad: float  # 相对candidate orientation的最大姿态漂移，单位rad
    min_joint_limit_margin_rad: float  # 所有有效关节到最近limit的最小余量，单位rad
    max_target_tracking_error_rms_rad: float  # PD target tracking RMS，单位rad
    max_joint_effort_rms_N_m: float  # 有效关节effort RMS，单位N·m
    min_basin_success_fraction: float  # local perturbation trials中完整point gate通过比例
    required_gravity_directions: int = 6  # AnyRotate强tier固定六个hand-frame主方向

    def __post_init__(self) -> None:
        r"""拒绝非法概率、负物理上界和空gravity stress。"""

        fractions = (
            self.min_tip_ge_2_fraction,
            self.min_tip_ge_3_fraction,
            self.max_finger_non_tip_fraction,
            self.min_basin_success_fraction,
        )  # 四个字段均是[0,1]时间/试验比例
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("pregrasp gate fractions must lie in [0,1]")
        if self.min_tip_ge_2_fraction <= 0.0:  # sealed authority禁止contact gate退化成support-only
            raise ValueError("min_tip_ge_2_fraction must be explicitly non-zero")
        non_negative = (
            self.max_penetration_depth_m,
            self.max_anchor_distance_m,
            self.max_linear_velocity_rms_m_s,
            self.max_angular_velocity_rms_rad_s,
            self.max_object_orientation_drift_rad,
            self.min_joint_limit_margin_rad,
            self.max_target_tracking_error_rms_rad,
            self.max_joint_effort_rms_N_m,
        )  # 全部是有限非负物理量
        if any(not math.isfinite(value) or value < 0.0 for value in non_negative):
            raise ValueError("pregrasp gate physical thresholds must be finite and non-negative")
        if self.required_gravity_directions < 1:
            raise ValueError("required_gravity_directions must be positive")

    def to_dict(self) -> dict[str, Any]:
        r"""返回JSON-safe gate document。"""

        return asdict(self)

    @property
    def digest(self) -> str:
        r"""返回所有认证阈值的canonical摘要。"""

        return stable_digest(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PregraspGate:
        r"""从artifact恢复并重新验证gate。"""

        return cls(**dict(payload))


@dataclass(frozen=True)
class PregraspLookupKey:
    r"""不含scale interval与dataset row的物理cache查询身份。

    ``asset_id``只作人类provenance，不进入digest；真正命中依赖source/physical/canonical/routing/cube、
    support mode、gate、physics和search identity。改变任一科学字段都会形成另一lookup domain。
    """

    asset_id: str  # provenance label，可随dataset命名变化
    source_content_hash: str  # source bundle content SHA-256
    physical_geometry_hash: str  # 排除ghost后的真实几何SHA-256
    canonical_schema_digest: str  # canonical ABI schema SHA-256
    routing_digest: str  # active joint routing SHA-256
    cube_asset_id: str  # 人类可读object identity
    cube_asset_sha256: str  # 运行时解析的真实USD/object bytes SHA-256
    support_mode: Literal["palm_supported", "tip_only"]  # 当前主线为palm_supported
    gate_digest: str  # :class:`PregraspGate`的SHA-256
    physics_identity: Mapping[str, Any]  # solver/material/mass/inertia/contact阈值等
    search_identity: Mapping[str, Any]  # algorithm/version/seed/proposal与stress定义

    def __post_init__(self) -> None:
        r"""严格验证摘要、文本和递归finite JSON。"""

        if not self.asset_id or not self.cube_asset_id:
            raise ValueError("asset_id and cube_asset_id must be non-empty")
        for field_name in (
            "source_content_hash",
            "physical_geometry_hash",
            "canonical_schema_digest",
            "routing_digest",
            "cube_asset_sha256",
            "gate_digest",
        ):
            object.__setattr__(self, field_name, _validate_sha256(getattr(self, field_name), field_name))
        if self.support_mode not in {"palm_supported", "tip_only"}:
            raise ValueError("support_mode must be palm_supported or tip_only")
        if not self.physics_identity or not self.search_identity:
            raise ValueError("physics_identity and search_identity must be non-empty")
        object.__setattr__(self, "physics_identity", _freeze_json(self.physics_identity, "physics_identity"))
        object.__setattr__(self, "search_identity", _freeze_json(self.search_identity, "search_identity"))

    def to_dict(self) -> dict[str, Any]:
        r"""返回含provenance asset_id的完整JSON document。"""

        return {
            "asset_id": self.asset_id,
            "source_content_hash": self.source_content_hash,
            "physical_geometry_hash": self.physical_geometry_hash,
            "canonical_schema_digest": self.canonical_schema_digest,
            "routing_digest": self.routing_digest,
            "cube_asset_id": self.cube_asset_id,
            "cube_asset_sha256": self.cube_asset_sha256,
            "support_mode": self.support_mode,
            "gate_digest": self.gate_digest,
            "physics_identity": _thaw_json(self.physics_identity),
            "search_identity": _thaw_json(self.search_identity),
        }

    def identity_dict(self) -> dict[str, Any]:
        r"""返回真正参与lookup digest的字段；selection/provenance asset_id被排除。"""

        document = self.to_dict()
        document.pop("asset_id")  # 同一物理资产改名不产生重复cache
        return document

    @property
    def digest(self) -> str:
        r"""返回lookup domain的canonical SHA-256。"""

        return stable_digest(self.identity_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PregraspLookupKey:
        r"""从artifact恢复lookup key，不修复非法字段。"""

        return cls(**dict(payload))


@dataclass(frozen=True)
class PregraspCandidate:
    r"""一个canonical hand controller state与hand-frame object reset point。

    接触平衡不只由关节位置决定，还依赖隐式PD执行器的预载：实际状态$\mathbf q_s$与控制目标
    $\mathbf q_t$之间的误差产生维持指尖法向力的执行器力矩。因而cache必须同时保存二者：reset先写
    ``q_state_rad``，随后把action/controller内部target初始化为``q_target_rad``。把同一个q同时用于两者会
    消除预载，使原本稳定的TIP contact在replay时退化。
    """

    q_state_rad: tuple[float, ...]  # canonical actual reset state $\mathbf q_s\in\mathbb R^{16}$，单位rad
    q_target_rad: tuple[float, ...]  # canonical PD preload target $\mathbf q_t\in\mathbb R^{16}$，单位rad
    active_joint_mask: tuple[bool, ...]  # True=真实有效joint
    object_position_h_m: tuple[float, float, float]  # object origin在hand frame中的位置，单位m
    object_orientation_wxyz: tuple[float, float, float, float]  # $R_{ho}$ quaternion，顺序wxyz
    object_scale: float  # nominal anchor scale
    seed_source: str  # proposal/template/refinement lineage

    def __post_init__(self) -> None:
        r"""验证canonical axes、ghost零值、unit quaternion与positive scale。"""

        q_state = _finite_tuple(self.q_state_rad, length=16, field_name="q_state_rad")
        q_target = _finite_tuple(self.q_target_rad, length=16, field_name="q_target_rad")
        mask = tuple(bool(value) for value in self.active_joint_mask)
        if len(mask) != 16 or not any(mask):
            raise ValueError("active_joint_mask must contain 16 entries and at least one active joint")
        if any(
            not active and (state != 0.0 or target != 0.0)
            for state, target, active in zip(q_state, q_target, mask)
        ):
            raise ValueError("pregrasp ghost joint state and target coordinates must be exactly zero")
        position = _finite_tuple(self.object_position_h_m, length=3, field_name="object_position_h_m")
        quaternion = _finite_tuple(self.object_orientation_wxyz, length=4, field_name="object_orientation_wxyz")
        if abs(math.sqrt(sum(value * value for value in quaternion)) - 1.0) > 1.0e-5:
            raise ValueError("object_orientation_wxyz must be a unit quaternion")
        if not math.isfinite(self.object_scale) or self.object_scale <= 0.0:
            raise ValueError("object_scale must be finite and positive")
        if not self.seed_source:
            raise ValueError("seed_source must be non-empty")
        object.__setattr__(self, "q_state_rad", q_state)
        object.__setattr__(self, "q_target_rad", q_target)
        object.__setattr__(self, "active_joint_mask", mask)
        object.__setattr__(self, "object_position_h_m", position)
        object.__setattr__(self, "object_orientation_wxyz", quaternion)

    def to_dict(self) -> dict[str, Any]:
        r"""返回JSON-safe candidate。"""

        return {
            "q_state_rad": list(self.q_state_rad),
            "q_target_rad": list(self.q_target_rad),
            "active_joint_mask": list(self.active_joint_mask),
            "object_position_h_m": list(self.object_position_h_m),
            "object_orientation_wxyz": list(self.object_orientation_wxyz),
            "object_scale": self.object_scale,
            "seed_source": self.seed_source,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PregraspCandidate:
        r"""从artifact恢复candidate。"""

        return cls(
            q_state_rad=tuple(payload["q_state_rad"]),
            q_target_rad=tuple(payload["q_target_rad"]),
            active_joint_mask=tuple(payload["active_joint_mask"]),
            object_position_h_m=tuple(payload["object_position_h_m"]),
            object_orientation_wxyz=tuple(payload["object_orientation_wxyz"]),
            object_scale=float(payload["object_scale"]),
            seed_source=str(payload["seed_source"]),
        )


@dataclass(frozen=True)
class PregraspMetrics:
    r"""Nominal point在settle/stress窗口上的物理充分统计量。"""

    finite: bool  # 所有参与gate的simulator state是否finite
    dropped: bool  # 是否越过明确drop/fall边界
    penetration_depth_max_m: float  # object-hand最大非法穿透，单位m
    tip_ge_2_fraction: float  # 至少2 TIP active的时间占比
    tip_ge_3_fraction: float  # 至少3 TIP active的时间占比
    tip_active_count_mean: float  # active TIP数量的时间均值
    palm_occupancy_fraction: float  # 合法palm support时间占比
    finger_non_tip_occupancy_fraction: float  # 不含palm的坏接触时间占比
    tip_object_center_distance_mean_m: float  # TIP到object center平均距离，单位m
    object_anchor_distance_max_m: float  # object相对candidate anchor最大平移，单位m
    object_linear_velocity_rms_m_s: float  # 线速度RMS，单位m/s
    object_angular_velocity_rms_rad_s: float  # 角速度RMS，单位rad/s
    object_orientation_drift_max_rad: float  # 相对candidate orientation的最大姿态漂移，单位rad
    joint_limit_margin_min_rad: float  # 最小joint limit margin，单位rad
    target_tracking_error_rms_rad: float  # PD target tracking RMS，单位rad
    joint_effort_rms_N_m: float  # 有效关节effort RMS，单位N·m

    def __post_init__(self) -> None:
        r"""验证fraction与物理量，不把缺失数据伪装成零。"""

        fractions = (self.tip_ge_2_fraction, self.tip_ge_3_fraction, self.palm_occupancy_fraction, self.finger_non_tip_occupancy_fraction)
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("pregrasp metric fractions must lie in [0,1]")
        non_negative = (
            self.penetration_depth_max_m,
            self.tip_active_count_mean,
            self.tip_object_center_distance_mean_m,
            self.object_anchor_distance_max_m,
            self.object_linear_velocity_rms_m_s,
            self.object_angular_velocity_rms_rad_s,
            self.object_orientation_drift_max_rad,
            self.target_tracking_error_rms_rad,
            self.joint_effort_rms_N_m,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in non_negative):
            raise ValueError("pregrasp physical metrics must be finite and non-negative")
        if not math.isfinite(self.joint_limit_margin_min_rad):
            raise ValueError("joint_limit_margin_min_rad must be finite")

    def to_dict(self) -> dict[str, Any]:
        r"""返回JSON-safe point metrics。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PregraspMetrics:
        r"""从artifact恢复point metrics。"""

        return cls(**dict(payload))


@dataclass(frozen=True)
class ScaleStressSample:
    r"""一个明确scale上的复核结果。"""

    scale: float  # 实际prestartup USD scale
    passed: bool  # 是否通过当前tier的完整point gate
    reason_codes: tuple[str, ...]  # 失败原因；passed时必须为空
    physics_snapshot: Mapping[str, Any]  # 该scale下实测mass/inertia/COM等，不进入lookup domain

    def __post_init__(self) -> None:
        r"""验证scale与pass/reason一致性。"""

        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("scale stress sample must use a finite positive scale")
        reasons = tuple(str(reason) for reason in self.reason_codes)
        if bool(self.passed) == bool(reasons):
            raise ValueError("passed scale sample must have no reasons; failed sample must have reasons")
        if not self.physics_snapshot:
            raise ValueError("scale stress sample requires an actual physics snapshot")
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "physics_snapshot", _freeze_json(self.physics_snapshot, "physics_snapshot"))

    def to_dict(self) -> dict[str, Any]:
        r"""返回JSON-safe scale sample。"""

        return {
            "scale": self.scale,
            "passed": self.passed,
            "reason_codes": list(self.reason_codes),
            "physics_snapshot": _thaw_json(self.physics_snapshot),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ScaleStressSample:
        r"""从artifact恢复scale sample。"""

        return cls(
            scale=float(payload["scale"]),
            passed=bool(payload["passed"]),
            reason_codes=tuple(payload["reason_codes"]),
            physics_snapshot=dict(payload["physics_snapshot"]),
        )


@dataclass(frozen=True)
class ScaleCertificate:
    r"""一个candidate的连续scale interval与local basin充分统计量。"""

    anchor: Literal["1.1", "1.2", "1.25"]  # search anchor使用十进制字符串，避免float key歧义
    scale_min: float  # 闭区间下界
    scale_max: float  # 闭区间上界
    scale_samples: tuple[ScaleStressSample, ...]  # interval内实际复核点
    perturbation_trials: int  # local q/pose/velocity trials总数
    perturbation_successes: int  # 完整point gate通过数
    gravity_directions_passed: int  # 六轴强stress通过方向数；普通contact可为0

    def __post_init__(self) -> None:
        r"""验证anchor、闭区间、scale samples与binomial sufficient statistics。"""

        if self.anchor not in SCALE_ANCHORS:
            raise ValueError(f"scale anchor must be one of {SCALE_ANCHORS}")
        if not (math.isfinite(self.scale_min) and math.isfinite(self.scale_max)):
            raise ValueError("scale interval must be finite")
        anchor_value = float(self.anchor)  # 仅数值比较，artifact key仍保留canonical字符串
        if self.scale_min <= 0.0 or self.scale_max < self.scale_min or not self.scale_min <= anchor_value <= self.scale_max:
            raise ValueError("scale interval must be positive and contain its anchor")
        samples = tuple(self.scale_samples)
        if not samples or any(not self.scale_min <= sample.scale <= self.scale_max for sample in samples):
            raise ValueError("scale samples must be non-empty and lie inside the certified interval")
        if any(not sample.passed for sample in samples):
            raise ValueError("certified scale interval cannot contain a failed stress sample")
        if not any(abs(sample.scale - anchor_value) <= 1.0e-8 for sample in samples):
            raise ValueError("scale certificate must explicitly test its anchor")
        if self.perturbation_trials < 1 or not 0 <= self.perturbation_successes <= self.perturbation_trials:
            raise ValueError("perturbation successes must lie in [0,trials] with trials>0")
        if not 0 <= self.gravity_directions_passed <= 6:
            raise ValueError("gravity_directions_passed must lie in [0,6]")
        object.__setattr__(self, "scale_samples", samples)

    @property
    def basin_success_fraction(self) -> float:
        r"""返回local perturbation的binomial pass fraction。"""

        return self.perturbation_successes / self.perturbation_trials  # trials已验证为正

    def contains(self, scale: float) -> bool:
        r"""按闭区间判断一个runtime scale是否被认证。"""

        requested = float(scale)
        return math.isfinite(requested) and self.scale_min <= requested <= self.scale_max

    def to_dict(self) -> dict[str, Any]:
        r"""返回JSON-safe scale/basin certificate。"""

        return {
            "anchor": self.anchor,
            "scale_min": self.scale_min,
            "scale_max": self.scale_max,
            "scale_samples": [sample.to_dict() for sample in self.scale_samples],
            "perturbation_trials": self.perturbation_trials,
            "perturbation_successes": self.perturbation_successes,
            "gravity_directions_passed": self.gravity_directions_passed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ScaleCertificate:
        r"""从artifact恢复scale certificate。"""

        return cls(
            anchor=str(payload["anchor"]),  # type: ignore[arg-type]
            scale_min=float(payload["scale_min"]),
            scale_max=float(payload["scale_max"]),
            scale_samples=tuple(ScaleStressSample.from_dict(item) for item in payload["scale_samples"]),
            perturbation_trials=int(payload["perturbation_trials"]),
            perturbation_successes=int(payload["perturbation_successes"]),
            gravity_directions_passed=int(payload["gravity_directions_passed"]),
        )


def _support_reasons(metrics: PregraspMetrics, gate: PregraspGate) -> tuple[str, ...]:
    r"""计算support point的稳定物理拒绝原因。"""

    reasons: list[str] = []  # 顺序固定，便于failure histogram跨run比较
    if not metrics.finite:
        reasons.append("non_finite_state")
    if metrics.dropped:
        reasons.append("object_dropped")
    if metrics.penetration_depth_max_m > gate.max_penetration_depth_m:
        reasons.append("invalid_penetration")
    if metrics.object_anchor_distance_max_m > gate.max_anchor_distance_m:
        reasons.append("object_anchor_drift")
    if metrics.object_linear_velocity_rms_m_s > gate.max_linear_velocity_rms_m_s:
        reasons.append("object_linear_motion")
    if metrics.object_angular_velocity_rms_rad_s > gate.max_angular_velocity_rms_rad_s:
        reasons.append("object_angular_motion")
    if metrics.object_orientation_drift_max_rad > gate.max_object_orientation_drift_rad:
        reasons.append("object_orientation_drift")
    if metrics.joint_limit_margin_min_rad < gate.min_joint_limit_margin_rad:
        reasons.append("joint_limit_margin")
    if metrics.target_tracking_error_rms_rad > gate.max_target_tracking_error_rms_rad:
        reasons.append("target_tracking_error")
    if metrics.joint_effort_rms_N_m > gate.max_joint_effort_rms_N_m:
        reasons.append("joint_effort")
    return tuple(reasons)


def _infer_tier(metrics: PregraspMetrics, gate: PregraspGate, certificate: ScaleCertificate | None) -> tuple[PregraspTier, tuple[str, ...]]:
    r"""从point metrics与可选strong stress推导最高接触等级。"""

    support_reasons = _support_reasons(metrics, gate)
    if support_reasons:
        return PregraspTier.REJECTED, support_reasons
    contact_reasons: list[str] = []
    if metrics.tip_ge_2_fraction < gate.min_tip_ge_2_fraction:
        contact_reasons.append("insufficient_tip_persistence")
    if metrics.finger_non_tip_occupancy_fraction > gate.max_finger_non_tip_fraction:
        contact_reasons.append("finger_non_tip_contact")
    if contact_reasons:
        return PregraspTier.SUPPORT_BASIN, tuple(contact_reasons)
    gravity_reasons: list[str] = []
    if metrics.tip_ge_3_fraction < gate.min_tip_ge_3_fraction:
        gravity_reasons.append("insufficient_three_tip_persistence")
    if certificate is None or certificate.gravity_directions_passed < gate.required_gravity_directions:
        gravity_reasons.append("gravity_stress_incomplete")
    if gravity_reasons:
        return PregraspTier.CONTACT_BASIN, tuple(gravity_reasons)
    return PregraspTier.GRAVITY_ROBUST, ()


@dataclass(frozen=True)
class PregraspRecord:
    r"""一个candidate的自验证schema-2 point/basin认证artifact。"""

    lookup_key: PregraspLookupKey  # 不含scale interval的cache domain
    candidate: PregraspCandidate  # nominal q与$T_{ho}$
    metrics: PregraspMetrics  # nominal point物理统计
    gate: PregraspGate  # 认证阈值，digest必须与lookup一致
    tier: PregraspTier  # 由metrics/certificate推导，不由writer自由声明
    coverage: PregraspCoverage  # point或basin
    scale_certificate: ScaleCertificate | None  # 只有basin必须存在
    reason_codes: tuple[str, ...]  # rejected或未达到更高tier的原因

    def __post_init__(self) -> None:
        r"""重算所有身份与认证结论，拒绝自相矛盾artifact。"""

        tier = PregraspTier(self.tier)
        coverage = PregraspCoverage(self.coverage)
        reasons = tuple(str(reason) for reason in self.reason_codes)
        if self.lookup_key.gate_digest != self.gate.digest:
            raise ValueError("lookup gate digest disagrees with embedded gate")
        if self.lookup_key.routing_digest != active_mask_digest(self.candidate.active_joint_mask):
            raise ValueError("candidate active mask disagrees with routing digest")
        certificate = self.scale_certificate
        if coverage == PregraspCoverage.BASIN:
            if certificate is None:
                raise ValueError("basin coverage requires a scale certificate")
            if not certificate.contains(self.candidate.object_scale):
                raise ValueError("candidate scale must lie inside its scale certificate")
            if certificate.basin_success_fraction < self.gate.min_basin_success_fraction:
                raise ValueError("basin certificate fails the configured perturbation success fraction")
        elif certificate is not None:
            raise ValueError("point/rejected coverage cannot carry a basin scale certificate")
        inferred_tier, inferred_reasons = _infer_tier(self.metrics, self.gate, certificate)
        inferred_coverage = PregraspCoverage.REJECTED if inferred_tier == PregraspTier.REJECTED else coverage
        if tier != inferred_tier or coverage != inferred_coverage or reasons != inferred_reasons:
            raise ValueError("record certification disagrees with metrics, gate, coverage, or certificate")
        object.__setattr__(self, "tier", tier)
        object.__setattr__(self, "coverage", coverage)
        object.__setattr__(self, "reason_codes", reasons)

    def payload_dict(self) -> dict[str, Any]:
        r"""返回不含self digest的canonical payload。"""

        return {
            "artifact_type": PREGRASP_RECORD_ARTIFACT_TYPE,
            "schema_version": PREGRASP_SCHEMA_VERSION,
            "lookup_key": self.lookup_key.to_dict(),
            "lookup_digest": self.lookup_key.digest,
            "candidate": self.candidate.to_dict(),
            "metrics": self.metrics.to_dict(),
            "gate": self.gate.to_dict(),
            "tier": self.tier.value,
            "coverage": self.coverage.value,
            "scale_certificate": self.scale_certificate.to_dict() if self.scale_certificate is not None else None,
            "reason_codes": list(self.reason_codes),
        }

    @property
    def digest(self) -> str:
        r"""返回完整认证payload的content SHA-256。"""

        return stable_digest(self.payload_dict())

    def to_dict(self) -> dict[str, Any]:
        r"""返回带record digest的完整artifact。"""

        return {**self.payload_dict(), "record_digest": self.digest}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PregraspRecord:
        r"""严格恢复record并复核lookup与content digests。"""

        if payload.get("artifact_type") != PREGRASP_RECORD_ARTIFACT_TYPE:
            raise ValueError("unsupported pregrasp artifact_type")
        if payload.get("schema_version") != PREGRASP_SCHEMA_VERSION:
            raise ValueError("unsupported pregrasp schema_version")
        lookup_key = PregraspLookupKey.from_dict(payload["lookup_key"])
        if payload.get("lookup_digest") != lookup_key.digest:
            raise ValueError("pregrasp lookup digest mismatch")
        certificate_payload = payload.get("scale_certificate")
        record = cls(
            lookup_key=lookup_key,
            candidate=PregraspCandidate.from_dict(payload["candidate"]),
            metrics=PregraspMetrics.from_dict(payload["metrics"]),
            gate=PregraspGate.from_dict(payload["gate"]),
            tier=PregraspTier(str(payload["tier"])),
            coverage=PregraspCoverage(str(payload["coverage"])),
            scale_certificate=(
                ScaleCertificate.from_dict(certificate_payload) if isinstance(certificate_payload, Mapping) else None
            ),
            reason_codes=tuple(str(reason) for reason in payload["reason_codes"]),
        )
        if payload.get("record_digest") != record.digest:
            raise ValueError("pregrasp record digest mismatch")
        return record


def certify_pregrasp(
    *,
    lookup_key: PregraspLookupKey,
    candidate: PregraspCandidate,
    metrics: PregraspMetrics,
    gate: PregraspGate,
    coverage: PregraspCoverage,
    scale_certificate: ScaleCertificate | None,
) -> PregraspRecord:
    r"""由原始point/basin证据推导最高tier并构造自验证record。

    Args:
        lookup_key (PregraspLookupKey): 不含scale interval的物理查询身份。
        candidate (PregraspCandidate): nominal hand q与hand-frame object pose。
        metrics (PregraspMetrics): nominal point的物理充分统计量。
        gate (PregraspGate): 本次认证使用的显式数值门。
        coverage (PregraspCoverage): caller实际执行的是point还是basin协议。
        scale_certificate (ScaleCertificate | None): basin的scale/local perturbation证书。

    Returns:
        PregraspRecord: tier与reason由证据推导的schema-2记录。
    """

    requested_coverage = PregraspCoverage(coverage)
    inferred_tier, reasons = _infer_tier(metrics, gate, scale_certificate)
    actual_coverage = PregraspCoverage.REJECTED if inferred_tier == PregraspTier.REJECTED else requested_coverage
    retained_certificate = None if inferred_tier == PregraspTier.REJECTED else scale_certificate
    return PregraspRecord(
        lookup_key=lookup_key,
        candidate=candidate,
        metrics=metrics,
        gate=gate,
        tier=inferred_tier,
        coverage=actual_coverage,
        scale_certificate=retained_certificate,
        reason_codes=reasons,
    )


__all__ = [
    "PREGRASP_INDEX_ARTIFACT_TYPE",
    "PREGRASP_RECORD_ARTIFACT_TYPE",
    "PREGRASP_SCHEMA_VERSION",
    "SCALE_ANCHORS",
    "PregraspCandidate",
    "PregraspCoverage",
    "PregraspGate",
    "PregraspLookupKey",
    "PregraspMetrics",
    "PregraspRecord",
    "PregraspTier",
    "ScaleCertificate",
    "ScaleStressSample",
    "active_mask_digest",
    "canonical_json_bytes",
    "certify_pregrasp",
    "stable_digest",
    "tier_satisfies",
]
