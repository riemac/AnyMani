r"""Palm-supported good-pregrasp Top-K catalog与exact runtime lookup。

对hand asset $h_i$、object identity $o$与无量纲scale $s$，catalog保存耦合reset集合：

$$
(h_i,o,s)\mapsto\mathcal B_{i,o,s}
=\left\{(q_0^{(k)},u_0^{(k)},T_{ho,0}^{(k)},m^{(k)})\right\}_{k=0}^{7}.
$$

$q_0,u_0\in\mathbb R^{16}$使用canonical storage，active mask给出真实$n_i$维子空间；角度单位rad。
$T_{ho,0}\in SE(3)$把object frame接到hand semantic frame，translation单位m、quaternion顺序wxyz。
MVP要求$u_0=q_0$且object严格upright；TIP/JOINT/PALM contact只记录为质量元数据，不构成tier或准入门。

Catalog采用一项physical hand/object/scale key对应一个Top-8 payload。Index和payload都以同文件系统临时文件
原子发布；resolve重新核对key与payload内容摘要，避免中断写入或目录串包。SHA-256只服务持久化identity，
不参与pregrasp质量排序。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

GOOD_PREGRASP_SCHEMA_VERSION = "3.0.0"
"""与旧contact-tier schema分离的good-pregrasp catalog版本。"""

GOOD_PREGRASP_ENTRY_TYPE = "anymani.good_pregrasp.entry"
GOOD_PREGRASP_INDEX_TYPE = "anymani.good_pregrasp.index"
GOOD_PREGRASP_TOP_K = 8
CANONICAL_JOINT_COUNT = 16
CANONICAL_OWNER_COUNT = 21
UPRIGHT_QUATERNION_WXYZ = (1.0, 0.0, 0.0, 0.0)
_SHA256 = re.compile(r"[0-9a-f]{64}")


class GoodPregraspCatalogError(RuntimeError):
    r"""Good-pregrasp catalog的IO、冲突或内容完整性错误。"""


class GoodPregraspMissError(GoodPregraspCatalogError):
    r"""Exact hand/object/scale key没有已发布Top-8 entry。"""


class GoodPregraspConflictError(GoodPregraspCatalogError):
    r"""同一个exact key试图发布不同Top-8 payload。"""


def _sha256(value: str, field_name: str) -> str:
    r"""验证持久化identity使用的64位小写SHA-256。"""

    parsed = str(value)
    if _SHA256.fullmatch(parsed) is None:
        raise ValueError(f"{field_name} must be a 64-character lowercase SHA-256")
    return parsed


def _finite_tuple(values: Sequence[float], *, length: int, name: str) -> tuple[float, ...]:
    r"""规约固定宽度有限浮点序列。"""

    parsed = tuple(float(value) for value in values)
    if len(parsed) != length or not all(math.isfinite(value) for value in parsed):
        raise ValueError(f"{name} must contain {length} finite values")
    return parsed


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    r"""生成字段排序、无NaN的稳定JSON bytes。"""

    return json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(payload: Mapping[str, Any]) -> str:
    r"""返回持久化document的SHA-256内容身份。"""

    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclass(frozen=True)
class GoodPregraspKey:
    r"""一项hand/object/exact-scale reset catalog查询坐标。"""

    asset_id: str  # formal selection中的人类可读asset label
    source_content_hash: str  # source bundle内容身份
    physical_geometry_hash: str  # canonical active physical mapping身份
    canonical_schema_digest: str  # canonical storage/routing schema身份
    routing_digest: str  # active-joint mask身份
    object_asset_id: str  # 当前为``DexCube``
    object_asset_sha256: str  # 实际USD bytes身份
    object_scale: float  # exact无量纲scale；MVP为1.1
    physics_identity_digest: str  # dt/material/solver/mass等生成物理身份
    generation_identity_digest: str  # proposal/settle/cold-reset协议身份

    def __post_init__(self) -> None:
        r"""拒绝空标签、非法hash和非正scale。"""

        if not self.asset_id.strip() or not self.object_asset_id.strip():
            raise ValueError("good-pregrasp asset/object IDs must be non-empty")
        for field_name in (
            "source_content_hash",
            "physical_geometry_hash",
            "canonical_schema_digest",
            "routing_digest",
            "object_asset_sha256",
            "physics_identity_digest",
            "generation_identity_digest",
        ):
            object.__setattr__(self, field_name, _sha256(getattr(self, field_name), field_name))
        if not math.isfinite(self.object_scale) or self.object_scale <= 0.0:
            raise ValueError("good-pregrasp object_scale must be finite and positive")

    @property
    def digest(self) -> str:
        r"""返回exact query key的内容身份。"""

        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        r"""返回JSON-safe key。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GoodPregraspKey:
        r"""从持久化mapping恢复并重新验证key。"""

        return cls(**dict(payload))


@dataclass(frozen=True)
class GoodPregraspCandidate:
    r"""一个可直接写入训练reset的hand-object初始状态。"""

    q_state_rad: tuple[float, ...]  # canonical actual joint state$[16]$，rad
    q_target_rad: tuple[float, ...]  # canonical PD target$[16]$，MVP严格等于$q_0$
    active_joint_mask: tuple[bool, ...]  # canonical active subspace$[16]$
    object_position_h_m: tuple[float, float, float]  # object origin在hand frame的位置，m
    object_orientation_h_wxyz: tuple[float, float, float, float] = UPRIGHT_QUATERNION_WXYZ

    def __post_init__(self) -> None:
        r"""验证$q_0=u_0$、ghost zero与严格upright初始姿态。"""

        q_state = _finite_tuple(self.q_state_rad, length=CANONICAL_JOINT_COUNT, name="q_state_rad")
        q_target = _finite_tuple(self.q_target_rad, length=CANONICAL_JOINT_COUNT, name="q_target_rad")
        mask = tuple(bool(value) for value in self.active_joint_mask)
        if len(mask) != CANONICAL_JOINT_COUNT or not any(mask):
            raise ValueError("active_joint_mask must contain 16 entries and at least one active joint")
        if q_state != q_target:
            raise ValueError("good-pregrasp MVP requires q_target_rad to equal q_state_rad exactly")
        if any(value != 0.0 for value, active in zip(q_state, mask, strict=True) if not active):
            raise ValueError("inactive canonical joint states/targets must be exactly zero")
        position = _finite_tuple(self.object_position_h_m, length=3, name="object_position_h_m")
        orientation = _finite_tuple(
            self.object_orientation_h_wxyz,
            length=4,
            name="object_orientation_h_wxyz",
        )
        if orientation != UPRIGHT_QUATERNION_WXYZ:
            raise ValueError("good-pregrasp MVP object orientation must be exact hand-frame upright")
        object.__setattr__(self, "q_state_rad", q_state)
        object.__setattr__(self, "q_target_rad", q_target)
        object.__setattr__(self, "active_joint_mask", mask)
        object.__setattr__(self, "object_position_h_m", position)
        object.__setattr__(self, "object_orientation_h_wxyz", UPRIGHT_QUATERNION_WXYZ)

    def to_dict(self) -> dict[str, Any]:
        r"""返回JSON-safe candidate。"""

        return {
            "q_state_rad": list(self.q_state_rad),
            "q_target_rad": list(self.q_target_rad),
            "active_joint_mask": list(self.active_joint_mask),
            "object_position_h_m": list(self.object_position_h_m),
            "object_orientation_h_wxyz": list(self.object_orientation_h_wxyz),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GoodPregraspCandidate:
        r"""从JSON-safe mapping恢复candidate。"""

        return cls(
            q_state_rad=tuple(payload["q_state_rad"]),
            q_target_rad=tuple(payload["q_target_rad"]),
            active_joint_mask=tuple(payload["active_joint_mask"]),
            object_position_h_m=tuple(payload["object_position_h_m"]),
            object_orientation_h_wxyz=tuple(payload["object_orientation_h_wxyz"]),
        )


@dataclass(frozen=True)
class GoodPregraspMetrics:
    r"""Geometry与1 s cold-reset acceptance的紧凑充分统计。"""

    joint_limit_margin_fraction: float  # active joints到最近limit的最小归一化余量
    envelope_fingers: tuple[str, str, str]  # thumb＋两个non-thumb roles
    envelope_sector_min_deg: float  # 三指面内最小sector separation，degree
    envelope_tip_center_distance_m: tuple[float, float, float]  # 三指TIP到object center距离，m
    penetration_depth_max_m: float  # 初态/重放最大非法穿透，m
    object_displacement_max_m: float  # 1 s内相对初态最大位移，m
    object_tilt_max_deg: float  # 1 s内object z与hand z最大夹角，degree
    peak_linear_velocity_m_s: float  # cold reset前0.2 s峰值，m/s
    peak_off_axis_angular_velocity_rad_s: float  # cold reset前0.2 s非目标轴峰值，rad/s
    palm_contact_fraction: float  # 最后0.5 s的PALM support占比
    owner_contact_fraction: tuple[float, ...]  # PALM+JOINT16+TIP4 contact占比$[21]$
    peak_angular_velocity_rad_s: float | None = None  # 可选总角速度峰值；strict v5要求非空，rad/s

    def __post_init__(self) -> None:
        r"""验证单位区间、三指包络和所有finite物理统计。"""

        if len(self.envelope_fingers) != 3 or self.envelope_fingers[0] != "thumb":
            raise ValueError("envelope_fingers must be thumb followed by two non-thumb roles")
        if len(set(self.envelope_fingers)) != 3 or any(
            finger not in {"thumb", "index", "middle", "ring"} for finger in self.envelope_fingers
        ):
            raise ValueError("envelope_fingers must contain three distinct canonical finger roles")
        distances = _finite_tuple(
            self.envelope_tip_center_distance_m,
            length=3,
            name="envelope_tip_center_distance_m",
        )
        owner_contact = _finite_tuple(
            self.owner_contact_fraction,
            length=CANONICAL_OWNER_COUNT,
            name="owner_contact_fraction",
        )
        scalars = (
            self.joint_limit_margin_fraction,
            self.envelope_sector_min_deg,
            self.penetration_depth_max_m,
            self.object_displacement_max_m,
            self.object_tilt_max_deg,
            self.peak_linear_velocity_m_s,
            self.peak_off_axis_angular_velocity_rad_s,
            self.palm_contact_fraction,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in scalars):
            raise ValueError("good-pregrasp metrics must be finite and non-negative")
        if not 0.0 <= self.joint_limit_margin_fraction <= 0.5:
            raise ValueError("joint_limit_margin_fraction must lie in [0,0.5]")
        if not 0.0 <= self.palm_contact_fraction <= 1.0 or any(
            not 0.0 <= value <= 1.0 for value in owner_contact
        ):
            raise ValueError("contact fractions must lie in [0,1]")
        if any(value < 0.0 for value in distances):
            raise ValueError("envelope distances must be non-negative")
        if self.peak_angular_velocity_rad_s is not None and (
            not math.isfinite(self.peak_angular_velocity_rad_s) or self.peak_angular_velocity_rad_s < 0.0
        ):
            raise ValueError("peak total angular velocity must be finite and non-negative when provided")
        object.__setattr__(self, "envelope_tip_center_distance_m", distances)
        object.__setattr__(self, "owner_contact_fraction", owner_contact)

    def to_dict(self) -> dict[str, Any]:
        r"""返回JSON-safe physical metrics。"""

        payload = asdict(self)
        payload["envelope_fingers"] = list(self.envelope_fingers)
        payload["envelope_tip_center_distance_m"] = list(self.envelope_tip_center_distance_m)
        payload["owner_contact_fraction"] = list(self.owner_contact_fraction)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GoodPregraspMetrics:
        r"""从持久化mapping恢复metrics。"""

        values = dict(payload)
        values["envelope_fingers"] = tuple(values["envelope_fingers"])
        values["envelope_tip_center_distance_m"] = tuple(values["envelope_tip_center_distance_m"])
        values["owner_contact_fraction"] = tuple(values["owner_contact_fraction"])
        return cls(**values)


@dataclass(frozen=True)
class GoodPregraspMember:
    r"""Top-8中的一个有序候选及其完整验收统计。"""

    rank: int  # 0为MVP runtime消费项
    candidate: GoodPregraspCandidate
    metrics: GoodPregraspMetrics
    selection_score: tuple[float, ...]  # generator定义的词典序质量向量

    def __post_init__(self) -> None:
        r"""验证rank与有限非空selection score。"""

        if self.rank < 0:
            raise ValueError("good-pregrasp rank must be non-negative")
        score = tuple(float(value) for value in self.selection_score)
        if not score or not all(math.isfinite(value) for value in score):
            raise ValueError("selection_score must be a non-empty finite tuple")
        object.__setattr__(self, "selection_score", score)

    def to_dict(self) -> dict[str, Any]:
        r"""返回JSON-safe ranked member。"""

        return {
            "rank": self.rank,
            "candidate": self.candidate.to_dict(),
            "metrics": self.metrics.to_dict(),
            "selection_score": list(self.selection_score),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GoodPregraspMember:
        r"""从持久化mapping恢复ranked member。"""

        return cls(
            rank=int(payload["rank"]),
            candidate=GoodPregraspCandidate.from_dict(payload["candidate"]),
            metrics=GoodPregraspMetrics.from_dict(payload["metrics"]),
            selection_score=tuple(payload["selection_score"]),
        )


@dataclass(frozen=True)
class GoodPregraspEntry:
    r"""一个exact hand/object/scale key对应的完整Top-8 reset集合。"""

    key: GoodPregraspKey
    members: tuple[GoodPregraspMember, ...]

    def __post_init__(self) -> None:
        r"""要求恰好8项、rank连续且candidate互不重复。"""

        if len(self.members) != GOOD_PREGRASP_TOP_K:
            raise ValueError(f"published good-pregrasp entry requires exactly {GOOD_PREGRASP_TOP_K} members")
        if tuple(member.rank for member in self.members) != tuple(range(GOOD_PREGRASP_TOP_K)):
            raise ValueError("good-pregrasp member ranks must be contiguous 0..7")
        candidate_digests = [_digest(member.candidate.to_dict()) for member in self.members]
        if len(set(candidate_digests)) != GOOD_PREGRASP_TOP_K:
            raise ValueError("good-pregrasp Top-8 candidates must be unique")

    @property
    def digest(self) -> str:
        r"""返回entry payload内容身份。"""

        return _digest(self.to_dict())

    @property
    def primary(self) -> GoodPregraspMember:
        r"""返回MVP固定消费的rank-0 member。"""

        return self.members[0]

    def to_dict(self) -> dict[str, Any]:
        r"""返回schema-3 JSON-safe entry。"""

        return {
            "artifact_type": GOOD_PREGRASP_ENTRY_TYPE,
            "schema_version": GOOD_PREGRASP_SCHEMA_VERSION,
            "key": self.key.to_dict(),
            "members": [member.to_dict() for member in self.members],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GoodPregraspEntry:
        r"""从持久化document恢复并验证entry。"""

        if payload.get("artifact_type") != GOOD_PREGRASP_ENTRY_TYPE:
            raise ValueError("unexpected good-pregrasp entry artifact_type")
        if payload.get("schema_version") != GOOD_PREGRASP_SCHEMA_VERSION:
            raise ValueError("unsupported good-pregrasp schema_version")
        return cls(
            key=GoodPregraspKey.from_dict(payload["key"]),
            members=tuple(GoodPregraspMember.from_dict(member) for member in payload["members"]),
        )


@dataclass(frozen=True)
class GoodPregraspIndexEntry:
    r"""Catalog index中的exact key到content-addressed payload映射。"""

    key_digest: str
    entry_digest: str
    payload_relpath: str

    def __post_init__(self) -> None:
        _sha256(self.key_digest, "key_digest")
        _sha256(self.entry_digest, "entry_digest")
        expected = f"records/{self.entry_digest}.json"
        if self.payload_relpath != expected:
            raise ValueError(f"good-pregrasp payload path must be {expected!r}")


class GoodPregraspCatalog:
    r"""Atomic schema-3 Top-8 catalog publisher与fail-closed resolver。"""

    def __init__(self, root: str | Path) -> None:
        r"""绑定catalog根；目录只在publish时创建。"""

        self.root = Path(root).expanduser()
        self.index_path = self.root / "index.json"
        self.records_dir = self.root / "records"

    def _load_index(self) -> tuple[GoodPregraspIndexEntry, ...]:
        r"""读取并验证有序index；不存在表示空catalog。"""

        if not self.index_path.is_file():
            return ()
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise GoodPregraspCatalogError(f"cannot read good-pregrasp index: {error}") from error
        if payload.get("artifact_type") != GOOD_PREGRASP_INDEX_TYPE:
            raise GoodPregraspCatalogError("unexpected good-pregrasp index artifact_type")
        if payload.get("schema_version") != GOOD_PREGRASP_SCHEMA_VERSION:
            raise GoodPregraspCatalogError("unsupported good-pregrasp index schema_version")
        try:
            entries = tuple(GoodPregraspIndexEntry(**entry) for entry in payload["entries"])
        except (KeyError, TypeError, ValueError) as error:
            raise GoodPregraspCatalogError(f"invalid good-pregrasp index entry: {error}") from error
        if tuple(entry.key_digest for entry in entries) != tuple(sorted(entry.key_digest for entry in entries)):
            raise GoodPregraspCatalogError("good-pregrasp index entries must be sorted by key_digest")
        if len({entry.key_digest for entry in entries}) != len(entries):
            raise GoodPregraspCatalogError("good-pregrasp index contains duplicate exact keys")
        return entries

    @staticmethod
    def _index_document(entries: Sequence[GoodPregraspIndexEntry]) -> dict[str, Any]:
        r"""构造紧凑、稳定排序的index document。"""

        return {
            "artifact_type": GOOD_PREGRASP_INDEX_TYPE,
            "schema_version": GOOD_PREGRASP_SCHEMA_VERSION,
            "entries": [asdict(entry) for entry in sorted(entries, key=lambda item: item.key_digest)],
        }

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        r"""在目标文件同目录写临时文件并原子replace。"""

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)

    def publish(self, entry: GoodPregraspEntry) -> GoodPregraspIndexEntry:
        r"""幂等发布一个Top-8 entry；同key不同payload严格冲突。"""

        key_digest = entry.key.digest
        document = entry.to_dict()
        payload_bytes = _canonical_bytes(document)
        entry_digest = hashlib.sha256(payload_bytes).hexdigest()
        index_entry = GoodPregraspIndexEntry(
            key_digest=key_digest,
            entry_digest=entry_digest,
            payload_relpath=f"records/{entry_digest}.json",
        )
        entries = list(self._load_index())
        existing = next((item for item in entries if item.key_digest == key_digest), None)
        if existing is not None:
            if existing.entry_digest != entry_digest:
                raise GoodPregraspConflictError("exact good-pregrasp key already maps to another Top-8 payload")
            return existing
        self._atomic_write(self.root / index_entry.payload_relpath, payload_bytes + b"\n")
        entries.append(index_entry)
        self._atomic_write(self.index_path, _canonical_bytes(self._index_document(entries)) + b"\n")
        return index_entry

    def resolve(self, key: GoodPregraspKey) -> GoodPregraspEntry:
        r"""按exact key读取Top-8并重新核对payload digest与embedded key。"""

        return self.resolve_many((key,))[0]

    def resolve_many(self, keys: Sequence[GoodPregraspKey]) -> tuple[GoodPregraspEntry, ...]:
        r"""一次读取index并按输入顺序解析多个exact keys。

        ManagerBased full reset会同时解析80个assets。Index是共同不可变证据，不应为每个key重复读取80次；
        payload仍逐项执行content digest与embedded key验证。
        """

        requested = tuple(keys)
        if not requested:
            return ()
        index_by_digest = {entry.key_digest: entry for entry in self._load_index()}  # 单次index read
        resolved_by_digest: dict[str, GoodPregraspEntry] = {}
        output: list[GoodPregraspEntry] = []
        for key in requested:
            match = index_by_digest.get(key.digest)
            if match is None:
                raise GoodPregraspMissError(
                    f"no good-pregrasp entry for asset={key.asset_id} object={key.object_asset_id} scale={key.object_scale}"
                )
            entry = resolved_by_digest.get(match.entry_digest)
            if entry is None:
                path = self.root / match.payload_relpath
                try:
                    payload_bytes = path.read_bytes()
                    payload = json.loads(payload_bytes)
                    entry = GoodPregraspEntry.from_dict(payload)
                except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
                    raise GoodPregraspCatalogError(f"cannot restore good-pregrasp payload: {error}") from error
                if hashlib.sha256(_canonical_bytes(payload)).hexdigest() != match.entry_digest:
                    raise GoodPregraspCatalogError("good-pregrasp payload content digest mismatch")
                resolved_by_digest[match.entry_digest] = entry
            if entry.key != key or entry.key.digest != match.key_digest:
                raise GoodPregraspCatalogError("good-pregrasp payload embedded key mismatch")
            output.append(entry)
        return tuple(output)


__all__ = [
    "CANONICAL_JOINT_COUNT",
    "CANONICAL_OWNER_COUNT",
    "GOOD_PREGRASP_SCHEMA_VERSION",
    "GOOD_PREGRASP_TOP_K",
    "UPRIGHT_QUATERNION_WXYZ",
    "GoodPregraspCandidate",
    "GoodPregraspCatalog",
    "GoodPregraspCatalogError",
    "GoodPregraspConflictError",
    "GoodPregraspEntry",
    "GoodPregraspKey",
    "GoodPregraspMember",
    "GoodPregraspMetrics",
    "GoodPregraspMissError",
]
