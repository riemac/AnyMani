r"""为共享学生准备四个 source-level 严格留出候选层。

本脚本只处理资产身份、来源坐标和确定性角色选择，不启动 Isaac、PhysX、预抓取搜索或策略推理。
它读取最终 N040 extended512 训练身份、同一次 N040 的 evaluation manifest，以及两族 teacher 的
训练/开发/旧终验 canonical lock，输出四个 ``32`` 成员的候选计划：

* ``pure LEAP-right × evaluation.unseen_variant_set``；
* ``pure Allegro-right × evaluation.unseen_variant_set``；
* ``pure LEAP-right × evaluation.unseen_mother``；
* ``pure Allegro-right × evaluation.unseen_mother``。

``physical_geometry_hash`` 在不同生产链中不是一个可直接相交的全局命名空间。N040 训练身份目前
冻结的是 ``(asset_id, content_hash)`` 轴，而 canonical cohort 的物理身份来自 canonical lowering；
因此这里绝不把两套 hash 集合的空交集写成物理隔离证书。输出始终标为 ``pending-canonical``，并
保留 N040 原始 partition、asset/source path、mother lineage、variant set、源文件 SHA 和 Allegro
限位修订前身份，供后续同域 canonical lowering 与 strict pregrasp 入口消费。

选择规则不读取任何策略分数：先按 typed ``family``、``handedness``、``collection_kind``、精确
production group 与 partition 过滤，再按 base lineage 做资格判断；每个 base 内按稳定 source hash
排序，base 之间按稳定 hash 做均衡 round-robin。新 base 层每个 base 先保留 mother，再取 variant。
Allegro variant 层只保留确实进入 Allegro teacher train 的七个 base，配额差最多一项。

正式后续入口约定是：先把本计划中的 source coordinates 物化为新的 source/canonical cohort，再把
schema-1.2 canonical-final lock 交给 ``scripts/research/prepare_cohort_pregrasp_shards.py``；strict
Top-8 完整后才可由 ``anymani.pregrasp.scripts.release_cohort`` 发布角色 lock/catalog。本文件不伪造
这些尚未存在的 canonical 或 ready 产物。
"""

from __future__ import annotations

# 标准库负责命令行、哈希、JSON 与确定性容器；脚本不依赖 AnyMani runtime 或 Isaac Sim。
# collections 只用于 source-level 计数和 lineage 分组，不把统计值引入选择排序。
# dataclass 让 revision/exclusion 状态在函数之间显式传递，避免隐式全局禁集。
# PyYAML 仅读取 source/evaluation manifest；canonical writer 的 JSON 优先由 _load_document 解析。
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# 所有默认输入都指向最终 N040 extended512 run；formal384 只在其 resume lineage 中出现。
REPO_ROOT = Path(__file__).resolve().parents[3]
# source manifest 声明 train/evaluation partition 与 generated lineage 关系。
N040_TRAIN_MANIFEST = Path("source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1/ssl.yaml")
# evaluation manifest 是 extended512 checkpoint 的两个 held-out suite，而不是 formal384 输出。
N040_EVAL_MANIFEST = Path(
    "logs/ssl/geometry_ssl_density_material_jacobian_se3_v0_8_1_extended512_matched_evaluation/"
    "20260830T174001Z/asset_manifest.yaml"
)
# source_artifacts 每个训练 asset 有16个 bank-index realization，用稳定 asset_id 做 train exposure 索引。
N040_TRAIN_ARTIFACTS = Path(
    "logs/ssl/geometry_ssl_density_material_jacobian_se3_v0_8_1_extended512_matched/"
    "20260830T164445Z/source_artifacts.jsonl"
)

# teacher lock 既作为 exact asset/source exclusion，也作为 base-design资格的 lineage evidence。
# LEAP train 的 canonical parent：32 个 mother 加每组 3 个 variant，共128逻辑成员。
DEFAULT_TEACHER_LOCKS: tuple[tuple[str, Path], ...] = (
    (
        "leap_teacher_train",
        Path(
            "source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1/cohorts/"
            "leap-right-32x4-foundation-20260906.canonical.lock.yaml"
        ),
    ),
    (
        "leap_teacher_dev_full",
        Path(
            "source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1/cohorts/"
            "leap-right-32x2-development-20260907.canonical.lock.yaml"
        ),
    ),
    # ready63 是同一 LEAP development parent 的可运行子视图；它不替代完整64分母。
    (
        "leap_teacher_dev_ready",
        Path(
            "source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1/cohorts/"
            "leap-right-development-ready63-20260907.canonical.lock.yaml"
        ),
    ),
    (
        "leap_teacher_old_final",
        Path(
            "source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1/cohorts/"
            "leap-right-32x4-acceptance-20260907.canonical.lock.yaml"
        ),
    ),
    # Allegro 原始 teacher 三套 lock 代表修订前物理；修订后由单独 revision train lock 绑定。
    (
        "allegro_teacher_train_old",
        Path(
            "logs/benchmarks/cross_family_rotation/cohort-preparation-20260907/released/"
            "allegro-v1/training.canonical.lock.yaml"
        ),
    ),
    (
        "allegro_teacher_dev_old",
        Path(
            "logs/benchmarks/cross_family_rotation/cohort-preparation-20260907/released/"
            "allegro-v1/development.canonical.lock.yaml"
        ),
    ),
    (
        "allegro_teacher_old_final",
        Path(
            "logs/benchmarks/cross_family_rotation/cohort-preparation-20260907/released/"
            "allegro-v1/acceptance.canonical.lock.yaml"
        ),
    ),
    (
        "allegro_teacher_train_revised",
        Path(
            "logs/benchmarks/family_teacher_distillation/allegro-tuning-20260911/"
            "frozen-acceptance-3c3f2046/configuration/asset_revision/training.canonical.lock.yaml"
        ),
    ),
    (
        "allegro_teacher_train_revised_pregrasp_parent",
        Path(
            "logs/benchmarks/family_teacher_distillation/allegro-tuning-20260911/"
            "mcp2-223-assets-v1/unified-certified-parents-v1/training.canonical.lock.yaml"
        ),
    ),
)

# 该 mapping 只把 Allegro joint-limit parent/child 绑定为同一逻辑 lineage，不比较两个 hash 域。
ALLEGRO_REVISION = Path(
    "logs/benchmarks/family_teacher_distillation/allegro-tuning-20260911/"
    "frozen-acceptance-3c3f2046/configuration/asset_revision/revision.json"
)

# 生产组必须是精确 typed group；mixed records 即便顶层 family 字段为 allegro/leap 也不能进入 pure strata。
# source manifest 中的 group 名是 family composition 的显式字段，不使用目录名模糊匹配。
PURE_GROUP_BY_FAMILY = {
    "leap": "single_palm_leap",
    "allegro": "single_palm_allegro",
}
# 每个 tuple 是 (输出名, family, N040 suite, base-qualification mode)。
STRATUM_ORDER = (
    ("leap_right_variant", "leap", "unseen_variant_set", "variant"),
    ("allegro_right_variant", "allegro", "unseen_variant_set", "variant"),
    ("leap_right_mother", "leap", "unseen_mother", "new_base"),
    ("allegro_right_mother", "allegro", "unseen_mother", "new_base"),
)


@dataclass(frozen=True)
class RevisionIdentity:
    r"""保存 Allegro 限位修订的 parent/child 对，不把新 ID 误解释为新 lineage。"""

    path: Path | None  # revision.json 的 source provenance；None 表示本次没有修订输入
    parent_by_child: Mapping[str, str]  # revised child ID -> 原始 parent ID
    child_by_parent: Mapping[str, str]  # 原始 parent ID -> revised child ID


@dataclass
class ExclusionIndex:
    r"""收集可用于 source-level 污染定位的身份与 lineage 索引。"""

    asset_ids: set[str] = field(default_factory=set)  # exact source asset IDs 的禁集
    content_hashes: set[str] = field(default_factory=set)  # source content identity 的禁集
    source_paths: set[str] = field(default_factory=set)  # 规范化 bundle path 的禁集
    locations_by_asset_id: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))  # ID -> 来源 labels
    locations_by_content_hash: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )  # content -> labels
    locations_by_source_path: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))  # path -> labels
    bases_by_label: dict[str, set[tuple[str, str]]] = field(
        default_factory=lambda: defaultdict(set)
    )  # label -> (family,base)
    member_counts: dict[str, int] = field(default_factory=dict)  # lock 文件声明的成员数


def _resolve_path(path: Path, root: Path) -> Path:
    r"""将相对输入固定到 AnyMani root，保留绝对 source path 的物理位置语义。"""

    candidate = path if path.is_absolute() else root / path  # 输入路径只影响读取位置，不改变记录内原始 path
    return candidate.expanduser().resolve(strict=False)


def _display_path(path: Path, root: Path) -> str:
    r"""优先输出 root-relative 路径，root 外的外部来源保留绝对路径。"""

    resolved = path.resolve(strict=False)  # stable path spelling用于计划文件，不读取或修改目标
    try:
        return resolved.relative_to(root.resolve(strict=False)).as_posix()  # 便于复现命令在同一仓库运行
    except ValueError:
        return str(resolved)  # 外部 source root 不能伪装成仓库内路径


def _sha256_file(path: Path) -> str:
    r"""分块计算源文件 SHA-256；只读取 hand.yaml/hand.urdf 等已物化元数据文件。"""

    digest = hashlib.sha256()  # 文件内容身份，不与 canonical physical identity 混用
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)  # 分块读取避免把资产文件整体复制到内存
    return digest.hexdigest()


def _stable_hash(parts: Iterable[str]) -> str:
    r"""对带长度边界的 source 字段求稳定 SHA，用于不依赖策略表现的确定性排序。"""

    normalized = tuple(str(part) for part in parts)  # 将所有 source 字段显式规范为字符串
    payload = json.dumps(normalized, ensure_ascii=True, separators=(",", ":"))  # 长度边界由 JSON tuple 保持
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()  # 只使用 source identity，不读 policy score


def _load_document(path: Path) -> Any:
    r"""优先按 canonical writer 的 JSON 读取，必要时再安全读取人工 YAML。"""

    raw = path.read_bytes()  # 保留原始文件不变；调用者另行计算文件级 SHA
    text = raw.decode("utf-8")  # 所有计划输入是 UTF-8 JSON/YAML metadata
    try:
        document = json.loads(text)  # canonical lock 虽以后缀 yaml 保存，实际内容通常是 JSON
        return document  # JSON writer 的字段类型原样保留
    except json.JSONDecodeError:
        document = yaml.safe_load(text)  # source/evaluation manifest 使用 YAML 语法
        if document is None:
            raise ValueError(f"empty metadata document: {path}")
        return document


def _mapping(document: Any, *, label: str) -> Mapping[str, Any]:
    r"""要求输入顶层是 mapping，避免把错误的列表/标量悄悄解释成空 cohort。"""

    valid = isinstance(document, Mapping)  # 先记录类型判定，让错误信息指向输入合同而非后续 KeyError
    if not valid:
        raise ValueError(f"{label} must be a mapping")
    return document


def _source_asset_path(record: Mapping[str, Any], root: Path) -> Path:
    r"""按 generated asset contract 恢复 mother 或 variant bundle 的原始 source path。"""

    mother_path = _resolve_path(Path(str(record["mother_path"])), root)  # provenance中的母体目录是 lineage 锚点
    role = str(record.get("asset_role", ""))  # role 决定 generated bundle 是母体目录还是 variant 子目录
    if role == "variant":
        variant_set = str(record["variant_set"])  # variant set 是同一母体下的独立生成批次
        asset_id = str(record["asset_id"])  # asset_id 是 variant bundle 的 content-addressed 目录名
        return mother_path / variant_set / asset_id  # variant asset 的独立 bundle
    return mother_path  # mother 自身直接位于母体目录，不能拼接虚构的 asset-id 子目录


def _source_file_hashes(asset_path: Path) -> dict[str, str]:
    r"""记录已存在 bundle 的 source 文件 SHA；不遍历 meshes 或启动任何几何解析。"""

    hashes: dict[str, str] = {}  # 缺失文件保持空映射，由下游把它视为 source materialization gap
    source_filenames = ("hand.yaml", "hand.urdf")  # 不读 meshes，避免把 source map 变成几何遍历任务
    for filename in source_filenames:
        path = asset_path / filename
        if path.is_file():
            hashes[filename] = _sha256_file(path)  # hand sidecar 与 URDF 是 source-level 可复核证据
    return hashes


def _family_for_group(group_name: str) -> str | None:
    r"""只承认两个 pure production group；mixed group 永远返回 None。"""

    for family, expected in PURE_GROUP_BY_FAMILY.items():  # 只走两个显式 production group 分支
        if group_name == expected:
            return family
    return None


def _train_base_designs(train_manifest: Mapping[str, Any], family: str) -> set[str]:
    r"""读取 source manifest 的 train section，返回指定 family/right 的 base design 名称。"""

    train = train_manifest.get("train", {})  # N040 manifest 中 train 是唯一参数更新暴露分区
    runs = train.get("runs", {}) if isinstance(train, Mapping) else {}  # source manifest 的 run namespace
    default_run = runs.get("default", {}) if isinstance(runs, Mapping) else {}  # 当前 N040 只消费 default run
    groups = default_run.get("groups", {}) if isinstance(default_run, Mapping) else {}  # 取 typed production groups
    pure_group = groups.get(PURE_GROUP_BY_FAMILY[family], {}) if isinstance(groups, Mapping) else {}  # 排除 mixed group
    if not isinstance(pure_group, Mapping):
        return set()  # 输入缺少该 family 时显式产生空资格集，调用者会报告 pool gap
    return {
        str(name) for name in pure_group if str(name).startswith("right_")
    }  # handedness是字段前缀的 source contract


def _eval_records(
    eval_manifest: Mapping[str, Any], family: str, suite: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    r"""按 typed fields 提取 pure-right records，并同时返回整个 suite 的审计计数。"""

    evaluation = eval_manifest.get("evaluation", {})  # N040 final manifest 的 held-out 角色入口
    suite_entries = evaluation.get(suite, []) if isinstance(evaluation, Mapping) else []  # 只展开请求的一个 suite
    if not isinstance(suite_entries, list):
        raise ValueError(f"evaluation.{suite} must be a list")

    all_collection = Counter(  # 完整 suite 计数用于说明 mixed/left 为何没有进入 pure strata
        str(entry.get("collection_kind", "")) for entry in suite_entries if isinstance(entry, Mapping)
    )
    all_family_hand = Counter(  # 顶层 family 计数不能替代 collection_kind/group typed filter
        (str(entry.get("family", "")), str(entry.get("handedness", "")))
        for entry in suite_entries
        if isinstance(entry, Mapping)
    )
    pure_group = PURE_GROUP_BY_FAMILY[family]
    selected: list[dict[str, Any]] = []  # 这里只保留 source records，不读取任何 evaluation metrics
    for origin_row, raw in enumerate(suite_entries):  # suite list index is the reproducible origin source row
        if not isinstance(raw, Mapping):
            continue  # malformed rows are excluded from a source-level candidate pool
        if (  # 四个 typed identity 条件共同定义 pure-right evaluation candidate
            raw.get("family") == family
            and raw.get("handedness") == "right"
            and raw.get("collection_kind") == "groups"
            and raw.get("group_name") == pure_group
            and raw.get("partition") == f"evaluation.{suite}"
            and raw.get("source_kind") == "generated"
        ):
            selected_record = dict(raw)  # 保留原始 partition/path/base/variant/hash 字段供 source map 使用
            selected_record["__origin_source_row"] = origin_row  # 外部注册时绑定原 eval manifest 坐标
            selected.append(selected_record)
    stats = {  # 既报告完整 suite，也报告本 family/right 的真实 source pool
        "suite_total": len(suite_entries),
        "suite_collection_kind_counts": dict(sorted(all_collection.items())),
        "suite_family_handedness_counts": {
            f"{fam}|{hand}": count for (fam, hand), count in sorted(all_family_hand.items())
        },
        "typed_pure_right_count": len(selected),
    }
    return selected, stats  # caller随后执行 teacher/train/distill source-level exclusion


def _record_source_path(record: Mapping[str, Any], root: Path) -> str:
    r"""返回规范化 source asset path，用于跨 manifest 的同对象定位。"""

    return str(_source_asset_path(record, root))  # 路径只用于身份索引，不对 bundle 做递归扫描


def _lock_source_path(member: Mapping[str, Any], root: Path) -> str | None:
    r"""从 canonical lock 的 provenance 重建对应 source bundle path。"""

    provenance = member.get("provenance", {})  # canonical lock 仍保留原 source lineage provenance
    if not isinstance(provenance, Mapping) or not provenance.get("mother_path"):
        return None  # 极旧 lock 可能没有完整 provenance，此时仍可用 asset/content ID 定位
    record = {  # 复用同一 generated path contract，定位 lock member 的 source bundle
        "mother_path": provenance["mother_path"],
        "asset_role": provenance.get("asset_role", ""),
        "variant_set": provenance.get("variant_set", ""),
        "asset_id": member.get("asset_id", ""),
    }
    try:  # 旧 lock 若缺 variant/source 字段时只报告 ID/content 身份，不猜测路径
        return _record_source_path(record, root)  # 与 evaluation record 使用同一 generated path 规则
    except (KeyError, TypeError):
        return None


def _add_exclusion_member(index: ExclusionIndex, label: str, member: Mapping[str, Any], root: Path) -> None:
    r"""登记 teacher/distill member 的 source identity、path 和 base lineage。"""

    asset_id = str(member.get("asset_id", ""))  # asset_id 是跨 hash 域定位同一 source object 的主键
    content_hash = str(member.get("content_hash", ""))  # content identity 用于跨 alias 的重复检测
    provenance = member.get("provenance", {})
    group_name = (
        str(provenance.get("group_name", "")) if isinstance(provenance, Mapping) else ""
    )  # exact group推导family
    family = _family_for_group(group_name)
    base = str(provenance.get("mother_name", "")) if isinstance(provenance, Mapping) else ""
    if asset_id:
        index.asset_ids.add(asset_id)
        index.locations_by_asset_id[asset_id].add(label)  # 污染报告保留命中来源 label
    if content_hash:
        index.content_hashes.add(content_hash)
        index.locations_by_content_hash[content_hash].add(label)  # 不把 content hash 当 canonical physical hash
    source_path = _lock_source_path(member, root)
    if source_path:
        index.source_paths.add(source_path)
        index.locations_by_source_path[source_path].add(label)  # same source bundle 的精确路径定位
    if family is not None and base:
        index.bases_by_label[label].add((family, base))  # base 资格与 exact asset 排除保持两个层级


def _load_lock(index: ExclusionIndex, label: str, path: Path, root: Path) -> dict[str, Any]:
    r"""读取一个 canonical/source lock，并返回不含策略结果的文件摘要。"""

    document = _mapping(_load_document(path), label=label)  # lock 内容可能是 JSON-with-yaml-suffix
    members = document.get("members", [])  # canonical/source lock 的唯一成员轴
    if not isinstance(members, list):
        raise ValueError(f"{label} members must be a list")
    for member in members:  # 只读每个 member 的 identity/provenance，不碰 policy evidence
        if isinstance(member, Mapping):
            _add_exclusion_member(index, label, member, root)  # 只登记 identity/provenance，不读取 trajectory
    index.member_counts[label] = len(members)  # 用于输出“完整 train/dev/final lock”核对
    return {  # lock file digest绑定后续复核输入
        "label": label,
        "path": _display_path(path, root),
        "sha256": _sha256_file(path),
        "member_count": len(members),
        "cohort_id": str(document.get("cohort_id", "")),
    }


def _load_distill_exclusion(index: ExclusionIndex, path: Path, root: Path, ordinal: int) -> dict[str, Any]:
    r"""读取可选 distill/source lock；当前 case 没有该文件时不会凭空制造 exclusion。"""

    label = f"distill_exclusion_{ordinal:02d}"  # 稳定 label 使污染位置可复现
    return _load_lock(index, label, path, root)


def _load_revision(
    path: Path | None, root: Path, index: ExclusionIndex
) -> tuple[RevisionIdentity, dict[str, Any] | None]:
    r"""登记 Allegro 修订 parent/child ID 映射，避免把限位修订后的 hash/ID 当作新资产。"""

    if path is None:
        return RevisionIdentity(None, {}, {}), None  # 测试或外部 case 可显式表示没有修订链
    document = _mapping(_load_document(path), label="allegro_revision")  # revision JSON 是 lineage 证据，不是新 cohort
    mapping = document.get("mapping", [])  # 仅 modified parent/child 对，unchanged 由 lock 保留
    if not isinstance(mapping, list):
        raise ValueError("Allegro revision mapping must be a list")
    parent_by_child: dict[str, str] = {}  # child asset ID -> pre-revision parent asset ID
    child_by_parent: dict[str, str] = {}  # parent asset ID -> revised child asset ID
    for row in mapping:  # 每一对仅用于同一逻辑资产的双向索引
        if not isinstance(row, Mapping):
            continue
        parent = str(row.get("parent_asset_id", ""))  # 修订前 source identity
        child = str(row.get("asset_id", ""))  # 修订后 bundle identity
        if not parent or not child:
            raise ValueError("Allegro revision mapping requires parent_asset_id and asset_id")
        if child in parent_by_child and parent_by_child[child] != parent:
            raise ValueError(f"duplicate revised child identity: {child}")
        if parent in child_by_parent and child_by_parent[parent] != child:
            raise ValueError(f"duplicate revision parent identity: {parent}")
        parent_by_child[child] = parent  # candidate 命中 child 时可恢复 pre-revision ID
        child_by_parent[parent] = child  # candidate 命中 parent 时可定位 revised child
        index.asset_ids.update((parent, child))  # 两个 ID 属于同一逻辑 lineage，均进入污染定位禁集
        index.locations_by_asset_id[parent].add("allegro_revision_parent")
        index.locations_by_asset_id[child].add("allegro_revision_child")
    summary = {  # summary 只描述修订规模，不把 child hash 当新的实验样本
        "path": _display_path(path, root),
        "sha256": _sha256_file(path),
        "parent_lock": document.get("parent_lock"),
        "parent_lock_sha256": document.get("parent_lock_sha256"),
        "source_lock": document.get("source_lock"),
        "source_lock_sha256": document.get("source_lock_sha256"),
        "declared_assets": int(document.get("assets", 0)),
        "modified_assets": int(document.get("modified_assets", len(mapping))),
        "unchanged_assets": int(document.get("unchanged_assets", 0)),
        "mapping_count": len(mapping),
        "lineage_policy": "parent-and-revised-child-IDs-are-one-logical-asset",
    }
    return RevisionIdentity(path, parent_by_child, child_by_parent), summary


def _load_train_asset_ids(path: Path | None) -> tuple[set[str], dict[str, Any]]:
    r"""读取 N040 source_artifacts 的 stable asset IDs，不把 input fingerprint冒充 canonical physical hash。"""

    if path is None:
        return set(), {"status": "not-provided"}  # 缺少 train artifact 时污染状态必须显式未知
    rows = path.read_text(encoding="utf-8").splitlines()  # JSONL 是 source artifact index，不是模型输出
    asset_ids: set[str] = set()  # 每个 train asset 在 source_artifacts 中有16个 bank-index realization rows
    row_count = 0  # 保留物化行数，检查16×asset轴是否一致
    for line in rows:  # 逐行解析以避免读取任何二进制 checkpoint
        if not line.strip():
            continue
        row_count += 1
        document = _mapping(json.loads(line), label="N040 source artifact row")  # source artifact 字段是 JSON
        asset_id = str(document.get("asset_id", ""))  # 仅 asset_id 用来判断 train exposure
        if asset_id:
            asset_ids.add(asset_id)
    return asset_ids, {  # 不暴露 input_fingerprint 为 canonical physical identity
        "status": "source-id-indexed",
        "row_count": row_count,
        "unique_asset_id_count": len(asset_ids),
        "path": str(path),
    }


def _teacher_base_sets(index: ExclusionIndex) -> dict[str, dict[str, set[str]]]:
    r"""从 label 约定中恢复 family→train/dev/final base sets，保持资格与 exact exclusion 分离。"""

    result: dict[str, dict[str, set[str]]] = {  # 训练/开发/终验 base 分开保存，支持不同资格规则
        family: {"train": set(), "dev": set(), "final": set()} for family in PURE_GROUP_BY_FAMILY
    }
    for label, pairs in index.bases_by_label.items():  # labels 来自默认/CLI lock 注册顺序
        role = "final" if "final" in label else "dev" if "dev" in label else "train" if "train" in label else None
        if role is None:
            continue
        for family, base in pairs:  # 一个 lock 可覆盖多个 family/base，按 provenance 逐项归类
            result[family][role].add(base)  # old/revised train label并集仍代表同一训练 lineage
    return result


def _source_sort_key(record: Mapping[str, Any]) -> str:
    r"""构造只依赖 source identity 的确定性记录顺序。"""

    parts = (
        str(record.get("family", "")),
        str(record.get("handedness", "")),
        str(record.get("mother_path", "")),
        str(record.get("mother_name", "")),
        str(record.get("variant_set", "")),
        str(record.get("asset_id", "")),
        str(record.get("content_hash", "")),
    )  # source-level排序键完整保留 family/handedness/lineage/variant/source identity
    return _stable_hash(parts)  # 不读 policy metric，不读 trajectory


def _base_sort_key(base: str, records: Sequence[Mapping[str, Any]]) -> str:
    r"""用 lineage path/base 名称排序，不按任何策略表现或物理质量排序。"""

    mother_path = str(records[0].get("mother_path", "")) if records else ""  # 同一 base 必须共享 mother path
    return _stable_hash((base, mother_path))


def _source_pollution_reasons(
    record: Mapping[str, Any],
    *,
    root: Path,
    n040_train_ids: set[str],
    exclusions: ExclusionIndex,
) -> list[str]:
    r"""只检查同对象 source identity/path/content；绝不跨 hash 域比较 physical_geometry_hash。"""

    reasons: list[str] = []  # 每个 reason 同时作为后续 source map 的污染定位标签
    asset_id = str(record.get("asset_id", ""))  # source asset 主键
    content_hash = str(record.get("content_hash", ""))  # source content 主键
    source_path = _record_source_path(record, root)  # bundle 路径是第三种同对象定位证据
    if asset_id in n040_train_ids:
        reasons.append("n040_train_asset_id")  # train exposure 的稳定 source 坐标命中
    if asset_id in exclusions.asset_ids:
        reasons.extend(f"asset_id:{label}" for label in sorted(exclusions.locations_by_asset_id.get(asset_id, ())))
    if content_hash and content_hash in exclusions.content_hashes:
        reasons.extend(f"content_hash:{label}" for label in sorted(exclusions.locations_by_content_hash[content_hash]))
    if source_path in exclusions.source_paths:
        reasons.extend(f"source_path:{label}" for label in sorted(exclusions.locations_by_source_path[source_path]))
    return sorted(set(reasons))  # 同一候选可能同时命中多类 identity，报告中保留全部原因


def _base_reasons(
    record: Mapping[str, Any],
    *,
    family: str,
    kind: str,
    ssl_train_bases: set[str],
    teacher_bases: Mapping[str, Mapping[str, set[str]]],
) -> list[str]:
    r"""检查 variant/new-base 的 base-design 资格，不把同 base 的不同 variant误当 source duplicate。"""

    base = str(record.get("mother_name", ""))  # base design 是母体 topology 名，不使用 path substring
    reasons: list[str] = []  # base 条件和 exact asset 污染条件需要在输出中分别可见
    if kind == "variant":
        if base not in teacher_bases[family]["train"]:
            reasons.append("base_not_in_teacher_train")  # policy-unseen variant 必须已有 teacher base design
    else:
        if base in ssl_train_bases:
            reasons.append("base_in_n040_ssl_train")  # 新基型不能在 SSL train 中出现
        if base in teacher_bases[family]["train"]:
            reasons.append("base_in_teacher_train")  # 新基型不能在 teacher train 中出现
        if base in teacher_bases[family]["dev"]:
            reasons.append("base_in_teacher_dev")  # 新基型不能在 teacher dev 中出现
    return reasons


def _revision_source_identity(record: Mapping[str, Any], revision: RevisionIdentity, family: str) -> dict[str, Any]:
    r"""把 candidate 映射到修订前 source identity，明确 outside-revision 与 parent/child 状态。"""

    asset_id = str(record.get("asset_id", ""))  # 先绑定原始 source ID，再决定是否位于 revision map
    if family != "allegro" or revision.path is None:
        return {
            "status": "not-applicable" if family != "allegro" else "revision-input-not-provided",
            "pre_revision_asset_id": asset_id,
            "revised_asset_id": None,
        }
    if asset_id in revision.child_by_parent:
        return {
            "status": "parent-id-in-revision-map",
            "pre_revision_asset_id": asset_id,
            "revised_asset_id": revision.child_by_parent[asset_id],
        }
    if asset_id in revision.parent_by_child:
        return {
            "status": "revised-child-id-in-revision-map",
            "pre_revision_asset_id": revision.parent_by_child[asset_id],
            "revised_asset_id": asset_id,
        }
    return {
        "status": "outside-allegro-training-revision",
        "pre_revision_asset_id": asset_id,
        "revised_asset_id": None,
    }


def _source_map(
    record: Mapping[str, Any],
    *,
    root: Path,
    eval_manifest_path: Path,
    stratum: str,
    role: str,
    base_rank: int,
    revision: RevisionIdentity,
    reasons: Sequence[str] = (),
) -> dict[str, Any]:
    r"""序列化一个 selected/reserve/excluded source member，不添加伪 canonical identity。"""

    source_path = _source_asset_path(record, root)  # 保留真实 bundle 路径，便于后续 source lock 注册
    source_files = _source_file_hashes(source_path)  # 只读两个文本 source 文件，缺失时保留空映射
    source_record = {
        key: value for key, value in record.items() if key != "__origin_source_row"
    }  # 不污染原始 manifest字段
    origin_row = record.get("__origin_source_row")  # suite内的原始行号，后续写入 source-lock.selection
    return {
        "asset_id": str(record.get("asset_id", "")),
        "source_record": source_record,  # 原始 N040 evaluation fields完整保留，包括 partition 与 source hashes
        "source_manifest": {
            "path": _display_path(eval_manifest_path, root),
            "sha256": _sha256_file(eval_manifest_path),
        },
        "origin_source": {
            "partition": source_record.get("partition"),
            "manifest_row": origin_row,
            "asset_id": source_record.get("asset_id"),
            "mother_path": source_record.get("mother_path"),
            "variant_set": source_record.get("variant_set", ""),
        },
        "source_asset": {
            "path": _display_path(source_path, root),
            "source_file_sha256": source_files,  # 只含已存在 hand.yaml/hand.urdf
        },
        "pre_revision_source_identity": _revision_source_identity(record, revision, str(record.get("family", ""))),
        "canonical_physical_identity": {
            "status": "pending-canonical",
            "source_manifest_physical_geometry_hash": record.get("physical_geometry_hash"),
            "canonical_physical_geometry_hash": None,
        },
        "selection": {
            "stratum": stratum,
            "role": role,
            "base_rank": base_rank,
            "source_sort_key": _source_sort_key(record),
            "exclusion_reasons": list(reasons),
        },
    }


def _select_balanced(
    records: Sequence[Mapping[str, Any]],
    *,
    family: str,
    kind: str,
    target_count: int,
    new_base: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int], list[str]]:
    r"""按 base hash 均衡选取目标数量，并返回 selected/reserve、quotas 与 base 顺序。"""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)  # mother_name 是 source lineage 的稳定组键
    for record in records:  # 候选先按 base 聚合，再执行均衡名额
        grouped[str(record.get("mother_name", ""))].append(dict(record))
    for base, group in grouped.items():  # 每个 lineage 内独立排序，不把不同 base 的 source 混排
        if new_base:
            group.sort(key=lambda row: (0 if row.get("asset_role") == "mother" else 1, _source_sort_key(row)))
        else:
            group.sort(key=_source_sort_key)  # variant层没有 mother record，全部按 source hash 排序
    base_order = sorted(grouped, key=lambda base: _base_sort_key(base, grouped[base]))  # base hash固定顺序
    base_count = len(base_order)  # 配额分母是实际合格 base 数，不把被排除 base算入
    if base_count == 0:
        return [], [], {}, []  # 上层仍会输出明确的 source-pool-insufficient 状态
    quotient, remainder = divmod(target_count, base_count)  # floor/ceil 配额保证最大差一
    quotas = {base: quotient + int(index < remainder) for index, base in enumerate(base_order)}  # 前 remainder 个多一项
    selected_by_base: dict[str, list[dict[str, Any]]] = {base: [] for base in base_order}
    # 每一轮跨 base 取一项，保证同一目标数量下不会先填满一个 lineage。
    for _round_index in range(max(quotas.values(), default=0)):  # round-robin 轮次只由 source 配额决定
        for base in base_order:
            if len(selected_by_base[base]) < quotas[base] and len(selected_by_base[base]) < len(grouped[base]):
                selected_by_base[base].append(grouped[base][len(selected_by_base[base])])
    selected = [
        selected_by_base[base][round_index]
        for round_index in range(max(quotas.values(), default=0))
        for base in base_order
        if round_index < len(selected_by_base[base])
    ]  # 按轮次交错 base，避免输出顺序把某一 lineage 误看成优先暴露
    selected_ids = {str(row.get("asset_id", "")) for row in selected}  # reserve 不能重复 selected asset
    reserve = [row for base in base_order for row in grouped[base] if str(row.get("asset_id", "")) not in selected_ids]
    return selected, reserve, quotas, base_order


def _resource_summary(path: Path, root: Path, *, kind: str) -> dict[str, Any]:
    r"""只读摘要现有 canonical lock 或 strict catalog，不解析策略分数。"""

    result: dict[str, Any] = {"path": _display_path(path, root), "exists": path.is_file(), "kind": kind}
    if not path.is_file():
        return result  # 缺失开发产物显式保留，避免把空路径解释为可复用
    result["sha256"] = _sha256_file(path)
    document = _load_document(path)
    if kind == "canonical_lock" and isinstance(document, Mapping):
        result["member_count"] = (
            len(document.get("members", [])) if isinstance(document.get("members", []), list) else None
        )
        result["cohort_id"] = document.get("cohort_id")
    if kind == "catalog_index" and isinstance(document, Mapping):
        result["entry_count"] = (
            len(document.get("entries", [])) if isinstance(document.get("entries", []), list) else None
        )
        result["artifact_type"] = document.get("artifact_type")
        result["schema_version"] = document.get("schema_version")
    return result


def _development_resources(root: Path) -> dict[str, Any]:
    r"""登记 policy development 的可复用 lock/catalog，并区分 Allegro 旧物理与修订物理。"""

    leap_dev64 = (
        root
        / "source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1/cohorts/leap-right-32x2-development-20260907.canonical.lock.yaml"
    )
    leap_ready63 = (
        root
        / "source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1/cohorts/leap-right-development-ready63-20260907.canonical.lock.yaml"
    )
    leap_catalog = (
        root / "outputs/pregrasp/catalogs/heterogeneous_rotation/strict-v1/dexcube/scale-1p1-unseen-20260907/index.json"
    )
    allegro_old_dev = (
        root
        / "logs/benchmarks/cross_family_rotation/cohort-preparation-20260907/released/allegro-v1/development.canonical.lock.yaml"
    )
    allegro_old_catalog = (
        root
        / "logs/benchmarks/cross_family_rotation/cohort-preparation-20260907/released/allegro-v1/catalogs/development/index.json"
    )
    allegro_revised_train = (
        root
        / "logs/benchmarks/family_teacher_distillation/allegro-tuning-20260911/frozen-acceptance-3c3f2046/configuration/asset_revision/training.canonical.lock.yaml"
    )
    allegro_revised_catalog = (
        root
        / "logs/benchmarks/family_teacher_distillation/allegro-tuning-20260911/frozen-acceptance-3c3f2046/configuration/pregrasp/catalog/index.json"
    )
    return {
        "leap": {
            "development_lock_64": _resource_summary(leap_dev64, root, kind="canonical_lock"),
            "ready_view_lock_63": _resource_summary(leap_ready63, root, kind="canonical_lock"),
            "combined_unseen_catalog": _resource_summary(leap_catalog, root, kind="catalog_index"),
            "reuse_note": "catalog index contains dev-ready63 plus final-ready127; isolate by exact lock key",
        },
        "allegro": {
            "old_development_lock_64": _resource_summary(allegro_old_dev, root, kind="canonical_lock"),
            "old_development_catalog_64": _resource_summary(allegro_old_catalog, root, kind="catalog_index"),
            "revised_teacher_training_lock": _resource_summary(allegro_revised_train, root, kind="canonical_lock"),
            "revised_teacher_training_catalog": _resource_summary(allegro_revised_catalog, root, kind="catalog_index"),
            "revised_development_status": "no revised development canonical lock/catalog found; primary must revalidate under MCP2 rule",
        },
    }


def _stratum_plan(
    *,
    stratum: str,
    family: str,
    suite: str,
    kind: str,
    candidate_records: Sequence[Mapping[str, Any]],
    root: Path,
    eval_manifest_path: Path,
    n040_train_ids: set[str],
    exclusions: ExclusionIndex,
    ssl_train_bases: set[str],
    teacher_bases: Mapping[str, Mapping[str, set[str]]],
    revision: RevisionIdentity,
    target_count: int,
) -> dict[str, Any]:
    r"""构造单个 family×suite 的 source-level selection plan。"""

    eligible: list[dict[str, Any]] = []  # 同时满足 exact source identity 与 base-design 资格的候选
    excluded: list[tuple[dict[str, Any], list[str]]] = []  # 记录每个拒绝位置，方便后续 source 审计
    for raw in candidate_records:
        pollution = _source_pollution_reasons(raw, root=root, n040_train_ids=n040_train_ids, exclusions=exclusions)
        base_reasons = _base_reasons(
            raw,
            family=family,
            kind=kind,
            ssl_train_bases=ssl_train_bases,
            teacher_bases=teacher_bases,
        )
        reasons = sorted(set((*pollution, *base_reasons)))
        if reasons:
            excluded.append((dict(raw), reasons))  # 污染/资格失败都不进入 reserve，避免误称未见
        else:
            eligible.append(dict(raw))
    selected, reserve, quotas, base_order = _select_balanced(
        eligible,
        family=family,
        kind=kind,
        target_count=target_count,
        new_base=kind == "new_base",
    )
    selected_maps = [
        _source_map(
            row,
            root=root,
            eval_manifest_path=eval_manifest_path,
            stratum=stratum,
            role="selected",
            base_rank=base_order.index(str(row.get("mother_name", ""))),
            revision=revision,
        )
        for row in selected
    ]
    reserve_maps = [
        _source_map(
            row,
            root=root,
            eval_manifest_path=eval_manifest_path,
            stratum=stratum,
            role="reserve",
            base_rank=base_order.index(str(row.get("mother_name", ""))),
            revision=revision,
        )
        for row in reserve
    ]
    excluded_maps = [
        _source_map(
            row,
            root=root,
            eval_manifest_path=eval_manifest_path,
            stratum=stratum,
            role="excluded",
            base_rank=base_order.index(str(row.get("mother_name", "")))
            if str(row.get("mother_name", "")) in base_order
            else -1,
            revision=revision,
            reasons=reasons,
        )
        for row, reasons in excluded
    ]
    selected_by_base = Counter(str(row.get("mother_name", "")) for row in selected)
    return {
        "family": family,
        "handedness": "right",
        "source_suite": f"evaluation.{suite}",
        "base_rule": (
            "base-in-teacher-train; candidate asset/source identity not in N040 train, teacher locks or distill"
            if kind == "variant"
            else "base absent from N040 train and teacher train/dev; candidate asset/source identity excluded by exact source checks"
        ),
        "target_count": target_count,
        "source_pool_count": len(candidate_records),
        "eligible_pool_count": len(eligible),
        "eligible_base_count": len({str(row.get("mother_name", "")) for row in eligible}),
        "mother_role_count_in_source_pool": sum(row.get("asset_role") == "mother" for row in candidate_records),
        "selection_status": "source-level-ready" if len(eligible) >= target_count else "source-pool-insufficient",
        "base_order": base_order,
        "base_quotas": quotas,
        "selected_count": len(selected_maps),
        "selected_by_base": dict(sorted(selected_by_base.items())),
        "reserve_count": len(reserve_maps),
        "excluded_count": len(excluded_maps),
        "selected": selected_maps,
        "reserve": reserve_maps,
        "excluded": excluded_maps,
    }


def build_source_selection_plan(
    *,
    repo_root: Path = REPO_ROOT,
    ssl_train_manifest: Path = N040_TRAIN_MANIFEST,
    ssl_eval_manifest: Path = N040_EVAL_MANIFEST,
    ssl_train_artifacts: Path | None = N040_TRAIN_ARTIFACTS,
    teacher_locks: Sequence[tuple[str, Path]] = DEFAULT_TEACHER_LOCKS,
    allegro_revision: Path | None = ALLEGRO_REVISION,
    distill_exclusions: Sequence[Path] = (),
    target_count: int = 32,
) -> dict[str, Any]:
    r"""读取固定输入并返回可审阅的四-strata source-level plan。

    该函数是纯文件读取和内存选择逻辑。它不导入 AnyMani/Isaac runtime，因此可在默认 contract pytest
    中运行；真正 canonical lowering、physical collision compare 和 strict pregrasp 由后续拥有物理权限的
    primary 任务执行。

    Args:
        repo_root: AnyMani 根目录；测试可传入临时 source fixture 根目录。
        ssl_train_manifest: 最终 N040 train source manifest，通常是 ``ssl.yaml``。
        ssl_eval_manifest: 最终 extended512 evaluation asset manifest，含两个 held-out suites。
        ssl_train_artifacts: N040 train source artifact JSONL；缺失时 train pollution 状态保持未知。
        teacher_locks: ``(label, path)`` 序列；label 中的 train/dev/final 用于 base 资格分层。
        allegro_revision: Allegro parent/child revision mapping；没有修订时可传 ``None``。
        distill_exclusions: 已产生的 distill source/cohort lock；当前 case 尚为空序列。
        target_count: 每个 stratum 的逻辑成员目标，正式计划为 32。

    Returns:
        dict[str, Any]: schema-1.0 source-selection plan，``status`` 固定为 ``pending-canonical``。

    Raises:
        ValueError: 输入 schema、typed fields 或 target_count 不符合选择合同。
        FileNotFoundError: 必需 source/lock/manifest 不存在。
    """

    if target_count < 1:
        raise ValueError("target_count must be positive")
    root = repo_root.expanduser().resolve(strict=True)  # 所有默认相对路径以唯一仓库 root 解释
    train_manifest_path = _resolve_path(ssl_train_manifest, root)
    eval_manifest_path = _resolve_path(ssl_eval_manifest, root)
    if not train_manifest_path.is_file() or not eval_manifest_path.is_file():
        raise FileNotFoundError("N040 train/evaluation manifest is missing")
    train_manifest = _mapping(_load_document(train_manifest_path), label="N040 train manifest")
    eval_manifest = _mapping(_load_document(eval_manifest_path), label="N040 evaluation manifest")
    n040_train_ids, train_artifact_summary = _load_train_asset_ids(
        _resolve_path(ssl_train_artifacts, root) if ssl_train_artifacts is not None else None
    )
    exclusions = ExclusionIndex()  # teacher、revision、distill 全部落入同一 source-level 禁集
    teacher_summaries = []
    for label, relative_path in teacher_locks:
        path = _resolve_path(relative_path, root)
        if not path.is_file():
            raise FileNotFoundError(f"teacher lock is missing: {label}: {path}")
        teacher_summaries.append(_load_lock(exclusions, label, path, root))
    revision_path = _resolve_path(allegro_revision, root) if allegro_revision is not None else None
    revision, revision_summary = _load_revision(revision_path, root, exclusions)
    distill_summaries = [
        _load_distill_exclusion(exclusions, _resolve_path(path, root), root, index)
        for index, path in enumerate(distill_exclusions)
    ]
    teacher_bases = _teacher_base_sets(exclusions)  # 用 label 派生 train/dev/final base 集合
    ssl_train_bases = {family: _train_base_designs(train_manifest, family) for family in PURE_GROUP_BY_FAMILY}
    eval_manifest_sha = _sha256_file(eval_manifest_path)
    train_manifest_sha = _sha256_file(train_manifest_path)
    stratum_plans: dict[str, Any] = {}
    raw_suite_stats: dict[str, Any] = {}
    for stratum, family, suite, kind in STRATUM_ORDER:
        records, stats = _eval_records(eval_manifest, family, suite)
        raw_suite_stats[stratum] = stats  # 同时保留完整suite 1024与typed pure-right 128的差异
        stratum_plans[stratum] = _stratum_plan(
            stratum=stratum,
            family=family,
            suite=suite,
            kind=kind,
            candidate_records=records,
            root=root,
            eval_manifest_path=eval_manifest_path,
            n040_train_ids=n040_train_ids,
            exclusions=exclusions,
            ssl_train_bases=ssl_train_bases[family],
            teacher_bases=teacher_bases,
            revision=revision,
            target_count=target_count,
        )
    # source-only污染摘要只汇总位置，不声称跨 hash 域的 physical no-overlap。
    pollution_locations = {
        stratum: [
            {
                "asset_id": member["asset_id"],
                "reasons": member["selection"]["exclusion_reasons"],
                "source_path": member["source_asset"]["path"],
            }
            for member in details["excluded"]
            if member["selection"]["exclusion_reasons"]
        ]
        for stratum, details in stratum_plans.items()
    }
    plan: dict[str, Any] = {
        "artifact_type": "anymani.family_teacher_distillation.evaluation_cohort_source_selection",
        "schema_version": "1.0.0",
        "status": "pending-canonical",
        "policy_results_read": False,  # 选择阶段禁止打开 evaluation.json/gate.json/trajectory 结果
        "selection_algorithm": {
            "name": "typed-base-balanced-source-hash-round-robin-v1",
            "target_count_per_stratum": target_count,
            "variant_rule": "base must be in family teacher train; candidate exact asset/source identity remains held out",
            "new_base_rule": "base absent from N040 train and family teacher train/dev; mother first within each base",
            "allegro_variant_quota_rule": "floor/ceil base quotas; difference at most one",
            "source_sort_fields": [
                "family",
                "handedness",
                "mother_path",
                "mother_name",
                "variant_set",
                "asset_id",
                "content_hash",
            ],
            "policy_metric_sort": False,
        },
        "inputs": {
            "n040_train_manifest": {
                "path": _display_path(train_manifest_path, root),
                "sha256": train_manifest_sha,
                "source_schema_version": train_manifest.get("schema_version"),
                "train_base_design_counts": {family: len(bases) for family, bases in sorted(ssl_train_bases.items())},
            },
            "n040_extended512_eval_manifest": {
                "path": _display_path(eval_manifest_path, root),
                "sha256": eval_manifest_sha,
                "source_schema_version": eval_manifest.get("schema_version"),
            },
            "n040_train_source_artifacts": {
                **train_artifact_summary,
                "path": (
                    _display_path(_resolve_path(ssl_train_artifacts, root), root)
                    if ssl_train_artifacts is not None
                    else None
                ),
                "sha256": (
                    _sha256_file(_resolve_path(ssl_train_artifacts, root))
                    if ssl_train_artifacts is not None and _resolve_path(ssl_train_artifacts, root).is_file()
                    else None
                ),
            },
            "teacher_locks": teacher_summaries,
            "distill_exclusions": distill_summaries,
            "allegro_revision": revision_summary,
        },
        "observed_source_counts": raw_suite_stats,
        "base_eligibility": {
            "ssl_train_right_base_counts": {family: len(bases) for family, bases in sorted(ssl_train_bases.items())},
            "teacher_train_base_counts": {
                family: len(teacher_bases[family]["train"]) for family in sorted(teacher_bases)
            },
            "teacher_dev_base_counts": {family: len(teacher_bases[family]["dev"]) for family in sorted(teacher_bases)},
            "teacher_final_base_counts": {
                family: len(teacher_bases[family]["final"]) for family in sorted(teacher_bases)
            },
        },
        "source_level_pollution_locator": {
            "n040_train_unique_asset_id_count": len(n040_train_ids),
            "teacher_and_revision_exclusion_asset_id_count": len(exclusions.asset_ids),
            "physical_hash_comparison": "not-performed-across-domains",
            "locations_by_stratum": pollution_locations,
        },
        "canonical_physical_isolation": {
            "status": "pending-canonical",
            "reason": "N040 source/provider physical hashes and canonical cohort physical hashes are different identity domains",
            "hash_domain_evidence": {
                "n040_train_identity": "ordered asset_id/content_hash axis; source_artifacts has no canonical physical_geometry_hash",
                "n040_eval_physical_field": "source/provider-domain field retained under source_record.physical_geometry_hash",
                "canonical_cohort_field": "canonical-lowering-domain physical_geometry_hash",
                "direct_cross_domain_set_intersection": "forbidden",
            },
            "source_checks_done": [
                "typed partition/group/family/handedness",
                "asset_id",
                "content_hash",
                "source path",
                "base lineage",
            ],
            "canonical_checks_required": [
                "materialize candidate source coordinates into a new canonical-final cohort",
                "lower candidate and all relevant SSL exposure with one physical identity definition",
                "reject duplicate canonical physical_geometry_hash and publish strict Top-8 catalog only after validation",
            ],
        },
        "strata": stratum_plans,
        "development_resources": _development_resources(root),
        "canonical_pregrasp_entry": {
            "source_selection_is_not_canonical_lock": True,
            "required_canonical_lock_schema": "1.2.0",
            "canonical_input_fields": [
                "strata[*].selected[*].source_record",
                "strata[*].selected[*].source_asset.path",
                "strata[*].selected[*].pre_revision_source_identity",
                "strata[*].selected[*].source_asset.source_file_sha256",
            ],
            "source_selection_to_canonical_adapter": (
                "scripts/research/family_teacher_distillation/materialize_evaluation_cohorts.py"
            ),
            "materialization_contract": {
                "source_registration_mode": "evaluation_only_as_technical_train_for_cohort_api",
                "canonicalize_flag": "--canonicalize",
                "full_exposure_audit_flag": "--audit-exposures",
                "audit_only_flag": "--audit-only",
                "strict_pregrasp_status_before_release": "pending-pregrasp",
            },
            "prepare_command_template": [
                "/home/hac/isaac/IsaacLab/isaaclab.sh",
                "-p",
                "scripts/research/prepare_cohort_pregrasp_shards.py",
                "--cohort-lock",
                "<new-candidate.canonical.lock.yaml>",
                "--output-dir",
                "<shared-student-20260913>/cohorts/<stratum>/pregrasp",
                "--shard-assets",
                "16",
                "--catalog",
                "<shared-student-20260913>/cohorts/<stratum>/catalog",
            ],
            "release_entry": "source/anymani/anymani/pregrasp/scripts/release_cohort.py",
            "physical_owner": "primary; no canonical lowering or strict pregrasp executed by this script",
        },
    }
    plan["selection_digest"] = _stable_hash(
        (json.dumps(plan, ensure_ascii=True, sort_keys=True, separators=(",", ":")),)
    )
    return plan


def _parse_teacher_locks(values: Sequence[str], root: Path) -> tuple[tuple[str, Path], ...]:
    r"""解析 CLI 的 ``LABEL=PATH``，保留用户显式提供的排除集合顺序。"""

    parsed: list[tuple[str, Path]] = []
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or not label.strip() or not raw_path.strip():
            raise ValueError(f"--teacher-lock requires LABEL=PATH: {value!r}")
        parsed.append((label.strip(), _resolve_path(Path(raw_path), root)))
    return tuple(parsed)


def main(argv: Sequence[str] | None = None) -> int:
    r"""执行 source-level plan；``--dry-run``只打印 JSON，永远不写输出文件。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--ssl-train-manifest", type=Path, default=N040_TRAIN_MANIFEST)
    parser.add_argument("--ssl-eval-manifest", type=Path, default=N040_EVAL_MANIFEST)
    parser.add_argument("--ssl-train-artifacts", type=Path, default=N040_TRAIN_ARTIFACTS)
    parser.add_argument(
        "--teacher-lock",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="override defaults with explicit teacher/source exclusion locks; may be repeated",
    )
    parser.add_argument("--allegro-revision", type=Path, default=ALLEGRO_REVISION)
    parser.add_argument("--distill-exclusion", type=Path, action="append", default=[])
    parser.add_argument("--target-count", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true", help="print source-level plan without writing any file")
    parser.add_argument("--output", type=Path, default=None, help="write a new plan JSON; never a canonical lock")
    args = parser.parse_args(argv)
    if args.dry_run and args.output is not None:
        parser.error("--dry-run and --output are mutually exclusive")
    root = args.repo_root.expanduser().resolve(strict=True)
    teacher_locks = _parse_teacher_locks(args.teacher_lock, root) if args.teacher_lock else DEFAULT_TEACHER_LOCKS
    plan = build_source_selection_plan(
        repo_root=root,
        ssl_train_manifest=args.ssl_train_manifest,
        ssl_eval_manifest=args.ssl_eval_manifest,
        ssl_train_artifacts=args.ssl_train_artifacts,
        teacher_locks=teacher_locks,
        allegro_revision=args.allegro_revision,
        distill_exclusions=args.distill_exclusion,
        target_count=args.target_count,
    )
    serialized = json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = _resolve_path(args.output, root)
        if output.exists():
            parser.error(f"output must be a new file: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")  # 只发布 source-selection plan，不写 lock/catalog
    print(serialized, end="")
    return 0 if all(details["selection_status"] == "source-level-ready" for details in plan["strata"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
