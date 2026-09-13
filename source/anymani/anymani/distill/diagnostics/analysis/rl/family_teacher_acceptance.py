r"""同一冻结家族教师的两次固定评价验收。

本模块保持两个明确边界：

* family_teacher.evaluate_teacher 是一次评价的唯一原始能力归约，负责
  30 秒、R16、确定性 Actor 均值、strict rank-0、ADR0、方向性与旧 103/29
  强门；本模块不重复实现 reward 或逐步轨迹审计。
* reduce_repeat_acceptance 只对两次已验证报告按 dataset_row 求一圈可靠交集，
  执行默认 86/128 的重复门。两圈交集只作描述，永远不能替代一圈交集。

正式 file wrapper 要求两份 evaluation 都发布 trajectory HDF5 和 stride-1、
带 reward terms 的 dense trace，并调用现有 palm_rotation_trace auditor。动作
物理权限由评价引用的 checkpoint 在 CPU 上读取：method identity、canonical
manifest、TIP-only policy、1/24 rad/策略步以及 retained N040 artifact 都必须
与显式 expected SHA 和 cohort 相符。读取权重文件只为核对 metadata，不构造
Actor、Isaac 环境或 GPU 上的张量。

直接执行：

python -B <本文件> evaluate <evaluation-1.json> <evaluation-2.json> <canonical.lock.yaml>
    --expected-method-identity-digest <digest> --expected-checkpoint-sha256 <sha>

分析函数不写文件；CLI 的 --output 使用排他创建模式。所有 source module
均按绝对文件路径 raw-load，避免 anymani 包初始化和 Isaac/Kit 注册。
"""

from __future__ import annotations

import argparse  # 只有一个纯文件 evaluate 子命令。
import copy  # 发布旧报告的副本，防止改写历史强门。
import hashlib  # 原始 JSON、checkpoint 与输出 envelope 的摘要。
import importlib.util  # raw-module 装载，绕开 anymani 包初始化。
import json  # evaluation、cohort 和 checkpoint identity 的 JSON-safe 部分。
import math  # 有限性与 1/24 rad/策略步核验。
import pickle  # 捕获torch序列化文件的明确反序列化错误。
import platform  # 只记录 CPU 解释器信息。
import stat  # 普通文件与 inode 去重。
from collections import Counter  # 32 拓扑的 4 代表归约。
from collections.abc import Mapping, Sequence  # 窄的 JSON/report 输入协议。
from pathlib import Path  # 所有路径均由调用者或 evaluation 显式提供。
from typing import Any, cast  # 外部证据边界的异构 JSON 值。


class AcceptanceError(ValueError):
    r"""重复验收的身份、协议、文件或统计证据不合法。"""


PHASE_CLOCK_CONTRACT: dict[str, Any] = {
    "source": "physical-episode-length-buf",
    "encoding": ["sin", "cos"],
    "encoding_dtype": "float32",
    "reset": "physical-episode-counter-zero-before-returned-observation",
    "transport_key": "phase_clock",
    "actor_adapter": "zero-linear2to128-after-contextual-joints-before-existing-head-norm",
    "critic_adapter": "zero-linear2to896-after-readout-before-existing-value-norm",
}


def _phase_clock_contract(value: Any, where: str) -> dict[str, Any]:
    r"""核对 checkpoint 中可选的 phase contract，而不导入 runtime 生成它。"""

    _require(isinstance(value, dict), f"{where} 必须是 JSON 对象")
    contract = dict(value)
    expected_keys = set(PHASE_CLOCK_CONTRACT) | {"period_policy_steps", "increment_rad_per_policy_step"}
    _require(
        set(contract) == expected_keys,
        f"{where} phase_clock contract keys与当前合同不一致",
    )
    period = contract["period_policy_steps"]
    _require(
        type(period) is int and period >= 2,
        f"{where}.period_policy_steps必须是>=2的整数",
    )
    increment = contract["increment_rad_per_policy_step"]
    _require(
        type(increment) in (int, float) and math.isfinite(float(increment)),
        f"{where}.increment_rad_per_policy_step必须有限",
    )
    _require(
        math.isclose(
            float(increment), 2.0 * math.pi / period, rel_tol=0.0, abs_tol=1.0e-12
        ),
        f"{where}.increment_rad_per_policy_step不匹配2*pi/{period}",
    )
    for key in PHASE_CLOCK_CONTRACT:
        _require(contract[key] == PHASE_CLOCK_CONTRACT[key], f"{where}.{key}与当前phase合同不一致")
    return contract


def _require(condition: bool, message: str) -> None:
    r"""在第一处证据冲突停止，避免缺失事实被解释为科学失败。"""

    if not condition:  # 非法输入不能靠删行、补零或动态缩小分母通过。
        raise AcceptanceError(message)  # 与合法但未达86的 passed=False 分开。


def _object(value: Any, where: str) -> dict[str, Any]:
    r"""要求 JSON 节点为对象，拒绝 null、列表和隐式缺省配置。"""

    _require(isinstance(value, dict), f"{where} 必须是 JSON 对象，实际为 {type(value).__name__}")  # 身份层必须完整。
    return cast(dict[str, Any], value)  # 已由运行时检查收窄类型。


def _integer(value: Any, where: str, minimum: int = 0) -> int:
    r"""读取离散计数时拒绝 bool 和浮点数，保持分母为真正整数。"""

    _require(
        type(value) is int and value >= minimum, f"{where} 必须是 >= {minimum} 的整数，实际为 {value!r}"
    )  # bool 不能冒充0/1。
    return cast(int, value)  # Python int 足以表达128/32/16计数。


def _finite_number(value: Any, where: str, low: float = -math.inf, high: float = math.inf) -> float:
    r"""读取有限物理量或比例，不裁剪越界值以掩盖证据损坏。"""

    _require(type(value) in (int, float), f"{where} 必须是有限数值，实际为 {value!r}")  # 字符串数字和bool均拒绝。
    number = float(value)  # 在CPU双精度中复核JSON标量。
    _require(
        math.isfinite(number) and low <= number <= high, f"{where} 必须有限且位于 [{low}, {high}]，实际为 {value!r}"
    )  # NaN/Inf不能参加科学门。
    return number  # 返回未裁剪的真实值。


def _sha256_digest(value: Any, where: str) -> str:
    r"""要求外部身份锚点是64位小写SHA-256。"""

    _require(
        isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value),
        f"{where} 必须是64位小写SHA-256",
    )  # 路径名不能代替内容身份。
    return value  # 显式expected值保留原样。


def _canonical_json(value: Any, *, ascii_only: bool = False) -> str:
    r"""产生稳定JSON表示，用于协议或checkpoint identity的等值比较。"""

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=ascii_only, allow_nan=False
    )  # 键顺序不影响身份。


def _file_stamp(path: Path) -> dict[str, int]:
    r"""读取普通文件stat，供发布快照和inode重复检测使用。"""

    status = path.stat()  # follow resolve后的实体。
    _require(stat.S_ISREG(status.st_mode), f"证据必须是普通文件：{path}")  # 拒绝目录、管道和设备。
    return {
        "bytes": status.st_size,
        "mtime_ns": status.st_mtime_ns,
        "ctime_ns": status.st_ctime_ns,
        "device": status.st_dev,
        "inode": status.st_ino,
    }  # device+inode识别硬链接。


def _unchanged(path: Path, before: Mapping[str, int]) -> None:
    r"""要求文件在一次读取或审计期间保持同一实体和长度时间戳。"""

    after = _file_stamp(path)  # 只比较stat键，fingerprint可额外含path/hash。
    _require(all(before[key] == after[key] for key in after), f"证据在分析期间发生变化：{path}")  # 拒绝并发替换/追加。


def _read_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    r"""严格读取有限JSON并绑定原始字节hash。"""

    before = _file_stamp(path)  # 固定读取的文件实体。
    _require(before["bytes"] <= 16 * 1024**2, f"JSON超过16 MiB限制：{path}")  # 防止异常输入无限占用内存。
    data = path.read_bytes()  # evaluation摘要应远小于HDF5。
    _require(len(data) == before["bytes"], f"JSON在读取期间被截短：{path}")  # 字节必须对应初始stat。

    def reject_constant(value: str) -> Any:
        r"""拒绝JSON的NaN/Infinity常量。"""

        raise AcceptanceError(f"JSON含非有限常量：{path}: {value}")  # 非有限证据不能成为科学失败。

    try:  # 语法/编码错误统一落到输入不合法。
        value = json.loads(data, parse_constant=reject_constant)  # 不按后缀引入YAML隐式类型。
    except (ValueError, UnicodeError, RecursionError) as error:
        raise AcceptanceError(f"无法读取JSON：{path}: {error}") from error
    document = _object(value, str(path))  # 顶层必须是评价对象。
    _unchanged(path, before)  # 内容hash与stat对应同一发布快照。
    fingerprint = {"path": str(path), **before, "sha256": hashlib.sha256(data).hexdigest()}  # 完整输入指纹。
    return document, fingerprint  # wrapper结束时还会复核快照。


def _load_raw_module(module_name: str, source: Path) -> Any:
    r"""按绝对文件路径加载纯分析模块，不触发包初始化。"""

    _require(source.is_file(), f"raw module文件不存在：{source}")  # 不猜测安装路径。
    spec = importlib.util.spec_from_file_location(module_name, source)  # 直接定位源码。
    if spec is None or spec.loader is None:  # 显式分支也让类型检查器确认loader存在。
        raise AcceptanceError(f"无法建立raw module loader：{source}")  # 失败闭合在文件边界。
    module = importlib.util.module_from_spec(spec)  # 只保留标准库/轻量依赖模块对象。
    spec.loader.exec_module(module)  # 不导入anymani、IsaacLab或任务注册。
    return module  # 调用方只取既有分析入口。


def _sha256_file(path: Path) -> tuple[str, dict[str, int]]:
    r"""流式计算checkpoint字节hash，不将完整权重复制到第二份bytes。"""

    before = _file_stamp(path)  # 固定checkpoint实体。
    digest = hashlib.sha256()  # 内存开销独立于文件大小。
    with path.open("rb") as stream:  # CPU顺序读取，不创建GPU张量。
        for _ in range((before["bytes"] + 2**20 - 1) // 2**20):  # 读取次数由初始长度界定。
            block = stream.read(2**20)  # 每块至多1MiB。
            _require(bool(block), f"checkpoint读取到空块：{path}")  # 截短文件不能冒充完整权重。
            digest.update(block)  # 累加原始字节。
    _unchanged(path, before)  # hash期间不能发生替换。
    return digest.hexdigest(), before  # 返回sha和stat快照。


def _read_checkpoint_identity(
    path: Path,
    *,
    expected_method_identity_digest: str,
    expected_checkpoint_sha256: str,
    cohort_sha256: str,
) -> dict[str, Any]:
    r"""在CPU读取checkpoint的method/policy/manifest/N040元数据。

    该函数只核对 checkpoint 内的 JSON-safe identity；不调用 Actor、环境或
    optimizer。checkpoint 的字节 hash 必须等于外部 expected SHA，避免仅凭
    identity 自报内容。method identity 的 canonical digest使用现有
    palm_rotation_identity 的 ensure_ascii=True 规则复算。
    """

    actual_sha, stamp = _sha256_file(path)  # expected SHA与真实文件字节闭合。
    _require(
        actual_sha == expected_checkpoint_sha256,
        f"checkpoint SHA不匹配：期望 {expected_checkpoint_sha256}，实际 {actual_sha}",
    )  # 不同权重不能合并。
    try:  # torch只在这里懒加载，并固定map_location=cpu。
        import torch  # noqa: PLC0415  # checkpoint metadata读取，不启动Isaac/GPU。

        payload = torch.load(path, map_location="cpu", weights_only=False)  # 不执行模型forward或环境构造。
    except (
        OSError,
        RuntimeError,
        ValueError,
        EOFError,
        ImportError,
        pickle.UnpicklingError,
        AttributeError,
        ModuleNotFoundError,
    ) as error:
        raise AcceptanceError(f"无法CPU读取checkpoint metadata：{path}: {error}") from error
    _require(isinstance(payload, dict), f"checkpoint根必须是对象：{path}")  # 只接受训练入口发布的根结构。
    identity = _object(payload.get("anymani_identity"), f"{path}.anymani_identity")  # exact method identity。
    _require(
        identity.get("identity_schema_version") in {"3.0.0", "4.0.0"}, f"{path}.anymani_identity schema不支持"
    )  # 固定PPO identity版本。
    identity_digest = _sha256_digest(
        identity.get("identity_digest"), f"{path}.anymani_identity.identity_digest"
    )  # identity内容hash。
    identity_payload = {
        key: value for key, value in identity.items() if key != "identity_digest"
    }  # 与生产stable digest相同的payload。
    recomputed = hashlib.sha256(
        _canonical_json(identity_payload, ascii_only=True).encode("utf-8")
    ).hexdigest()  # canonical method identity SHA。
    _require(
        identity_digest == recomputed == expected_method_identity_digest,
        f"{path} method identity digest与expected不一致",
    )  # 不信任checkpoint自报短标签。
    manifest = _object(identity.get("manifest"), f"{path}.anymani_identity.manifest")  # canonical资产轴。
    _require(
        manifest.get("sha256") == cohort_sha256, f"{path} manifest.sha256与canonical cohort不一致"
    )  # 同一128成员锁。
    _require(
        manifest.get("support_asset_count") == 128 and type(manifest.get("support_asset_count")) is int,
        f"{path} manifest support_asset_count必须是128",
    )  # 原始A128分母。
    _require(
        manifest.get("selected_rows") == list(range(128)), f"{path} manifest.selected_rows必须保持0..127顺序"
    )  # 不允许资产重排。
    policy = _object(identity.get("policy"), f"{path}.anymani_identity.policy")  # actor物理接口。
    _require(
        policy.get("actor_contact") == "tip-only-binary", f"{path} policy.actor_contact必须是tip-only-binary"
    )  # TIP-only权限。
    authority = _finite_number(
        policy.get("action_authority_rad_per_policy_step"), f"{path} policy.action_authority_rad_per_policy_step", 0.0
    )  # rad/策略步。
    _require(
        math.isclose(authority, 1.0 / 24.0, rel_tol=0.0, abs_tol=1e-12),
        f"{path} action physical authority必须是1/24 rad/策略步",
    )  # 物理动作幅度固定。
    provider = _object(
        identity.get("geometry_provider"), f"{path}.anymani_identity.geometry_provider"
    )  # N040 provider identity。
    retained = _object(
        provider.get("retained_artifact"), f"{path}.geometry_provider.retained_artifact"
    )  # 冻结N040 artifact。
    _require(
        retained.get("schema_version") == "5.0.0" and retained.get("artifact_type") == "retained_geometry_encoder",
        f"{path}必须绑定schema-5 N040 retained artifact",
    )  # 不把hash-Z或旧表征当作N040。
    retained_sha = _sha256_digest(
        retained.get("sha256"), f"{path}.geometry_provider.retained_artifact.sha256"
    )  # N040 artifact内容身份。
    policy_phase = None
    if "phase_clock" in policy:
        policy_phase = _phase_clock_contract(policy["phase_clock"], f"{path}.anymani_identity.policy.phase_clock")
    _unchanged(path, stamp)  # metadata读取期间checkpoint不能替换。
    policy_metadata = {
        "actor_contact": policy["actor_contact"],
        "action_authority_rad_per_policy_step": authority,
    }
    if policy_phase is not None:
        policy_metadata["phase_clock"] = policy_phase
    return {
        "path": str(path),
        "sha256": actual_sha,
        "bytes": stamp["bytes"],
        "identity_digest": identity_digest,
        "manifest_sha256": manifest["sha256"],
        "selected_rows": list(manifest["selected_rows"]),
        "policy": policy_metadata,
        "retained_artifact": {
            "schema_version": retained["schema_version"],
            "artifact_type": retained["artifact_type"],
            "sha256": retained_sha,
        },
    }  # 只发布必要JSON-safe metadata，不发布model tensors。


def _identity_and_protocol(
    document: Mapping[str, Any], cohort_sha256: str, *, where: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    r"""提取evaluation身份，并把dense trace边界交给正式wrapper。"""

    identity = _object(document.get("evaluation_identity"), f"{where}.evaluation_identity")  # 评价身份层。
    _require(
        identity.get("manifest_sha256") == cohort_sha256, f"{where}.manifest_sha256与canonical cohort不一致"
    )  # 同一有序成员锁。
    _sha256_digest(identity.get("method_identity_digest"), f"{where}.method_identity_digest")  # method形状。
    _sha256_digest(identity.get("checkpoint_sha256"), f"{where}.checkpoint_sha256")  # checkpoint形状。
    protocol = _object(identity.get("protocol"), f"{where}.protocol")  # 旧family_teacher负责固定值逐项核验。
    _require(
        type(protocol.get("trace_stride")) is int and protocol["trace_stride"] == 1,
        f"{where}.protocol.trace_stride必须是1",
    )  # 稀疏trace不能做完整性证明。
    _require(
        protocol.get("trace_rewards") is True, f"{where}.protocol.trace_rewards必须为true"
    )  # reward terms必须存在。
    return (
        identity,
        protocol,
        {
            "trace_stride": 1,
            "trace_rewards": True,
            "protocol_sha256": hashlib.sha256(_canonical_json(protocol).encode("utf-8")).hexdigest(),
        },
    )  # 30s/R16/mean/rank0/ADR0等由既有单次工具核验。


def _required_evidence_paths(document: Mapping[str, Any], where: str) -> dict[str, Path]:
    r"""要求正式评价同时发布terminal trajectory和dense step trace。"""

    trajectory_value = document.get("trajectory_hdf5")  # [A,R] terminal snapshot。
    if not isinstance(trajectory_value, str) or not trajectory_value.strip():  # 让类型边界与证据边界同步。
        raise AcceptanceError(f"{where}.trajectory_hdf5缺失")  # 正式门不能无HDF5。
    trajectory = Path(trajectory_value).expanduser().resolve()  # resolve用于路径别名去重。
    _file_stamp(trajectory)  # 发布文件必须已存在且是普通文件。
    trace_meta = _object(document.get("step_trace"), f"{where}.step_trace")  # [T,A,R,...] dense trace声明。
    trace_value = trace_meta.get("path")  # trace路径。
    if not isinstance(trace_value, str) or not trace_value.strip():  # trace路径必须明确。
        raise AcceptanceError(f"{where}.step_trace.path缺失")  # stride-1审计的输入文件。
    trace = Path(trace_value).expanduser().resolve()  # 统一路径身份。
    _file_stamp(trace)  # trace文件必须已经发布。
    _require(trajectory != trace, f"{where} trajectory和trace不能复用同一HDF5")  # 两层事实不能同文件冒充。
    return {"trajectory": trajectory, "trace": trace}  # wrapper随后做跨run inode去重。


def _path_entity(path: Path, where: str) -> tuple[str, tuple[int, int]]:
    r"""返回已存在HDF5的resolve路径与(device,inode)实体键。"""

    stamp = _file_stamp(path)  # 路径存在性和普通文件边界。
    return str(path), (stamp["device"], stamp["inode"])  # 硬链接共享实体键。


def _reject_duplicate_entities(entities: Sequence[tuple[str, tuple[int, int]]], where: str) -> None:
    r"""跨两份评价拒绝同一resolve路径或同一HDF5 inode。"""

    _require(
        len({item[0] for item in entities}) == len(entities), f"{where}重复引用同一resolve路径"
    )  # symlink/..别名。
    _require(len({item[1] for item in entities}) == len(entities), f"{where}重复引用同一inode")  # hard link别名。


def _row_index(value: Any, where: str) -> int:
    r"""读取0..127的selection-local dataset row。"""

    row = _integer(value, where, 0)  # bool/负数均拒绝。
    _require(row < 128, f"{where}必须位于0..127，实际为{row}")  # A128分母固定。
    return row  # row是两次交集的唯一键。


def _asset_axis(report: Mapping[str, Any], *, where: str) -> tuple[dict[int, dict[str, Any]], dict[int, str]]:
    r"""验证family_teacher报告的完整128资产、R16和32×4拓扑轴。

    这是已归约结果证书的形状检查；不从reward或HDF5重新生成净圈。
    """

    raw = report.get("asset_results")  # family_teacher的每资产物理中心。
    _require(isinstance(raw, list) and len(raw) == 128, f"{where}.asset_results必须恰有128行")  # 失败资产不能删除。
    assets: dict[int, dict[str, Any]] = {}  # row到结果。
    topology: dict[int, str] = {}  # row到group/mother拓扑。
    for position, value in enumerate(cast(list[Any], raw)):  # 逐项闭合分母和有限值。
        item = _object(value, f"{where}.asset_results[{position}]")  # 非null资产。
        row = _row_index(item.get("dataset_row"), f"{where}.asset_results[{position}].dataset_row")  # 0..127。
        _require(row not in assets, f"{where}.asset_results重复dataset_row={row}")  # 重复row不能增加票数。
        _require(
            type(item.get("replica_count")) is int and item.get("replica_count") == 16,
            f"{where}.asset_results[{position}].replica_count必须是16",
        )  # R16分母。
        _require(item.get("finite") is True, f"{where}.asset_results[{position}].finite必须为true")  # 非有限值拒绝。
        _finite_number(
            item.get("net_turns_median"), f"{where}.asset_results[{position}].net_turns_median"
        )  # 圈数可带符号。
        _finite_number(
            item.get("absolute_path_turns_median"), f"{where}.asset_results[{position}].absolute_path_turns_median", 0.0
        )  # 路径圈非负。
        _finite_number(
            item.get("directional_consistency"), f"{where}.asset_results[{position}].directional_consistency", 0.0, 1.0
        )  # D∈[0,1]。
        _finite_number(
            item.get("safe_replica_fraction"), f"{where}.asset_results[{position}].safe_replica_fraction", 0.0, 1.0
        )  # S∈[0,1]。
        label = item.get("topology_id")  # 新报告的完整拓扑标签。
        if not isinstance(label, str) or not label.strip():  # 旧结果至少有mother_id。
            label = item.get("mother_id")
        _require(
            isinstance(label, str) and bool(label.strip()),
            f"{where}.asset_results[{position}]缺少topology_id/mother_id",
        )  # 禁止未知拓扑。
        assets[row] = item  # 保留已验证结果。
        topology[row] = str(label)  # 两次报告须按row使用同一拓扑。
    _require(set(assets) == set(range(128)), f"{where}.asset_results必须覆盖完整0..127轴")  # 原A128分母。
    counts = Counter(topology.values())  # 32拓扑各4代表。
    _require(
        len(counts) == 32 and set(counts.values()) == {4},
        f"{where}.asset_results必须是32个拓扑且每拓扑4代表，实际{dict(counts)}",
    )  # 结构分母。
    return assets, topology  # 不按照输入列表位置猜测row。


def _coverage_rows(
    report: Mapping[str, Any],
    name: str,
    assets: Mapping[int, Mapping[str, Any]],
    *,
    where: str,
) -> list[int]:
    r"""验证一圈/两圈通过集合、阈值和N/D/S资产门闭合。"""

    coverage = _object(report.get(name), f"{where}.{name}")  # family_teacher已归约的coverage。
    _require(coverage.get("finite") is True, f"{where}.{name}.finite必须为true")  # 非有限run不能参加重复门。
    _require(
        type(coverage.get("asset_count")) is int and coverage.get("asset_count") == 128,
        f"{where}.{name}.asset_count必须是128",
    )  # A128。
    _require(
        type(coverage.get("topology_count")) is int and coverage.get("topology_count") == 32,
        f"{where}.{name}.topology_count必须是32",
    )  # 32拓扑。
    raw_rows = coverage.get("passed_asset_rows")  # 已归约通过集合。
    _require(isinstance(raw_rows, list), f"{where}.{name}.passed_asset_rows必须是列表")  # 缺失不当空集合。
    rows = [
        _row_index(value, f"{where}.{name}.passed_asset_rows[]") for value in cast(list[Any], raw_rows)
    ]  # 逐row严格解析。
    _require(len(rows) == len(set(rows)), f"{where}.{name}.passed_asset_rows含重复row")  # 集合不能重复加权。
    _require(
        coverage.get("passed_asset_count") == len(rows), f"{where}.{name}.passed_asset_count与row集合不一致"
    )  # 计数闭合。
    thresholds = _object(coverage.get("thresholds"), f"{where}.{name}.thresholds")  # 一圈/两圈门声明。
    turns = 1.0 if name == "one_turn" else 2.0  # two-turn只提高净圈目标。
    _require(
        math.isclose(
            _finite_number(thresholds.get("horizon_s"), f"{where}.{name}.thresholds.horizon_s"),
            30.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        f"{where}.{name}必须是30秒",
    )  # 时间窗口不变。
    _require(
        type(thresholds.get("replicas_per_asset")) is int and thresholds["replicas_per_asset"] == 16,
        f"{where}.{name}必须是R16",
    )  # 副本分母不变。
    _require(
        math.isclose(
            _finite_number(thresholds.get("net_turns_min"), f"{where}.{name}.thresholds.net_turns_min"),
            turns,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        f"{where}.{name}.net_turns_min不匹配",
    )  # one/two区分。
    for key, expected in (
        ("directional_consistency_min", 0.7),
        ("safe_replica_fraction_min", 0.75),
        ("topology_representative_fraction_min", 0.5),
    ):  # 方向与安全门不变。
        _require(
            math.isclose(
                _finite_number(thresholds.get(key), f"{where}.{name}.thresholds.{key}"),
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"{where}.{name}.thresholds.{key}不匹配",
        )  # 不用two-turn .85替代。
    expected_rows = sorted(
        row
        for row, asset in assets.items()
        if _finite_number(asset["net_turns_median"], f"{where}.asset_results[{row}].net_turns_median") >= turns
        and _finite_number(
            asset["directional_consistency"], f"{where}.asset_results[{row}].directional_consistency", 0.0, 1.0
        )
        >= 0.7
        and _finite_number(
            asset["safe_replica_fraction"], f"{where}.asset_results[{row}].safe_replica_fraction", 0.0, 1.0
        )
        >= 0.75
    )  # 只闭合N/D/S，不重算reward。
    _require(
        rows == expected_rows, f"{where}.{name}.passed_asset_rows与N/D/S资产门不一致"
    )  # 方向或安全不足不能自报通过。
    return rows  # 统一升序，便于审计和求交。


def _topology_intersection(rows: Sequence[int], topology: Mapping[int, str]) -> dict[str, Any]:
    r"""在固定32拓扑分母上统计共同通过代表和2/4拓扑门。"""

    selected = set(rows)  # row集合语义，不按R16展开。
    passed_by_topology = Counter(topology[row] for row in selected)  # 每个代表一票。
    labels = sorted(set(topology.values()))  # 0票拓扑也保留。
    table = [
        {
            "topology_id": label,
            "asset_count": 4,
            "passed_asset_count": passed_by_topology[label],
            "required_asset_count": 2,
            "passed": passed_by_topology[label] >= 2,
        }
        for label in labels
    ]  # 与既有R的4代表半数语义一致。
    return {
        "asset_count": 128,
        "passed_asset_count": len(selected),
        "passed_asset_rows": sorted(selected),
        "asset_fraction": len(selected) / 128.0,
        "topology_count": 32,
        "passed_topology_count": sum(item["passed"] for item in table),
        "topology_fraction": sum(item["passed"] for item in table) / 32.0,
        "topology_results": table,
    }  # 共同统计的原始分母仍是128/32。


def _optional_report_identity(report: Mapping[str, Any], *, where: str) -> tuple[str, str] | None:
    r"""若旧报告带inputs身份，则在纯reducer中也拒绝跨checkpoint拼接。"""

    inputs = report.get("inputs")  # family_teacher当前报告保留expected身份。
    if inputs is None:  # 最小纯reducer夹具可省略输入层。
        return None
    inputs = _object(inputs, f"{where}.inputs")  # 身份层类型。
    method = _sha256_digest(
        inputs.get("expected_method_identity_digest"), f"{where}.inputs.expected_method_identity_digest"
    )  # method信任根。
    checkpoint = _sha256_digest(
        inputs.get("expected_checkpoint_sha256"), f"{where}.inputs.expected_checkpoint_sha256"
    )  # checkpoint信任根。
    return method, checkpoint  # 相同method/checkpoint才可求交。


def _single_result(report: Mapping[str, Any], label: str, trace_report: Mapping[str, Any]) -> dict[str, Any]:
    r"""复制一次旧分析结果并附上trace完整性证书；历史门在顶层单独报告。"""

    copied = copy.deepcopy(dict(report))  # 不对旧报告做原地修改。
    return {
        "evaluation": label,
        "trace_audit": copy.deepcopy(dict(trace_report)),
        "analysis_report": copied,
    }  # 单独结果完整可追溯，且旧字段不被新门重写。


def reduce_repeat_acceptance(
    first_report: Mapping[str, Any],
    second_report: Mapping[str, Any],
    *,
    required_assets: int = 86,
    first_evaluation: str = "first",
    second_evaluation: str = "second",
    trace_reports: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    r"""对两份已验证报告按dataset_row执行86资产重复验收。

    新门只使用
    $A_{\mathrm{common}}=A^{(1)}_{\mathrm{one\ turn}}\cap A^{(2)}_{\mathrm{one\ turn}}$。
    其中N/D/S、128资产与R16闭合由本函数验证，原始轨迹/reward审计仍归既有
    family_teacher与独立trace auditor所有。
    """

    required_assets = _integer(required_assets, "required_assets", 1)  # 新门阈值。
    _require(required_assets <= 128, "required_assets不能超过128分母")  # 禁止改变原始分母。
    if trace_reports is None:  # 纯reducer测试明确表达未提供trace层。
        traces = [
            {"status": "not_declared", "scope": "pure reducer only"},
            {"status": "not_declared", "scope": "pure reducer only"},
        ]  # 该状态不被正式file wrapper使用。
    else:  # 正式wrapper传入两次独立audit结果。
        _require(len(trace_reports) == 2, "trace_reports必须恰有两项")  # 一一对应。
        traces = [copy.deepcopy(dict(item)) for item in trace_reports]  # 发布副本。
    first_assets, first_topology = _asset_axis(first_report, where="first")  # 第一份完整128轴。
    second_assets, second_topology = _asset_axis(second_report, where="second")  # 第二份完整128轴。
    _require(first_topology == second_topology, "两份报告dataset_row到topology映射不一致")  # 成员顺序/拓扑不可漂移。
    first_identity = _optional_report_identity(first_report, where="first")  # 可选旧报告身份。
    second_identity = _optional_report_identity(second_report, where="second")  # 可选旧报告身份。
    _require((first_identity is None) == (second_identity is None), "两份报告inputs身份声明不一致")  # 信任层形状一致。
    if first_identity is not None and second_identity is not None:  # 实际family_teacher报告进入此分支。
        _require(first_identity == second_identity, "两份报告method/checkpoint identity不一致")  # 不能跨冻结策略拼接。
    first_one = _coverage_rows(first_report, "one_turn", first_assets, where="first")  # 一圈主门集合。
    second_one = _coverage_rows(second_report, "one_turn", second_assets, where="second")  # 第二次一圈集合。
    first_two = _coverage_rows(first_report, "two_turn", first_assets, where="first")  # 两圈描述集合。
    second_two = _coverage_rows(second_report, "two_turn", second_assets, where="second")  # 两圈独立求交。
    common_one = sorted(set(first_one) & set(second_one))  # 两次可靠资产交集。
    common_two = sorted(set(first_two) & set(second_two))  # 两次two-turn交集，仅作描述。
    first_history = _object(first_report.get("strong_gate"), "first.strong_gate")  # 旧历史门原样来源。
    second_history = _object(second_report.get("strong_gate"), "second.strong_gate")  # 第二次旧历史门。
    for label, history in (
        ("first.strong_gate", first_history),
        ("second.strong_gate", second_history),
    ):  # 只核对旧门身份。
        _require(
            history.get("required_assets") == 103 and history.get("required_topologies") == 29, f"{label}必须保留103/29"
        )  # 新门与旧门分离。
        _require(
            type(history.get("asset_gate_passed")) is bool and type(history.get("topology_gate_passed")) is bool,
            f"{label}缺少独立布尔门",
        )  # 不用交集改写旧结果。
    repeat_gate = {
        "required_assets": required_assets,
        "denominator_assets": 128,
        "first_passed_assets": len(first_one),
        "second_passed_assets": len(second_one),
        "common_passed_assets": len(common_one),
        "first_asset_gate_passed": len(first_one) >= required_assets,
        "second_asset_gate_passed": len(second_one) >= required_assets,
        "common_asset_gate_passed": len(common_one) >= required_assets,
        "passed": len(first_one) >= required_assets
        and len(second_one) >= required_assets
        and len(common_one) >= required_assets,
    }  # 共同门与两次单独门均显式报告。
    return {
        "artifact_type": "anymani.family_teacher.repeat_acceptance",
        "schema_version": "1.0.0",
        "passed": repeat_gate["passed"],  # 顶层便捷判定，唯一来源仍是repeat_gate。
        "repeat_gate": repeat_gate,
        "common_reliable_asset_rows": common_one,
        "common_reliable_topology": _topology_intersection(common_one, first_topology),
        "two_turn_intersection": {
            **_topology_intersection(common_two, first_topology),
            "source": "intersection of both two_turn reports; descriptive only",
            "substitutes_one_turn": False,
        },
        "single_results": [
            _single_result(first_report, first_evaluation, traces[0]),
            _single_result(second_report, second_evaluation, traces[1]),
        ],
        "historical_103_29_gate": {
            "required_assets": 103,
            "required_topologies": 29,
            "runs": [copy.deepcopy(first_history), copy.deepcopy(second_history)],
        },
        "protocol": {
            "asset_denominator": 128,
            "topology_denominator": 32,
            "replicas_per_asset": 16,
            "one_turn_is_acceptance_basis": True,
            "two_turn_is_descriptive_only": True,
        },
        "scope": "repeatability of one frozen fixed evaluation; no reward reconstruction",
    }  # 纯reducer不写文件且不改变输入报告。


def _run_trace_audit(trace_module: Any, evaluation: Path) -> dict[str, Any]:
    r"""调用既有独立dense trace审计，不把完整性改写成能力通过。"""

    try:  # 既有auditor以ValueError等表示文件/生命周期错误。
        report = trace_module.audit_palm_rotation_trace(
            evaluation, action_mode="mean"
        )  # trace负责HDF5、reward和terminal闭合。
    except (ValueError, OSError, KeyError, TypeError) as error:
        raise AcceptanceError(f"trace audit失败：{evaluation}: {error}") from error
    _require(
        isinstance(report, dict) and report.get("status") == "passed", f"trace audit未通过：{evaluation}"
    )  # 只有完整性通过才发布。
    return cast(dict[str, Any], report)  # 保留独立audit原字段。


def _check_phase_trace_matches_checkpoint(
    trace_report: Mapping[str, Any], checkpoint_metadata: Mapping[str, Any], *, where: str
) -> dict[str, Any] | None:
    r"""要求每次 trace 的 phase 证据与同一 CPU checkpoint 的可选契约成对一致。"""

    policy = _object(checkpoint_metadata.get("policy"), f"{where}.checkpoint.policy")
    checkpoint_phase = policy.get("phase_clock")
    trace_phase = trace_report.get("phase_clock")
    if checkpoint_phase is None and trace_phase is None:
        return None  # 旧 checkpoint 与旧 trace 均无 phase，维持历史验收标准。
    _require(
        checkpoint_phase is not None and trace_phase is not None,
        f"{where} phase contract missing from checkpoint or trace evidence",
    )
    checkpoint_contract = _phase_clock_contract(checkpoint_phase, f"{where}.checkpoint.policy.phase_clock")
    trace_contract = _phase_clock_contract(trace_phase, f"{where}.trace.phase_clock")
    for key in set(PHASE_CLOCK_CONTRACT) - {"increment_rad_per_policy_step"}:
        _require(
            checkpoint_contract[key] == trace_contract[key],
            f"{where} checkpoint/trace phase contract differs at {key}",
        )
    _require(
        math.isclose(
            float(checkpoint_contract["increment_rad_per_policy_step"]),
            float(trace_contract["increment_rad_per_policy_step"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ),
        f"{where} checkpoint/trace phase increment differs",
    )
    max_error = trace_report.get("phase_clock_max_abs_error")
    if not isinstance(max_error, (int, float)) or isinstance(max_error, bool):
        raise AcceptanceError(f"{where} trace phase max_abs_error is missing or exceeds 1e-6")
    _require(
        math.isfinite(float(max_error)) and float(max_error) <= 1.0e-6,
        f"{where} trace phase max_abs_error is missing or exceeds 1e-6",
    )
    return trace_contract


def evaluate_repeat_acceptance(
    evaluations: Sequence[Path | str],
    cohort_path: Path | str,
    *,
    expected_method_identity_digest: str,
    expected_checkpoint_sha256: str,
    required_assets: int = 86,
) -> dict[str, Any]:
    r"""正式读取两份完整evaluation并执行同一冻结checkpoint的86门。

    evaluation.json、trajectory HDF5、dense trace和canonical cohort都是只读
    输入。两份评价的JSON路径/inode必须不同；trajectory和trace的路径/inode也
    必须跨run不同。不同run的HDF5内容相同仍允许，因为deterministic mean的
    字节重复不等于同一次运行。
    """

    method = _sha256_digest(expected_method_identity_digest, "expected_method_identity_digest")  # 显式method信任根。
    checkpoint_sha = _sha256_digest(expected_checkpoint_sha256, "expected_checkpoint_sha256")  # 显式checkpoint信任根。
    _require(
        not isinstance(evaluations, (str, bytes, Path)) and len(evaluations) == 2, "evaluations必须恰有两份路径"
    )  # 禁止一份文件自重复。
    evaluation_paths = tuple(Path(value).expanduser().resolve() for value in evaluations)  # resolve后检查别名。
    _require(evaluation_paths[0] != evaluation_paths[1], "两份evaluation解析到同一路径")  # symlink/..不算独立运行。
    evaluation_stamps = [_file_stamp(path) for path in evaluation_paths]  # 同一evaluation hard link也拒绝。
    _require(
        (evaluation_stamps[0]["device"], evaluation_stamps[0]["inode"])
        != (evaluation_stamps[1]["device"], evaluation_stamps[1]["inode"]),
        "两份evaluation使用同一inode",
    )  # 文件实体去重。
    cohort = Path(cohort_path).expanduser().resolve()  # 显式canonical lock。
    cohort_document, cohort_input = _read_json(cohort)  # 既有family_teacher按同一JSON语义解析。
    _require(
        cohort_document.get("schema_version") == "1.2.0", "cohort必须是schema-1.2.0 canonical lock"
    )  # 成员轴由既有单次工具完整核验。
    cohort_sha = cast(str, cohort_input["sha256"])  # evaluation manifest和checkpoint manifest共同绑定。
    root = Path(__file__).resolve().parent  # raw分析文件目录。
    family_teacher = _load_raw_module(
        "anymani_family_teacher_acceptance_source", root / "family_teacher.py"
    )  # 复用旧一次门。
    trace_module = _load_raw_module(
        "anymani_palm_rotation_trace_acceptance_source", root / "palm_rotation_trace.py"
    )  # 正式入口固定调用独立trace audit。
    documents: list[dict[str, Any]] = []  # 原始evaluation文档。
    evaluation_inputs: list[dict[str, Any]] = []  # JSON字节指纹。
    protocols: list[dict[str, Any]] = []  # 跨run协议等值比较。
    protocol_evidence: list[dict[str, Any]] = []  # trace/protocol摘要。
    evidence_paths: list[dict[str, Path]] = []  # 两层HDF5路径。
    checkpoint_paths: list[Path] = []  # 每次评价声明的checkpoint路径。
    checkpoint_metadata: list[dict[str, Any]] = []  # CPU checkpoint identity证书。
    checkpoint_cache: dict[Path, dict[str, Any]] = {}  # 同一个冻结checkpoint只读一次metadata。
    for index, evaluation in enumerate(evaluation_paths):  # 先读raw身份和HDF5去重，再调用旧分析。
        document, fingerprint = _read_json(evaluation)  # 严格JSON快照。
        identity, protocol, evidence = _identity_and_protocol(
            document, cohort_sha, where=f"evaluation[{index}]"
        )  # trace stride/reward严格要求。
        _require(
            identity.get("method_identity_digest") == method, f"evaluation[{index}] method identity与expected不一致"
        )  # 同一方法。
        _require(
            identity.get("checkpoint_sha256") == checkpoint_sha, f"evaluation[{index}] checkpoint SHA与expected不一致"
        )  # 同一冻结权重。
        paths = _required_evidence_paths(document, f"evaluation[{index}]")  # 正式入口不接受无HDF5 summary。
        checkpoint_value = document.get("checkpoint")  # evaluator发布的checkpoint来源。
        if not isinstance(checkpoint_value, str) or not checkpoint_value.strip():  # 不从expected反推路径。
            raise AcceptanceError(f"evaluation[{index}].checkpoint缺失")  # CPU metadata必须有真实文件。
        checkpoint_path = Path(checkpoint_value).expanduser().resolve()  # 真实文件路径。
        if checkpoint_path not in checkpoint_cache:  # 同一checkpoint可以服务两次独立评价。
            checkpoint_cache[checkpoint_path] = _read_checkpoint_identity(
                checkpoint_path,
                expected_method_identity_digest=method,
                expected_checkpoint_sha256=checkpoint_sha,
                cohort_sha256=cohort_sha,
            )  # CPU metadata独立核对action/N040。
        documents.append(document)  # trace auditor使用原始JSON。
        evaluation_inputs.append(fingerprint)  # 发布路径/hash/stat。
        protocols.append(protocol)  # 完整协议后面等值比较。
        protocol_evidence.append(evidence)  # trace字段证据。
        evidence_paths.append(paths)  # HDF5实体待跨run去重。
        checkpoint_paths.append(checkpoint_path)  # identity路径。
        checkpoint_metadata.append(copy.deepcopy(checkpoint_cache[checkpoint_path]))  # 每次单独记录来源。
    _require(
        _canonical_json(protocols[0]) == _canonical_json(protocols[1]), "两份evaluation protocol不一致"
    )  # 30s/R16/mean/rank0/ADR0等由旧工具验证后再要求相同。
    for key in ("trajectory", "trace"):  # 两层HDF5都必须是两个run的不同实体。
        entities = [
            _path_entity(paths[key], f"evaluation[{index}].{key}") for index, paths in enumerate(evidence_paths)
        ]  # resolve/inode。
        _reject_duplicate_entities(entities, f"{key} HDF5")  # 同路径或同inode均拒绝。
    reports: list[dict[str, Any]] = []  # 两份旧分析报告。
    trace_reports: list[dict[str, Any]] = []  # 两份独立trace完整性报告。
    for index, evaluation in enumerate(evaluation_paths):  # 每份都传入同样显式expected。
        try:  # 旧入口已区分输入错误与合法passed=False。
            report = family_teacher.evaluate_teacher(
                evaluation, cohort, expected_method_identity_digest=method, expected_checkpoint_sha256=checkpoint_sha
            )  # type: ignore[attr-defined]  # raw module公开入口。
        except (ValueError, OSError, KeyError, TypeError) as error:
            raise AcceptanceError(f"evaluation[{index}] family_teacher复核失败：{error}") from error
        _require(isinstance(report, dict), f"evaluation[{index}] family_teacher未返回报告")  # reducer输入类型。
        reports.append(cast(dict[str, Any], report))  # 不修改旧报告。
        trace_reports.append(_run_trace_audit(trace_module, evaluation))  # 正式入口无跳过开关。
    phase_contracts = [
        _check_phase_trace_matches_checkpoint(trace_report, checkpoint_metadata[index], where=f"evaluation[{index}]")
        for index, trace_report in enumerate(trace_reports)
    ]
    result = reduce_repeat_acceptance(
        reports[0],
        reports[1],
        required_assets=required_assets,
        first_evaluation=str(evaluation_paths[0]),
        second_evaluation=str(evaluation_paths[1]),
        trace_reports=trace_reports,
    )  # 新门只使用dataset_row交集。
    result["inputs"] = {
        "evaluations": evaluation_inputs,
        "cohort": cohort_input,
        "checkpoint_paths": [str(path) for path in checkpoint_paths],
        "checkpoint_metadata": checkpoint_metadata,
        "expected_method_identity_digest": method,
        "expected_checkpoint_sha256": checkpoint_sha,
        "hdf5_entities": [
            {
                key: {
                    "path": _path_entity(paths[key], f"evaluation[{index}].{key}")[0],
                    "entity": _path_entity(paths[key], f"evaluation[{index}].{key}")[1],
                }
                for key in ("trajectory", "trace")
            }
            for index, paths in enumerate(evidence_paths)
        ],
    }  # 输入、checkpoint和HDF5实体共同形成可复查边界。
    result["identity_evidence"] = {
        "same_method_identity": True,
        "same_checkpoint_sha256": True,
        "method_identity_digest": method,
        "checkpoint_sha256": checkpoint_sha,
        "canonical_cohort_sha256": cohort_sha,
        "checkpoint_policy_verified": True,
        "checkpoint_manifest_order_verified": True,
        "checkpoint_n040_verified": True,
    }  # method、a/24、TIP-only、manifest和N040来自CPU metadata。
    result["identity_evidence"]["phase_clock"] = copy.deepcopy(phase_contracts[0])
    result["protocol_evidence"] = {
        "same_protocol": True,
        "runs": copy.deepcopy(protocol_evidence),
        "canonical_protocol_sha256": hashlib.sha256(_canonical_json(protocols[0]).encode("utf-8")).hexdigest(),
    }  # 正式dense trace protocol与两次evaluation逐字段一致。
    for path, stamp in zip(evaluation_paths, evaluation_stamps, strict=True):  # JSON不能在审计期间被替换。
        _unchanged(path, stamp)  # 绑定发布字节。
    _unchanged(cohort, cohort_input)  # canonical lock不能漂移。
    result["execution"] = {
        "device": "cpu",
        "python": platform.python_version(),
        "machine": platform.machine(),
    }  # 不查询GPU/Isaac。
    envelope = _canonical_json(
        {
            key: result[key]
            for key in (
                "artifact_type",
                "schema_version",
                "inputs",
                "identity_evidence",
                "protocol_evidence",
                "repeat_gate",
            )
        }
    )  # 结果输入指纹。
    result["input_fingerprint_sha256"] = hashlib.sha256(
        envelope.encode("utf-8")
    ).hexdigest()  # 不冒充method/checkpoint SHA。
    return result  # 合法但少于86时返回passed=False。


def main(argv: Sequence[str] | None = None) -> int:
    r"""执行正式重复验收CLI，并以排他方式创建新JSON。"""

    parser = argparse.ArgumentParser(description="同一冻结family teacher的两次固定评价重复验收")  # 不包含训练参数。
    commands = parser.add_subparsers(dest="command", required=True)  # 仅evaluate。
    evaluate = commands.add_parser("evaluate", help="消费两份evaluation.json和一个canonical cohort lock")  # 正式入口。
    evaluate.add_argument("evaluation_1", type=Path)  # 第一次评价。
    evaluate.add_argument("evaluation_2", type=Path)  # 第二次评价。
    evaluate.add_argument("cohort", type=Path)  # canonical lock。
    evaluate.add_argument("--expected-method-identity-digest", required=True)  # method expected。
    evaluate.add_argument("--expected-checkpoint-sha256", required=True)  # checkpoint expected。
    evaluate.add_argument("--required-assets", type=int, default=86)  # 新门默认86/128。
    evaluate.add_argument("--output", type=Path, help="只创建新JSON；省略时打印报告")  # 排他创建。
    args = parser.parse_args(argv)  # argparse负责缺参错误。
    try:  # 先读完全部输入，合法后才写输出。
        report = evaluate_repeat_acceptance(
            [args.evaluation_1, args.evaluation_2],
            args.cohort,
            expected_method_identity_digest=args.expected_method_identity_digest,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            required_assets=args.required_assets,
        )  # 不修改任何旧产物。
        text = (
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
        )  # 发布JSON不含NaN。
        if args.output is None:  # 默认无磁盘副作用。
            print(text, end="")  # stdout供人和管道使用。
        else:  # 不创建父目录、不覆盖既有结果。
            with args.output.expanduser().open("x", encoding="utf-8") as stream:  # OS级排他创建。
                stream.write(text)  # 只写用户明确的新结果路径。
            print(
                json.dumps({"output": str(args.output.expanduser().absolute())}, ensure_ascii=False)
            )  # 返回产物位置。
    except (AcceptanceError, OSError, ValueError) as error:
        parser.error(str(error))  # 非法输入status=2，合法科学失败仍status=0。
    return 0  # 完成分析不等于86门通过。


if __name__ == "__main__":  # 推荐直接执行保持raw-module边界。
    raise SystemExit(main())  # 不启动Isaac、Kit或GPU。


__all__ = [
    "AcceptanceError",
    "evaluate_repeat_acceptance",
    "main",
    "reduce_repeat_acceptance",
]
