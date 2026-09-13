r"""FlashSAC 检查点的原子发布与严格恢复边界，独立于环境和日志系统。

Learner 拥有网络、优化器、奖励统计与随机流；StaticHandBank 拥有静态手型证据；
CompactReplay 拥有完整转移环、显式终点和历史段边界；experiment 保存调用方的
抽样随机流、进度和编排状态。此模块仅负责把这些 owner 已形成的状态同步封存。

模型快照允许 replay=None，metadata 明示 model_snapshot；full_resume 必须包含
完整回放状态。full_resume 表示算法与回放恢复资格，物理环境重建和 reset_streams
边界仍由训练入口负责；文件读取不会运行环境、评价回放或 logger。
"""

from __future__ import annotations

import fcntl
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .identity import _ALGORITHM, _IDENTITY_SCHEMA, _json_value, _stable_digest

_SCHEMA = "flash-sac-checkpoint-1"  # 原生算法状态封装协议，不与其它学习器共用版本号。
_FIELDS = {"schema_version", "learner", "identity", "static_bank", "replay", "experiment", "metadata"}  # 完整外层键。


def _checked_identity(identity: Mapping, label: str) -> dict:
    r"""检查原生 schema、必要科学字段和完整摘要，返回调用方数据的独立副本。

    已保存摘要与期望摘要都须按内容重算，不能仅比较两个可能过期的字符串标签。
    恢复时不重读来源资产；当前实际实现/资产由调用方新建 expected_identity 核对。
    """
    if not isinstance(identity, Mapping):  # 缺失或异类对象不可作为方法凭据。
        raise ValueError(f"{label} identity must be a mapping")  # 裸权重不包含实验科学合同。
    result = _json_value(identity)  # 深拷贝JSON metadata，不包含模型/回放tensor。
    if result.get("identity_schema_version") != _IDENTITY_SCHEMA or result.get("algorithm") != _ALGORITHM:
        raise ValueError(f"{label} identity schema/algorithm is not native FlashSAC")  # 算法域先于参数shape。
    if not isinstance(result.get("task_id"), str) or not result["task_id"].strip():
        raise ValueError(f"{label} identity requires task_id")  # 相同网络可以对应不同任务。
    for name in (
        "task_contract",  # 实际物理和奖励参数。
        "policy",  # 动作分布与信息边界。
        "manifest",  # 有序支持域。
        "pregrasp",  # 初态分布。
        "geometry_provider",  # 冻结表示身份。
        "transport_abi",  # 具名状态布局。
        "training",  # 算法与入口配置。
        "implementation",  # 实际源码字节。
    ):
        if not isinstance(result.get(name), dict) or not result[name]:  # 明确拒绝空兼容身份。
            raise ValueError(f"{label} identity requires nonempty {name}")  # 未知物理条件不能续接。
    if not isinstance(result["training"].get("config"), dict) or not result["training"]["config"]:
        raise ValueError(f"{label} identity requires training.config")  # 显式绑定Bellman与预算参数。
    if not isinstance(result["training"].get("run_contract"), dict) or not result["training"]["run_contract"]:
        raise ValueError(f"{label} identity requires training.run_contract")  # 入口条件同样必须完整。
    files = result["implementation"].get("files")  # 精确源码内容映射，不能用Git标签占位。
    if not isinstance(files, dict) or not files:
        raise ValueError(f"{label} identity requires implementation.files provenance")  # 版本标签不能代替源码。
    for path, digest in files.items():  # 字段类型错误不能进入严格resume比较。
        if (
            not path  # 必须指向一个可审计的源文件。
            or not isinstance(digest, str)  # 固定文本协议，不是对象repr。
            or len(digest) != 64  # 256bit对应64位十六进制文本。
            or any(c not in "0123456789abcdef" for c in digest)  # 规范小写字符集。
        ):
            raise ValueError(f"{label} implementation source digest is invalid: {path}")  # 定位具体源文件。
    body = {key: value for key, value in result.items() if key != "identity_digest"}  # 排除自引用字段。
    if result.get("identity_digest") != _stable_digest(body):
        raise ValueError(f"{label} identity digest is inconsistent with its contents")  # 内容改变不能沿用旧身份。
    return result  # 所有后续比较使用已自洽且与外部对象解耦的树。


def _compare_identity(actual: dict, expected: dict) -> None:
    r"""完整方法逐字段一致才允许续接或同方法 latest 覆盖，不按网络shape猜测兼容性。"""
    if actual["identity_digest"] != expected["identity_digest"]:  # 两端均已独立校验摘要。
        fields = sorted(  # 按科学合同层级定位差异，字典插入顺序不影响诊断。
            key
            for key in actual.keys() | expected.keys()  # 包括一端缺失的合同字段。
            if key != "identity_digest"  # 摘要变化只是内容变化的结果。
            and _stable_digest({key: actual.get(key)}) != _stable_digest({key: expected.get(key)})
        )
        raise ValueError(f"FlashSAC identity mismatch: {fields}")  # 错误指出科学合同层级。


def _validate_payload(payload: Any, *, expected_identity: Mapping | None, require_replay: bool) -> dict:
    r"""统一读写前置闸门：schema、配置双写一致性、回放资格及完整方法身份。"""
    if not isinstance(payload, dict) or payload.keys() != _FIELDS or payload.get("schema_version") != _SCHEMA:
        raise ValueError("FlashSAC checkpoint schema/fields mismatch")  # 缺键不等于可选状态None。
    for name in ("learner", "static_bank", "experiment"):  # experiment可为空，但仍须是可解释mapping。
        if not isinstance(payload[name], Mapping):
            raise ValueError(f"FlashSAC checkpoint {name} must be a mapping")  # 保留各owner命名空间。
    if not payload["learner"] or not payload["static_bank"]:  # 不接受空模型或空手型表冒充检查点。
        raise ValueError("FlashSAC checkpoint requires nonempty learner and static_bank")  # 快照也须可重建输入。
    identity = _checked_identity(payload["identity"], "checkpoint")  # 保存内容自身必须先自洽。
    configuration = payload["learner"].get("config")  # 学习器的固定配置字段。
    if not isinstance(configuration, Mapping) or _stable_digest(configuration) != _stable_digest(
        identity["training"]["config"]  # 双写配置必须具有相同规范JSON，防止同shape异算法。
    ):
        raise ValueError("FlashSAC learner config must match identity.training.config")  # 早于参数/Adam写入。

    # 元数据按实际presence形成；空mapping不是“完整回放”，缺键也不是显式快照None。
    replay = payload["replay"]  # 保留回放tensor原引用，不遍历或复制数GiB环池。
    if replay is not None and (not isinstance(replay, Mapping) or not replay):
        raise ValueError("FlashSAC replay must be a nonempty complete state mapping or None")
    has_replay = replay is not None  # 仅声明携带资格；内部序列完整性由CompactReplay.load_state_dict核验。
    metadata = {"kind": "full_resume" if has_replay else "model_snapshot", "has_replay": has_replay}  # 两种明确产物。
    if payload["metadata"] != metadata:  # 不能宣称携带与实际内容不同的resume能力。
        raise ValueError("FlashSAC checkpoint metadata disagrees with replay presence")  # 资格与实际内容同源。
    if require_replay and not has_replay:  # 只读模型消费可省略，完整resume不能退化为空池。
        raise ValueError("FlashSAC full resume requires replay; this is a model snapshot")
    if expected_identity is not None:  # caller用当前实现和实际配置重建的科学条件。
        _compare_identity(identity, _checked_identity(expected_identity, "expected"))  # 全方法严格比较。
    payload["identity"] = identity  # 返回规范独立metadata，状态tensor仍由各owner消费。
    return payload  # 此时尚未调用任何learner/环境的恢复方法。


def read_checkpoint(path: Path, *, expected_identity: Mapping | None = None, require_replay: bool = False) -> dict:
    r"""在 CPU 读取原生检查点，并在交付状态之前拒绝不匹配或不完整恢复。

    Args:
        path: 由write_checkpoint发布的torch zip检查点。
        expected_identity: 当前实际方法身份；完整resume调用方应提供并保持逐字段一致。
        require_replay: True要求携带完整回放，False允许只读模型快照。

    Returns:
        dict: schema_version、learner、identity、static_bank、replay、experiment和metadata。
            tensor通过map_location='cpu'加载；mmap避免检查既有latest时再复制整个回放。

    Raises:
        ValueError: schema、摘要、配置、完整方法或回放资格不一致。
        OSError: 文件读取失败；原始序列化错误同样向调用方传播。

    Notes:
        返回tensor使用只影响本进程的私有映射；恢复owner应按其load_state_dict复制。
        此函数不解释或复原物理环境状态，也不推进logger、随机流或训练时钟。
    """
    payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True, mmap=True)  # 仅CPU状态读取。
    return _validate_payload(
        payload, expected_identity=expected_identity, require_replay=require_replay
    )  # 先核对后交付。


def write_checkpoint(
    path: Path,  # 发布后代表完整方法拥有的状态文件。
    *,
    learner_state: Mapping,  # Actor/双Q/温度/Adam/统计/随机流。
    identity: Mapping,  # 科学条件与源码凭据。
    static_bank_state: Mapping,  # A种手型的静态证据。
    replay_state: Mapping | None = None,  # None明确声明不具完整回放恢复资格。
    experiment_state: Mapping | None = None,  # 调用方时钟与抽样随机流。
) -> Path:
    r"""在同目录同步临时文件，再原子替换同一方法拥有的检查点路径。

    Args:
        path: 发布位置；同identity的latest可以替换，不同方法的既有目标一律拒绝。
        learner_state: FlashSACLearner.state_dict()；固定config必须等于identity.training.config。
        identity: build_method_identity生成的完整原生身份。
        static_bank_state: StaticHandBank.state_dict()，保持同一cohort成员轴。
        replay_state: 完整CompactReplay.state_dict()或模型快照的None。
        experiment_state: caller拥有的进度、抽样RNG和编排状态；None规范为{}。

    Returns:
        Path: 已完成原子发布的路径。

    Raises:
        ValueError: 状态/身份自矛盾，或目标已属于不同方法。
        OSError: 序列化/内容fsync/replace失败；发布前失败保留旧目标并清除本次临时文件。

    Notes:
        调用者须暂停更新/append直到此同步调用返回，不能边保存边原地改写共享tensor。
        同目录writer通过目录描述符的flock串行化身份检查和发布；不产生永久锁侧文件。
        replace是提交点，内容fsync在其之前完成；这里不声称保存了仿真物理快照。
    """
    # 先核对调用者状态，任何科学合同错误都不能创建或覆盖产物。
    for label, value in (("learner_state", learner_state), ("static_bank_state", static_bank_state)):
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} must be a mapping")  # 不接受实际模型对象代替state_dict。
    if experiment_state is not None and not isinstance(experiment_state, Mapping):
        raise ValueError("experiment_state must be a mapping or None")  # None规范成空实验metadata。
    has_replay = replay_state is not None  # 产物资格源于实际payload，而不是调用者标签。
    payload = {
        "schema_version": _SCHEMA,  # 外层存储协议。
        "learner": dict(learner_state),  # 只复制轻量字典容器，模型tensor保持同步快照引用。
        "identity": identity,  # validator将其规范成独立JSON树。
        "static_bank": dict(static_bank_state),  # 小型静态证据由调用者owner形成。
        "replay": replay_state,  # 不复制整池，下一append须等待本调用结束。
        "experiment": dict(experiment_state) if experiment_state is not None else {},  # caller自有编排状态。
        "metadata": {"kind": "full_resume" if has_replay else "model_snapshot", "has_replay": has_replay},
    }
    payload = _validate_payload(payload, expected_identity=None, require_replay=False)  # 身份/config先于文件操作。
    path = Path(path).expanduser()  # checkpoint路径按caller给出的目录解释。
    path.parent.mkdir(parents=True, exist_ok=True)  # 允许训练入口交付尚未创建的checkpoint子目录。
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)  # Linux目录inode作为同目录writer锁。
    temporary: Path | None = None  # 只清理由本次调用取得所有权的临时文件。
    try:  # 锁覆盖现有identity核验、内容序列化和最终发布的整个窗口。
        fcntl.flock(directory, fcntl.LOCK_EX)  # 不同方法的两个并行writer不能都通过“目标尚不存在”。
        if path.exists():  # 原文件必须可读且完整自洽，不能用损坏/异类文件覆盖许可回退。
            read_checkpoint(path, expected_identity=payload["identity"])  # mmap只访问轻量metadata。
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)  # 同一文件系统内的独立临时文件，绝不先截断旧latest。
            torch.save(payload, stream)  # tensor与全部owner状态同步完成序列化。
            stream.flush()  # Python用户态缓冲先写入内核。
            os.fsync(stream.fileno())  # 新文件完整内容已同步，发布线之前仍只暴露旧目标。
        if path.exists():  # 同时防止保存期间外部非本writer的目标变化被静默覆盖。
            read_checkpoint(path, expected_identity=payload["identity"])  # 失败仍保留旧目标。
        os.replace(temporary, path)  # 唯一发布提交点；读者只见完整旧版或完整新版。
    finally:  # 成功时临时名已消失；失败时清掉本调用的部分字节，不触碰旧目标。
        try:
            if temporary is not None:
                temporary.unlink(missing_ok=True)  # 不按glob删除，避免误清其他writer文件。
        finally:
            os.close(directory)  # 释放目录flock，即使保存/清理失败也不保留锁。
    return path  # 到达这里表示身份核对、内容fsync和原子发布均成功。
