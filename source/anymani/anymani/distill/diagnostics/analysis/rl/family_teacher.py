r"""家族 fullKD 教师的有界、CPU、产物只读分析。

候选选择只消费已发布的 metrics.parquet 和调用者明确给出的 nn 目录。
训练首窗代理按 $(n_1,n_2,\widetilde N,\bar S,-u)$ 字典序选优；每资产至少 16 窗，
且观测、资格和资产分母均为 128。默认完整终点 $u=2000$，每 update 有 61440 个新 transitions。
这里的训练首窗可能跨策略版本，只用于安排最多两次固定评价，不能据此宣称能力达门。

固定门消费正式 evaluation JSON 和 schema-1.2 canonical cohort 的原始 JSON 字节。
正式资产方向性为 $D_i=\operatorname{clip}(\max(N_i,0)/\max(P_i,2^{-23}),0,1)$；
$N_i,P_i$ 分别是 16 副本净圈与路径圈的中位数，单位均为圈。安全比例 $S_i$ 的分母为 16。
一圈可靠门为 $N_i\ge1,D_i\ge0.7,S_i\ge0.75$；两圈只把净圈门改为 $N_i\ge2$。
拓扑有 4 个代表，至少 2 个通过才被覆盖；强门是一圈资产至少 103/128 且拓扑至少 29/32。
比值中位数与中位数之比是不同统计量：训练首窗方向定义留在 recorder，正式方向定义在本文件逐项复核。

文件入口：python -B <本文件> select <metrics.parquet> <nn目录> --output <新JSON>
固定门入口：python -B <本文件> evaluate <evaluation.json> <canonical.lock.yaml>
    --expected-method-identity-digest <digest> --expected-checkpoint-sha256 <sha> --output <新JSON>
直接执行文件可保持标准库与 Polars 的导入闭包，避免顶层包初始化中的环境注册。
checkpoint 只做文件名匹配和 stat；主 agent 在真正评价前须另验其内容哈希、epoch/frame 和方法元数据。
分析函数不写文件；CLI 仅以排他创建方式发布显式 --output，省略时打印 JSON。
"""

from __future__ import annotations

import argparse  # 仅 select/evaluate 两个窄入口。
import hashlib  # 输入字节与候选文件目录快照的可复盘摘要。
import json  # canonical lock 虽以 .yaml 结尾，实际格式仍为 JSON。
import math  # 有限性、比例区间与正式方向定义。
import platform  # 只记录 CPU 解释器信息，不查询训练进程或 GPU。
import re  # 保存点仅按已有命名规则定位。
import stat  # stat 检查普通文件，不读取 checkpoint 内容。
import statistics  # 128 资产等权中心。
from collections import Counter  # 拓扑的 4 代表分母。
from collections.abc import Sequence  # CLI 参数的窄序列接口。
from pathlib import Path  # 所有输入路径均由调用者显式提供。
from typing import Any, cast  # 外部 JSON/Parquet 的异构标量与已验证类型边界。

import polars as pl  # 惰性投影与 global 谓词下推；不导入训练组件。

_JSON_LIMIT = 16 * 1024**2  # 单份评价/集合 JSON 上限 16 MiB，足够保存 A128 的可读证据。
_METRICS_LIMIT = 2 * 1024**3  # 完成标量表上限 2 GiB；哈希流式读取，内存不随文件大小增长。
_INVENTORY_LIMIT = 10000  # 仅枚举一个显式 nn 目录，且不递归。
_EPSILON = 2**-23  # FP32 epsilon，无量纲；与正式 physical evaluator 一致。
_TOLERANCE = 1e-12  # JSON 双精度标量的复算绝对容差，不放宽科学阈值。
_FIRST30_INTS = (
    "first30_asset_count",  # 所选资产全集，不是副本数。
    "first30_observed_assets",  # 至少一窗的资产数。
    "first30_qualified_assets",  # 达到 recorder 配置资格门的资产数。
    "first30_window_count",  # 留存窗口总数。
    "first30_windows_min",  # 缺测资产贡献 0，不能被过滤。
    "first30_windows_max",  # 用于检查窗口总数的合法区间。
    "first30_one_turn_proxy_assets",  # 一圈训练代理计数。
    "first30_two_turn_proxy_assets",  # 两圈训练代理计数。
)
_FIRST30_FLOATS = (
    "first30_net_median",  # median_asset(median_window(net))，单位圈。
    "first30_goal_median",  # 严格目标次数，只验证合法性。
    "first30_direction_median",  # 训练逐窗方向比的二层中位数，范围 [-1,1]。
    "first30_safe_fraction",  # 资产等权安全比例，范围 [0,1]。
)
_FIRST30_VERSIONS = ("first30_policy_start_min", "first30_policy_end_max")  # 留存窗口涉及的策略版本界。
_FIRST30 = (*_FIRST30_INTS, *_FIRST30_FLOATS, *_FIRST30_VERSIONS)  # 只读取首窗相关统计。
_R_THRESHOLDS = {
    "horizon_s": 30.0,  # 首轨迹测量窗口，秒。
    "replicas_per_asset": 16,  # 安全比例的固定分母。
    "net_turns_min": 1.0,  # 一圈可靠门的净圈下界。
    "directional_consistency_min": 0.7,  # 一圈/两圈共用方向门。
    "safe_replica_fraction_min": 0.75,  # 12/16 副本安全。
    "topology_representative_fraction_min": 0.5,  # ceil(4×0.5)=2 个代表。
}
_PROTOCOL = {
    "num_assets": 128,  # 固定资产宇宙。
    "policy_steps": 600,  # 600 个控制周期。
    "policy_dt_s": 0.05,  # 20 Hz，每步模拟时间为 0.05 秒。
    "horizon_s": 30.0,  # 600×0.05=30 秒。
    "replicas_per_asset": 16,  # 每资产首轨迹数。
    "deterministic_actor_mean": True,  # 固定 Actor 均值动作。
    "first_trajectory_only": True,  # automatic reset 不增补样本。
    "pregrasp_rank": 0,  # 固定 rank-0 初态。
    "adr_enabled": False,  # 评价固定 ADR0。
    "actor_contact": "tip-only-binary",  # Actor 的指尖二值接触输入。
    "evaluation_role": "capability",  # 正式能力角色。
    "actor_relay": None,  # 单个冻结 Actor，无接力。
    "actor_contact_intervention": "none",  # 无观察干预。
    "residual_off_intervention": False,  # 无动作分支干预。
    "direct_logit_gain_intervention": 1.0,  # 无动作幅值干预。
    "cohort_transfer": False,  # 当前教师的训练集合。
    "reliable_topology_coverage_protocol_matched": True,  # 必须有正式 R 证据。
    "goal_advance": "qualified-pose",  # fullKD 的完整姿态/位置推进门。
    "goal_reference": "current_object",  # fullKD 的目标参考。
}


class AnalysisError(ValueError):
    r"""协议、身份或证据不合法；与合法输入下的科学 passed=False 分开。"""


def _require(condition: bool, message: str) -> None:
    r"""在第一处证据冲突停止，并保留含字段路径的错误原因。"""
    if not condition:  # 不能通过过滤非法行提高科学覆盖率。
        raise AnalysisError(message)  # 调用者可以单独处理输入无效。


def _object(value: Any, where: str) -> dict[str, Any]:
    r"""必要的 JSON 对象缺失、为 null 或为其它类型时拒绝。"""
    _require(isinstance(value, dict), f"{where} 必须是 JSON 对象，实际为 {type(value).__name__}")  # 必要对象不能用空缺省值补齐。
    return value  # Any 只留在外部证据边界。


def _integer(value: Any, where: str, minimum: int = 0) -> int:
    r"""计数/索引必须为真整数；不把 bool 或浮点数隐式转换成分母。"""
    _require(type(value) is int and value >= minimum, f"{where} 必须是 >= {minimum} 的整数，实际为 {value!r}")  # 样本数必须可解释为离散事件计数。
    return value  # Python int 不受机器整型乘法溢出影响。


def _number(value: Any, where: str, low: float = -math.inf, high: float = math.inf) -> float:
    r"""物理标量必须有限且在声明区间；圈数可带符号，比例必须有界。"""
    _require(type(value) in (int, float), f"{where} 必须是有限数值，实际为 {value!r}")  # 禁止字符串数字和 bool 改变物理量的类型。
    try:  # 超大 JSON 整数也不能溢出为合法物理量。
        number = float(value)  # 统一为双精度 CPU 复算。
    except OverflowError as error:  # 保持清楚的字段错误。
        raise AnalysisError(f"{where} 超出有限浮点范围") from error
    _require(math.isfinite(number) and low <= number <= high, f"{where} 必须有限且位于 [{low}, {high}]，实际为 {value!r}")  # 非有限量不是可排序的科学失败样本。
    return number  # 不裁剪输入本身，不掩盖超界统计。


def _digest(value: Any, where: str) -> str:
    r"""身份锚点必须是显式的 64 位小写十六进制 SHA-256。"""
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None, f"{where} 必须是 SHA-256 摘要")  # 摘要格式先于身份等值比较验证。
    return value  # 不用路径名或自报短标签替代内容身份。


def _expect_fields(actual: dict[str, Any], expected: dict[str, Any], where: str) -> None:
    r"""核对固定协议/计数叶节点；missing 与显式 null 不等价。"""
    for key, reference in expected.items():  # 每个强门必要字段都有单独错误路径。
        label = f"{where}.{key}"  # 返回用户可以逐字段核对的位置。
        _require(key in actual, f"{label} 缺失，期望 {reference!r}")  # 缺失的干预声明不能被解释为无干预。
        value = actual[key]  # 不通过 get(None) 接受缺失 actor_relay。
        if type(reference) is float:  # 数值单位已由字段名和固定协议规定。
            valid = math.isclose(_number(value, label), reference, rel_tol=0.0, abs_tol=_TOLERANCE)
        else:  # 整数、bool、字符串与 None 均使用严格类型相等。
            valid = type(value) is type(reference) and value == reference  # False 不能冒充 pregrasp_rank=0。
        _require(valid, f"{label} 不匹配：期望 {reference!r}，实际 {value!r}")  # 固定协议错误不进入科学计分。


def _stamp(path: Path) -> dict[str, int]:
    r"""仅读取文件系统元数据；用于检测分析期间产物更换或继续写入。"""
    status = path.stat()  # 不打开 checkpoint 内容；遵循显式路径中的链接。
    _require(stat.S_ISREG(status.st_mode), f"输入必须是普通文件：{path}")  # 拒绝目录、管道或设备等非有限字节产物。
    return {
        "bytes": status.st_size,  # 文件长度，字节。
        "mtime_ns": status.st_mtime_ns,  # 最后内容写入时间，纳秒。
        "ctime_ns": status.st_ctime_ns,  # inode 状态改变时间，纳秒。
        "device": status.st_dev,  # 与 inode 联合定位文件实体。
        "inode": status.st_ino,  # 原子替换也会改变快照身份。
    }


def _unchanged(path: Path, before: dict[str, Any]) -> None:
    r"""要求整个读取期间的文件元数据保持不变；这不证明外部进程已退出。"""
    _require(all(before[key] == value for key, value in _stamp(path).items()), f"产物在分析期间发生变化：{path}")  # 报告只能绑定单一完成快照。


def _hash_metrics(path: Path) -> dict[str, Any]:
    r"""流式获取完整 Parquet 的字节指纹；标量解码仍仅投影 global 首窗字段。"""
    before = _stamp(path)  # 哈希之前固定文件实体。
    _require(before["bytes"] <= _METRICS_LIMIT, f"metrics.parquet 超过 {_METRICS_LIMIT} 字节的只读分析上限")  # 总 I/O 预算有显式上界。
    digest = hashlib.sha256()  # 哈希内存开销与文件大小无关。
    with path.open("rb") as stream:  # 唯一完整顺序读取是原始字节校验。
        for offset in range(0, before["bytes"], 1024**2):  # 总读取次数也受最初 stat 大小约束。
            size = min(1024**2, before["bytes"] - offset)  # 每次至多 1 MiB，且不跟随文件增长。
            block = stream.read(size)  # 已完成表的有界顺序字节读取。
            _require(len(block) == size, f"metrics.parquet 在读取期间被截短：{path}")  # 哈希必须覆盖最初声明的全部字节。
            digest.update(block)  # 不反序列化 Parquet 内的未投影列。
    _unchanged(path, before)  # 完成表不能在读取时被继续更新。
    return {"path": str(path), **before, "sha256": digest.hexdigest()}  # 完整输入快照。


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    r"""拒绝重复 JSON 键，防止同一协议字段在字节证据中有两个声明。"""
    result: dict[str, Any] = {}  # 每个对象的唯一键集合。
    for key, value in pairs:  # object_pairs_hook 保留重复键以供验证。
        _require(key not in result, f"JSON 重复键：{key}")  # 单个字段只能对应一个协议或身份事实。
        result[key] = value  # 只有唯一声明才进入语义对象。
    return result  # 不接受 JSON 解析器默认的“最后一个键获胜”。


def _reject_constant(value: str) -> Any:
    r"""NaN/Infinity 不是合法科学 JSON 数值，必须在解析阶段停止。"""
    raise AnalysisError(f"JSON 含非有限常量：{value}")  # 不将其改写成零或缺测。


def _read_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    r"""读取至多 16 MiB 的 JSON，并对被解析的同一份字节计算 hash。"""
    before = _stamp(path)  # 与文件字节长度共同约束读取范围。
    _require(before["bytes"] <= _JSON_LIMIT, f"JSON 超过 {_JSON_LIMIT} 字节上限：{path}")  # 限制结构化证据的解析内存。
    with path.open("rb") as stream:  # 不按照 .yaml 后缀引入 YAML 隐式类型转换。
        data = stream.read(_JSON_LIMIT + 1)  # 并发增大文件时也有固定内存边界。
    _require(len(data) <= _JSON_LIMIT, f"JSON 超过读取上限：{path}")  # 并发增长时也不能绕过字节边界。
    _unchanged(path, before)  # 同一文件快照用于 JSON 与 SHA-256。
    try:  # JSON 解码错误也附加确切文件路径。
        value = json.loads(data, object_pairs_hook=_json_pairs, parse_constant=_reject_constant, parse_float=lambda token: _number(float(token), "JSON 数值"))
    except (ValueError, UnicodeError, RecursionError) as error:  # 包括指数溢出、重复键和过深嵌套。
        raise AnalysisError(f"{path}: {error}") from error
    document = _object(value, str(path))  # 文档必须是对象，不能是数组或单个标量。
    return document, {"path": str(path), **before, "sha256": hashlib.sha256(data).hexdigest()}  # hash 绑定原始字节。


def _checkpoint_inventory(nn_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    r"""按既有文件名定位保存点，要求实际被选择的优先类别具有唯一来源。

    周期evaluation文件优先于last；同一预算终点可由不同保存路径生成多种last后缀。
    唯一evaluation已经明确来源时，这些last只留作库存记录；没有evaluation时仍拒绝多last歧义。
    """
    _require(nn_dir.is_dir(), f"nn_dir 必须是明确目录：{nn_dir}")  # 候选保存点的发现范围只有这一层目录。
    patterns = {
        "evaluation": re.compile(r"evaluation_.+_ep_(\d{5,})\.pth"),  # 既有零填充 update 格式。
        "last": re.compile(r"last_.+_ep_(\d+)_rew.*\.pth"),  # reward 字符串仅是文件名的一部分。
    }
    inventory: list[dict[str, Any]] = []  # 仅保存路径和 stat，无权重字节。
    ignored: list[str] = []  # 非上述命名的文件不猜测其 update。
    counts: Counter[tuple[int, str]] = Counter()  # 按(update,类别)核对优先来源的唯一性。
    for index, path in enumerate(nn_dir.iterdir()):  # 不递归，也不发现其它 run。
        _require(index < _INVENTORY_LIMIT, f"nn_dir 超过 {_INVENTORY_LIMIT} 项的枚举上限")  # 不跟随无限增长的目录。
        for kind, pattern in patterns.items():  # 同一 basename 最多匹配一种类别。
            match = pattern.fullmatch(path.name)  # 必须完整匹配已有命名。
            if match is None:  # 无匹配时继续尝试另一类。
                continue
            update = _integer(int(match[1]), f"{path.name}.update", 1)  # 文件名声明的更新位置。
            counts[(update, kind)] += 1  # 先收集完整库存，再判断哪种来源具有选择优先权。
            inventory.append({"path": str(path), "kind": kind, "update": update, **_stamp(path)})  # 仅定位。
            break  # 已经完整匹配文件名。
        else:  # 不识别普通 best-reward 保存名或其它扩展名。
            ignored.append(path.name)  # 保留排除理由的可审计输入范围。
    for (update, kind), count in counts.items():
        unique_source = count == 1 or (kind == "last" and counts[(update, "evaluation")] == 1)
        _require(unique_source, f"checkpoint 优先来源歧义：update={update}, kind={kind}, nn_dir={nn_dir}")
    return sorted(inventory, key=lambda item: (item["update"], item["kind"], item["path"])), sorted(ignored)  # 多last也按路径稳定排序。


def _proxy_eligibility(row: dict[str, Any], asset_count: int) -> list[str]:
    r"""验证首窗统计合法性，再报告样本资格缺口；合法低样本不是数据损坏。

    $0\le n_2\le n_1\le A_{obs}\le A$，$A w_{min}\le W\le A w_{max}$。
    浮点描述量只在 $A_{obs}=0$ 时为 null；此时全部窗口计数与版本界也必须为空。
    最佳候选额外要求 $A=A_{obs}=A_{qualified}=128$ 且 $w_{min}\ge16$。
    """
    where = f"global[update={row['update']}]"  # 将错误定位到真实保存/训练位置。
    counts = {key: _integer(row[key], f"{where}.{key}") for key in _FIRST30_INTS}  # 不截断浮点计数。
    total = counts["first30_asset_count"]  # 此 global 声明的资产分母。
    observed = counts["first30_observed_assets"]  # 有数据资产数。
    qualified = counts["first30_qualified_assets"]  # recorder 配置资格计数。
    low, high = counts["first30_windows_min"], counts["first30_windows_max"]  # 最弱/最强资产窗口数。
    windows = counts["first30_window_count"]  # 总留存窗口数。
    one, two = counts["first30_one_turn_proxy_assets"], counts["first30_two_turn_proxy_assets"]  # 两个训练代理计数。
    _require(total > 0 and 0 <= qualified <= observed <= total, f"{where}: first30 资产/观测/资格分母不合法")  # 资格集合必须包含于观测集合。
    _require(0 <= two <= one <= observed, f"{where}: first30 一圈/两圈代理计数不合法")  # 两圈代理集合必须包含于一圈集合。
    _require(low <= high and total * low <= windows <= total * high and windows >= observed, f"{where}: first30 窗口分母不合法")  # 各资产计数加和必须与总窗口数闭合。
    _require((low == 0) == (observed < total), f"{where}: first30_windows_min 与缺测资产不一致")  # 缺测资产必须在最弱样本门中显式贡献零。
    if observed == 0:  # 官方首窗 schema 的合法空状态。
        _require(windows == high == 0, f"{where}: 无观测却声明非零窗口")  # 空样本不能有任何留存事件。
        _require(all(row[key] is None for key in (*_FIRST30_FLOATS, *_FIRST30_VERSIONS)), f"{where}: 无观测的描述量/版本界必须为 null")  # 未定义中位数与真实零旋转是不同事实。
    else:  # 有观测时所有统计必须有限且合法，不能以 NaN 隐式失去排序资格。
        _require(high + (observed - 1) * max(low, 1) <= windows <= observed * high, f"{where}: 窗口总数与观测资产/最大窗数不一致")  # 最大计数至少被一个资产取得，其余观测资产至少一窗。
        if observed == total:  # 完整观测时至少有一个资产取得最小窗口数。
            _require(windows <= low + (total - 1) * high, f"{where}: 窗口总数与最小窗数不一致")  # 最小计数也必须被实际资产取得。
        _number(row["first30_net_median"], f"{where}.first30_net_median")  # 净圈保留正负号。
        _number(row["first30_goal_median"], f"{where}.first30_goal_median", 0.0)  # 次数不能为负。
        _number(row["first30_direction_median"], f"{where}.first30_direction_median", -1.0, 1.0)  # 训练方向为有符号统计。
        _number(row["first30_safe_fraction"], f"{where}.first30_safe_fraction", 0.0, 1.0)  # 资产等权比例。
        start, end = [_integer(row[key], f"{where}.{key}") for key in _FIRST30_VERSIONS]  # 策略版本范围。
        _require(start <= end, f"{where}: first30 策略版本范围反向")  # 留存窗口时间轴保持因果顺序。
    reasons = []  # 合法但不足以参加最佳排序的样本条件。
    for key in ("first30_asset_count", "first30_observed_assets", "first30_qualified_assets"):  # 三个分母缺一不可。
        if counts[key] != asset_count:  # 低样本/非目标分母只影响最佳资格。
            reasons.append(f"{key}={counts[key]}，要求 {asset_count}")  # 保留数值缺口。
    if low < 16:  # 固定最佳候选的每资产最少窗口门。
        reasons.append(f"first30_windows_min={low}，要求 >=16")  # 不以组平均样本量替代最弱资产。
    return reasons  # 空列表表示有限、合法且具有完整样本资格。


def _score(row: dict[str, Any]) -> tuple[int, int, float, float, int]:
    r"""最佳排序 $(n_1,n_2,\widetilde N,\bar S,-u)$；所有项均已通过合法性检查。"""
    return (
        row["first30_one_turn_proxy_assets"],  # 第一优先：一圈覆盖代理。
        row["first30_two_turn_proxy_assets"],  # 第二优先：两圈覆盖代理。
        row["first30_net_median"],  # 第三优先：资产净圈中心。
        row["first30_safe_fraction"],  # 第四优先：资产等权安全比例。
        -row["update"],  # 完全平局时偏好更早保存点。
    )


def _envelope(kind: str, inputs: dict[str, Any]) -> dict[str, Any]:
    r"""两个窄入口共用的 CPU 证据头；输入指纹包含路径、hash/stat 和显式参数。"""
    encoded = json.dumps(inputs, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")  # 确定字节编码。
    return {
        "artifact_type": f"anymani.family_teacher.{kind}",  # 分析结果有别于训练/正式评价产物。
        "schema_version": "1.0.0",  # 本只读入口的结果 schema。
        "execution": {"device": "cpu", "python": platform.python_version(), "machine": platform.machine(), "polars": pl.__version__},  # 不查询 GPU。
        "inputs": inputs,  # 完整可追溯输入范围。
        "input_fingerprint_sha256": hashlib.sha256(encoded).hexdigest(),  # 快照指纹，不冒充方法身份。
    }


def select_candidates(
    metrics_path: Path | str,
    nn_dir: Path | str,
    *,
    final_update: int = 2000,
    asset_count: int = 128,
    transitions_per_update: int = 61440,
) -> dict[str, Any]:
    r"""从完成表与已保存文件选择去重的训练最佳和最终，最多两个候选。

    Args:
        metrics_path: 已完成 run 的 metrics.parquet；不读取实时日志或补读 shards。
        nn_dir: 明确的保存点目录；只读文件名与 stat，不读取模型字节。
        final_update: 完整预算的终点，默认 2000；global 最大 update 必须恰为该值。
        asset_count: 最佳资格的三项资产分母，默认 128。
        transitions_per_update: 每 update 新环境转移数，默认 61440；$T_u=u\times61440$。

    Returns:
        JSON 安全报告：candidates、逐保存点资格理由、CPU 信息和输入指纹。

    Raises:
        AnalysisError: 身份/预算/统计不合法、缺最终点、保存点歧义或快照变化。
        OSError: 输入路径不存在或不可读。最终元数据校验由实际评价调用方另做。
    """
    final_update = _integer(final_update, "final_update", 1)  # 完整预算坐标。
    asset_count = _integer(asset_count, "asset_count", 1)  # 候选资格分母。
    transitions_per_update = _integer(transitions_per_update, "transitions_per_update", 1)  # 新样本预算。
    metrics = Path(metrics_path).expanduser().resolve()  # 完成表必须由调用者指明。
    nn = Path(nn_dir).expanduser().resolve()  # 禁止猜测训练目录或搜索其它 run。
    _require(metrics.name == "metrics.parquet", "仅接受已完成 run 的 metrics.parquet")  # recorder 的完成发布点是唯一指标入口。
    fingerprint = _hash_metrics(metrics)  # 对完整完成表的原始字节定锚。
    inventory, ignored = _checkpoint_inventory(nn)  # 文件名定位不加载权重。
    columns = ["schema_version", "identity_digest", "scope", "update", "transitions", *_FIRST30]  # 无 reward 投影。
    try:  # Polars 的底层解码错误统一附上输入路径。
        scan = pl.scan_parquet(metrics)  # 惰性执行，先读取 schema 元数据。
        missing = sorted(set(columns) - set(scan.collect_schema().names()))  # 必要首窗字段不能静默缺省。
        _require(not missing, f"metrics.parquet 缺少字段：{missing}")  # 缺首窗字段不能退化成奖励排序。
        rows = scan.filter(pl.col("scope") == "global").select(columns).limit(final_update + 1).collect().to_dicts()  # 内存只保留有界 global 投影。
    except pl.exceptions.PolarsError as error:  # 不回退到实时 shards 或 reward。
        raise AnalysisError(f"无法读取 metrics.parquet：{error}") from error
    _require(bool(rows), "metrics.parquet 没有 global 行，无法确认 final 终点")  # 终点必须有可核验的预算行。
    _require(len(rows) <= final_update, "global 行数超过 final_update，终点或唯一性不合法")  # u 从 1 起，合法唯一行数最多为终点值。
    by_update: dict[int, dict[str, Any]] = {}  # 每 update 恰好一个 global 事实。
    eligibility: dict[int, list[str]] = {}  # 全部 global 的首窗统计都需合法。
    identities: set[str] = set()  # 只核对所投影 global 层的训练方法身份。
    schemas: set[str] = set()  # 记录消费到的实际指标 schema。
    for row in rows:  # 固定资产层不会混入候选排序。
        update = _integer(row["update"], "global.update", 1)  # 禁止非整数版本坐标。
        _require(update <= final_update, f"global update={update} 超过 final 终点 {final_update}")  # 不截取更长 run 来冒充指定完整预算。
        _require(update not in by_update, f"update={update} 必须对应唯一 global 行")  # 相同内容的重复行同样破坏唯一对应关系。
        transitions = _integer(row["transitions"], f"global[{update}].transitions")  # 真实新样本量。
        _require(transitions == update * transitions_per_update, f"global[{update}].transitions={transitions}，期望 {update * transitions_per_update}")  # 更新次数与新样本预算精确闭合。
        identities.add(_digest(row["identity_digest"], f"global[{update}].identity_digest"))  # 同一方法身份。
        _require(isinstance(row["schema_version"], str) and bool(row["schema_version"]), f"global[{update}].schema_version 缺失")
        schemas.add(row["schema_version"])  # 字段合同已逐项验证，不猜测版本号对应的缺省值。
        eligibility[update] = _proxy_eligibility(row, asset_count)  # 低样本保留理由；非法数据抛错。
        by_update[update] = row  # 已验证的唯一 global。
    _require(len(identities) == 1, "global 行混合了多个 identity_digest")  # 不跨方法身份拼接一条最佳曲线。
    _require(max(by_update) == final_update, f"完成表缺少完整 final 终点 update={final_update}")  # 最大已发布 update 必须是明确完整终点。
    checkpoints: dict[int, dict[str, Any]] = {}  # 同 update 在两类间优先 evaluation。
    for checkpoint in inventory:  # 每个识别出的保存点都要有对应 global，不能只核对获选者。
        update = checkpoint["update"]  # 此值仍仅来自文件名。
        _require(update in by_update, f"checkpoint update={update} 没有唯一 global 行或超过 final 终点")  # 未获选保存点也必须与标量事实轴一致。
        if update not in checkpoints or checkpoint["kind"] == "evaluation":  # 跨类别的唯一优先规则。
            checkpoints[update] = checkpoint  # 禁止用奖励后缀消解同类歧义。
    _require(final_update in checkpoints, f"缺少完整 final 终点 checkpoint：update={final_update}")  # 终点指标不能替代终点权重文件。
    eligible = [update for update in checkpoints if not eligibility[update]]  # 只有实际保存的合格 update。
    best = max(eligible, key=lambda update: _score(by_update[update])) if eligible else None  # 不从未保存的曲线峰值造候选。
    chosen = list(dict.fromkeys([best, final_update] if best is not None else [final_update]))  # 最佳优先、终点去重、最多两项。
    candidates = []  # 简洁的下游正式评价任务输入。
    for update in chosen:  # 每个候选都保留科学选择依据和定位限制。
        roles = (["best_first30"] if update == best else []) + (["final"] if update == final_update else [])  # 去重不丢理由。
        candidates.append({
            "update": update,  # 文件名与唯一 global 的共同位置。
            "transitions": by_update[update]["transitions"],  # 精确预算坐标。
            "checkpoint": checkpoints[update],  # 只含文件系统元数据。
            "roles": roles,  # 训练最佳和/或完整终点。
            "reasons": ["first30 合格保存点按 (一圈代理,两圈代理,净圈中心,安全比例,-update) 取最大" if role == "best_first30" else f"完整预算终点 update={final_update}" for role in roles],  # 人类可核对。
            "first30_statistics": {key: by_update[update][key] for key in _FIRST30},  # 分母与排名事实同存。
        })
    _unchanged(metrics, fingerprint)  # CPU 选择期间不能改变完成表。
    _require((inventory, ignored) == _checkpoint_inventory(nn), "nn 保存点目录在分析期间发生变化")  # 候选集合及文件元数据必须来自同一静态快照。
    inputs = {"metrics": fingerprint, "nn_dir": str(nn), "checkpoint_inventory": inventory, "ignored_nn_entries": ignored, "parameters": {"final_update": final_update, "asset_count": asset_count, "transitions_per_update": transitions_per_update}}  # 快照与预算共同定锚。
    return {
        **_envelope("candidate_selection", inputs),  # 不包含任何模型运行证据。
        "method_identity_digest": next(iter(identities)),  # global 共同声明的方法身份。
        "metrics_schema_versions": sorted(schemas),  # 原始宽表版本证据。
        "global_row_count": len(rows),  # 本次只投影 global 的实际行数。
        "eligible_best_count": len(eligible),  # 仅计算已保存点。
        "candidates": candidates,  # 结果长度为 1 或 2。
        "saved_update_audit": [{"update": update, "eligible_as_best": not eligibility[update], "exclusion_reasons": eligibility[update]} for update in sorted(checkpoints)],  # 样本不足可解释。
        "checkpoint_metadata_verified": False,  # stat/文件名不能验证方法、epoch、frame 或权重 hash。
        "checkpoint_binding_note": "仅文件命名定位；主 agent 在正式评价前须另验 checkpoint SHA-256、方法 identity、epoch/frame 元数据。",  # 下游责任边界。
        "completion_evidence": "只接受 metrics.parquet 的完整终点与对应保存点；调用者负责提供已停止写入的完成 run。",  # 不查询进程或实时日志。
    }


def _cohort_axis(cohort: dict[str, Any]) -> list[tuple[str, str]]:
    r"""由 canonical 成员位置恢复 $(group,mother)$ 轴，要求 128=32×4。"""
    _expect_fields(cohort, {"schema_version": "1.2.0"}, "cohort")  # 只接受 canonical-final 合同。
    members = cohort.get("members")  # evaluator 使用成员列表位置作为 dataset_row。
    _require(isinstance(members, list) and len(members) == 128, "cohort.members 必须恰有 128 个成员")  # 不以动态观测成员数重定义资产分母。
    members = cast(list[Any], members)  # 上方已验证有序成员列表和长度。
    axis: list[tuple[str, str]] = []  # 每个 dataset_row 的拓扑标签与母体名。
    for index, value in enumerate(members):  # 不重新排序，以保留 evaluator 的真实位置轴。
        member = _object(value, f"cohort.members[{index}]")  # 必须是完整成员对象。
        _expect_fields(member, {"cohort_index": index}, f"cohort.members[{index}]")  # cohort_index 与成员位置一致。
        provenance = _object(member.get("provenance"), f"cohort.members[{index}].provenance")  # 真实 group/mother 来源。
        labels = [provenance.get(key) for key in ("group_name", "mother_name")]  # 跨 family 的母体可能同名。
        _require(all(isinstance(label, str) and label.strip() and "/" not in label for label in labels), f"cohort.members[{index}] 的 group_name/mother_name 必须非空且不含 /")
        group, mother = cast(list[str], labels)  # 每个名称均已验证为单个路径标签。
        axis.append((f"{group}/{mother}", mother))  # 与正式 R 的 topology_id 拼接方式一致。
    counts = Counter(topology for topology, _ in axis)  # 每拓扑统计资产代表，而非 simulation 副本。
    _require(len(counts) == 32 and set(counts.values()) == {4}, "cohort 必须为 32 拓扑、每拓扑恰好 4 代表")  # 29/32 与 2/4 门的两个分母在此固定。
    return axis  # 返回值位置就是 selection-local dataset_row。


def _physical_assets(document: dict[str, Any], axis: list[tuple[str, str]]) -> list[dict[str, Any]]:
    r"""验证 128 原始资产并复算正式方向；有限性/分母错误均使输入无效。

    安全比例来自 16 次首轨迹的二值存活事实，故 $16S_i$ 必须为整数。
    $D_i$ 比对允许 $10^{-12}$ 的序列化误差；实际门使用未放宽的复算值与 .7 比较。
    """
    physical = _object(document.get("physical_rotation"), "physical_rotation")  # 不读取 support/scale 字段。
    _expect_fields(physical, {"finite_and_identity_valid": True}, "physical_rotation")  # 全轴必须有效。
    raw = physical.get("asset_results")  # 唯一用于重新判门的逐资产事实源。
    _require(isinstance(raw, list) and len(raw) == 128, "physical_rotation.asset_results 必须恰有 128 行")  # 坏资产或失败资产不能通过删行消失。
    raw = cast(list[Any], raw)  # 类型收窄只发生在科学分母检查之后。
    results: dict[int, dict[str, Any]] = {}  # 保留失败资产；最终按 dataset_row 输出。
    for position, value in enumerate(raw):  # 输入顺序可不同，但身份必须一一对应。
        where = f"physical_rotation.asset_results[{position}]"  # 输入错误的确切位置。
        row = _object(value, where)  # 不能接受 null 或隐式缺测。
        index = _integer(row.get("dataset_row"), f"{where}.dataset_row")  # 0..127 的局部资产轴。
        _require(index < 128 and index not in results, f"{where}.dataset_row 必须唯一且位于 0..127")  # 128 个合法唯一整数自动构成完整局部轴。
        topology, mother = axis[index]  # cohort 显式绑定的母体。
        _expect_fields(row, {"mother_id": mother, "replica_count": 16, "finite": True}, where)  # 每资产身份/分母/有效性。
        net = _number(row.get("net_turns_median"), f"{where}.net_turns_median")  # 带符号的净圈中位数，圈。
        path = _number(row.get("absolute_path_turns_median"), f"{where}.absolute_path_turns_median", 0.0)  # 非负路径圈中位数。
        stated_direction = _number(row.get("directional_consistency"), f"{where}.directional_consistency", 0.0, 1.0)  # 正式声明值。
        direction = min(max(net, 0.0) / max(path, _EPSILON), 1.0)  # $D_i=clip(max(N_i,0)/max(P_i,2^{-23}),0,1)$。
        _require(math.isclose(stated_direction, direction, rel_tol=0.0, abs_tol=_TOLERANCE), f"{where}.directional_consistency={stated_direction} 与正式中位数之比 {direction} 不一致")  # 拒绝逐轨比中位数冒充正式资产方向。
        safe = _number(row.get("safe_replica_fraction"), f"{where}.safe_replica_fraction", 0.0, 1.0)  # 生存副本数/16。
        _require(math.isclose(safe * 16, round(safe * 16), rel_tol=0.0, abs_tol=_TOLERANCE), f"{where}.safe_replica_fraction 的分母必须是 16")  # 每副本贡献只能是 0 或 1 个安全事件。
        results[index] = {
            "dataset_row": index, "topology_id": topology, "mother_id": mother,  # 同时保留局部轴与拓扑身份。
            "replica_count": 16, "finite": True,  # 已验证的统计分母。
            "net_turns_median": net, "absolute_path_turns_median": path,  # 资产物理中心，单位圈。
            "directional_consistency": direction, "safe_replica_fraction": safe,  # 无量纲比例。
            "gaps": {"net_turns_to_one": max(1.0 - net, 0.0), "net_turns_to_two": max(2.0 - net, 0.0), "direction_to_0p7": max(0.7 - direction, 0.0), "safe_fraction_to_0p75": max(0.75 - safe, 0.0)},  # 逐资产缺口，不删除科学失败。
        }
    return [results[index] for index in range(128)]  # 长度与唯一范围已保证完整 128 行。


def _coverage(assets: list[dict[str, Any]], turns: int) -> dict[str, Any]:
    r"""独立重算一圈/两圈覆盖；$I_i=[N_i\ge k\land D_i\ge.7\land S_i\ge.75]$。

    资产分母为 128，拓扑分母为 32；每拓扑至少 2/4 代表通过。
    turns=2 只用于并列能力描述，最终强门使用 turns=1 的结果。
    """
    passed = [row for row in assets if row["net_turns_median"] >= turns and row["directional_consistency"] >= 0.7 and row["safe_replica_fraction"] >= 0.75]  # 三个科学门的交集。
    passed_rows = [row["dataset_row"] for row in passed]  # 保留通过资产集合。
    counts = Counter(row["topology_id"] for row in passed)  # 每个拓扑的通过代表数。
    topologies = sorted({row["topology_id"] for row in assets})  # 包括 0 个通过代表的拓扑。
    table = [{"topology_id": topology, "asset_count": 4, "passed_asset_count": counts[topology], "required_asset_count": 2, "passed": counts[topology] >= 2} for topology in topologies]  # 与真实 R 表逐字段同义。
    topology_count = sum(row["passed"] for row in table)  # 拓扑内半数门的等权覆盖。
    return {
        "thresholds": {**_R_THRESHOLDS, "net_turns_min": float(turns)},  # 两圈保持 .7/.75 不变。
        "finite": True, "asset_count": 128, "topology_count": 32,  # 固定完整分母。
        "passed_asset_count": len(passed), "passed_asset_rows": passed_rows,  # 资产覆盖的计数和集合。
        "passed_topology_count": topology_count, "topology_results": table,  # 拓扑覆盖的计数和逐项事实。
        "asset_fraction": len(passed) / 128, "topology_fraction": topology_count / 32,  # 分母绝不随失败变化。
        "failed_asset_rows": [row["dataset_row"] for row in assets if row["dataset_row"] not in passed_rows],  # 完整科学失败集合。
        "uncovered_topology_ids": [row["topology_id"] for row in table if not row["passed"]],  # 拓扑缺口位置。
    }


def _audit_reliable(reported: dict[str, Any], computed: dict[str, Any]) -> None:
    r"""R 的阈值、有限性、分母、通过集合和每一拓扑表项必须与重计一致。"""
    expected = {key: computed[key] for key in ("finite", "asset_count", "topology_count", "passed_asset_count", "passed_topology_count")}  # 不信任总计数。
    _expect_fields(reported, {"schema_version": "1.0.0", **expected}, "reliable_topology_coverage")  # 正式 R schema。
    _expect_fields(_object(reported.get("thresholds"), "R.thresholds"), _R_THRESHOLDS, "R.thresholds")  # 不消费 legacy 门。
    rows = reported.get("passed_asset_rows")  # 必须是集合意义的唯一局部索引列表。
    _require(isinstance(rows, list), "R.passed_asset_rows 必须是列表")
    rows = cast(list[Any], rows)  # 不接受 missing/null 等隐式空集合。
    values = [_integer(value, "R.passed_asset_rows") for value in rows]  # bool 不可冒充 row0/row1。
    _require(len(values) == len(set(values)) and sorted(values) == computed["passed_asset_rows"], "R.passed_asset_rows 与 128 原始资产重计不一致")
    raw_table = reported.get("topology_results")  # 验证具体拓扑而不是仅验证总覆盖数。
    _require(isinstance(raw_table, list) and len(raw_table) == 32, "R.topology_results 必须恰有 32 行")
    raw_table = cast(list[Any], raw_table)  # 已验证完整 32 拓扑分母。
    actual_table: dict[str, dict[str, Any]] = {}  # 允许表行重排，不允许同拓扑重复。
    for value in raw_table:  # 逐拓扑的通过代表计数必须可审计。
        row = _object(value, "R.topology_results[]")  # 不能接受缺表或 null 表项。
        label = row.get("topology_id")  # 正式 group/mother 完整身份。
        _require(isinstance(label, str) and label not in actual_table, "R.topology_results.topology_id 必须唯一")
        actual_table[cast(str, label)] = row  # 同名短 mother 不会被自动合并。
    _require(set(actual_table) == {row["topology_id"] for row in computed["topology_results"]}, "R.topology_results 与 cohort 的 group_name/mother_name 拓扑集合不一致")
    for reference in computed["topology_results"]:  # 同时核对 asset_count/required_count/passed_count/passed。
        _expect_fields(actual_table[reference["topology_id"]], reference, f"R.topology_results[{reference['topology_id']}]")


def evaluate_teacher(
    evaluation_path: Path | str,
    cohort_path: Path | str,
    *,
    expected_method_identity_digest: str,
    expected_checkpoint_sha256: str,
) -> dict[str, Any]:
    r"""复核正式 fullKD 教师固定强门，纯读取 JSON，不运行任何实际评价。

    Args:
        evaluation_path: 已结束的真实固定能力评价 JSON。
        cohort_path: schema-1.2 canonical lock 路径；以原始 JSON 文件 hash 绑定集合。
        expected_method_identity_digest: 调用者外部核定的方法 digest，必须显式给出。
        expected_checkpoint_sha256: 调用者外部核定的 checkpoint SHA-256，必须显式给出。

    Returns:
        合法输入返回 passed、独立 1/2 圈覆盖、资产净圈中心/安全、缺口与证据指纹。
        passed=False 只表示一圈覆盖不足 103/128 或拓扑不足 29/32，不表示数据错误。

    Raises:
        AnalysisError: 身份/协议/有限性/分母/拓扑或 R 自报内容不符。
        OSError: 显式文件路径不可读。函数不会从评价文档反向推导 expected 身份。
    """
    method = _digest(expected_method_identity_digest, "expected_method_identity_digest")  # 独立方法锚点。
    checkpoint = _digest(expected_checkpoint_sha256, "expected_checkpoint_sha256")  # 独立权重锚点。
    evaluation = Path(evaluation_path).expanduser().resolve()  # 只读明确的已结束评价。
    cohort_file = Path(cohort_path).expanduser().resolve()  # 只读明确的 canonical 集合。
    document, evaluation_input = _read_json(evaluation)  # JSON 与其 hash 使用同一字节快照。
    cohort, cohort_input = _read_json(cohort_file)  # 文件扩展名不改变解析协议。
    _expect_fields(document, {"artifact_type": "anymani.palm_rotation_support_fixed_evaluation"}, "evaluation")  # 排除诊断角色产物。
    identity = _object(document.get("evaluation_identity"), "evaluation_identity")  # 必要的身份层。
    _expect_fields(identity, {"manifest_sha256": cohort_input["sha256"], "method_identity_digest": method, "checkpoint_sha256": checkpoint}, "evaluation_identity")  # 三个独立身份锚点。
    protocol = _object(identity.get("protocol"), "evaluation_identity.protocol")  # 必要的固定测量合同。
    _expect_fields(protocol, _PROTOCOL, "evaluation_identity.protocol")  # fullKD、ADR0、TIP-only、无干预等逐项核对。
    _expect_fields(_object(protocol.get("reliable_topology_coverage_thresholds"), "protocol.reliable_topology_coverage_thresholds"), _R_THRESHOLDS, "protocol.reliable_topology_coverage_thresholds")  # 双位置阈值声明同义。
    axis = _cohort_axis(cohort)  # 独立恢复 32×4 拓扑结构。
    assets = _physical_assets(document, axis)  # 原始 128 行，禁止删除失败或非有限行。
    one = _coverage(assets, 1)  # 一圈可靠覆盖是最终强门事实。
    two = _coverage(assets, 2)  # 两圈独立重算，方向门仍为 .7。
    _audit_reliable(_object(document.get("reliable_topology_coverage"), "reliable_topology_coverage"), one)  # 不信任 R 自报通过数。
    asset_gate = one["passed_asset_count"] >= 103  # 强门的资产条件。
    topology_gate = one["passed_topology_count"] >= 29  # 强门的独立拓扑条件。
    net = [row["net_turns_median"] for row in assets]  # 128 个资产等权净圈中心，单位圈。
    safe = [row["safe_replica_fraction"] for row in assets]  # 128 个 R16 安全比例。
    for path, fingerprint in ((evaluation, evaluation_input), (cohort_file, cohort_input)):  # 返回前再次固定输入快照。
        _unchanged(path, fingerprint)  # 拒绝分析期间改写的产物。
    inputs = {"evaluation": evaluation_input, "cohort": cohort_input, "expected_method_identity_digest": method, "expected_checkpoint_sha256": checkpoint}  # 显式 expected 进入证据指纹。
    return {
        **_envelope("fixed_gate_analysis", inputs),  # CPU 产物分析身份。
        "passed": asset_gate and topology_gate,  # 103/128 与 29/32 的交集。
        "strong_gate": {"required_assets": 103, "required_topologies": 29, "asset_gate_passed": asset_gate, "topology_gate_passed": topology_gate},  # 两门独立报告。
        "one_turn": one, "two_turn": two,  # 完整计数、比例、通过 rows、失败 rows、拓扑表。
        "asset_summary": {
            "asset_count": 128, "replicas_per_asset": 16, "replica_count": 128 * 16,  # 总副本分母为 2048。
            "net_turns_median": math.fsum(value / 2 for value in sorted(net)[63:65]),  # 两个中心资产先除以 2，避免中位数求和溢出；单位圈。
            "net_turns_mean": math.fsum(value / 128 for value in net),  # 先除以分母，避免可避免的求和溢出。
            "net_turns_min": min(net), "net_turns_max": max(net),  # 完整资产轴的净圈范围，圈。
            "safe_fraction_mean": math.fsum(safe) / 128, "safe_fraction_median": statistics.median(safe), "safe_fraction_min": min(safe),  # 资产等权安全描述。
            "safe_replica_count": sum(round(value * 16) for value in safe),  # 已核对 R16 格点的真实安全副本计数。
        },
        "gaps": {"assets_to_103": max(103 - one["passed_asset_count"], 0), "topologies_to_29": max(29 - one["passed_topology_count"], 0)},  # 强门缺口只用一圈可靠覆盖。
        "asset_results": assets,  # 每资产物理中心、安全、方向与 1/2 圈连续缺口。
        "reliable_topology_coverage_verified": True,  # 阈值、有限性、分母、集合、拓扑表均完成复核。
    }


def main(argv: Sequence[str] | None = None) -> int:
    r"""只读分析 CLI；合法科学未达标返回 0，输入/身份/输出冲突退出 2。

    --output 必须是已有目录内的新文件；使用 x 模式保证不能覆盖已有结果或符号链接。
    省略 --output 时只打印完整 JSON。两个分析函数本身均无文件写入副作用。
    """
    parser = argparse.ArgumentParser(description="家族 fullKD 教师的 CPU 产物只读分析")  # 不包含训练参数。
    commands = parser.add_subparsers(dest="command", required=True)  # 恰好两个明确职责。
    select = commands.add_parser("select", help="从已完成 metrics.parquet 和显式 nn 选择最多两个候选")  # 训练代理选择。
    select.add_argument("metrics", type=Path)  # 不接受 run 自动搜索。
    select.add_argument("nn_dir", type=Path)  # 调用者明确保存目录。
    select.add_argument("--final-update", type=int, default=2000)  # 完整预算终点。
    select.add_argument("--asset-count", type=int, default=128)  # 候选资格的资产分母。
    select.add_argument("--transitions-per-update", type=int, default=61440)  # 新样本预算。
    evaluate = commands.add_parser("evaluate", help="复核已完成评价 JSON 的固定强门，不运行实际评价")  # 事后复核。
    evaluate.add_argument("evaluation", type=Path)  # 真实评价文档路径。
    evaluate.add_argument("cohort", type=Path)  # canonical JSON 路径。
    evaluate.add_argument("--expected-method-identity-digest", required=True)  # 禁止由输入自报反推。
    evaluate.add_argument("--expected-checkpoint-sha256", required=True)  # 必须由主 agent 另验绑定。
    for command in (select, evaluate):  # 两个子命令使用相同的排他输出语义。
        command.add_argument("--output", type=Path, help="只创建新 JSON；省略时打印到标准输出")  # 无隐式输出路径。
    args = parser.parse_args(argv)  # 缺必填 expected 等参数由 argparse 明确拒绝。
    try:  # 先完成全部分析与序列化，合法后才可能创建结果文件。
        if args.command == "select":  # 不存在调用模型的执行分支。
            report = select_candidates(args.metrics, args.nn_dir, final_update=args.final_update, asset_count=args.asset_count, transitions_per_update=args.transitions_per_update)
        else:  # evaluate 表示复核既有证据，而不是启动评价器。
            report = evaluate_teacher(args.evaluation, args.cohort, expected_method_identity_digest=args.expected_method_identity_digest, expected_checkpoint_sha256=args.expected_checkpoint_sha256)
        text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"  # 不发布 NaN JSON。
        if args.output is None:  # 默认没有磁盘副作用。
            print(text, end="")  # 人类/管道均能读取完整报告。
        else:  # 排他创建，不创建父目录、不使用覆盖式 rename。
            output = args.output.expanduser()  # 不 resolve 输出链接，x 模式会拒绝任何已有目录项。
            with output.open("x", encoding="utf-8") as stream:  # 操作系统保证不能覆盖已有文件。
                stream.write(text)  # 只写显式的新结果路径。
            print(json.dumps({"output": str(output.absolute())}, ensure_ascii=False))  # 返回清楚的产物位置。
    except (AnalysisError, OSError, ValueError) as error:  # 科学未达标本身不会走错误分支。
        parser.error(str(error))  # 标准 CLI 输入错误状态 2。
    return 0  # 表示只读分析成功完成，不表示教师通过强门。


if __name__ == "__main__":  # 推荐直接执行文件以保持无环境导入的 CPU 边界。
    raise SystemExit(main())  # 无训练、仿真或 GPU 生命周期。
