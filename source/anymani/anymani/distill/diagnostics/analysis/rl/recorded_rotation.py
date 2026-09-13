r"""只读训练回合的掌旋对比：随机动作、跨策略版本、实际交互预算与低样本分母。

CLI只接收 ``--manifest Path --output_dir Path``。JSON manifest包含project_root及
runs[{name, run_dirs, total_transitions, reward_label}]；相对run_dirs以project_root为根。
前两项依次是候选方法与同预算、同奖励的首要参照，后续项仅作明确标注的背景参照。
一个逻辑run可以合并多个物理恢复段；策略版本已经是原始全局transition坐标，不添加偏移。
仅消费各run_dirs/episodes/*.parquet，不导入训练、模型、任务、Isaac或Torch模块。

科学合同：C为实际总transition数，选episode起点v满足0.80C <= v < 0.95C。
首30秒已观察完整，或在此前真实drop/axis，才形成可观察窗口；停训删失不补零。
令F=terminal drop OR axis，B=first30_complete，D=duration_s，则
$$n_{30}=B?n_{\mathrm{first30}}:n_{\mathrm{terminal}},\quad
s_{30}=B\land(D>30.00001\lor\neg F).$$
回合终点30秒之后的失败不追溯污染已完成的首30秒；恰在30秒失败则不安全。
每资产先求episode净圈中位数，再在每族128资产间求中位数及Q25/Q75；安全率先求
每资产safe fraction，再按资产等权求均值。分位数使用NumPy的linear插值。
asset_index 0..127为LEAP，128..255为Allegro；缺窗口的资产保留空值与覆盖分母。

全程自然结束回合另报N、终点净圈>=1/2与max；后期end_version>=0.80C且安全完整
120秒的回合另报同一回合net120-net30的中位数，明确其存活条件与实际分母。
本工具不给出冻结R16评价或正式reliable结论。资产配对只采用调用方声明的共同索引。
manifest可附expected_checks（summary点分路径到数值）与anchor_atol；锚点仅用于
独立核对，绝不参与数据或分母计算。不一致时保留真实统计、列出差异并以退出码2返回。
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

# 时间与资产合同直接来自本次科学接口；数值容差只用于已落盘浮点时间的闭合。
_DT = 0.05  # 每policy step的模拟时间，单位秒；30秒=600步，120秒=2400步。
_EPS = 1e-5  # recorder允许的绝对物理量舍入误差，安全边界仍严格使用30.00001。
_FAMILIES = {"LEAP": range(128), "Allegro": range(128, 256)}  # 每族固定128个声明资产。
_KEYS = ["segment_id", "env_id", "episode_id"]  # 恢复段命名空间内的真实回合身份。
_INTS = ("env_id", "episode_id", "asset_index", "policy_version_start", "policy_version_end", "policy_steps")  # 身份与交互计数必须为整数。
_FLOATS = ("duration_s", "net_turns", "absolute_path_turns", "max_positive_net_turns", "net_turns_first30", "policy_dt_s")  # 物理量按秒或圈保存。
_BOOLS = ("termination_drop", "termination_axis", "termination_timeout", "censored", "first30_complete")  # 联合事件标志决定观察性与安全性。
_STRINGS = ("schema_version", "identity_digest", "segment_id")  # 格式版本与方法/进程身份均须显式保留。
_OUTPUTS = ("manifest.json", "summary.json", "per_asset.csv", "paired_per_asset.csv", "late_first30.png")  # 派生产物的完整写入范围。


def _require(condition: bool | np.bool_, message: str) -> None:
    r"""科学合同失败即停止；不通过删除坏行、去重或改写预算继续计算。"""
    if not condition:  # 合同是不可放宽的逻辑命题，不是允许失败行继续进入计算的筛选器。
        raise ValueError(message)  # 错误信息绑定原始run/字段，供调用方返回最终真源核查。


def _protocol() -> dict[str, Any]:
    r"""给每个JSON保留独立可读的随机训练语义、公式与分母解释。"""
    return {  # 每个独立JSON携带同一科学合同，避免脱离代码后把训练证据读成固定评估。
        "evidence": "随机训练回合；跨策略版本；仅分析已记录Parquet",  # 随机训练属性来自调用方声明。
        "policy_coordinate": "原始policy_version_start/end，以已消费transitions计；不重置恢复段坐标",  # 坐标是交互数，不是优化器步数。
        "budget": "每run实际sum(policy_steps)，必须等于manifest.total_transitions",  # 不使用计划预算补分母。
        "family_mapping": "asset_index 0..127=LEAP，128..255=Allegro，各声明128资产",  # 索引划分由调用方给定。
        "late_first30": "0.80*C <= policy_version_start < 0.95*C；first30_complete或更早真实drop/axis",  # 起点版本使用半开区间。
        "net30": "first30_complete ? net_turns_first30 : terminal net_turns",  # 有符号净转，单位圈。
        "safe30": "first30_complete AND (duration_s > 30.00001 OR NOT (termination_drop OR termination_axis))",  # 超过30秒的后续失败不追溯扣安全。
        "net_aggregation": "每资产episode median -> 已覆盖资产median；另报资产间Q25/Q75",  # 中心以资产为统计单位。
        "safety_aggregation": "每资产safe fraction -> 已覆盖资产等权mean；另报资产间Q25/Q75",  # 高失败频率资产不会因回合多而增权。
        "quantile_method": "linear",  # Q25/Q75描述资产分布，不是置信区间。
        "low_sample_denominator": "训练低样本窗口；逐资产报告N，缺测保留null，覆盖不足不补0；不假定独立重复",  # 覆盖与估计不混淆。
        "full_episode": "全训练期间自然结束回合，终点net>=1/2及max；与首30秒窗口分母分开",  # 整圈尾部使用自然终点净转。
        "late_safe120": "end_version>=0.80*C，未删失、timeout、无drop/axis、2400步；同回合net120-net30",  # 条件存活总体。
        "pairing": "前两run同预算同reward_label；按声明的共同asset_index配对；delta=candidate-reference",  # 配对方向固定。
        "interpretation": "非冻结R16；不提供正式reliable结论；单次训练描述性比较，不能推断算法因果优劣",  # 结论边界随数据发布。
    }


def _stats(values: Sequence[float] | np.ndarray) -> dict[str, Any]:
    r"""返回实际数值总体的N、均值和分位数；空总体的所有估计量为null。"""
    x = np.asarray(values, dtype=np.float64)  # 只在已审计的观测值上计算，单位由调用方字段说明。
    if x.size == 0:  # 分母为零时任何数值中心都不可估计。
        return dict.fromkeys(("min", "q25", "median", "q75", "max", "mean")) | {"n": 0}  # 空估计量为None，序列化为null。
    # 线性分位数与偶数样本中位数一致；不把资产内回合视为新增独立资产。
    quantiles = np.quantile(x, [0, 0.25, 0.5, 0.75, 1], method="linear")  # 五个分位点，单位与x相同。
    return dict(zip(("min", "q25", "median", "q75", "max"), map(float, quantiles), strict=True)) | {  # 转为JSON可保存的Python标量。
        "n": int(x.size),  # 分母来自真正传入的观测总体。
        "mean": float(x.mean()),  # 安全资产率的等权中心使用此均值。
    }


def _audit(table: pl.DataFrame, budget: int, label: str) -> dict[str, Any]:
    r"""在全部原始行上验证身份、类型、有限性、时间、版本和合法终止关系。

terminal=drop OR axis OR timeout必须恰为NOT censored；真实终止标志允许并存。
首30秒完成位必须等价于至少600个已观察步；不根据净圈值猜测完成位。
时间闭合沿用producer的rtol=1e-6、atol=1e-5秒；不修改落盘时间或policy_steps。
"""
    required = set(_INTS + _FLOATS + _BOOLS + _STRINGS)  # 只要求分析所需的完整联合事实。
    _require(table.height > 0 and required <= set(table.columns), f"{label}: empty/missing columns {required - set(table.columns)}")  # 联合列缺失不能重建真值。
    _require(sum(table.null_count().row(0)) == 0, f"{label}: null in raw evidence")  # 坏行不能静默丢弃。
    for name, dtype in table.schema.items():  # 核对全部原始列，而非只核对最终会被选中的回合。
        # 所有原始数值列（包括不用于比较的奖励积分）也须有限，确保不是局部挑好列。
        if dtype.is_numeric():  # NaN与Inf对净圈、奖励或计数均没有合法物理含义。
            _require(bool(np.isfinite(table[name].to_numpy()).all()), f"{label}: nonfinite {name}")  # 任一坏值使整次审计失败。
        if name in _INTS or name in ("goal_count", "frontier_count", "orientation_goal_count", "adr_position_level"):  # 附加计数列也保留整数合同。
            _require(dtype.is_integer() and bool((table[name] >= 0).all()), f"{label}: invalid nonnegative integer {name}")  # 禁止浮点ID截断后混入统计。
        if name in _FLOATS:  # 时长与净转接受数值schema，但不接受可转换的字符串。
            _require(dtype.is_numeric(), f"{label}: nonnumeric physical field {name}")  # 禁止字符串到数值的隐式修复。
        if name in _BOOLS:  # 事件真假必须来自producer的显式布尔字段。
            _require(dtype == pl.Boolean, f"{label}: {name} must be boolean")  # 不把0/1数值代替schema布尔量。
        if name in _STRINGS:  # 空身份不能确定方法或恢复段归属。
            _require(dtype == pl.String and bool((table[name].str.len_chars() > 0).all()), f"{label}: invalid {name}")  # 所有身份字符串必须非空。
    # 一个逻辑训练run只能有一个方法identity；多个进程段共享它，但保留不同segment_id。
    identities = table["identity_digest"].unique().to_list()  # SHA身份只来源于原始episode列。
    _require(len(identities) == 1 and re.fullmatch(r"[0-9a-f]{64}", identities[0]) is not None, f"{label}: identity not single SHA256")  # 不混合方法身份。
    _require(table.select(_KEYS).unique().height == table.height, f"{label}: duplicate episode keys across shards/segments")  # unique仅用于审计，不改原始行。
    _require(set(table["schema_version"].unique().to_list()) == {"1.1.0"}, f"{label}: unsupported episode schema")  # 当前首30秒字段对应1.1.0。
    a = {name: table[name].to_numpy() for name in _INTS + _FLOATS + _BOOLS}  # [E]列，不导入任何环境张量。
    terminal = a["termination_drop"] | a["termination_axis"] | a["termination_timeout"]  # 三终止可重叠。
    _require(bool((terminal == ~a["censored"]).all()), f"{label}: terminal/censored are not exclusive and exhaustive")  # 每行恰为自然结束或右删失。
    _require(bool((a["policy_steps"] >= 1).all()), f"{label}: episode has no observed steps")  # 每行必须承载真实交互。
    _require(bool((a["asset_index"] < 256).all()), f"{label}: asset outside declared 256-asset cohort")  # 非负性已核对，合法范围为0..255。
    _require(bool((a["policy_dt_s"] == _DT).all()), f"{label}: policy_dt_s differs from 0.05 s")  # 每步模拟时间必须一致。
    _require(bool(np.allclose(a["duration_s"], _DT * a["policy_steps"], rtol=1e-6, atol=_EPS)), f"{label}: duration != .05*steps")  # $D=0.05K$，保留producer舍入容差。
    _require(bool((a["first30_complete"] == (a["policy_steps"] >= 600)).all()), f"{label}: first30 flag/observed steps disagree")  # $B\iff K\ge600$。
    _require(bool((~a["termination_timeout"] | (a["policy_steps"] == 2400)).all()), f"{label}: timeout is not at 120 s")  # timeout应对应名义120秒horizon。
    _require(bool((a["policy_version_start"] <= a["policy_version_end"]).all()), f"{label}: reversed policy versions")  # 版本坐标随交互单调。
    _require(bool((a["policy_version_end"] < budget).all()), f"{label}: policy version outside actual budget")  # 最后一步动作的版本仍在已消费总预算之前。
    # 净转与累计路径的基本几何不等式也在全体行核对，首30秒净圈不会大于全回合绝对路径。
    _require(bool((a["absolute_path_turns"] >= 0).all()), f"{label}: negative absolute path")  # 路径长度非负。
    _require(bool((a["absolute_path_turns"] + _EPS >= np.abs(a["net_turns"])).all()), f"{label}: path < abs(net)")  # 累计路径不小于净位移绝对值。
    _require(bool((a["max_positive_net_turns"] + _EPS >= np.maximum(a["net_turns"], 0)).all()), f"{label}: invalid positive frontier")  # 正向历史峰值包含当前正向净转。
    _require(bool((~a["first30_complete"] | (np.abs(a["net_turns_first30"]) <= a["absolute_path_turns"] + _EPS)).all()), f"{label}: first30 exceeds absolute path")  # 已锁存前缀受全回合路径界约束。
    # 同一物理进程的env_id固定绑定资产；合法恢复段可重新建立env命名空间。
    bindings = table.group_by("segment_id", "env_id").agg(pl.col("asset_index").n_unique())  # 每env的资产数。
    _require(bool((bindings["asset_index"] == 1).all()), f"{label}: env asset binding changed within segment")  # 同一env不能在回合间暗换手型。
    steps = int(table["policy_steps"].sum())  # 完整回合与删失尾段都进入交互预算。
    _require(steps == budget, f"{label}: sum(policy_steps)={steps} != total_transitions={budget}")  # 整数精确闭合，不容许预算近似。
    return {  # 审计事实与物理恢复段信息一同进入summary。
        "status": "passed",  # 上述断言全部通过才发布统计。
        "rows": table.height, "complete": int((~a["censored"]).sum()), "censored": int(a["censored"].sum()),  # 全部行=完整行+删失行。
        "sum_policy_steps": steps, "declared_total_transitions": budget, "discarded_invalid_rows": 0,  # 交互分母包括删失尾段。
        "unique_episode_keys": True, "identity_digest": identities[0], "identity_count": len(identities),  # 单一方法身份和全段回合键闭合。
        "segments": sorted(table["segment_id"].unique().to_list()),  # 物理恢复段数量与UUID可复查。
        "schema_versions": sorted(table["schema_version"].unique().to_list()),  # 原始分片schema可追溯。
        "asset_count": table["asset_index"].n_unique(),  # 缺资产由覆盖报告公开，不伪造观测。
        "policy_version_start_min": int(a["policy_version_start"].min()),  # 全run最早起点版本。
        "policy_version_end_max": int(a["policy_version_end"].max()),  # 全run最晚已记录终点版本。
        "cross_policy_version_episodes": int((a["policy_version_end"] > a["policy_version_start"]).sum()),  # 跨策略版本属性的实际计数。
        "duration_max_abs_error_s": float(np.max(np.abs(a["duration_s"] - _DT * a["policy_steps"]))),  # 报告实际舍入误差而非只报通过。
        "flag_combinations": table.group_by(list(_BOOLS)).len().sort(list(_BOOLS)).to_dicts(),  # 保留复合终止事实。
    }


def _read_run(root: Path, spec: dict[str, Any], used_files: set[Path]) -> tuple[pl.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    r"""读取声明路径的episode字节、计算SHA256并合并；恢复段从不去重或补偏移。"""
    tables, sources, directories = [], [], []  # 分片表、精确字节身份、物理目录分母。
    for relative in spec["run_dirs"]:  # manifest提供全部物理恢复段，不从目录名猜测遗漏段。
        directory = (root / relative).resolve()  # Path对绝对relative保留其绝对含义。
        files = sorted((directory / "episodes").glob("*.parquet"))  # 输入范围只到episode直接子文件。
        _require(bool(files), f"{spec['name']}: no episodes/*.parquet in {directory}")  # 缺失目录不能当成零transition段。
        local = []  # 当前物理目录的原始分片，不跨目录重新解释policy_version。
        for path in files:  # 文件名排序只稳定来源报告，不改变回合统计权重。
            path = path.resolve()  # 检测重复路径或多个manifest项指向同一原始分片。
            _require(path not in used_files, f"duplicated input file: {path}")  # 防止两个路径入口重复扩大分母。
            used_files.add(path)  # 同一字节证据只能被一个逻辑run消费一次。
            before = path.stat()  # 记录读取前的文件状态，分析末尾核对输入稳定。
            payload = path.read_bytes()  # 只读；统计与哈希必须对应同一批字节。
            table = pl.read_parquet(io.BytesIO(payload))  # CPU内存读取；保留所有行与原始列类型。
            after = path.stat()  # 拒绝在读取中被追加/替换的证据。
            _require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), f"input changed while reading: {path}")  # 一次分析需要稳定字节快照。
            sources.append({  # 分片身份包含SHA、行数和读取时刻文件状态。
                "path": str(path), "sha256": hashlib.sha256(payload).hexdigest(),  # 精确来源字节指纹。
                "bytes": len(payload), "mtime_ns": after.st_mtime_ns, "rows": table.height,  # 字节量、纳秒mtime与原始回合数。
            })
            local.append(table)  # 不筛选、不去重，合并后整体审计。
        combined = pl.concat(local, how="vertical")  # 不同schema会显式失败，不放宽类型拼接。
        tables.append(combined)  # 多物理段作为一个逻辑训练预算。
        directories.append({  # 每物理目录对逻辑run总预算的贡献单独发布。
            "run_dir": str(directory), "files": len(files), "rows": combined.height,  # 真实绝对路径及分片/回合数。
            "complete": int((~combined["censored"]).sum()), "sum_policy_steps": int(combined["policy_steps"].sum()),  # 自然结束数与实际交互数。
            "segment_ids": sorted(combined["segment_id"].unique().to_list()),  # 每目录保留真实进程身份。
        })
    table = pl.concat(tables, how="vertical")  # 所有物理段的完整原始总体。
    audit = _audit(table, spec["total_transitions"], spec["name"])  # 审计优先于任何科学窗口筛选。
    audit["run_directories"] = directories  # 报告每段对总交互预算的贡献。
    return table, audit, sources  # 原始联合表只读传递，不用审计摘要重建回合。


def _summarize(spec: dict[str, Any], table: pl.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    r"""先形成真实可观察窗口，再按资产、家族两层聚合；完整回合与条件存活另列。

缺测资产仍输出一行，窗口数为0但median/safe fraction为null。每族中心的实际资产
分母与声明128同时报告；Q25/Q75基于资产中位数，不能用回合池化的分位数替代。
"""
    a = {name: table[name].to_numpy() for name in _INTS + _FLOATS + _BOOLS}  # 各字段均为[E]。
    budget = spec["total_transitions"]  # C为已通过预算闭合的总transition数。
    late = (100 * a["policy_version_start"] >= 80 * budget) & (100 * a["policy_version_start"] < 95 * budget)  # 精确整数半开区间。
    failure = a["termination_drop"] | a["termination_axis"]  # 真失败，包括两者同时发生。
    natural = ~a["censored"]  # 已审计为至少一种自然终止，包含timeout。
    observed = a["first30_complete"] | (natural & failure)  # B或更早真实失败；未满30秒删失不能成为观察窗口。
    selected = late & observed  # 首要总体；其他回合只进入自己的分母。
    net30 = np.where(a["first30_complete"], a["net_turns_first30"], a["net_turns"])  # 每窗口有符号圈。
    safe30 = a["first30_complete"] & ((a["duration_s"] > 30.00001) | ~failure)  # 精确实现边界时刻语义。
    end_late = 100 * a["policy_version_end"] >= 80 * budget  # 条件120秒总体只限制回合终点版本。
    full120 = natural & (a["policy_steps"] == 2400)  # 120秒真实完整回合，不接受停训时刻的假结束。
    safe120 = end_late & full120 & a["termination_timeout"] & ~failure & a["first30_complete"]  # 同回合存活条件。
    delta120 = a["net_turns"] - a["net_turns_first30"]  # 每个条件回合的后90秒净增量，单位圈。
    assets, families = [], {}  # 恰好输出每run的256行声明资产以及两个家族摘要。
    for family, indices in _FAMILIES.items():  # LEAP与Allegro各自归约，家族之间不混合分母。
        family_rows = []  # 逐资产统计是家族中心的唯一输入。
        family_mask = np.isin(a["asset_index"], list(indices))  # 由用户给定的固定索引划分家族。
        for index in indices:  # 声明的128个资产全部输出，覆盖不足也保留索引。
            own = a["asset_index"] == index  # 当前资产所有原始回合。
            window = own & selected  # 当前资产的可观察late首30秒窗口。
            candidate_n = int((own & late).sum())  # 未筛观察性之前的真实候选数。
            n = int(window.sum())  # 每资产不同N不能池化成家族权重。
            net_stats = _stats(net30[window])  # 当前资产内回合净圈的分布。
            safe_n = int((window & safe30).sum())  # 安全首30秒的真实窗口数。
            completed = own & natural  # 全程完整回合单独总体。
            continued = own & safe120  # 后期安全完整120秒的条件总体。
            row = {  # 当前资产一行，保存可重新形成家族中心的统计量与分母。
                "run": spec["name"], "reward_label": spec["reward_label"], "total_transitions": budget,  # 方法/奖励/实际预算随每行保存。
                "evidence": "stochastic_training_cross_policy_versions_low_sample",  # CSV独立阅读也不丢证据属性。
                "family": family, "asset_index": index, "expected_assets_in_family": len(indices),  # 保留配对键与家族声明分母。
                "late30_candidate_n": candidate_n, "late30_observed_n": n, "late30_unobserved_n": candidate_n - n,  # 窗口候选=已观察+观察不足。
                "late30_net_median": net_stats["median"], "late30_net_q25": net_stats["q25"], "late30_net_q75": net_stats["q75"],  # 资产内净圈分布，单位圈。
                "late30_safe_n": safe_n, "late30_safe_fraction": safe_n / n if n else None,  # 无窗口不伪造0安全率。
                "late30_first30_complete_n": int((window & a["first30_complete"]).sum()),  # 满30秒的观察窗口数。
                "late30_early_drop_axis_n": int((window & ~a["first30_complete"]).sum()),  # 未满30秒但真实失败的观察窗口数。
                "late30_cross_policy_version_n": int((window & (a["policy_version_end"] > a["policy_version_start"])).sum()),  # 该资产窗口跨版本的实际数量。
                "late30_single_window_net_ge1_n": int((net30[window] >= 1).sum()),  # 单个真实窗口>=1圈，不是资产中位数门。
                "late30_single_window_net_ge2_n": int((net30[window] >= 2).sum()),  # 单窗口至少2圈的尾部计数。
                "all_complete_n": int(completed.sum()), "all_complete_net_ge1_n": int((a["net_turns"][completed] >= 1).sum()),  # 全程自然结束数及其中整圈数。
                "all_complete_net_ge2_n": int((a["net_turns"][completed] >= 2).sum()),  # 完整回合终点至少2圈的实际数量。
                "all_complete_max_net": _stats(a["net_turns"][completed])["max"],  # 终点max，不读取历史正向峰值代替。
                "late_safe120_n": int(continued.sum()), "late_safe120_delta_median": _stats(delta120[continued])["median"],  # 条件存活数量与同回合增量中心。
            }
            family_rows.append(row)  # 所有声明资产保留，包括零覆盖资产。
        # 资产等权聚合只接受非空资产估计量；并公开缺测索引、低样本数量及全部窗口N分布。
        covered = [row for row in family_rows if row["late30_observed_n"] > 0]  # 实际可估计的资产分母。
        centers = _stats([row["late30_net_median"] for row in covered])  # 资产中位数的中位数/四分位数。
        safety = _stats([row["late30_safe_fraction"] for row in covered])  # 资产安全率的等权均值/四分位数。
        natural_net = a["net_turns"][family_mask & natural]  # 全程自然结束终点净圈，独立分母。
        late120_mask = family_mask & safe120  # 后期安全120秒条件回合。
        families[family] = {  # 家族的三个不同科学总体明确分开，避免交叉借用分母。
            "expected_assets": len(indices),  # 声明分母始终保留128，估计时缺测不填0。
            "late_first30": {  # 首要late80%-95%首30秒窗口总体。
                "candidate_n": sum(row["late30_candidate_n"] for row in family_rows),  # 起点版本落入窗口的全部候选数。
                "observed_n": sum(row["late30_observed_n"] for row in family_rows),  # 真正可形成net30/safe30的回合数。
                "unobserved_n": int((family_mask & late & ~observed).sum()),  # 观察不足条目只记覆盖损失。
                "unobserved_censored_n": int((family_mask & late & ~observed & a["censored"]).sum()),  # 停训/恢复导致的未满30秒删失。
                "unobserved_other_n": int((family_mask & late & ~observed & ~a["censored"]).sum()),  # 短timeout也不冒充失败。
                "covered_assets": len(covered), "assets_with_at_least_two_windows": sum(row["late30_observed_n"] >= 2 for row in family_rows),  # 实际资产分母及至少双窗口覆盖数。
                "missing_asset_indices": [row["asset_index"] for row in family_rows if row["late30_observed_n"] == 0],  # 可以定位缺测资产，不仅给出比例。
                "one_window_asset_indices": [row["asset_index"] for row in family_rows if row["late30_observed_n"] == 1],  # 单窗口资产不足以描述资产内波动。
                "windows_per_declared_asset": _stats([row["late30_observed_n"] for row in family_rows]),  # 含零覆盖资产的N分布。
                "center": centers["median"], "asset_net_median_distribution": centers,  # 主净圈中心与其资产间分布。
                "safety_asset_equal_mean": safety["mean"], "asset_safety_fraction_distribution": safety,  # $|A|^{-1}\sum_a k_a/n_a$。
                "safe_window_n": sum(row["late30_safe_n"] for row in family_rows),  # 此数仅作审计，不用于池化安全中心。
                "assets_net_median_ge0_5_n": sum(row["late30_net_median"] >= 0.5 for row in covered),  # 至少半圈的资产中位数门。
                "assets_net_median_ge1_n": sum(row["late30_net_median"] >= 1 for row in covered),  # 至少一圈的资产中位数门。
                "single_window_net_ge1_n": sum(row["late30_single_window_net_ge1_n"] for row in family_rows),  # 单窗口整圈数，不等于通过资产数。
                "single_window_net_ge2_n": sum(row["late30_single_window_net_ge2_n"] for row in family_rows),  # 单窗口两圈尾部数量。
            },
            "naturally_completed": {  # 全训练时段自然结束回合；可与审计complete数量相加闭合。
                "n": int(natural_net.size), "net_ge1_n": int((natural_net >= 1).sum()),  # 完整回合分母与终点整圈数。
                "net_ge2_n": int((natural_net >= 2).sum()), "max_net": _stats(natural_net)["max"],  # 两圈尾部和最大的终点净转。
            },
            "late_safe120": {  # 只回答活到120秒条件下，首30秒之后又增加多少净圈。
                "end_late_n": int((family_mask & end_late).sum()),  # 逐步公开条件收缩的原始分母。
                "end_late_complete120_n": int((family_mask & end_late & full120).sum()),  # 后期完整120秒回合数，尚未剔除同步失败。
                "conditional_episode_n": int(late120_mask.sum()), "conditional_asset_n": int(np.unique(a["asset_index"][late120_mask]).size),  # 无drop/axis的条件回合与覆盖资产数。
                "delta_net120_minus_net30": _stats(delta120[late120_mask]),  # 同回合差值的median，不能用两个median相减。
                "same_episode_net120": _stats(a["net_turns"][late120_mask]),  # 同一条件总体的120秒终点净圈。
                "same_episode_net30": _stats(a["net_turns_first30"][late120_mask]),  # 同一条件总体已经锁存的首30秒净圈。
            },
        }
        assets.extend(family_rows)  # CSV保留能重新形成全部主中心的逐资产原子统计。
    return families, assets  # 汇总和逐资产表共用一次归约结果，避免两条不同统计口径。


def _paired(specs: list[dict[str, Any]], assets: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    r"""按共同资产索引比较前两run；改善/下降/相同采用精确差值符号，不作显著性宣称。"""
    candidate, reference = specs[:2]  # manifest顺序声明首要比较，不按结果大小挑参照。
    lookup = {(row["run"], row["asset_index"]): row for row in assets}  # 每个run每资产恰有一行。
    rows, families = [], {}  # 缺测配对仍保留行，只让其delta为空。
    for family, indices in _FAMILIES.items():  # 配对在家族内部进行，名义上每族128对。
        own_rows = []  # 两族独立形成128对，不混成256资产中心。
        for index in indices:  # 当前共同索引是一项资产配对，而非一次回合配对。
            left, right = lookup[candidate["name"], index], lookup[reference["name"], index]  # 同一声明资产。
            observed = left["late30_observed_n"] > 0 and right["late30_observed_n"] > 0  # 两边均有窗口才能配对。
            row = {  # 同时保存两边估计值和N，使每个delta可直接重算。
                "candidate": candidate["name"], "reference": reference["name"],  # 差值方向由候选减首要参照定义。
                "candidate_total_transitions": candidate["total_transitions"], "reference_total_transitions": reference["total_transitions"],  # 两边实际预算显式保留。
                "reward_label": candidate["reward_label"], "evidence": "stochastic_training_cross_policy_versions_low_sample",  # 共同奖励标签和随机训练语义。
                "family": family, "asset_index": index, "paired_observed": observed,  # 无共同覆盖时不伪造delta。
                "candidate_window_n": left["late30_observed_n"], "reference_window_n": right["late30_observed_n"],  # 两边资产内窗口数可以不同。
                "candidate_net_median": left["late30_net_median"], "reference_net_median": right["late30_net_median"],  # 两个资产中位数，单位圈。
                "net_median_delta": left["late30_net_median"] - right["late30_net_median"] if observed else None,  # 圈。
                "candidate_safe_fraction": left["late30_safe_fraction"], "reference_safe_fraction": right["late30_safe_fraction"],  # 各自以本资产观察窗口为分母。
                "safe_fraction_delta": left["late30_safe_fraction"] - right["late30_safe_fraction"] if observed else None,  # 概率差。
            }
            own_rows.append(row)  # 每行同时给出两边的低样本分母。
        families[family] = {"expected_pairs": len(indices)}  # 缺配对不能当作相同或下降。
        for metric in ("net_median_delta", "safe_fraction_delta"):  # 运动中心与安全率分开计数改善、下降和相同。
            delta = np.asarray([row[metric] for row in own_rows if row["paired_observed"]], dtype=np.float64)  # [有效资产对]。
            families[family][metric] = _stats(delta) | {  # 配对delta的中位数不等于两个家族中位数之差。
                "improved_n": int((delta > 0).sum()), "declined_n": int((delta < 0).sum()), "tied_n": int((delta == 0).sum()),  # 精确零为相同，三数之和是有效配对N。
            }
        rows.extend(own_rows)  # 供逐资产核对，包含没有可估计delta的缺测行。
    return {  # 同预算、同奖励前提在入口验证，此处只发布对应的配对描述性证据。
        "candidate": candidate["name"], "reference": reference["name"], "delta_direction": "candidate - reference",  # 正值统一表示候选更高。
        "budget_matched": True, "reward_label_matched": True, "total_transitions_each": candidate["total_transitions"],  # 匹配只指声明预算/奖励。
        "asset_identity_basis": "调用方声明的共同asset_index；episode文件没有physical asset hash，不额外读取资产/检查点",  # 不把索引匹配冒称为额外的资产哈希审计。
        "families": families,  # LEAP/Allegro各自保留实际配对分母。
    }, rows


def _check_anchors(summary: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    r"""独立锚点只与已经算完的结果比较；任何差异原样返回，绝不回写原始统计。"""
    checks = []  # 每个核对锚保留期望、实测、容差和通过状态。
    tolerance = float(manifest.get("anchor_atol", 1e-7))  # 七位小数锚的舍入容差，整数必须精确相等。
    _require(np.isfinite(tolerance) and tolerance >= 0, "invalid anchor_atol")  # 无穷或负容差不能形成可解释的核验。
    for path, expected in manifest.get("expected_checks", {}).items():  # 未提供的锚不凭空推断为已通过。
        actual: Any = summary  # 点分路径只引用当前summary，不读取外部文件。
        for key in path.split("."):  # 逐层定位已计算结果；锚点不访问原始回合数组。
            actual = actual[key]  # 锚点拼写错误应显式失败，不跳过未知字段。
        atol = 0.0 if type(expected) is int else tolerance  # 行数/预算/覆盖数不能用近似掩盖差异。
        passed = actual is not None and bool(np.isclose(actual, expected, rtol=0, atol=atol))  # 只允许指定绝对容差。
        checks.append({"path": path, "expected": expected, "actual": actual, "atol": atol, "passed": passed})  # 每项实测与期望并排保存。
    return {"status": "passed" if checks and all(row["passed"] for row in checks) else "not_requested" if not checks else "mismatch", "checks": checks}  # 完整保留核对证据。


def _plot(output: Path, summary: dict[str, Any], assets: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> None:
    r"""绘制一张同预算两族图：资产中心/四分位、等权安全中心、首要参照配对。

横向抖动由资产索引确定，仅分开重叠点，不重采样。每个点对应一个资产估计量。
只纳入与首项相同实际预算的run；背景奖励差异与逐资产低样本分母同时标注。
Matplotlib使用Agg CPU后端，临时字体缓存仅在output_dir内产生并自动移除。
"""
    with tempfile.TemporaryDirectory(prefix=".matplotlib-", dir=output.parent) as cache:  # 只有临时CPU字体缓存，不形成额外科研产物。
        os.environ["MPLCONFIGDIR"] = cache  # 防止字体缓存写到任务拥有范围之外。
        import matplotlib

        matplotlib.use("Agg")  # 静态位图不需要显示服务、GPU或回放。
        from matplotlib import pyplot as plt

        primary = summary["pairwise"]  # 第一、二run已经通过同预算同奖励校验。
        budget = primary["total_transitions_each"]  # 图上唯一的实际预算。
        names = [name for name, run in summary["runs"].items() if run["total_transitions"] == budget]  # 排除大预算背景。
        colors = ["#2369a8", "#d87527", "#559445", "#925bb0"]  # 固定方法顺序，不按效果改变颜色。
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))  # 两族三视图，一张必要对比图。
        footer = []  # 各方法/家族的真实窗口数与每资产范围随图保存。
        for row_index, family in enumerate(_FAMILIES):  # 图的行固定对应LEAP和Allegro。
            net_ax, safe_ax, paired_ax = axes[row_index]  # 同族净圈、安全与配对。
            for position, name in enumerate(names):  # 仅显示实际预算匹配的方法。
                run = summary["runs"][name]  # 只使用同预算run的统计摘要。
                data = [row for row in assets if row["run"] == name and row["family"] == family and row["late30_observed_n"]]  # 缺测资产无可画估计量，覆盖仍在脚注报告。
                net = np.asarray([row["late30_net_median"] for row in data])  # 每点是一个资产的episode median。
                safe = np.asarray([row["late30_safe_fraction"] for row in data])  # 每点是一个资产safe fraction。
                jitter = np.asarray([((row["asset_index"] * 37) % 127) / 126 - 0.5 for row in data]) * 0.32  # 确定性显示偏移。
                color = colors[position % len(colors)]  # 不随运行时随机数变化。
                for axis, values, center_key in ((net_ax, net, "median"), (safe_ax, safe, "mean")):  # 净转用资产中位数，安全用资产均值。
                    axis.scatter(position + jitter, values, color=color, alpha=0.4, s=15, linewidths=0)  # 全部已覆盖资产点。
                    stats = _stats(values)  # 图中心严格复用与JSON相同的估计量定义。
                    if values.size:  # 无覆盖时不在图上画零中心或四分位条。
                        axis.vlines(position, stats["q25"], stats["q75"], color="black", linewidth=4)  # 资产间IQR，不是误差条CI。
                        axis.scatter([position], [stats[center_key]], marker="D", s=65, color=color, edgecolor="black", zorder=4)  # 对应主中心。
                window = run["families"][family]["late_first30"]  # 图下注明低样本条件的精确分母。
                counts = window["windows_per_declared_asset"]  # 包括缺测资产的0窗口计数。
                footer.append(f"{family} {name}: {window['covered_assets']}/128 assets; {window['observed_n']} windows; N/asset {counts['min']:.0f}-{counts['max']:.0f} (median {counts['median']:.1f}); unobserved {window['unobserved_n']}")  # 独立图像携带真实低样本N。
            # 图中方法标签直接带奖励核，007的不同奖励无需读外部文档即可识别。
            labels = [f"{name}\n{summary['runs'][name]['reward_label']}" for name in names]  # 同预算不等于同奖励。
            for axis in (net_ax, safe_ax):  # 两个中心视图使用相同方法顺序与家族分母。
                axis.set_xticks(range(len(names)), labels, fontsize=9)  # 资产等权分布按manifest方法顺序排列。
                axis.grid(axis="y", alpha=0.2)  # 辅助读取量级，不插入任何历史评估点。
                axis.set_xlim(-0.6, len(names) - 0.4)  # 显示抖动点且留出边距。
            net_ax.set_title(f"{family}: first30 net\nDiamond = median of asset medians")  # 图标题直接说明两层中位数归约。
            net_ax.set_ylabel("Signed net turns; each dot = one asset")  # 有符号圈，不能解释成绝对路径。
            safe_ax.set_title(f"{family}: first30 safety\nDiamond = asset-equal mean")  # 菱形明确是安全率资产等权均值。
            safe_ax.set_ylabel("Safe-window fraction per asset")  # 明确概率分母在资产内部。
            safe_ax.set_ylim(-0.04, 1.04)  # 两族统一概率尺度。
            own_pairs = [row for row in pairs if row["family"] == family and row["paired_observed"]]  # 缺测配对不画伪点。
            x = np.asarray([row["reference_net_median"] for row in own_pairs])  # 横轴为首要参照008。
            y = np.asarray([row["candidate_net_median"] for row in own_pairs])  # 纵轴为候选SAC。
            paired_ax.scatter(x, y, color="#2369a8", alpha=0.65, s=24)  # 每点一对共同资产中位数。
            bound = max(float(np.max(np.abs(np.r_[x, y]))) if x.size else 0.0, 0.25) * 1.12  # 同一正负量纲边界。
            low = float(np.min(np.r_[x, y])) if x.size else 0.0  # 保留真实负净圈，不截断为0。
            low = min(low - 0.03, 0.0)  # 留出少量视觉边距。
            paired_ax.plot([low, bound], [low, bound], "--", color="gray", linewidth=1)  # y=x表示两方法资产中位数相同。
            paired_ax.set(xlim=(low, bound), ylim=(low, bound), xlabel=f"{primary['reference']}: asset median net turns", ylabel=f"{primary['candidate']}: asset median net turns")  # 横纵轴使用共同范围，保留净圈符号。
            comparison = primary["families"][family]["net_median_delta"]  # 精确配对改善/下降计数。
            median_text = "missing" if comparison["median"] is None else f"{comparison['median']:+.4f}"  # 缺测图不格式化null。
            paired_ax.set_title(f"{family}: {comparison['n']}/128 paired assets\nImproved/declined/tied = {comparison['improved_n']}/{comparison['declined_n']}/{comparison['tied_n']}; median delta {median_text}")  # 图内配对数字与JSON完全同源。
            paired_ax.grid(alpha=0.2)  # 配对离散点不拟合回归或宣称统计显著。
        fig.suptitle(f"Recorded stochastic training | cross-policy-version episodes\nActual matched budget: {budget:,} transitions/run | episode start in [0.80C, 0.95C)", fontsize=15, y=0.99)  # 随机训练、跨版本、实际预算和版本窗口均在主标题。
        fig.text(0.02, 0.185, "Low sample denominators; dots are descriptive asset estimates. Black bars: asset Q25-Q75, not confidence intervals.\nPPO007: different-reward background. Frozen R16 / formal reliable evaluation: not applicable.", fontsize=9)  # IQR与训练能力结论的边界随图保留。
        fig.text(0.02, 0.025, "\n".join(footer), fontsize=8.5, linespacing=1.35)  # 图独立保留实际N，而非只有图例。
        fig.tight_layout(rect=(0, 0.23, 1, 0.94))  # 给两行证据说明与六组分母留足空间。
        with output.open("wb") as stream:  # 只再生成已命名的派生PNG，不读取或覆盖原始训练媒体。
            fig.savefig(stream, format="png", dpi=170, metadata={"Description": json.dumps(_protocol(), ensure_ascii=False)})  # 仅创建指定的一张PNG。
        plt.close(fig)  # 释放CPU画布，不留下后台显示进程。


def analyze(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    r"""执行只读审计与对比，创建manifest、summary、两张逐资产表和一张PNG。

Args:
    manifest_path: JSON科学接口，前两run是同预算同奖励的候选/首要参照。
    output_dir: 独立派生产物目录；不得位于任何输入run内，同名派生产物可重新生成。

Returns:
    完整summary；expected_checks不一致时status=anchor_mismatch，实测值仍是真源。

Raises:
    ValueError: 环境、原始证据或科学接口不满足合同；坏行不会被剔除继续执行。
"""
    # 明确的CPU资源合同在读取任何episode前检查；此入口从不导入模拟器或模型。
    for key, expected in (("CUDA_VISIBLE_DEVICES", ""), ("OMP_NUM_THREADS", "1"), ("POLARS_MAX_THREADS", "1")):  # 项目要求禁用CUDA并限制CPU线程。
        _require(os.environ.get(key) == expected, f"CPU contract requires {key}={expected!r}")  # 缺失环境声明也不能默认为合同成立。
    manifest_bytes = manifest_path.read_bytes()  # 配置只读，输出使用另一个manifest.json文件。
    manifest = json.loads(manifest_bytes)  # 不解析Research或run params/checkpoint来补充事实。
    root = Path(manifest["project_root"]).resolve()  # project_root必须绑定当前声明项目。
    specs = manifest["runs"]  # 算法/奖励/预算只来自显式manifest。
    _require(isinstance(specs, list) and len(specs) >= 2, "manifest requires candidate and primary reference runs")  # 至少有明确的候选与首要参照。
    _require(len({spec["name"] for spec in specs}) == len(specs), "run names must be unique")  # 重名会破坏summary与配对映射。
    output_dir = output_dir.resolve()  # 输出路径先解析，防止符号链接把产物写回run。
    for spec in specs:  # 每个run的声明在读取分片前验证，避免错误标签传播到图表。
        _require(isinstance(spec["name"], str) and bool(spec["name"]), "invalid run name")  # 非空方法名称。
        _require(type(spec["total_transitions"]) is int and spec["total_transitions"] > 0, "budget must be positive integer transitions")  # C是精确计数，不接受bool/浮点。
        _require(isinstance(spec["reward_label"], str) and bool(spec["reward_label"]), "reward_label is required")  # 奖励区别必须随结果发布。
        _require(isinstance(spec["run_dirs"], list) and bool(spec["run_dirs"]), "each run needs physical run_dirs")  # 恢复段列表不可为空。
        for directory in spec["run_dirs"]:  # 检查每个输入源与派生目录的只读边界。
            _require(not output_dir.is_relative_to((root / directory).resolve()), "output_dir must be outside input run directories")  # 训练run中不产生分析文件。
    _require(specs[0]["total_transitions"] == specs[1]["total_transitions"], "primary pair budgets differ")  # 首要比较必须精确预算匹配。
    _require(specs[0]["reward_label"] == specs[1]["reward_label"], "primary pair reward labels differ")  # 首要比较不能混用奖励核。
    _require(all(not (output_dir / name).is_symlink() for name in _OUTPUTS), "derived outputs must not be symlinks")  # 防止输出别名指向原始证据。
    _require(manifest_path.resolve() not in {output_dir / name for name in _OUTPUTS}, "input manifest must be distinct from generated outputs")  # 配置本身仍为只读。
    results, all_assets, all_sources, key_tables = {}, [], [], []  # 只持有CPU统计、来源与全局键审计所需证据。
    used_files: set[Path] = set()  # 所有算法间也禁止重复消费同一分片。
    for index, spec in enumerate(specs):  # manifest顺序决定比较角色，绝不按得分择优排序。
        table, audit, sources = _read_run(root, spec, used_files)  # 每个逻辑run全量审计并闭合预算。
        families, assets = _summarize(spec, table)  # 审计通过后才按科学窗口聚合。
        same_budget = spec["total_transitions"] == specs[0]["total_transitions"]  # 背景run的实际预算必须显式对比。
        same_reward = spec["reward_label"] == specs[0]["reward_label"]  # reward不同只作背景。
        role = "candidate" if index == 0 else "primary_reference" if index == 1 else "background"  # 不按统计结果更换首要参照。
        results[spec["name"]] = {  # 所有run均有同结构摘要，背景预算差异显式可检索。
            "total_transitions": spec["total_transitions"], "reward_label": spec["reward_label"], "role": role,  # 实际预算及奖励语义。
            "budget_matched_to_candidate": same_budget, "reward_label_matched_to_candidate": same_reward,  # 两种匹配是不同声明。
            "late_start_version_interval": [0.80 * spec["total_transitions"], 0.95 * spec["total_transitions"]],  # 输出真实transition坐标上下界。
            "audit": audit, "families": families,  # 大预算背景不进入首要配对或同预算图。
        }
        all_assets.extend(assets)  # 每run保留两个家族的256行。
        all_sources.extend(dict(source, run=spec["name"]) for source in sources)  # 文件字节身份绑定逻辑run。
        key_tables.append(table.select(_KEYS))  # 全manifest再次审计回合键，不只在各run内部去查。
    global_keys = pl.concat(key_tables, how="vertical")  # 所有物理段的episode key联合总体。
    _require(global_keys.unique().height == global_keys.height, "duplicate episode keys across manifest runs")  # 各逻辑run之间也不能重复共享回合键。
    pairwise, paired_assets = _paired(specs, all_assets)  # 同资产配对保留两边各自的窗口分母。
    summary = {  # status仅表达数据/计算合同，能力与视觉结论另行约束。
        "schema_version": "1.0.0", "protocol": _protocol(), "status": "passed",  # 指数据/计算合同，不是能力判定。
        "runs": results, "pairwise": pairwise, "global_unique_episode_keys": True,  # 全量审计、分族聚合与首要配对一并发布。
        "output_files": list(_OUTPUTS), "visual_acceptance": "awaiting_user_inspection",  # 图的最终可读性由用户确认。
    }
    summary["anchor_validation"] = _check_anchors(summary, manifest)  # 锚点不参与前述任何计算。
    if summary["anchor_validation"]["status"] == "mismatch":  # 任何锚不一致都必须传回调用方。
        summary["status"] = "anchor_mismatch"  # 发布真实差异，但不能宣称锚点通过。
    # 核对读取后原始集合未新增分片、原始文件未被改写；这些检查只读metadata。
    final_files = {(root / directory / "episodes" / path.name).resolve() for spec in specs for directory in spec["run_dirs"] for path in (root / directory / "episodes").glob("*.parquet")}  # 重新枚举同一窄输入集合，只检查是否改变。
    _require(final_files == used_files, "episode shard set changed during analysis")  # 新增/删除分片都会使本次快照失效。
    for source in all_sources:  # 检查每一个已哈希分片，不能只抽查首末文件。
        status = Path(source["path"]).stat()  # 对应本次哈希过的原始文件。
        _require((status.st_size, status.st_mtime_ns) == (source["bytes"], source["mtime_ns"]), f"input changed: {source['path']}")  # 拒绝统计过程中发生字节长度或mtime变化。
    output_dir.mkdir(parents=True, exist_ok=True)  # 所有源数据合同通过才创建派生产物目录。
    resolved_manifest = {  # 复现身份以真实输入、工具源码与CPU依赖为边界。
        "schema_version": "1.0.0", "protocol": _protocol(), "project_root": str(root), "runs": specs,  # 原始预算声明与完整科学协议。
        "input_manifest": str(manifest_path.resolve()), "input_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),  # 原始输入配置字节指纹。
        "analysis_source": str(Path(__file__).resolve()), "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),  # 本次真正执行的分析公式实现。
        "python": sys.executable, "numpy_version": np.__version__, "polars_version": pl.__version__,  # 数值与Parquet解码环境。
        "cpu_environment": {key: os.environ[key] for key in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "POLARS_MAX_THREADS")},  # 实际资源限制可复查。
        "expected_checks": manifest.get("expected_checks", {}), "anchor_atol": manifest.get("anchor_atol", 1e-7),  # 独立核对锚原样保留。
        "sources": all_sources, "input_snapshot_stable": True, "output_dir": str(output_dir),  # 字节、集合和预算共同闭合。
    }
    # CSV保留空估计量，JSON禁止NaN；不引入额外的回放、检查点或训练依赖。
    for name, rows in (("per_asset.csv", all_assets), ("paired_per_asset.csv", paired_assets)):  # 两张表分别以资产和共同资产对为统计单位。
        with (output_dir / name).open("w", encoding="utf-8", newline="") as stream:  # 只生成派生表，可在同输出目录复算。
            pl.DataFrame(rows, infer_schema_length=None).write_csv(stream)  # 所有资产均保留，空字段表示缺测。
    _plot(output_dir / "late_first30.png", summary, all_assets, paired_assets)  # 一张同预算对比图，009只在表/JSON单列。
    for name, document in (("manifest.json", resolved_manifest), ("summary.json", summary)):  # 精确来源与科学归约各有独立JSON入口。
        with (output_dir / name).open("w", encoding="utf-8") as stream:  # 不修改输入manifest或任何原始run文件。
            json.dump(document, stream, indent=2, ensure_ascii=False, allow_nan=False)  # 每个JSON有完整科学标签。
            stream.write("\n")  # 标准文本终止换行。
    return summary  # CLI打印摘要并将锚点异常转换为非零退出码。


def main() -> int:
    r"""CPU CLI；正常合同通过返回0，数据错误或独立锚点差异返回2并给出具体证据。"""
    parser = argparse.ArgumentParser(description=__doc__)  # help直接展示科学合同与只读边界。
    parser.add_argument("--manifest", type=Path, required=True, help="JSON manifest: project_root and ordered runs")  # 输入算法顺序定义候选与首要参照。
    parser.add_argument("--output_dir", type=Path, required=True, help="Derived-artifact directory outside input runs")  # 派生产物统一落在显式目录。
    args = parser.parse_args()  # 科学接口仅包含两个显式路径参数。
    try:  # 文件/科学合同异常返回错误，不转成缺测行或伪零。
        summary = analyze(args.manifest, args.output_dir)  # 全部检查与单次分析共用同一批真实输入。
    except (ValueError, OSError, KeyError, TypeError, pl.exceptions.PolarsError) as error:  # 保留具体失败原因而非静默跳过。
        print(f"recorded_rotation: {error}", file=sys.stderr)  # 不隐去数据异常后继续生成貌似完整的结论。
        return 2  # 调用方可以据退出码停止下游研究汇总。
    print(json.dumps({"status": summary["status"], "output_dir": str(args.output_dir.resolve()), "anchor_status": summary["anchor_validation"]["status"], "pairwise": summary["pairwise"]}, ensure_ascii=False, indent=2))  # 终端只打印完成状态与紧凑首要配对。
    for check in summary["anchor_validation"]["checks"]:  # 失败锚逐项返回，不只给总状态。
        if not check["passed"]:  # 数值差异没有任何自动调整或容差重试。
            print(f"ANCHOR MISMATCH: {check}", file=sys.stderr)  # 实测值与锚点并排，原始证据仍为最终真源。
    return 0 if summary["status"] == "passed" else 2  # 数据通过且所有请求锚一致才返回0。


if __name__ == "__main__":
    raise SystemExit(main())  # 直接文件入口不触发anymani包的环境注册或任何训练启动。
