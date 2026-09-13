r"""逐回合联合统计的不可覆盖Parquet分片，不运行环境或重新计算奖励。

调用方在reset前收集一批CPU数组，批量写入；不在每个环境的每次reset中同步写文件。
一个回合由(segment_id, env_id, episode_id)标识，资产索引由训练cohort定义。
策略版本起止必须保留：跨PPO更新的训练回合不能冒充冻结checkpoint评价。
回合表在终止或停训删失时发布，首窗表在首30秒完成或此前物理失败时发布；相同回合键不代表相同统计单位。
两种表都保存逐事件联合事实，不按观察时长外推圈数，也不在写入时做资产加权或成功率归约。
"""

from __future__ import annotations

import math
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import polars as pl

# 整数列是标识、策略版本或事件计数；回合ID只在同一segment/environment内唯一。
_INTEGER_COLUMNS = (
    "env_id",  # 本进程段内的并行物理环境索引
    "episode_id",  # 同一环境中递增的回合编号，需与segment_id联合解释
    "asset_index",  # 训练资产集合中的手型索引，不等于环境索引
    "policy_version_start",  # 回合首个动作的策略版本坐标
    "policy_version_end",  # 回合结算时的策略版本坐标，可与起点不同
    "policy_steps",  # 回合已实际经过的策略步数，不是计划horizon
    "goal_count",  # 已完成的严格目标事件数，不能代替轴向净转圈数
    "frontier_count",  # 环境给出的旋转前沿事件计数，与严格目标分开保存
)
# 角度保存为圈，时间为秒；净转可负，绝对路径与历史正向前沿非负。
_FLOAT_COLUMNS = ("duration_s", "net_turns", "absolute_path_turns", "max_positive_net_turns")  # 时长(s)、净转/路径/正向前沿(圈)
# drop与axis允许同时成立；censored表示停训时未自然结束的回合，不计为完整成功。
_BOOL_COLUMNS = ("termination_drop", "termination_axis", "termination_timeout", "censored")  # 物理失败、时间截断与停训删失
_EXTRA_INTEGER_COLUMNS = ("orientation_goal_count", "adr_position_level", "goal_count_first30")  # 仅朝向目标数、位置课程档位、首窗严格目标数
_EXTRA_FLOAT_COLUMNS = (
    "adr_position_offset_x_h_m", "adr_position_offset_y_h_m", "net_turns_first30", "absolute_path_turns_first30",  # x/y位置偏移(m)、首窗净转/路径(圈)
)  # 首30秒的路径与净圈使用相同起止时刻。
_EXTRA_BOOL_COLUMNS = ("first30_complete", "first30_safe")  # 完成30秒与窗口内安全分别记录，完整不等于安全
_FIRST30_COLUMNS = {
    "net_turns_first30", "absolute_path_turns_first30", "goal_count_first30", "first30_complete", "first30_safe",
}  # 新的联合窗口必须完整记录，才能解释方向、目标次数与安全的交集。


def write_episode_evidence(
    destination: Path,
    columns: Mapping[str, np.ndarray],
    *,
    reward_sums: Mapping[str, np.ndarray],
    identity_digest: str,
    segment_id: str,
    policy_dt_s: float = 0.05,
) -> Path:
    r"""验证并发布一批逐回合证据；只接受已形成的CPU事实。

    columns的每列均为[E]，E是本批回合数；奖励列为已含weight与dt的episode sum，
    不除以计划时长，也不再次乘权重。零权重项可显式保存零；缺失项不能伪造成零。
    输出保留原始联合记录，使资产中位数、下尾与安全/运动交集可以事后重算。
    first30联合字段保留同回合首窗前缀；完整回合可晚于30秒终止，后续失败不追溯改写已锁存的安全性。

    Args:
        destination: 新.parquet文件；已存在时失败，不覆盖先前证据。
        columns: 必需整数、物理量和终止列均为[E]；仅接受已声明的可选列，联合首窗扩展须完整提供。
        reward_sums: reward term名到[E]已加权累计贡献的映射。
        identity_digest: 产生这些数据的方法SHA-256身份。
        segment_id: 本次进程/恢复段的稳定标识，用于区分重新创建的环境。
        policy_dt_s: 策略步长，单位秒；当前掌旋为6/120=0.05。

    Returns:
        Path: 成功发布的新分片路径；每行携带schema、方法身份、进程段及策略时间尺度。

    Raises:
        ValueError: 身份、列类型/形状、时间积分或终止与首窗事实不满足合同。
        OSError: 文件写入或不可覆盖发布失败；调用方仍需保留尚未确认发布的批次。
    """
    # 发布前先确认分片身份和时间尺度，避免将不可解释的证据写成正式产物。
    destination = Path(destination)  # 本批回合证据的唯一发布路径
    if destination.suffix != ".parquet" or destination.exists():  # 已有分片代表此前已发布的事实
        raise ValueError("episode evidence requires a new .parquet destination")  # 不覆盖既有回合记录
    if not re.fullmatch(r"[0-9a-f]{64}", identity_digest) or not segment_id:  # 64个十六进制字符的方法摘要与非空进程段
        raise ValueError("episode evidence requires method identity and segment ID")  # 身份缺失不能拼接训练段
    if not math.isfinite(policy_dt_s) or policy_dt_s <= 0:  # 秒/策略步必须是有限正数
        raise ValueError("policy_dt_s must be finite and positive")  # 时间尺度不由记录器猜测

    # 固定schema避免mean-only记录与逐episode记录被错误拼接，且不接受隐式截断的浮点ID。
    expected = set(_INTEGER_COLUMNS + _FLOAT_COLUMNS + _BOOL_COLUMNS)  # 所有回合共有的必需联合事实
    optional = set(_EXTRA_INTEGER_COLUMNS + _EXTRA_FLOAT_COLUMNS + _EXTRA_BOOL_COLUMNS)  # 声明过的扩展字段全集
    if not expected <= set(columns) or set(columns) - expected - optional:  # 不补缺失列，也不接收未知列
        raise ValueError(f"episode columns differ from schema: {set(columns) ^ expected}")  # 防止混入其他统计单位
    integers = (*_INTEGER_COLUMNS, *(key for key in _EXTRA_INTEGER_COLUMNS if key in columns))  # 实际提供的整数列
    floats = (*_FLOAT_COLUMNS, *(key for key in _EXTRA_FLOAT_COLUMNS if key in columns))  # 实际提供的物理数值列
    bools = (*_BOOL_COLUMNS, *(key for key in _EXTRA_BOOL_COLUMNS if key in columns))  # 实际提供的布尔事件列
    arrays = {name: np.asarray(value) for name, value in columns.items()}  # 统一访问CPU数组，保留原始dtype供校验
    count = arrays["env_id"].size  # E为本批终止或删失回合数
    if count == 0 or any(value.shape != (count,) for value in arrays.values()):  # 每一列必须与同一批回合逐行对齐
        raise ValueError("episode evidence requires nonempty aligned [E] arrays")  # 不发布空分片或广播标量
    for name in integers:  # 标识、策略版本及事件计数均保留整数语义
        if arrays[name].dtype.kind not in "iu" or np.any(arrays[name] < 0):  # bool和浮点ID均不属于合法整数列
            raise ValueError(f"{name} must contain nonnegative integer values")  # 不通过cast掩盖非法原值
    for name in bools:  # 安全与终止标志必须来自明确的布尔事实
        if arrays[name].dtype.kind != "b":  # 数值0/1不隐式推断为终止判据
            raise ValueError(f"{name} must be boolean")  # 区分事件与数值计数
    for name in floats:  # 此写入器接受浮点或整数表示的有限实物理量
        if arrays[name].dtype.kind not in "fiu" or not np.isfinite(arrays[name]).all():  # NaN/inf不能伪装成零运动
            raise ValueError(f"{name} must contain finite physical values")  # 保留缺测与有效物理量的边界

    # 完整回合与右删失回合分开；相同回合不能在同一批中重复计数。
    terminal = arrays["termination_drop"] | arrays["termination_axis"] | arrays["termination_timeout"]  # [E]任一终止成立
    if np.any(terminal == arrays["censored"]):  # 恰为terminal与censored异或；同时真或同时假都非法
        raise ValueError("a row must be either terminal or censored, not both")  # 删失不计作完整回合
    keys = np.stack((arrays["env_id"], arrays["episode_id"]), axis=1)  # [E,2]段内联合回合键
    if np.unique(keys, axis=0).shape[0] != count:  # 同批不能重复计入同一环境的同一回合
        raise ValueError("duplicate episode identity within shard")  # 跨分片去重由调用方生命周期负责
    if np.any(arrays["policy_version_end"] < arrays["policy_version_start"]):  # 只要求逐回合起止版本不倒序
        raise ValueError("episode policy versions must be monotone")  # 允许一个回合跨多个策略版本
    if np.any(arrays["policy_steps"] < 1) or not np.allclose(
        arrays["duration_s"], arrays["policy_steps"] * policy_dt_s, rtol=1e-6, atol=1e-5  # duration=步数×dt；绝对容差1e-5秒
    ):
        raise ValueError("duration must agree with observed policy steps, not planned horizon")  # 不用计划时长归一化短回合
    if np.any(arrays["absolute_path_turns"] + 1e-5 < np.abs(arrays["net_turns"])):  # 路径三角不等式，绝对容差1e-5圈
        raise ValueError("absolute path cannot be smaller than signed net rotation")  # 正反抵消不能减少已走路径
    if np.any(arrays["max_positive_net_turns"] + 1e-5 < np.maximum(arrays["net_turns"], 0)):  # 历史正向前沿至少覆盖当前正净转
        raise ValueError("positive frontier cannot precede current positive net rotation")  # 负净转时正向下界为0

    # 首30秒在时间边界锁存；后续回合终止不改变已发生的窗口安全性。
    joint_window = bool(set(columns) & {"absolute_path_turns_first30", "goal_count_first30", "first30_safe"})  # 联合首窗扩展标志
    if joint_window:  # 任一联合扩展字段出现，就要求完整五元首窗证据
        if not _FIRST30_COLUMNS.issubset(columns):  # 净转、路径、目标、完成与安全必须同源
            raise ValueError("joint first30 episode evidence requires the complete window fields")  # 不拼接缺失交集
        if np.any(arrays["first30_complete"] != (arrays["duration_s"] >= 30.0)):  # 是否真的观察到完整30秒
            raise ValueError("first30 completion must correspond to thirty actual seconds")  # 奖励课程时长不是首窗时钟
        if np.any(arrays["absolute_path_turns_first30"] + 1e-5 < np.abs(arrays["net_turns_first30"])):  # 同起止前缀的路径界
            raise ValueError("first30 absolute path cannot be smaller than signed net rotation")  # 容差仍为1e-5圈
        if np.any(arrays["goal_count_first30"] > arrays["goal_count"]):  # 首窗目标是同一回合目标计数的前缀
            raise ValueError("first30 goals cannot exceed the episode goal count")  # 不能超出整回合累计量
        unsafe_prefix = (~arrays["first30_complete"]) | (  # 未满30秒不能被标为安全完成
            (arrays["duration_s"] <= 30.0) & (arrays["termination_drop"] | arrays["termination_axis"])  # 30秒内含边界物理失败
        )  # 恰在30秒发生物理失败也不安全；30秒后的失败不进入这一谓词。
        if np.any(arrays["first30_safe"] & unsafe_prefix):  # 校验安全标志的必要条件，不由晚期terminal反推首窗
            raise ValueError("first30 safety contradicts observed completion or boundary failure")  # 不把早失败当安全窗口

    # 奖励保留每个真实term的积分，避免由全局均值反推家族/资产贡献。
    for name, value in reward_sums.items():  # 每个term各存一列，不把回合奖励拆成无法配对的均值
        value = np.asarray(value)  # [E]已含权重与dt的回合累计贡献
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):  # term名形成稳定的reward_sum列后缀
            raise ValueError(f"invalid reward term name: {name}")  # 拒绝不符合列命名合同的名称
        if value.shape != (count,) or value.dtype.kind not in "fiu" or not np.isfinite(value).all():  # 与回合键逐行对齐
            raise ValueError(f"reward sum {name} must be a finite [E] array")  # 缺测奖励不默认零
        arrays[f"reward_sum/{name}"] = value.astype(np.float64)  # 仅规范存储精度，不再乘weight、dt或除时长
    table = pl.DataFrame(arrays).with_columns(  # 每行一回合，元数据广播到本分片所有行
        [pl.col(name).cast(pl.Int64) for name in integers]  # 标识、版本和事件计数采用统一整数存储
        + [pl.col(name).cast(pl.Float64) for name in floats]  # 时间、圈数及位置偏移采用双精度存储
        + [
            pl.lit("1.2.0" if joint_window else ("1.1.0" if set(columns) & optional else "1.0.0")).alias("schema_version"),  # 联合首窗/一般扩展/基础列
            pl.lit(identity_digest).alias("identity_digest"),  # 本批数据所属训练方法身份
            pl.lit(segment_id).alias("segment_id"),  # 环境与回合编号的进程段命名空间
            pl.lit(policy_dt_s).alias("policy_dt_s"),  # 秒/策略步，可独立复核duration与policy_steps
        ]
    )

    return _publish_table(destination, table)  # 完成整表验证后才进行不可覆盖发布


def write_first_window_evidence(
    destination: Path,
    columns: Mapping[str, np.ndarray],
    *,
    identity_digest: str,
    segment_id: str,
    policy_dt_s: float = 0.05,
) -> Path:
    r"""发布首30秒或此前物理失败的窗口事实，不要求整个episode已经结束。

    完成窗口的duration为30秒；提前drop/axis保留实际短时终点，不乘30/duration。
    首30秒前的纯timeout不构成可观测完整窗口，不能写入此表。窗口与episode的键相同，
    但统计单位不同：正常存活窗口在第600步即可发布，后续episode仍可继续到120秒。

    Args:
        destination: 独立首窗目录中的新.parquet分片路径。
        columns: 每列均为[E]，含窗口键、起止策略版本、实际步数/秒数、净转/路径、严格goal与事件标志。
        identity_digest: 方法SHA-256身份；一个窗口可以跨策略更新，摘要不代表冻结策略版本。
        segment_id: 物理进程段标识，与env_id及episode_id共同确定窗口来源。
        policy_dt_s: 秒/策略步，默认0.05；30秒完整窗口对应600个已完成策略步。

    Returns:
        Path: 成功发布的首窗分片路径；window_seconds=30表示名义观察上限，duration_s保留真实终点。

    Raises:
        ValueError: 身份、列合同、时间边界、安全/终止或路径/计数/版本事实不一致。
        OSError: 写入或发布失败；不能据此释放调用方待发布窗口。
    """
    destination = Path(destination)  # 调用方显式提供独立first30目录中的新分片。
    integers = ("env_id", "episode_id", "asset_index", "policy_version_start", "policy_version_end", "policy_steps", "goal_count")  # 窗口来源、版本与计数
    floats = ("duration_s", "net_turns", "absolute_path_turns")  # 秒与有符号/绝对圈数。
    bools = ("complete", "safe", "termination_drop", "termination_axis", "termination_timeout")  # 完成时钟与三类终止独立保存
    if destination.suffix != ".parquet" or destination.exists():  # 窗口分片也不能覆盖先前证据
        raise ValueError("first-window evidence requires a new .parquet destination")  # 使用新的分片编号
    if not re.fullmatch(r"[0-9a-f]{64}", identity_digest) or not segment_id:  # 方法摘要与物理进程段缺一不可
        raise ValueError("first-window evidence requires method identity and segment ID")  # 起止版本另存于逐窗列
    if not math.isfinite(policy_dt_s) or policy_dt_s <= 0 or set(columns) != set(integers + floats + bools):  # 精确首窗列集合
        raise ValueError("first-window evidence has invalid time scale or columns")  # 不接受回合表的额外列
    arrays = {name: np.asarray(value) for name, value in columns.items()}  # 只消费CPU原始记录。
    count = arrays["env_id"].size  # E为本批窗口数，不是环境数或完整episode数。
    if count < 1 or any(value.shape != (count,) for value in arrays.values()):  # 每个窗口必须具有完整且对齐的一行
        raise ValueError("first-window columns must be nonempty aligned vectors")  # 空事件批不产生空分片
    for name in integers:  # 先校验原dtype，再由列式存储统一整数精度
        if arrays[name].dtype.kind not in "iu" or np.any(arrays[name] < 0):  # 浮点索引和bool计数都不合法
            raise ValueError(f"{name} must contain nonnegative integer values")  # 不隐式截断计数或版本
    for name in floats:  # 窗口物理量允许实数整数表示，但必须有限
        if arrays[name].dtype.kind not in "fiu" or not np.isfinite(arrays[name]).all():  # 路径缺测不能化为方向0
            raise ValueError(f"{name} must contain finite physical values")  # 保持逐窗事实可归约
    if any(arrays[name].dtype.kind != "b" for name in bools):  # 三类terminal及complete/safe均为显式布尔事实
        raise ValueError("first-window event columns must be boolean")  # 不从奖励或计数推断安全

    # 一次窗口必须在首个30秒边界或更早的真实物理终点结束，不能把晚期状态错贴为首窗口。
    duration = arrays["duration_s"]  # 实际观察时长；20Hz的完整窗口为600步。
    failure = arrays["termination_drop"] | arrays["termination_axis"]  # 两种物理失败可能同时发生。
    if np.any(duration <= 0) or np.any(duration > 30.0 + 1e-5):  # 上界只容许1e-5秒舍入误差
        raise ValueError("first-window duration must be in (0,30] seconds")  # 晚期回合终点不能冒充首窗
    if np.any(arrays["policy_steps"] < 1) or not np.allclose(duration, arrays["policy_steps"] * policy_dt_s, rtol=1e-6, atol=1e-5):  # 时钟与实际步数一致
        raise ValueError("first-window duration and observed step count disagree")  # 不以名义600步替代早失败步数
    if np.any(arrays["complete"] != (duration >= 30.0)) or np.any(~arrays["complete"] & ~failure):  # 短窗必须有drop/axis
        raise ValueError("an incomplete first-window row must end in a physical failure")  # 纯早timeout无法提供完整观察窗
    if np.any(arrays["safe"] != (arrays["complete"] & ~failure)):  # 安全恰为完成30秒且该窗口终点无物理失败
        raise ValueError("first-window safety must match completion and physical termination")  # 30秒边界失败也为不安全
    if np.any(arrays["absolute_path_turns"] + 1e-5 < np.abs(arrays["net_turns"])):  # 同一窗口路径至少覆盖净转绝对值
        raise ValueError("first-window path must bound signed net rotation")  # 保留净转符号，容差1e-5圈
    if np.any(arrays["policy_version_end"] < arrays["policy_version_start"]):  # 窗口可跨版本，但版本坐标不倒序
        raise ValueError("first-window policy versions must be monotone")  # 不要求起止为同一冻结策略
    if np.unique(np.stack((arrays["env_id"], arrays["episode_id"]), axis=1), axis=0).shape[0] != count:  # [E,2]段内窗口键
        raise ValueError("duplicate first-window episode key")  # 一回合仅应产生一个首窗事件
    table = pl.DataFrame(arrays).with_columns(  # 每行一个已结算窗口，保留实际观察时长
        [pl.col(name).cast(pl.Int64) for name in integers]  # 精确索引、版本与计数。
        + [pl.col(name).cast(pl.Float64) for name in floats]  # 保留已形成物理量的精度。
        + [pl.lit("1.0.0").alias("schema_version"), pl.lit(identity_digest).alias("identity_digest"),  # 首窗表版本及方法身份
           pl.lit(segment_id).alias("segment_id"), pl.lit(policy_dt_s).alias("policy_dt_s"),  # 物理进程段与秒/策略步
           pl.lit(30.0).alias("window_seconds")]  # 名义30秒上限；早失败仍按duration_s保留实际短时观察
    )  # 此schema始终表示训练窗口，不能作为冻结R16的资格证书。
    return _publish_table(destination, table)  # 首窗发布与完整回合是否结束无关


def _publish_table(destination: Path, table: pl.DataFrame) -> Path:
    r"""在同目录原子发布不可覆盖的Parquet分片，记录成功后才允许调用方释放缓存。"""
    # exclusive hard link保证已有分片不能被覆盖；临时文件始终清理。
    destination.parent.mkdir(parents=True, exist_ok=True)  # 调用方指定的证据目录
    descriptor, temporary_name = tempfile.mkstemp(prefix=".episode-", suffix=".tmp", dir=destination.parent)  # 与目标同文件系统
    os.close(descriptor)  # 后续由Parquet writer按路径管理文件句柄
    temporary = Path(temporary_name)  # 未发布的完整表暂存路径
    try:  # 正式文件名只在整表写完后出现
        table.write_parquet(temporary, compression="zstd")  # 无损压缩逐行原始事实
        os.link(temporary, destination)  # 原子建立新文件名；目标已存在时失败而不覆盖
    finally:  # 写入或发布失败都清理临时名字，既有正式分片保持独立
        temporary.unlink(missing_ok=True)  # 成功后正式硬链接仍指向已写好的数据
    return destination  # 返回即表示本分片已成功发布
