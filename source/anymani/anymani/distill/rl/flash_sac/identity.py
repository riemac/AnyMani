r"""FlashSAC 的原生方法身份：把实验条件、成员轴和实际实现封成可复算凭据。

完整续训必须保持同一软 Bellman 目标、数据支持域、初态分布、观察和动作语义。
身份因此绑定 config、入口合同、MDP 合同、canonical cohort、strict rank-0 查询键、
冻结 N040 provider 以及实际执行路径的源码字节。Git 提交号不代替源码内容身份。

默认实验锚点是 256 资产、8192000 条新交互、TIP-only、History30、20 Hz 控制和
120 Hz 物理步、每策略步 1/24 rad 动作幅度。资产数和预算取自 FlashSACConfig；
任务物理/奖励合同由调用者从实际装配交付，本模块不创建环境或推断训练表现。
源码和 catalog 只读；所有相对数据路径锚定 AnyMani 项目根，而不是当前工作目录。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .config import FlashSACConfig

_IDENTITY_SCHEMA = "flash-sac-1.0.0"  # 首个原生 FlashSAC 方法身份协议。
_ALGORITHM = "flash_sac_anymani_adaptation"  # 结构化观察与异构有效关节上的 SAC 适配。
_TASK_ID = "AnyMani-Hetero-Generated-PalmRotation-MVP-RLGames-v0"  # 已注册的任务 alias，不代表学习算法。
_PACKAGE_PATH = Path("source/anymani/anymani")  # 从项目根到 Python 包的固定源码布局。

# 共享源文件按实际数值/物理职责显式列出；读取字节不会执行这些模块的环境导入。
# FlashSAC 自有包另整体封存，使新训练入口加入时也自动进入同一 provenance。
_SHARED_SOURCES = (
    # 网络依赖：结构化Actor、History30、图注意力和冻结几何的输入/状态合同。
    "distill/models/palm_rotation_policy.py",  # direct-token Actor 与条件潜高斯尺度。
    "distill/models/temporal_encoder.py",  # 逐关节TCN历史编码。
    "distill/models/backbones/geometry_transformer.py",  # owner图关系偏置与attention。
    "distill/models/policy.py",  # canonical静态证据bank的数据合同。
    "distill/models/structured_heterogeneous.py",  # 共享geometry provider的具名数据视图。
    "distill/models/input_adapters/geometry.py",  # static evidence公开导出。
    "distill/models/input_adapters/evidence.py",  # canonical owner/joint证据与padding。
    "distill/models/input_adapters/encoder.py",  # retained frontend与backbone组装。
    "distill/models/input_adapters/se3_invariant_encoder.py",  # N040 proper-SE(3) encoder。
    # 几何依赖：只读artifact、源实现与当前q条件下的冻结特征计算。
    "distill/methods/density_material_jacobian/artifact.py",  # schema-5 N040恢复与校验。
    "distill/methods/multi_anchor_gaussian_implicit_field/artifact.py",  # artifact共享摘要/加载报告。
    "distill/representations/sources/geometry_source.py",  # typed geometry source物化。
    "distill/representations/sources/kinematics.py",  # source FK/owner frame。
    "distill/representations/sources/anchor_sampling.py",  # 静态anchor bank。
    "distill/representations/sources/collision_geometry.py",  # home collision surface证据。
    "distill/rl/canonical_evidence.py",  # source→canonical owner证据散射。
    "distill/rl/runtime/source_config.py",  # N040 source采样参数真源。
    "distill/rl/runtime/retained_geometry.py",  # 冻结encoder与静态frontend缓存。
    "distill/rl/runtime/structured_geometry.py",  # cohort到冻结N040的绑定。
    "distill/rl/runtime/palm_rotation_geometry.py",  # BF16 encoder/FP32 token边界。
    "distill/rl/runtime/palm_rotation_precision.py",  # 显式精度执行合同。
    # 传输依赖：算法只消费具名状态，prototype_index只用于查表与均衡采样。
    "distill/rl/runtime/palm_rotation_vecenv.py",  # 当前任务的shared named transport。
    "distill/diagnostics/recording/rl/training_evidence.py",  # shared transport的只读训练事实记录。
    "distill/diagnostics/recording/rl/episode_evidence.py",  # 原始回合分母与不可覆盖分片发布。
    "distill/diagnostics/recording/rl/runtime.py",  # 实际资源观测与生命周期记录。
    # 物理资产与预抓取依赖：成员轴、ghost路由以及初态catalog的消费规则。
    "assets/bank/cohort.py",  # canonical-final cohort顺序和来源校验。
    "assets/asset_schema_geometry.py",  # typed geometry与物理身份定义。
    "assets/canonical_runtime.py",  # canonical artifact、routing与ghost语义。
    "robots/hand_spawn.py",  # 同一canonical资产的运行时spawn合同。
    "pregrasp/schema.py",  # exact key和摘要规范。
    "pregrasp/good_catalog.py",  # strict候选catalog消费与数据校验。
    "pregrasp/isaac_runtime.py",  # reset姿态的手/世界坐标变换。
    "tasks/hetero/config/generated/asset_binding.py",  # 支持资产到环境副本轴。
    "tasks/hetero/config/generated/scene.py",  # 场景/碰撞/物理装配。
    "tasks/hetero/config/generated/palm_rotation_mvp_env_cfg.py",  # 实际任务配置基底。
    "tasks/hetero/config/generated/pregrasp_identity.py",  # 120Hz与材料物理锚点。
    "tasks/hetero/config/generated/good_pregrasp_identity.py",  # DexCube尺度与物理合同。
    "tasks/hetero/config/generated/strict_good_pregrasp_identity.py",  # strict初态准入身份。
    "tasks/hetero/config/generated/cohort_good_pregrasp_identity.py",  # 成员级strict catalog绑定。
    # MDP依赖：动作、观察、接触、回合、命令、奖励、终止和课程状态。
    "tasks/hetero/contact_layout.py",  # 有效contact owner与结构碰撞关系。
    "tasks/hetero/contact_sensors.py",  # 实际接触读数的取得方式。
    "tasks/hetero/mdp/runtime_state.py",  # ghost masks与pregrasp运行状态。
    "tasks/hetero/mdp/actions.py",  # preload-aware 1/24rad动作。
    "tasks/hetero/mdp/observation_state.py",  # History30重建与归一化坐标。
    "tasks/hetero/mdp/observations.py",  # Actor/privileged Critic信息分流。
    "tasks/hetero/mdp/contact_state.py",  # 20Hz接触统计与TIP bits。
    "tasks/hetero/mdp/object_state.py",  # 物体姿态与速度表示。
    "tasks/hetero/mdp/events.py",  # strict reset、ghost锁定与collision filter。
    "tasks/hetero/mdp/episode_horizon.py",  # 计划回合长度与有限时域timeout。
    "tasks/hetero/mdp/commands.py",  # 旋转目标与进展状态。
    "tasks/hetero/mdp/orientation_goal.py",  # SO(3)目标及姿态奖励核配置。
    "tasks/hetero/mdp/rewards.py",  # 实际奖励项与有效关节归约。
    "tasks/hetero/mdp/task_math.py",  # 姿态、进展、终止的纯张量数学。
    "tasks/hetero/mdp/terminations.py",  # 失败终止合同。
    "tasks/hetero/mdp/adr.py",  # 实际随机化与课程条件。
    "tasks/hetero/mdp/curriculum_state.py",  # 资产/形态层面的课程状态。
    "tasks/hetero/mdp/curriculums.py",  # 奖励释放的实际更新规则。
    "tasks/hetero/mdp/diagnostics.py",  # command所消费的回合充分统计。
)

# 每个shape不含batch轴B；维度对应当前shared named transport，并由CPU合同独立核对。
# 这里只声明数据ABI，不导入会间接注册环境的producer模块。
_FLOAT_SHAPES = {
    "actor_jnt_current": [16, 5],  # q/pi、u/pi、上一动作、非TIP位、TIP位。
    "actor_jnt_history": [30, 16, 5],  # 1.5s oldest-to-latest，含当前帧。
    "actor_jnt_limits": [16, 2],  # 下/上限除以pi。
    "actor_owner_contact": [21, 1],  # PALM/JOINT/TIP的二元接触。
    "critic_jnt_state": [16, 4],  # 特权关节动力学状态。
    "critic_owner_contact": [21, 2],  # 全owner的force/bit，Actor不能读取。
    "critic_obj": [1, 15],  # 物体姿态和twist。
    "critic_task": [1, 8],  # 任务命令与进展。
    "critic_reward_release": [1],  # 当前奖励释放系数。
    "geometry_tokens": [21, 128],  # 冻结N040当前q特征，交付FP32。
}
_BOOL_SHAPES = {
    "jnt_valid": [16],  # 有效可控关节集合。
    "tip_valid": [4],  # 有效手指末端集合。
    "owner_valid": [21],  # PALM/JOINT/TIP统一有效实体集合。
}
_INT16_SHAPES = {
    "shortest_path": [21, 21],  # 无向图距离bucket。
    "parent_direction": [21, 21],  # 父向距离bucket。
    "child_direction": [21, 21],  # 子向距离bucket。
    "prototype_index": [1],  # selection-local来源索引，仅routing metadata。
}


def _json_value(value: Any) -> Any:
    r"""取得独立 JSON 数据树，保留序列顺序并拒绝非字符串键及非有限数值。

    元组和列表表达相同的有序配置；整数键不能被JSON隐式变成字符串后与另一键碰撞。
    Tensor、模块和任意Python对象不属于方法metadata，应由各状态owner另行保存。
    """
    if isinstance(value, Mapping):  # provider/run/task均允许任意只读Mapping实现。
        if any(not isinstance(key, str) for key in value):  # 防止1与"1"变成同一JSON键。
            raise TypeError("FlashSAC identity mappings require string keys")
        return {key: _json_value(item) for key, item in value.items()}  # 深拷贝，不持有caller可变引用。
    if isinstance(value, (tuple, list)):  # 保留cohort/graph/policy参数的实际顺序。
        return [_json_value(item) for item in value]  # 规范JSON数组。
    if value is None or type(value) in (str, bool, int):  # JSON原生离散标量。
        return value  # 不发生类型猜测或数值强制转换。
    if type(value) is float and math.isfinite(value):  # 有单位的连续实验参数必须有限。
        return value  # 保持原始双精度值。
    raise ValueError(f"FlashSAC identity requires finite JSON values, got {type(value).__name__}")


def _stable_digest(payload: Mapping[str, Any]) -> str:
    r"""对排除 identity_digest 自身的规范 JSON 求 SHA-256，可独立复算。"""
    encoded = json.dumps(
        _json_value(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")  # 键顺序不影响方法身份。
    return hashlib.sha256(encoded).hexdigest()  # 文件摘要与整体摘要均使用SHA-256。


def _sha256(path: Path) -> str:
    r"""分块只读文件字节；大 catalog 不必一次复制到额外缓冲。"""
    digest = hashlib.sha256()  # 每个文件的独立内容指纹。
    with path.open("rb") as stream:  # 缺失文件直接失败，不能填入占位摘要。
        for block in iter(lambda: stream.read(1024 * 1024), b""):  # 1MiB流式读取。
            digest.update(block)  # 保留精确字节，包括注释/空白。
    return digest.hexdigest()  # 不依赖Git状态或mtime。


def _resolve_path(path: str | Path, root: Path) -> Path:
    r"""把相对资产/catalog路径锚定项目根，并要求所指对象真实存在。"""
    expanded = Path(path).expanduser()  # 显式支持用户目录写法。
    return (expanded if expanded.is_absolute() else root / expanded).resolve(strict=True)  # cwd不参与解释。


def _record_path(path: Path, root: Path) -> str:
    r"""项目内用相对路径，项目外保留规范绝对路径，保证两种写法得到相同身份。"""
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)  # 不重定位或改写任何资产。


def _implementation_files(root: Path) -> dict[str, str]:
    r"""封存 FlashSAC 实包和显式共享数值路径；所有声明均须存在且读取真实字节。"""
    flash = Path(__file__).resolve().parent  # 当前真实加载的FlashSAC源码目录。
    paths = {_PACKAGE_PATH / name for name in _SHARED_SOURCES}  # 显式共享职责边界。
    paths.update(path.relative_to(root) for path in flash.rglob("*.py"))  # 实包包含入口、算法、身份和I/O。
    return {str(path): _sha256(root / path) for path in sorted(paths)}  # 稳定有序映射，不遍历其他算法目录。


def _manifest_identity(path: Path, root: Path, asset_count: int) -> dict[str, Any]:
    r"""读取 schema-1.2 canonical lock 的来源键和连续局部轴，冻结顺序与原始字节。

    此窄入口不重复资产lowering或物理准入；上游已验证的锁文件整体摘要保留其全部证书。
    这里额外阻止重复来源键、错排cohort_index和算法资产基数矛盾造成恢复时错配手型。
    """
    content = path.read_bytes()  # 解析与sha来自同一字节快照，不进行两次竞态读取。
    document = yaml.safe_load(content)  # 只解析安全YAML值，不导入资产/Isaac模块。
    if not isinstance(document, Mapping) or document.get("schema_version") != "1.2.0":
        raise ValueError("FlashSAC requires a schema-1.2.0 canonical cohort lock")
    members = document.get("members")  # 文件顺序定义A轴。
    if not isinstance(members, list) or not members or len(members) != asset_count:
        raise ValueError("cohort members must be nonempty and match config.asset_count")
    cohort_id = document.get("cohort_id")  # 人类可读支持集名称仍须有明确值。
    if not isinstance(cohort_id, str) or not cohort_id.strip():
        raise ValueError("cohort lock requires a nonempty cohort_id")
    keys: list[str] = []  # 来源键顺序与local tensor axis一一对应。
    for index, member in enumerate(members):  # 不按外部source_row排序，保留冻结成员顺序。
        if (
            not isinstance(member, Mapping)
            or type(member.get("cohort_index")) is not int
            or member["cohort_index"] != index
        ):
            raise ValueError("cohort_index must follow the dense ordered member axis")
        alias, row = member.get("source_alias"), member.get("source_row")  # 两段来源坐标。
        if not isinstance(alias, str) or not alias.strip() or type(row) is not int or row < 0:
            raise ValueError("cohort members require source_alias and nonnegative integer source_row")
        keys.append(f"{alias}#{row}")  # 同一source内部的row身份，绝非网络输入。
    if len(set(keys)) != len(keys):  # 重复source键会暗中改变训练分布的资产权重。
        raise ValueError("cohort source member keys must be unique")
    return {  # 完整文件sha封存其余canonical physical/source证书。
        "path": _record_path(path, root),  # 数据位置可审计。
        "sha256": hashlib.sha256(content).hexdigest(),  # 与解析内容同一快照。
        "schema_version": "1.2.0",  # canonical-final锁协议。
        "cohort_id": cohort_id,  # 可读支持集名称。
        "support_asset_count": asset_count,  # A，等额回放采样的支持基数。
        "source_member_keys": keys,  # ordered source_alias#source_row。
        "member_order": list(range(asset_count)),  # 显式连续cohort_index轴。
    }


def build_method_identity(
    *,
    config: FlashSACConfig,
    provider_identity: Mapping,
    cohort_lock: Path,
    pregrasp: Any,
    task_contract: Mapping,
    run_contract: Mapping,
) -> dict:
    r"""构造不执行环境代码的原生 FlashSAC 精确方法身份。

    Args:
        config: 真实解析的FlashSAC配置；预算包含预热新交互，默认8192000条。
        provider_identity: 冻结N040权重、source、precision等provider自描述身份。
        cohort_lock: schema-1.2 canonical lock；相对路径以AnyMani根为基准。
        pregrasp: 含catalog_root、bindings(each.key_json)、rank、require_strict的结构对象。
        task_contract: 实际MDP物理/奖励/回合参数，原样深拷贝到身份。
        run_contract: 实际入口条件；task_id可显式覆盖现有任务alias。

    Returns:
        dict: JSON-safe独立数据树；training固定含config和run_contract，identity_digest封存全部字段。

    Raises:
        ValueError: 支持集/strict初态/重复配置矛盾，或metadata不是有限JSON值。
        FileNotFoundError: cohort、catalog index或声明源码不存在。

    Notes:
        返回值只证明声明输入和文件字节已绑定；物理准入由pregrasp/provider负责，
        仿真/训练效果仍须由真实训练产物判断。构造身份不会查询Git或加载Isaac。
    """
    # 配置和任务值先取得独立metadata快照，调用者之后改动对象不能追改已发布身份。
    if not isinstance(config, FlashSACConfig):
        raise TypeError("config must be a FlashSACConfig")
    for label, value in (
        ("provider_identity", provider_identity),
        ("task_contract", task_contract),
        ("run_contract", run_contract),
    ):
        if not isinstance(value, Mapping) or not value:
            raise ValueError(f"{label} must be a nonempty mapping")
    configuration = _json_value(config.to_dict())  # 完整算法与预算配置，不用选择性字段摘要。
    provider, task, run = map(_json_value, (provider_identity, task_contract, run_contract))  # 深副本。
    for name in configuration.keys() & run.keys():  # 入口重复声明seed/N等参数时必须与learner一致。
        if _stable_digest({name: configuration[name]}) != _stable_digest({name: run[name]}):
            raise ValueError(f"run_contract/config mismatch for {name}")
    task_id = run.get("task_id", _TASK_ID)  # 现有alias只是缺省值，显式实际ID优先。
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("task_id must be a nonempty string")

    # 由当前文件的固定source布局确定项目根，避免引入父包注册或bank资产解析副作用。
    root = Path(__file__).resolve().parents[6]  # AnyMani/source/anymani/anymani/distill/rl/flash_sac。
    manifest = _manifest_identity(_resolve_path(cohort_lock, root), root, config.asset_count)  # 来源轴只从锁读取。
    if "source_member_keys" in run and run["source_member_keys"] != manifest["source_member_keys"]:
        raise ValueError("run_contract source_member_keys disagree with the cohort order")
    if "cohort_id" in run and run["cohort_id"] != manifest["cohort_id"]:  # 两个可读支持集标签不能矛盾。
        raise ValueError("run_contract cohort_id disagrees with the cohort lock")
    if pregrasp.require_strict is not True or type(pregrasp.rank) is not int or pregrasp.rank != 0:
        raise ValueError("FlashSAC pregrasp requires strict rank-0")
    bindings = tuple(pregrasp.bindings)  # 与members同序的exact查询键。
    if len(bindings) != config.asset_count:
        raise ValueError("pregrasp requires one binding per cohort member")
    catalog = _resolve_path(pregrasp.catalog_root, root)  # 相对catalog同样锚定项目根。
    index = catalog / "index.json"  # catalog发布集合的commit marker。
    key_digests: list[str] = []  # 顺序保留，不按摘要排序。
    for binding in bindings:  # 每个成员必须具有可审计的JSON key。
        key = binding.key_json  # exact key的原始文本，不当成文件路径。
        if not isinstance(key, str):
            raise ValueError("pregrasp key_json must encode a JSON object")
        key_value = json.loads(key)  # 只解析一次；散列仍使用原始exact-key文本。
        if not isinstance(key_value, dict):
            raise ValueError("pregrasp key_json must encode a JSON object")
        _json_value(key_value)  # 同样拒绝NaN等没有稳定物理意义的key。
        key_digests.append(hashlib.sha256(key.encode("utf-8")).hexdigest())  # exact-key字节身份。

    # 身份同时描述SAC联合密度与每有效关节熵正则，ghost绝不进入概率或动作空间。
    payload = {
        "identity_schema_version": _IDENTITY_SCHEMA,  # 原生方法协议。
        "algorithm": _ALGORITHM,  # 可读算法域。
        "task_id": task_id,  # 真实任务alias。
        "task_contract": task,  # 任务装配方交付的完整物理/MDP合同。
        "policy": {
            "actor_variant": config.actor_variant,  # 结构化或Flash MLP，不能互换续接。
            "actor_contact": "tip-only-binary",  # 当前帧、历史和owner三个通路共同限制。
            "distribution": "tanh-squashed-active-joint-diagonal-normal",  # 合法动作域[-1,1]^active。
            "log_probability": "sum-active",  # log pi沿真实关节求和。
            "entropy_regularization": "mean-active",  # 不同DoF使用每有效关节温度尺度。
            "action_authority_rad_per_policy_step": 1 / 24,  # rad/策略步，任务动作真值。
            "ghost": "invalid-zero-action-excluded-from-probability",  # ghost只提供存储padding。
            "history_steps": config.history_steps,  # L=30，1.5s控制历史。
            "history_order": "oldest-to-latest-including-current-with-reset-padding",  # 与CompactReplay一致。
            "structured_sigma_bounds": [0.05, math.exp(-0.43)] if config.actor_variant == "structured" else None,
            "mlp_log_std_bounds": [-10.0, 2.0] if config.actor_variant == "flash_mlp" else None,  # 潜Gaussian尺度。
        },
        "manifest": manifest,  # ordered支持域与其来源字节。
        "pregrasp": {
            "catalog_root": _record_path(catalog, root),  # 数据位置。
            "index_path": _record_path(index, root),  # catalog集合发布标记。
            "index_sha256": _sha256(index),  # 完整index的真实内容摘要。
            "ordered_key_digests": key_digests,  # 按cohort轴排列的exact-key摘要。
            "rank": 0,  # 默认初态唯一消费rank-0。
            "require_strict": True,  # 不放宽strict物理准入。
        },
        "geometry_provider": provider,  # 保留冻结encoder/source/precision的全部自描述。
        "transport_abi": {  # 具名producer侧dtype/shape；网络侧图索引无损转long。
            "float_shapes": _json_value(_FLOAT_SHAPES),  # FP32观察。
            "bool_shapes": _json_value(_BOOL_SHAPES),  # 实际物理集合。
            "int16_shapes": _json_value(_INT16_SHAPES),  # 离散routing与图关系。
        },
        "training": {"config": configuration, "run_contract": run},  # 两个owner的记录不可互相覆盖。
        "implementation": {"files": _implementation_files(root)},  # 真实执行路径的项目源码。
    }
    return {**payload, "identity_digest": _stable_digest(payload)}  # 摘要本身不参与自身散列。
