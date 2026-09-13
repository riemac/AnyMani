r"""FlashSAC 原生身份的 CPU 合同：冻结科学输入、成员轴与真实源码字节。

身份是一次实验的可复现约束，不是学习表现的评价。小型临时 cohort/catalog 只提供
有序 metadata；源码摘要始终读取实际 AnyMani 项目。复用已有 learner 导入夹具，
不新增包替身，不加载任务环境、Isaac、Kit 或任何优化算法替身。
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import yaml
from test_flash_sac_learner import api as api
from test_flash_sac_learner import small_config as small_config


def _identity_case(tmp_path: Path, config: Any) -> dict[str, Any]:
    r"""形成与 A 种手型一一对应的锁文件和 strict rank-0 查询键。

    这里只验证锁文件轴与摘要，不把合成条目当成通过物理准入的真实资产。
    source_alias#source_row 是成员来源键；cohort_index 才是局部张量顺序。
    """
    members = [  # 来源 row 故意不连续，避免把外部 row 错当成本地 tensor index。
        {"cohort_index": index, "source_alias": "train", "source_row": 11 + 3 * index}  # 外部row与局部轴区分。
        for index in range(config.asset_count)  # A 与配置声明一致，每资产一个查询键。
    ]
    manifest = tmp_path / "cohort.yaml"  # 仅此测试拥有的冻结支持集文件。
    document = {"schema_version": "1.2.0", "cohort_id": "cpu-contract", "members": members}  # 原生 canonical lock。
    manifest.write_text(yaml.safe_dump(document), encoding="utf-8")  # 持久化后才计算身份。
    catalog = tmp_path / "catalog"  # 临时 catalog 根，独立于正式预抓取资产。
    catalog.mkdir(exist_ok=True)  # 同一临时 case 可重复建立输入。
    (catalog / "index.json").write_text('{"schema_version": 3, "entries": {}}\n', encoding="utf-8")  # index 字节锚点。
    return {  # 生产函数的全部参数均通过明确的关键词交付。
        "config": config,  # 真正 FlashSACConfig，不用同名替身。
        "provider_identity": {"identity_digest": "a" * 64, "encoder": "N040", "precision": "bf16-to-fp32"},
        "cohort_lock": manifest,  # 成员顺序只从该文件获得。
        "pregrasp": SimpleNamespace(
            catalog_root=str(catalog),  # 与 binding.key_json 同属这个固定 catalog。
            bindings=tuple(  # 每资产一个exact key，顺序与cohort_index相同。
                SimpleNamespace(key_json=json.dumps({"physical": index})) for index in range(config.asset_count)
            ),
            rank=0,  # 每资产固定取排名零的安全初态。
            require_strict=True,  # strict 准入不得静默放宽。
        ),
        "task_contract": {  # 合同由调用方交付，identity 不构造或执行环境。
            "policy_hz": 20,  # 0.05 s 一次控制动作。
            "physics_hz": 120,  # 一次控制周期对应六个物理步。
            "action_authority_rad_per_policy_step": 1 / 24,  # 弧度/策略步。
            "rotation_progress_reward_weight": 20.0,  # reward/rad 的任务锚点。
        },
        "run_contract": {"seed": config.seed, "num_envs": config.num_envs, "actor_contact": "tip"},
    }


@pytest.fixture(scope="module")
def identity_module(api: SimpleNamespace) -> ModuleType:
    r"""在已有 CPU 导入边界内载入真实 identity 模块，并核对新增依赖。"""
    before = set(sys.modules)  # 只观察本模块新引入的依赖。
    module = importlib.import_module("anymani.distill.rl.flash_sac.identity")  # 正常执行生产源码。
    forbidden = {"isaaclab", "isaacsim", "omni", "pxr", "rl_games"}  # 原生身份不需要训练/仿真后端。
    added = set(sys.modules) - before  # api 已证明其自身只载入真实 CPU 算法依赖。
    assert not [name for name in added if name.split(".")[0] in forbidden], "identity 导入了运行时后端"
    assert "anymani.distill.rl.runtime.palm_rotation_identity" not in added, "identity 借用了 PPO builder"
    return module  # 不替换任何身份函数或实际源码读取。


@pytest.fixture
def identity_inputs(tmp_path: Path, small_config: Any) -> dict[str, Any]:
    r"""四资产 CPU 配置与小网络恢复测试共享相同的身份输入。"""
    return _identity_case(tmp_path, small_config)  # 只有 metadata 合成，算法对象仍是生产实现。


def test_native_identity_binds_order_budget_and_actual_sources(
    identity_module: ModuleType,  # 真实原生身份实现。
    api: SimpleNamespace,  # 真实FlashSAC配置，未初始化正式规模learner。
    tmp_path: Path,  # 本测试独占的cohort/catalog目录。
) -> None:
    r"""默认 256 资产/8192000 新交互进入身份，源码集合真实且没有 PPO 优化文件。"""
    inputs = _identity_case(tmp_path, api.config.FlashSACConfig())  # 不构造256环境或分配正式回放。
    identity = identity_module.build_method_identity(**inputs)  # 仅 CPU 文件读取与规范 JSON。
    assert identity["identity_schema_version"] == "flash-sac-1.0.0", "FlashSAC identity schema 不符"
    assert identity["algorithm"] == "flash_sac_anymani_adaptation", "算法身份必须原生"  # 任务alias不决定算法。
    assert identity["task_id"] == "AnyMani-Hetero-Generated-PalmRotation-MVP-RLGames-v0", "默认任务 alias 改变"
    assert identity["training"]["config"] == inputs["config"].to_dict(), "完整配置未进入 training.config"
    assert identity["training"]["config"]["total_transitions"] == 8_192_000, "预算必须按新交互计数"
    assert identity["training"]["run_contract"] == inputs["run_contract"], "入口合同丢失"
    assert identity["task_contract"] == inputs["task_contract"], "不能猜测或覆盖任务物理合同"
    assert identity["manifest"]["support_asset_count"] == 256, "正式支持集数量不符"  # A决定每资产采样权重。
    assert identity["manifest"]["source_member_keys"] == [f"train#{11 + 3 * index}" for index in range(256)]
    assert identity["manifest"]["member_order"] == list(range(256)), "cohort 局部顺序丢失"  # 静态查表轴。
    assert identity["manifest"]["sha256"] == hashlib.sha256(inputs["cohort_lock"].read_bytes()).hexdigest()
    pregrasp = identity["pregrasp"]  # rank 与 exact-key 摘要属于初态分布身份。
    assert pregrasp["rank"] == 0 and pregrasp["require_strict"] is True, "strict rank-0 门禁未冻结"
    expected = [hashlib.sha256(item.key_json.encode()).hexdigest() for item in inputs["pregrasp"].bindings]
    assert pregrasp["ordered_key_digests"] == expected, "pregrasp 查询键顺序未绑定成员轴"  # 同手型读同初态。
    assert identity["policy"]["actor_contact"] == "tip-only-binary", "Actor 信息边界改变"
    assert identity["policy"]["action_authority_rad_per_policy_step"] == 1 / 24, "控制幅度单位改变"

    # 源码 provenance 独立复算，不把非空字符串或同名空模块当成真实实现证据。
    root = Path(__file__).resolve().parents[7]  # 实际 AnyMani 项目根，和临时数据根完全独立。
    files = identity["implementation"]["files"]  # project-relative path → SHA-256。
    package = root / "source/anymani/anymani/distill/rl/flash_sac"  # 实际生产包。
    assert {str(path.relative_to(root)) for path in package.rglob("*.py")} <= files.keys(), "FlashSAC 源码漏记"
    for name, digest in files.items():  # 每个声明都必须对应实际存在且非空的源文件。
        content = (root / name).read_bytes()  # 正式源码只读。
        assert content and digest == hashlib.sha256(content).hexdigest(), f"源码 provenance 不实：{name}"
    forbidden = {
        "palm_rotation_ppo.py",  # PPO优化主循环。
        "masked_ppo.py",  # PPO兼容/优化合同。
        "structured_ppo.py",  # 另一PPO更新实现。
        "ppo_batch.py",  # PPO优势/采样规约。
        "cagrad.py",  # 未使用的梯度处理。
        "popart.py",  # 未使用的价值归一化。
        "task_gradients.py",  # 未使用的PPO分任务梯度路径。
        "palm_rotation_identity.py",  # 其它算法的身份builder。
        "palm_rotation_network.py",  # rl_games策略分布适配。
    }  # 不属于 SAC 学习路径。
    assert not {Path(name).name for name in files} & forbidden, "原生身份混入 PPO 学习/身份源码"
    encoded = json.dumps(  # 独立实现规范JSON协议，避免仅与被测散列helper自比。
        {key: value for key, value in identity.items() if key != "identity_digest"},  # 摘要排除自引用。
        sort_keys=True,  # 字典插入顺序不属于科学身份。
        separators=(",", ":"),  # 规范紧凑JSON。
        ensure_ascii=True,  # 与非ASCII实验名称保持确定编码。
        allow_nan=False,  # 非有限实验值必须拒绝。
    ).encode()
    assert identity["identity_digest"] == hashlib.sha256(encoded).hexdigest(), "整身份摘要无法独立复算"


def test_transport_shapes_match_real_named_producer_without_import(
    identity_module: ModuleType,  # 只读身份构造器。
    identity_inputs: dict[str, Any],  # 可复算的具名输入合同。
) -> None:
    r"""只读取共享 transport 常量 AST，证伪身份中的维度与真实 producer 不一致。"""
    root = Path(__file__).resolve().parents[7]  # AST 读取不会执行 Isaac/rl_games 导入。
    source = root / "source/anymani/anymani/distill/rl/runtime/palm_rotation_vecenv.py"  # 共享具名传输真源。
    constants = {  # 从真实源码取完整shape字典，不复制一份预期shape替代producer。
        node.target.id: ast.literal_eval(node.value)  # 只解释字面量，不执行生产模块。
        for node in ast.parse(source.read_text()).body  # 只读顶层声明，不执行导入或函数。
        if isinstance(node, ast.AnnAssign)  # producer使用带类型的字典常量。
        and isinstance(node.target, ast.Name)  # 固定符号名决定对应dtype组。
        and node.value is not None  # 声明必须携带实际shape值。
        and node.target.id.startswith("PALM_ROTATION_")  # 只取公开传输常量。
    }  # 仅字面量 shape 声明。
    abi = identity_module.build_method_identity(**identity_inputs)["transport_abi"]  # 原生身份的声明。
    for kind in ("float", "bool", "int16"):  # FP32、逻辑集合与无损离散图索引三类。
        expected = {key: list(shape) for key, shape in constants[f"PALM_ROTATION_{kind.upper()}_SHAPES"].items()}
        assert abi[f"{kind}_shapes"] == expected, f"{kind} transport shape 与实际 producer 漂移"


@pytest.mark.parametrize("field", ["config", "run", "task", "provider", "catalog", "keys", "members"])
def test_each_scientific_input_changes_identity(
    identity_module: ModuleType,  # 前后两次使用同一原生算法。
    identity_inputs: dict[str, Any],  # 其他科学条件保持一致。
    field: str,  # 每条用例只改变一个科学输入。
) -> None:
    r"""方法、随机实验、MDP、冻结几何、初态 catalog 和成员重排分别产生不同身份。"""
    before = identity_module.build_method_identity(**identity_inputs)  # 干预前固定身份。
    changed = deepcopy(identity_inputs)  # 干预不得通过引用回写基线合同。
    if field == "config":  # 折扣改变软 Bellman 目标，网络形状却不改变。
        changed["config"] = replace(changed["config"], gamma=0.98)  # 形状兼容仍不可续接。
    elif field == "run":  # 独立实验条件也必须纳入 resume 身份。
        changed["run_contract"]["episode_seconds"] = 60.0  # s。
    elif field == "task":  # 奖励系数变化是 MDP 干预。
        changed["task_contract"]["rotation_progress_reward_weight"] = 5.0  # reward/rad。
    elif field == "provider":  # 冻结表示改变也会改变实际策略函数。
        changed["provider_identity"]["identity_digest"] = "b" * 64  # 不把 N040 标签当成完整权重身份。
    elif field == "catalog":  # Catalog index 是已发布候选集合的字节身份。
        index = Path(changed["pregrasp"].catalog_root) / "index.json"  # 临时测试数据。
        index.write_text(index.read_text() + "\n")  # 内容摘要绑定字节，不只解释后的 JSON。
    elif field == "keys":  # 相同 catalog 中更换物理/搜索查询键仍是初态变化。
        changed["pregrasp"].bindings[0].key_json = '{"physical": 99}'  # 唯一受干预的 binding。
    else:  # 来源集合相同，但张量轴重排必须产生新身份。
        document = yaml.safe_load(changed["cohort_lock"].read_text())  # 小型真实锁文件。
        document["members"].reverse()  # 调换来源顺序。
        for index, member in enumerate(document["members"]):  # 新锁仍使用合法连续局部轴。
            member["cohort_index"] = index  # 所指物理成员变了，局部轴重新从零编号。
        changed["cohort_lock"].write_text(yaml.safe_dump(document))  # 仅覆盖本条测试的临时锁。
    after = identity_module.build_method_identity(**changed)  # 同一个生产算法处理新合同。
    assert before["identity_digest"] != after["identity_digest"], f"{field} 变化被错误地视为同一方法"


def test_project_relative_paths_canonical_copy_and_task_override(
    identity_module: ModuleType, identity_inputs: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    r"""改变 cwd 不改变路径语义；返回身份拥有自己的 JSON 数据，显式任务 alias 得到保留。"""
    identity_inputs["run_contract"]["task_id"] = "AnyMani-FlashSAC-Explicit-Task"  # 调用者拥有实际 task ID。
    first = identity_module.build_method_identity(**identity_inputs)  # 绝对路径参照。
    root = Path(__file__).resolve().parents[7]  # 相对路径锚定项目根，允许 ../ 外部临时资产。
    identity_inputs["cohort_lock"] = Path(os.path.relpath(identity_inputs["cohort_lock"], root))  # 字面相对路径。
    pregrasp = identity_inputs["pregrasp"]  # 同一结构对象的路径字段。
    pregrasp.catalog_root = os.path.relpath(pregrasp.catalog_root, root)  # 同一真实 index。
    monkeypatch.chdir(tmp_path)  # 故意从与项目根不同的位置执行。
    second = identity_module.build_method_identity(**identity_inputs)  # 路径与字典顺序不能造成身份噪声。
    assert first == second and second["task_id"] == "AnyMani-FlashSAC-Explicit-Task", "相对路径或 task ID 漂移"
    identity_inputs["task_contract"]["policy_hz"] = 10  # 返回之后改变外部对象。
    identity_inputs["provider_identity"]["encoder"] = "changed"  # 同样改变嵌套 provider 对象。
    assert (
        second["task_contract"]["policy_hz"] == 20 and second["geometry_provider"]["encoder"] == "N040"
    )  # 返回数据独立。


@pytest.mark.parametrize(
    "fault",  # 每次只破坏一个必要前提，以区分拒绝机制。
    ["schema", "empty", "duplicate", "order", "count", "rank", "strict", "bindings", "key", "run_config", "nonfinite"],
)
def test_invalid_or_contradictory_identity_inputs_fail_closed(
    identity_module: ModuleType,  # 不允许错误输入回退为空身份。
    identity_inputs: dict[str, Any],  # 资产轴、初态与运行条件的完整参照。
    fault: str,  # 当前要破坏的唯一前提。
) -> None:
    r"""锁轴、strict 初态或重复配置矛盾时拒绝建立身份，不发布看似完整的 resume 凭据。"""
    document = yaml.safe_load(identity_inputs["cohort_lock"].read_text())  # 修改范围仅在临时输入。
    if fault == "schema":  # 新训练只使用 canonical-final lock。
        document["schema_version"] = "1.1.0"  # 缺 canonical-final 身份约束。
    elif fault == "empty":  # 没有支持资产就没有实验成员轴。
        document["members"] = []  # 不允许空身份。
    elif fault == "duplicate":  # 不同局部 index 不能暗中指向同一来源键。
        document["members"][1]["source_row"] = document["members"][0]["source_row"]  # 重复成员。
    elif fault == "order":  # 局部轴顺序必须与文件顺序一致。
        document["members"][0]["cohort_index"] = 2  # 非连续有序轴。
    elif fault == "count":  # 算法的等额采样资产基数必须与锁一致。
        document["members"].pop()  # A=3 与 config A=4 不一致。
    elif fault == "rank":  # 初态排名不允许暗中切换。
        identity_inputs["pregrasp"].rank = 1  # 此原生方法固定rank-0。
    elif fault == "strict":  # 初态准入不得放宽。
        identity_inputs["pregrasp"].require_strict = False  # 非 strict。
    elif fault == "bindings":  # 一项资产不能缺少初态查询键。
        identity_inputs["pregrasp"].bindings = identity_inputs["pregrasp"].bindings[:-1]  # 长度不同。
    elif fault == "key":  # exact key 应是有效 JSON，而非随意文件名。
        identity_inputs["pregrasp"].bindings[0].key_json = "invalid-json"  # 不可审计查询键。
    elif fault == "run_config":  # 入口与 learner 不能声称不同并行数量。
        identity_inputs["run_contract"]["num_envs"] *= 2  # 与 FlashSACConfig 冲突。
    else:  # NaN 无稳定科学语义，不能进入 JSON 身份。
        identity_inputs["task_contract"]["policy_hz"] = float("nan")  # 非有限控制时钟。
    identity_inputs["cohort_lock"].write_text(yaml.safe_dump(document))  # 只改本测试的临时数据。
    with pytest.raises((ValueError, TypeError)):  # 所有失败早于 checkpoint 或环境创建。
        identity_module.build_method_identity(**identity_inputs)  # 不用空 identity 回退。
