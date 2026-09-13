r"""MVP80 method/run identity对task、policy与PPO配置的fail-closed合同。"""

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from anymani.distill.rl.masked_ppo import validate_anymani_checkpoint_identity
from anymani.distill.rl.runtime import palm_rotation_identity as identity_module
from anymani.distill.rl.runtime.palm_rotation_identity import build_palm_rotation_method_identity


@dataclass(frozen=True)
class _Binding:
    r"""测试用schema-3 catalog key surface。"""

    key_json: str


@dataclass(frozen=True)
class _Pregrasp:
    r"""测试用strict rank-0 pregrasp identity配置。"""

    catalog_root: str
    bindings: tuple[_Binding, ...]
    rank: int = 0
    require_strict: bool = True


def _identity(
    tmp_path: Path, *, learning_rate: float, reward_release_start_turns: float = 1.0, arm: str = "residual", **overrides: Any
) -> dict[str, Any]:
    r"""构造只改变base LR的一组完整identity。"""

    manifest = tmp_path / "ppo_mvp80.yaml"
    manifest.write_text("selected_rows: []\n", encoding="utf-8")
    catalog = tmp_path / "catalog"
    catalog.mkdir(exist_ok=True)
    catalog.joinpath("index.json").write_text("{}\n", encoding="utf-8")
    return build_palm_rotation_method_identity(
        provider_identity={"identity_digest": "p" * 64},
        manifest_path=manifest,
        selected_rows=tuple(range(80)),
        pregrasp=_Pregrasp(
            catalog_root=str(catalog),
            bindings=tuple(_Binding(f'{{"row":{row}}}') for row in range(80)),
        ),
        arm=arm,
        run_contract={
            "seed": 42,
            "actor_base_lr": learning_rate,
            "num_envs": 2560,
            "reward_release_start_turns": reward_release_start_turns,
            "reward_release_end_turns": 2.0,
            "reward_release_ema_alpha": 0.05,
            **overrides,
        },
    )


def test_identity_binds_film_contact_reward_and_training_contract(tmp_path: Path) -> None:
    r"""Reward归约、all-owner bits、FiLM residual与PPO配置必须共同进入checkpoint身份。"""

    first = _identity(tmp_path, learning_rate=3.0e-4)
    changed = _identity(tmp_path, learning_rate=1.0e-4)
    assert first["identity_schema_version"] == "4.0.0"
    assert first["task_contract"]["stable_joint_reduction"] == "reference-dof-16"
    assert first["task_contract"]["training_mdp_anchor"] == "N000-gm-tactile-rotation-v0.5.0"
    assert first["task_contract"]["rotation_frontier_reward_weight"] == 0.0
    assert first["task_contract"]["strict_tracking_reward_weight"] == 10.0
    assert first["task_contract"]["reward_release"] == {
        "aggregation": "per-asset-ema-to-handedness-inclusive-cell-median",
        "ema_alpha": 0.05,
        "end_turns": 2.0,
        "start_turns": 1.0,
        "floor": 0.0,
        "reference_seconds": 120.0,
    }
    assert first["policy"]["actor_contact"] == "all-owner-binary-no-force"
    assert "dynamic-film-base" in first["policy"]["residual_decomposition"]
    assert first["manifest"]["support_asset_count"] == 80
    assert first["identity_digest"] != changed["identity_digest"]


def test_phase_identity_binds_runtime_clock_without_changing_physics_or_geometry(tmp_path: Path) -> None:
    base = _identity(tmp_path, learning_rate=3e-5, arm="direct_token")
    explicit_off = _identity(tmp_path, learning_rate=3e-5, arm="direct_token", phase_period_steps=None)
    clock = _identity(tmp_path, learning_rate=3e-5, arm="direct_token", phase_period_steps=43)
    changed_period = _identity(tmp_path, learning_rate=3e-5, arm="direct_token", phase_period_steps=44)
    assert base == explicit_off
    assert clock["task_contract"] == base["task_contract"]
    assert clock["geometry_provider"] == base["geometry_provider"]
    assert clock["transport_abi"]["float_shapes"]["phase_clock"] == [2]
    assert "phase_clock" not in base["transport_abi"]["float_shapes"]
    assert clock["policy"]["phase_clock"]["period_policy_steps"] == 43
    assert clock["policy"]["phase_clock"]["encoding"] == ["sin", "cos"]
    assert len({base["identity_digest"], clock["identity_digest"], changed_period["identity_digest"]}) == 3
    with pytest.raises(ValueError, match="direct"):
        _identity(tmp_path, learning_rate=3e-5, arm="residual", phase_period_steps=43)


def test_limit_recovery_distribution_enters_method_identity(tmp_path: Path) -> None:
    r"""同样的Actor张量配合不同局部探索规则仍是不同方法，且物理动作权限保持。"""
    baseline = _identity(tmp_path, learning_rate=3e-5)
    recovery = _identity(tmp_path, learning_rate=3e-5, recovery_sigma_floor=0.6, max_log_std=-0.5)
    assert baseline["identity_digest"] != recovery["identity_digest"]
    rule = recovery["policy"]["recovery_exploration"]
    assert rule["sigma_floor"] == 0.6
    assert rule["limit_margin_rad"] == 0.02
    assert rule["previous_outward_action_min"] == 0.05
    assert rule["contact_scope"] == "all-valid-tips-zero"
    assert recovery["policy"]["action_authority_rad_per_policy_step"] == 1 / 24
    assert "recovery_exploration" not in baseline["policy"]


@pytest.mark.parametrize("floor", [0.0, -0.1, float("nan"), float("inf"), 0.7])
def test_limit_recovery_identity_rejects_invalid_floor(tmp_path: Path, floor: float) -> None:
    r"""声明不能静默超过原潜高斯上限，也不能保存非法尺度。"""
    with pytest.raises(ValueError, match="recovery"):
        _identity(tmp_path, learning_rate=3e-5, recovery_sigma_floor=floor, max_log_std=-0.5)


def test_identity_binds_early_reward_release_schedule(tmp_path: Path) -> None:
    r"""从0圈渐进释放稳定项是训练MDP干预，必须与历史1圈dead-zone产生不同method identity。"""

    baseline = _identity(tmp_path, learning_rate=3.0e-4)
    early = _identity(tmp_path, learning_rate=3.0e-4, reward_release_start_turns=0.0)

    assert early["task_contract"]["reward_release"]["start_turns"] == 0.0
    assert baseline["identity_digest"] != early["identity_digest"]


def test_tip_only_short_horizon_and_release_floor_are_explicit_method_changes(tmp_path: Path) -> None:
    r"""删除不可部署触觉、缩短回合和固定塑形各自改变身份，不能伪装完整续训。"""

    baseline = _identity(tmp_path, learning_rate=1.0e-4)
    tip = _identity(tmp_path, learning_rate=1.0e-4, actor_contact="tip")
    short = _identity(tmp_path, learning_rate=1.0e-4, episode_seconds_min=20.0, episode_seconds_max=60.0)
    floor = _identity(tmp_path, learning_rate=1.0e-4, reward_release_floor=1.0)
    assert tip["policy"]["actor_contact"] == "tip-only-binary"
    assert short["task_contract"]["episode_seconds"] == 60.0
    assert short["task_contract"]["episode_seconds_min"] == 20.0
    assert floor["task_contract"]["reward_release"]["floor"] == 1.0
    assert len({value["identity_digest"] for value in (baseline, tip, short, floor)}) == 4


def test_progress_clip_is_a_task_contract_parameter_not_just_training_metadata(tmp_path: Path) -> None:
    r"""每步0.025与0.04 rad对应不同进展奖励，评价和恢复必须显式绑定该差异。"""

    baseline = _identity(tmp_path, learning_rate=1.0e-4)
    faster = _identity(tmp_path, learning_rate=1.0e-4, rotation_progress_clip_rad_per_step=0.04)
    assert baseline["task_contract"]["rotation_progress_clip_rad_per_step"] == 0.025
    assert faster["task_contract"]["rotation_progress_clip_rad_per_step"] == 0.04
    assert baseline["identity_digest"] != faster["identity_digest"]


@pytest.mark.parametrize("weight", (0.0, 20.0))
def test_progress_reward_weight_is_distinct_from_clip_and_action_authority(tmp_path: Path, weight: float) -> None:
    r"""进展系数只改变$w_\psi\operatorname{clip}(\Delta\psi,\pm c)$，不能改写动作或截断。

    $w_\psi$单位为reward/rad；0表示显式消融，20表示旋转激励增强。
    历史checkpoint未保存该字段时，其任务系数仍为5，不能在重放时隐式改成新值。
    """

    baseline = _identity(tmp_path, learning_rate=3.0e-4)  # 历史任务的默认进展系数为5
    changed = _identity(tmp_path, learning_rate=3.0e-4, rotation_progress_reward_weight=weight)
    assert "rotation_progress_reward_weight" not in baseline["task_contract"]  # 保持已发布任务字段布局
    assert changed["task_contract"]["rotation_progress_reward_weight"] == weight  # 新系数必须属于MDP身份
    remaining = dict(changed["task_contract"])
    remaining.pop("rotation_progress_reward_weight")
    assert remaining == baseline["task_contract"]  # 包括clip、reward-release、终止等合同均不变
    assert changed["policy"] == baseline["policy"]  # 不借奖励干预改变动作authority或观测
    assert changed["identity_digest"] != baseline["identity_digest"]


@pytest.mark.parametrize("weight", (0.0, 0.25))
def test_full_pose_weight_changes_shaping_without_changing_progress_or_physics(tmp_path: Path, weight: float) -> None:
    r"""物体全位姿kernel系数独立于进展20与关节初姿罚；0是整项塑形消融。

    $r_t^{pose}=w_{pose}K_{pose}(s_t,g_t)\Delta t$；kernel无量纲，系数单位为reward/s。
    它同时包含物体位置和姿态，不是joint_pose_anchor，也不改变硬物理终止。
    """

    baseline = _identity(tmp_path, learning_rate=3.0e-4, rotation_progress_reward_weight=20.0)
    changed = _identity(
        tmp_path, learning_rate=3.0e-4, rotation_progress_reward_weight=20.0, pose_keypoint_reward_weight=weight
    )
    assert "pose_keypoint_reward_weight" not in baseline["task_contract"]  # 历史默认1保持字段布局
    assert changed["task_contract"]["pose_keypoint_reward_weight"] == weight
    remaining = dict(changed["task_contract"])
    remaining.pop("pose_keypoint_reward_weight")
    assert remaining == baseline["task_contract"]  # 保留进展、goal、release与物理边界
    assert changed["policy"] == baseline["policy"]
    assert changed["identity_digest"] != baseline["identity_digest"]


def test_pose_mode_default_is_canonical_and_position_only_is_an_explicit_task_change(tmp_path: Path) -> None:
    r"""相同系数下，位置核与全位姿核是不同MDP；默认full_pose保持隐式字段布局。

    模式只决定pose_keypoint的测量几何，Actor信息、动作authority与strict goal合同不随它改变。
    显式full_pose与缺省full_pose需产生相同语义身份，以免CLI默认值制造额外的checkpoint分支。
    """

    baseline = _identity(tmp_path, learning_rate=3.0e-4)  # 历史默认使用全位姿kernel
    explicit = _identity(tmp_path, learning_rate=3.0e-4, pose_keypoint_mode="full_pose")  # 同一数学模式
    assert baseline == explicit  # 包括training字段：默认模式应规范化为省略字段
    position = _identity(tmp_path, learning_rate=3.0e-4, pose_keypoint_mode="position_only")  # 新的稠密塑形
    assert position["task_contract"]["pose_keypoint_mode"] == "position_only"  # 任务身份必须绑定测量模式
    assert position["training"]["pose_keypoint_mode"] == "position_only"  # 评价应从训练记录恢复实际模式
    other = dict(position["task_contract"])  # 除模式外的任务参数应一致
    other.pop("pose_keypoint_mode")  # 去掉唯一被干预的任务字段
    assert other == baseline["task_contract"] and position["policy"] == baseline["policy"]  # strict goal与信息边界
    with pytest.raises(RuntimeError, match="identity mismatch"):
        validate_anymani_checkpoint_identity(runtime_identity=position, checkpoint_identity=baseline)  # 不跨MDP续训
    with pytest.raises(RuntimeError, match="semantic identity mismatch"):
        identity_module.validate_palm_rotation_evaluation_identity(
            runtime_identity=position, checkpoint_identity=baseline
        )  # 新模式不能冒充旧checkpoint训练时的奖励语义


@pytest.mark.parametrize("mode", (None, "position", "", 1))
def test_pose_mode_rejects_ambiguous_or_unknown_measurement_geometry(tmp_path: Path, mode: Any) -> None:
    r"""测量几何只允许full_pose或position_only，不能将拼写错误静默回退。"""

    with pytest.raises(ValueError, match="pose keypoint mode"):
        _identity(tmp_path, learning_rate=3.0e-4, pose_keypoint_mode=mode)  # 非法模式必须在身份构造时拒绝


@pytest.mark.parametrize("entry", ("train_palm_rotation_mvp.py", "evaluate_palm_rotation_mvp.py"))
@pytest.mark.parametrize("mode", (None, "full_pose", "position_only"))
def test_pose_mode_runtime_wiring_does_not_modify_strict_goal_radius(entry: str, mode: str | None) -> None:
    r"""执行真实入口的reward参数赋值；评价旧字段缺失时恢复full_pose。

    只执行该赋值AST，命令的0.05 m keypoint半径仍归strict goal使用。
    完整的配置装配、奖励输出与保存恢复由全cohort运行检查验证。
    """

    path = Path(__file__).resolve().parents[3] / "rl" / entry  # 两个真实生产入口
    tree = ast.parse(path.read_text())  # 读取实际代码，而不是复制一个恢复表达式
    target_name = "env_cfg.rewards.pose_keypoint.params['position_only']"  # 仅reward term的局部参数
    statements: list[ast.stmt] = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and any(ast.unparse(t) == target_name for t in node.targets)
    ]  # mode不应通过修改command全局半径实现
    assert len(statements) == 1, "entry must configure reward-only pose mode"  # 入口有且仅有一处明确赋值
    command_cfg = SimpleNamespace(keypoint_radius_m=0.05)  # strict goal的原始几何尺度，m
    reward = SimpleNamespace(params={})  # 模式必须真实写入term.params
    namespace = {
        "env_cfg": SimpleNamespace(
            rewards=SimpleNamespace(pose_keypoint=reward), commands=SimpleNamespace(goal_pose=command_cfg)
        ),  # reward与command配置是独立对象
        "args_cli": SimpleNamespace(pose_keypoint_mode=mode or "full_pose"),
        "run_contract": {} if mode is None else {"pose_keypoint_mode": mode},
    }  # 训练默认由CLI形成；旧评价允许缺字段
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(path), "exec"), namespace)  # 不启动AppLauncher
    assert reward.params["position_only"] is (mode == "position_only")  # 位置模式显式True，其余False
    assert command_cfg.keypoint_radius_m == 0.05  # strict orientation尺度未改变


@pytest.mark.parametrize("field", ("rotation_progress_reward_weight", "pose_keypoint_reward_weight"))
@pytest.mark.parametrize("weight", (-1.0, float("nan"), float("inf"), -float("inf")))
def test_positive_reward_weights_reject_nonfinite_or_reversed_objective(
    tmp_path: Path, field: str, weight: float
) -> None:
    r"""进展和全位姿的奖励系数须有限且非负；负号不是这两项的同目标干预。"""

    with pytest.raises(ValueError, match="reward weight"):
        _identity(tmp_path, learning_rate=3.0e-4, **{field: weight})


@pytest.mark.parametrize(
    "term,contract,expected",
    (
        ("rotation_progress", {}, 5.0),
        ("rotation_progress", {"rotation_progress_reward_weight": 0.0}, 0.0),
        ("rotation_progress", {"rotation_progress_reward_weight": 20.0}, 20.0),
        ("pose_keypoint", {}, 1.0),
        ("pose_keypoint", {"pose_keypoint_reward_weight": 0.0}, 0.0),
        ("pose_keypoint", {"pose_keypoint_reward_weight": 0.25}, 0.25),
    ),
)
def test_evaluation_restores_reward_weights_without_starting_simulator(
    term: str, contract: dict, expected: float
) -> None:
    r"""执行评价入口的真实赋值：缺字段按历史进展5/全位姿1，显式零值不得丢失。

    只提取该赋值AST，不import会启动AppLauncher的评价模块；真实环境链由独立canary覆盖。
    """

    path = Path(__file__).resolve().parents[3] / "rl" / "evaluate_palm_rotation_mvp.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))  # 生产入口的源码，不复制一份恢复公式
    assignments: list[ast.stmt] = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(ast.unparse(target) == f"env_cfg.rewards.{term}.weight" for target in node.targets)
    ]
    assert len(assignments) == 1, f"evaluation must restore the recorded {term} reward weight"
    reward = SimpleNamespace(weight=-999.0)  # sentinel不能被环境默认值掩盖
    namespace = {
        "env_cfg": SimpleNamespace(rewards=SimpleNamespace(**{term: reward})),
        "run_contract": contract,
    }
    exec(compile(ast.Module(body=assignments, type_ignores=[]), str(path), "exec"), namespace)
    assert reward.weight == expected  # 零系数是合法消融，不能被truthy fallback改回历史值


def test_strict_goal_weight_is_explicit_without_adding_frontier_reward(tmp_path: Path) -> None:
    r"""Strict-only降权必须单独进入任务身份，物理frontier仍只有诊断用途。"""

    baseline = _identity(tmp_path, learning_rate=1.0e-4)
    lower = _identity(tmp_path, learning_rate=1.0e-4, strict_goal_reward_weight=1.0)
    assert baseline["task_contract"]["strict_tracking_reward_weight"] == 10.0
    assert lower["task_contract"]["strict_tracking_reward_weight"] == 1.0
    assert lower["task_contract"]["rotation_frontier_reward_weight"] == 0.0
    assert baseline["identity_digest"] != lower["identity_digest"]


def test_joint_anchor_override_is_explicit_and_preserves_other_task_semantics(tmp_path: Path) -> None:
    r"""取消软关节初姿惩罚是单独奖励干预，不能顺便改变物体、动作或其他任务规则。"""

    baseline = _identity(tmp_path, learning_rate=1.0e-4)
    relaxed = _identity(tmp_path, learning_rate=1.0e-4, joint_pose_anchor_weight=0.0)
    assert "joint_pose_anchor_weight" not in baseline["task_contract"]
    assert relaxed["task_contract"]["joint_pose_anchor_weight"] == 0.0
    other = dict(relaxed["task_contract"])
    other.pop("joint_pose_anchor_weight")
    assert other == baseline["task_contract"]
    assert relaxed["policy"] == baseline["policy"]


def test_identity_accepts_explicit_single_asset_closure_with_one_strict_binding(tmp_path: Path) -> None:
    r"""Single-embodiment closure应复用同一identity schema，并显式记录支持集基数1。"""

    manifest = tmp_path / "ppo_mvp80.yaml"
    manifest.write_text("selected_rows: [1966]\n", encoding="utf-8")
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    catalog.joinpath("index.json").write_text("{}\n", encoding="utf-8")
    identity = build_palm_rotation_method_identity(
        provider_identity={"identity_digest": "p" * 64},
        manifest_path=manifest,
        selected_rows=(1966,),
        pregrasp=_Pregrasp(catalog_root=str(catalog), bindings=(_Binding('{"row":1966}'),)),
        arm="residual",
        run_contract={"seed": 42, "num_envs": 1280},
    )

    assert identity["identity_schema_version"] == "4.0.0"
    assert identity["manifest"]["support_asset_count"] == 1
    assert identity["manifest"]["selected_rows"] == [1966]


def test_identity_distinguishes_local_skip_and_token_only_direct_heads(tmp_path: Path) -> None:
    r"""Checkpoint identity必须区分feature bypass与只读取$H^a_{t,j}$的canonical Direct。"""

    manifest = tmp_path / "ppo_mvp80.yaml"
    manifest.write_text("selected_rows: [0]\n", encoding="utf-8")
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    catalog.joinpath("index.json").write_text("{}\n", encoding="utf-8")
    common = {
        "provider_identity": {"identity_digest": "p" * 64},
        "manifest_path": manifest,
        "selected_rows": (0,),
        "pregrasp": _Pregrasp(catalog_root=str(catalog), bindings=(_Binding('{"row":0}'),)),
        "run_contract": {"seed": 42, "num_envs": 128},
    }

    local_skip = build_palm_rotation_method_identity(**common, arm="direct")
    token_only = build_palm_rotation_method_identity(**common, arm="direct_token")

    assert local_skip["policy"]["direct_decomposition"] == "full-authority-contextual-plus-local-skip"
    assert token_only["policy"]["direct_decomposition"] == "full-authority-contextual-token-only"
    assert local_skip["identity_digest"] != token_only["identity_digest"]


def test_commit_provenance_does_not_change_identical_implementation_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""仅提交Git、未改执行源码时，method和Parquet续接身份必须保持不变。"""

    monkeypatch.setattr(identity_module, "_git_head", lambda _root: "a" * 40)
    first = _identity(tmp_path, learning_rate=3.0e-4)
    assert identity_module.palm_rotation_code_provenance() == {"git_head": "a" * 40}
    monkeypatch.setattr(identity_module, "_git_head", lambda _root: "b" * 40)
    second = _identity(tmp_path, learning_rate=3.0e-4)
    assert identity_module.palm_rotation_code_provenance() == {"git_head": "b" * 40}
    assert first == second and "git_head" not in first["implementation"]
    validate_anymani_checkpoint_identity(runtime_identity=second, checkpoint_identity=first)


def test_readonly_replay_requires_exact_certificate_and_never_weakens_resume(tmp_path: Path) -> None:
    r"""旧schema跨实现只允许带证书的评估；任务漂移和完整续训均不能借此通过。"""

    runtime = _identity(tmp_path, learning_rate=3.0e-4)
    legacy = deepcopy(runtime)
    legacy["identity_schema_version"] = "3.0.0"
    legacy["implementation"] = {"git_head": "a" * 40, "files": {"legacy.py": "b" * 64}}
    legacy["identity_digest"] = identity_module._stable_digest(
        {key: value for key, value in legacy.items() if key != "identity_digest"}
    )
    certificate = {
        "artifact_type": "anymani.palm_rotation.refactor_equivalence",
        "schema_version": "1.0.0",
        "passed": True,
        "reference_implementation_files": legacy["implementation"]["files"],
        "current_implementation_files": runtime["implementation"]["files"],
    }
    with pytest.raises(RuntimeError, match="certificate is required"):
        identity_module.validate_palm_rotation_evaluation_identity(runtime_identity=runtime, checkpoint_identity=legacy)
    identity_module.validate_palm_rotation_evaluation_identity(
        runtime_identity=runtime, checkpoint_identity=legacy, implementation_certificate=certificate
    )
    with pytest.raises(RuntimeError, match="identity mismatch"):
        validate_anymani_checkpoint_identity(runtime_identity=runtime, checkpoint_identity=legacy)
    wrong_certificate = {**certificate, "current_implementation_files": {"wrong.py": "c" * 64}}
    with pytest.raises(RuntimeError, match="does not cover"):
        identity_module.validate_palm_rotation_evaluation_identity(
            runtime_identity=runtime, checkpoint_identity=legacy, implementation_certificate=wrong_certificate
        )
    changed = _identity(tmp_path, learning_rate=1.0e-4)
    with pytest.raises(RuntimeError, match="semantic identity mismatch"):
        identity_module.validate_palm_rotation_evaluation_identity(
            runtime_identity=changed, checkpoint_identity=legacy, implementation_certificate=certificate
        )


def test_rejected_action_auxiliary_identity(tmp_path: Path) -> None:
    r"""辅助目标身份和MDP合同分别记录，显式零权重保持旧默认布局。"""
    baseline = _identity(tmp_path, learning_rate=3e-5)
    zero = _identity(tmp_path, learning_rate=3e-5, rejected_action_weight=0.0)
    changed = _identity(tmp_path, learning_rate=3e-5, rejected_action_weight=0.05)
    assert zero == baseline
    assert changed["task_contract"] == baseline["task_contract"]
    assert changed["policy"] == baseline["policy"]
    assert changed["transport_abi"] == baseline["transport_abi"]
    assert changed["identity_digest"] != baseline["identity_digest"]
    contract = changed["training"]["rejected_action_regularization"]
    assert contract["weight"] == 0.05
    assert contract["applied_to"] == "actor-objective-only-not-environment-reward"
    assert contract["action_authority_rad_per_policy_step"] == 1 / 24
    for invalid in [-1.0, float("nan"), float("inf")]:
        with pytest.raises(ValueError, match="rejected action weight"):
            _identity(tmp_path, learning_rate=3e-5, rejected_action_weight=invalid)
