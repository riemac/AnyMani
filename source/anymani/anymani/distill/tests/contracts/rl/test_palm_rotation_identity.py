r"""MVP80 method/run identity对task、policy与PPO配置的fail-closed合同。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
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
    tmp_path: Path, *, learning_rate: float, reward_release_start_turns: float = 1.0, **overrides: Any
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
        arm="residual",
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
