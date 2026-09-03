r"""MVP80 method/run identity对task、policy与PPO配置的fail-closed合同。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def _identity(tmp_path: Path, *, learning_rate: float) -> dict[str, Any]:
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
        run_contract={"seed": 42, "actor_base_lr": learning_rate, "num_envs": 2560},
    )


def test_identity_binds_film_contact_reward_and_training_contract(tmp_path: Path) -> None:
    r"""Reward归约、all-owner bits、FiLM residual与PPO配置必须共同进入checkpoint身份。"""

    first = _identity(tmp_path, learning_rate=3.0e-4)
    changed = _identity(tmp_path, learning_rate=1.0e-4)
    assert first["identity_schema_version"] == "3.0.0"
    assert first["task_contract"]["stable_joint_reduction"] == "reference-dof-16"
    assert first["policy"]["actor_contact"] == "all-owner-binary-no-force"
    assert "dynamic-film-base" in first["policy"]["residual_decomposition"]
    assert first["identity_digest"] != changed["identity_digest"]
