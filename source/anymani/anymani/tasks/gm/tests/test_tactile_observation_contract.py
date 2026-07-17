r"""Tactile rotation privileged terms 与 observation module ownership contracts。"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

OBS_DIR = Path(__file__).resolve().parents[1] / "mdp" / "observations"
r"""GM observation modules 目录；用于锁定 state/tactile/priv 三类语义所有权。"""


def _load_privileged_observation_module() -> types.ModuleType:
    r"""用最小 math/command/ADR stubs 加载 `observations_priv.py`。"""

    module_path = OBS_DIR / "observations_priv.py"
    package_names = (
        "anymani",
        "anymani.tasks",
        "anymani.tasks.gm",
        "anymani.tasks.gm.mdp",
        "anymani.tasks.gm.mdp.observations",
        "anymani.tasks.gm.mdp.commands",
    )
    replacements: dict[str, types.ModuleType] = {}
    for package_name in package_names:
        package = types.ModuleType(package_name)
        package.__path__ = []  # type: ignore[attr-defined]
        replacements[package_name] = package

    math_stub = types.ModuleType("isaaclab.utils.math")
    math_stub.quat_apply_inverse = lambda _quat, vector: vector
    math_stub.matrix_from_quat = lambda quat: torch.eye(3).repeat(quat.shape[0], 1, 1)
    assets_stub = types.ModuleType("isaaclab.assets")
    assets_stub.Articulation = object
    assets_stub.RigidObject = object
    managers_stub = types.ModuleType("isaaclab.managers")
    managers_stub.SceneEntityCfg = lambda name, **kwargs: SimpleNamespace(name=name, **kwargs)
    command_stub = types.ModuleType("anymani.tasks.gm.mdp.commands.tactile_rotation_command")
    command_stub.ensure_post_physics_progress_updated = lambda env, _name: env.command
    adr_stub = types.ModuleType("anymani.tasks.gm.mdp.adr_state")
    adr_stub.get_gm_adr_state = lambda env, _action_dim=16: SimpleNamespace(values=env.adr_state)
    replacements.update(
        {
            "isaaclab": types.ModuleType("isaaclab"),
            "isaaclab.assets": assets_stub,
            "isaaclab.managers": managers_stub,
            "isaaclab.utils": types.ModuleType("isaaclab.utils"),
            "isaaclab.utils.math": math_stub,
            "anymani.tasks.gm.mdp.commands.tactile_rotation_command": command_stub,
            "anymani.tasks.gm.mdp.adr_state": adr_stub,
        }
    )
    previous = {name: sys.modules.get(name) for name in replacements}
    try:
        sys.modules.update(replacements)
        spec = importlib.util.spec_from_file_location(
            "anymani.tasks.gm.mdp.observations.observations_priv_contract", module_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load privileged observations from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


def test_object_goal_and_training_context_terms_have_exact_shapes() -> None:
    r"""Privileged module 应独立提供 15D task、48D ADR 与 1D reward-release state。"""

    module = _load_privileged_observation_module()
    batch = 2
    robot = SimpleNamespace(
        data=SimpleNamespace(root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(batch, 1))
    )
    object_asset = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=torch.zeros(batch, 3),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(batch, 1),
            root_lin_vel_w=torch.full((batch, 3), 0.25),
            root_ang_vel_w=torch.full((batch, 3), 0.5),
        )
    )
    env = SimpleNamespace(
        num_envs=batch,
        device="cpu",
        scene={"robot": robot, "object": object_asset},
        command=SimpleNamespace(
            position_anchor_w=torch.zeros(batch, 3),
            goal_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(batch, 1),
        ),
        adr_state=torch.arange(batch * 48, dtype=torch.float32).reshape(batch, 48),
        _gm_reward_curriculum_lambda=torch.tensor(0.75),
    )

    task_state = module.object_goal_task_state(env, command_name="goal_pose")
    adr_state = module.adr_actual_state(env)
    reward_release = module.reward_release_coefficient(env)

    assert task_state.shape == (batch, 15)
    assert adr_state.shape == (batch, 48)
    assert reward_release.shape == (batch, 1)
    torch.testing.assert_close(task_state[:, :3], torch.zeros(batch, 3))
    torch.testing.assert_close(task_state[:, 3:9], torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]).repeat(batch, 1))
    torch.testing.assert_close(task_state[:, 9:12], torch.full((batch, 3), 0.25))
    torch.testing.assert_close(task_state[:, 12:15], torch.full((batch, 3), 0.5))
    torch.testing.assert_close(adr_state, env.adr_state)
    torch.testing.assert_close(reward_release, torch.full((batch, 1), 0.75))


def test_observation_modules_follow_state_tactile_privileged_ownership() -> None:
    r"""Contact 文件与 task-level composite helpers 均应出清，只保留语义模块。"""

    assert not (OBS_DIR / "observations_contact.py").exists()
    state_source = (OBS_DIR / "observations_state.py").read_text(encoding="utf-8")
    tactile_source = (OBS_DIR / "observations_tactile.py").read_text(encoding="utf-8")
    priv_source = (OBS_DIR / "observations_priv.py").read_text(encoding="utf-8")
    all_source = state_source + tactile_source + priv_source

    assert "def joint_target(" in state_source
    assert "def tip_contact_bits_ema(" in tactile_source
    assert "def object_goal_task_state(" in priv_source
    assert "tactile_rotation_policy_frame" not in all_source
    assert "tactile_rotation_critic_state" not in all_source
