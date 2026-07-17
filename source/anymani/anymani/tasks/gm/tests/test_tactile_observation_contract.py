r"""52D actor / 103D privileged task / 152D central critic 的纯 tensor contracts。"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_observation_module():
    r"""用最小 math/state stubs 加载 composite observation 文件。"""

    module_path = Path(__file__).resolve().parents[1] / "mdp" / "observations" / "observations_tactile.py"
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
    contact_stub = types.ModuleType("anymani.tasks.gm.mdp.tactile_contact_state")
    contact_stub.get_tactile_contact_state = lambda env, *_args, **_kwargs: env.contact
    adr_stub = types.ModuleType("anymani.tasks.gm.mdp.adr_state")
    adr_stub.gm_adr_state_observation = lambda env, action_dim=16: env.adr_state
    replacements.update(
        {
            "isaaclab": types.ModuleType("isaaclab"),
            "isaaclab.assets": assets_stub,
            "isaaclab.managers": managers_stub,
            "isaaclab.utils": types.ModuleType("isaaclab.utils"),
            "isaaclab.utils.math": math_stub,
            "anymani.tasks.gm.mdp.commands.tactile_rotation_command": command_stub,
            "anymani.tasks.gm.mdp.tactile_contact_state": contact_stub,
            "anymani.tasks.gm.mdp.adr_state": adr_stub,
        }
    )
    previous = {name: sys.modules.get(name) for name in replacements}
    try:
        sys.modules.update(replacements)
        spec = importlib.util.spec_from_file_location(
            "anymani.tasks.gm.mdp.observations.observations_tactile_contract", module_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load tactile observations from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


def _robot(batch: int) -> SimpleNamespace:
    r"""构造 16-DOF fake articulation data。"""

    return SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=torch.arange(batch * 16, dtype=torch.float32).reshape(batch, 16),
            joint_vel=torch.full((batch, 16), 2.0),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(batch, 1),
        )
    )


def _action_manager(batch: int) -> SimpleNamespace:
    r"""构造暴露 target/raw policy action 的 fake action manager。"""

    action_term = SimpleNamespace(
        current_targets=torch.full((batch, 16), torch.pi),
        raw_actions=torch.full((batch, 16), -0.5),
    )
    return SimpleNamespace(get_term=lambda _name: action_term)


def _contact(batch: int) -> SimpleNamespace:
    r"""构造 4 tip / 19 finger-non-tip / 1 palm 的共享 contact snapshot。"""

    return SimpleNamespace(
        tip_bits=torch.tensor([[True, False, True, False]]).repeat(batch, 1),
        tip_force_ema=torch.arange(4, dtype=torch.float32).repeat(batch, 1),
        palm_force_ema=torch.full((batch, 1), 7.0),
        finger_non_tip_bits=torch.zeros(batch, 19, dtype=torch.bool),
    )


def test_actor_frame_is_52d_in_exact_order_and_needs_no_privileged_state() -> None:
    r"""Actor 仅给 robot/action/tip snapshot 即可构造，证明没有隐式 object/ADR/goal 访问。"""

    module = _load_observation_module()
    batch = 2
    robot = _robot(batch)
    env = SimpleNamespace(
        num_envs=batch,
        device="cpu",
        scene={"robot": robot},  # 故意不提供 object
        action_manager=_action_manager(batch),
        contact=_contact(batch),
    )
    robot_cfg = SimpleNamespace(name="robot", joint_ids=list(range(16)))

    frame = module.tactile_rotation_policy_frame(
        env,
        fingertip_sensor_names=("t0", "t1", "t2", "t3"),
        finger_non_tip_sensor_names=tuple(f"n{i}" for i in range(19)),
        palm_sensor_name="palm",
        robot_cfg=robot_cfg,
    )

    assert frame.shape == (batch, 52)
    assert torch.allclose(frame[:, :16], robot.data.joint_pos / torch.pi)
    assert torch.allclose(frame[:, 16:32], torch.ones(batch, 16))  # target $u=\pi$ -> 1
    assert torch.allclose(frame[:, 32:48], torch.full((batch, 16), -0.5))
    assert torch.equal(frame[:, 48:].bool(), env.contact.tip_bits)


def test_privileged_and_full_critic_shapes_are_103_and_152() -> None:
    r"""Critic schema 必须闭合为 103 task/contact + 48 ADR + 1 curriculum。"""

    module = _load_observation_module()
    batch = 2
    robot = _robot(batch)
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
        action_manager=_action_manager(batch),
        contact=_contact(batch),
        adr_state=torch.arange(batch * 48, dtype=torch.float32).reshape(batch, 48),
        command=SimpleNamespace(
            position_anchor_w=torch.zeros(batch, 3),
            goal_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(batch, 1),
        ),
        _gm_reward_curriculum_lambda=torch.tensor(0.75),
    )
    robot_cfg = SimpleNamespace(name="robot", joint_ids=list(range(16)))
    object_cfg = SimpleNamespace(name="object")
    kwargs = {
        "command_name": "goal_pose",
        "fingertip_sensor_names": ("t0", "t1", "t2", "t3"),
        "finger_non_tip_sensor_names": tuple(f"n{i}" for i in range(19)),
        "palm_sensor_name": "palm",
        "robot_cfg": robot_cfg,
        "object_cfg": object_cfg,
    }

    task_state = module.tactile_rotation_privileged_task_state(env, **kwargs)
    critic_state = module.tactile_rotation_critic_state(env, **kwargs)

    assert task_state.shape == (batch, 103)
    assert critic_state.shape == (batch, 152)
    assert torch.allclose(critic_state[:, :103], task_state)
    assert torch.allclose(critic_state[:, 103:151], env.adr_state)
    assert torch.allclose(critic_state[:, 151], torch.full((batch,), 0.75))
