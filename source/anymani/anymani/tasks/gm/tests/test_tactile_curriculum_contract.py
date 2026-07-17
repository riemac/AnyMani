r"""Net-rotation reward curriculum 与 ADR promotion 的纯 tensor contracts。"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_curriculum_module():
    r"""以最小 IsaacLab/command stubs 加载 curriculum 文件，不启动 Kit。"""

    module_path = Path(__file__).resolve().parents[1] / "mdp" / "curriculums.py"
    package_names = ("anymani", "anymani.tasks", "anymani.tasks.gm", "anymani.tasks.gm.mdp", "anymani.tasks.gm.mdp.commands")
    replacements: dict[str, types.ModuleType] = {}
    for package_name in package_names:
        package = types.ModuleType(package_name)
        package.__path__ = []  # type: ignore[attr-defined]
        replacements[package_name] = package

    command_stub = types.ModuleType("anymani.tasks.gm.mdp.commands.tactile_rotation_command")
    command_stub.ensure_post_physics_progress_updated = lambda env, _name: env.command
    adr_state_stub = types.ModuleType("anymani.tasks.gm.mdp.adr_state")
    adr_state_stub.get_gm_adr_state = lambda _env: SimpleNamespace(set=lambda *_args, **_kwargs: None)
    envs_stub = types.ModuleType("isaaclab.envs")
    envs_stub.ManagerBasedRLEnv = object
    managers_stub = types.ModuleType("isaaclab.managers")
    managers_stub.CurriculumTermCfg = object
    managers_stub.ManagerTermBase = object
    replacements.update(
        {
            "anymani.tasks.gm.mdp.commands.tactile_rotation_command": command_stub,
            "anymani.tasks.gm.mdp.adr_state": adr_state_stub,
            "isaaclab": types.ModuleType("isaaclab"),
            "isaaclab.envs": envs_stub,
            "isaaclab.managers": managers_stub,
        }
    )
    previous = {name: sys.modules.get(name) for name in replacements}
    try:
        sys.modules.update(replacements)
        spec = importlib.util.spec_from_file_location("anymani.tasks.gm.mdp.curriculums_contract", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load curriculum module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


class _MissingEventManager:
    r"""模拟未启用 optional ADR range EventTerms 的 env。"""

    def get_term_cfg(self, _name: str):
        r"""所有 optional term 均缺失，scheduler 应安全跳过。"""

        raise ValueError

    def set_term_cfg(self, _name: str, _cfg) -> None:
        r"""防御接口；当前测试不应调用。"""

        raise AssertionError("Missing event term must not be updated")


def test_net_rotation_reward_release_is_zero_one_linear_two_saturated() -> None:
    r"""正向净旋转 EMA 应在 1 圈前为 0、1→2 圈线性、2 圈后为 1。"""

    module = _load_curriculum_module()
    turns = torch.tensor([0.0, 1.0, 1.5, 2.0, 3.0])

    release = module.net_rotation_reward_release(turns, 1.0, 2.0)

    assert torch.allclose(release, torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))


def test_net_rotation_rate_uses_sampled_full_horizon_seconds() -> None:
    r"""同样净圈数在更长 sampled horizon 下 competence 必须更低。"""

    module = _load_curriculum_module()
    turns = torch.tensor(4.0)

    assert torch.allclose(module.net_rotation_rate_turns_per_s(turns, torch.tensor(100.0)), torch.tensor(0.04))
    assert torch.allclose(module.net_rotation_rate_turns_per_s(turns, torch.tensor(20.0)), torch.tensor(0.2))


def test_adr_level_zero_bootstraps_then_uses_cooldown_and_rate_threshold() -> None:
    r"""第 0 档首次检查自动升级；后续档必须满足 960 checks 与 0.08 turns/s。"""

    module = _load_curriculum_module()
    scheduler = module.LeapADRByNetRotationRate.__new__(module.LeapADRByNetRotationRate)
    scheduler.increment = 0
    scheduler.net_turns_ema = torch.tensor(0.0)
    scheduler.net_rotation_rate = torch.tensor(0.0)
    scheduler.reset_checks_since_increase = 0
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        step_dt=0.05,
        command=SimpleNamespace(net_rotation_turns=torch.zeros(2)),
        leap_adr_episode_lengths=torch.full((2,), 2000),  # sampled full horizon = 100 s
        event_manager=_MissingEventManager(),
    )

    scheduler(env, torch.tensor([0, 1]), command_name="goal_pose")
    assert scheduler.increment == 1  # 无 competence 也只允许 0→1 bootstrap
    assert env.gm_adr_com_half_width == 0.01 / 25.0

    scheduler.reset_checks_since_increase = 960
    env.command.net_rotation_turns[:] = 10.0  # alpha=1 时 10 turns / 100 s = 0.1 turns/s
    scheduler(
        env,
        torch.tensor([0, 1]),
        command_name="goal_pose",
        ema_alpha=1.0,
        threshold_turns_per_s=0.08,
    )
    assert scheduler.increment == 2
    assert scheduler.net_turns_ema.item() == 0.0  # promotion 后新档重新证明 competence
