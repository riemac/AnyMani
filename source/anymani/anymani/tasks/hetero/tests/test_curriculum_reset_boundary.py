r"""执行生产reset adapter的纯Torch方法，证伪空回合/非终止reset污染课程。

只提取__call__方法以隔离Isaac ManagerTermBase导入；方法体没有替代实现，真实Manager调用顺序另由短训练验证。
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import torch

from anymani.tasks.hetero.mdp.curriculum_state import HETERO_REWARD_RELEASE_STATE_ATTR, HeterogeneousRewardReleaseState
from anymani.tasks.hetero.mdp.episode_horizon import EPISODE_HORIZON_STEPS_ATTR, reference_horizon_turns


def test_only_nonempty_terminal_episodes_update_restored_curriculum() -> None:
    r"""冷启动空回合、stale done和手动非终止reset都不改EMA；真实terminal只更新所属资产。"""

    path = Path(__file__).resolve().parents[1] / "mdp" / "curriculums.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RewardReleaseByAssetMedianCell"
    )
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__call__")
    namespace = {
        "torch": torch,
        "HETERO_REWARD_RELEASE_STATE_ATTR": HETERO_REWARD_RELEASE_STATE_ATTR,
        "HeterogeneousRewardReleaseState": HeterogeneousRewardReleaseState,
        "EPISODE_HORIZON_STEPS_ATTR": EPISODE_HORIZON_STEPS_ATTR,
        "reference_horizon_turns": reference_horizon_turns,
        "get_rotation_command": lambda env, _: env.command,
    }
    exec(compile(ast.Module(body=[tree.body[1], method], type_ignores=[]), str(path), "exec"), namespace)
    state = HeterogeneousRewardReleaseState(
        dataset_rows_by_asset=(0, 1), cell_ids_by_asset=(0, 0), asset_index_by_env=(0, 1, 0, 1), device="cpu"
    )
    state.asset_net_turns_ema.fill_(2.0)
    state.asset_episode_updates.fill_(10)
    env = SimpleNamespace(
        num_envs=4,
        device="cpu",
        step_dt=0.05,
        max_episode_length=2400,
        episode_length_buf=torch.zeros(4, dtype=torch.long),
        termination_manager=SimpleNamespace(dones=torch.tensor((True, False, False, False))),
        command=SimpleNamespace(positive_net_rotation_turns=torch.tensor((100.0, 4.0, 100.0, 100.0))),
    )
    setattr(env, HETERO_REWARD_RELEASE_STATE_ATTR, state)
    call = namespace["__call__"]
    kwargs = dict(
        command_name="goal_pose",
        dataset_rows_by_asset=(0, 1),
        cell_ids_by_asset=(0, 0),
        asset_index_by_env=(0, 1, 0, 1),
        release_floor=1.0,
    )
    call(None, env, torch.arange(4), **kwargs)
    assert torch.equal(state.asset_episode_updates, torch.tensor((10, 10))), (
        "cold reset must not fabricate completed episodes"
    )
    assert torch.equal(state.asset_net_turns_ema, torch.tensor((2.0, 2.0)))

    env.episode_length_buf = torch.tensor((0, 10, 5, 0))
    env.termination_manager.dones = torch.tensor((True, True, False, False))
    call(None, env, torch.arange(4), **kwargs)
    assert torch.equal(state.asset_episode_updates, torch.tensor((10, 11)))
    torch.testing.assert_close(state.asset_net_turns_ema, torch.tensor((2.0, 2.1)))
    assert torch.equal(state.env_lambda, torch.ones(4))
