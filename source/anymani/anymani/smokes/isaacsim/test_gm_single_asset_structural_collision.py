r"""IsaacSim smoke for generated-hand structural collision filtering.

本文件是运行时 smoke，不属于默认 `pytest` contract suite。它必须通过显式路径运行，
因为模块导入阶段会启动 `AppLauncher(headless=True)`：

```bash
cd /home/hac/isaac/AnyMani
source /home/hac/isaac/env_isaaclab/bin/activate
timeout --kill-after=20s 240s /home/hac/isaac/IsaacLab/isaaclab.sh -p -m pytest \
  source/anymani/anymani/smokes/isaacsim/test_gm_single_asset_structural_collision.py -q -s
```

该 smoke 针对 2026-06-24 单资产标定台消融暴露出的“伪测试”风险：纯 contract test
只能证明 link-pair 集合 $\mathcal{F}$ 的组合逻辑正确，不能证明
`PhysicsFilteredPairsAPI` 已经在 USD stage 中 author，也不能证明 PhysX 初始化后环境可
reset / step。这里显式启动 IsaacSim，验证 `prestartup` event 的 stage 级副作用。
"""

from __future__ import annotations

# ruff: noqa: I001
# IsaacLab smoke 必须先创建 `AppLauncher`，再导入 `omni` / task runtime 相关模块；
# 因此本文件有意不服从普通 isort 的“所有 import 必须在文件顶部”排序模型。

from collections.abc import Mapping
from typing import Any

from isaaclab.app import AppLauncher

# IsaacLab 官方测试约定：所有 `omni` / `pxr` / task cfg import 都必须在 AppLauncher 之后。
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
import pytest
import torch
from isaaclab_tasks.utils import parse_env_cfg

import anymani.tasks.gm  # noqa: F401  # 注册 `AnyMani-GM-SingleAsset-v0`，不依赖父包自动发现
from anymani.tasks.gm.config.single_asset.single_asset_env_cfg import GM_SINGLE_ASSET_CONTACT_LAYOUT
from anymani.tasks.gm.mdp.events import generated_structural_collision_filter_pairs

TASK_ID = "AnyMani-GM-SingleAsset-v0"
r"""被验证的单资产 GM 环境；训练侧 MLP alias 复用同一个 env cfg，因此这里测 gm-owned task 即可。"""

SMOKE_NUM_ENVS = 2
r"""headless smoke 的最小并行规模；两个 env 可同时验证 pairwise authoring 覆盖 cloned env roots。"""

SMOKE_STEPS = 64
r"""随机动作步数；约半秒仿真时间，用于覆盖连续 PhysX step，但不扩展成训练稳定性测试。"""


def teardown_module() -> None:
    r"""关闭 IsaacSim app，避免显式 smoke 结束后遗留 Kit 进程。"""

    simulation_app.close()


@pytest.mark.isaacsim
def test_single_asset_structural_collision_filter_authors_usd_and_steps() -> None:
    r"""验证 generated structural collision filter 真的进入 USD stage 并支持 reset/step。

    科研语义：
        generated hand 的结构过滤集合为
        $$
        \mathcal{F}
          = \{(\mathrm{palm},l)\mid l\in\cup_f F_f\}
            \cup \bigcup_f \{(a,b)\mid a,b\in F_f,\ a\ne b\}.
        $$
        纯 contract test 只检查 $\mathcal{F}$；本 smoke 进一步检查每个参与 link
        对应的 `PhysicsFilteredPairsAPI` 是否在 stage 中存在，并用少量随机动作证明
        PhysX 初始化、reset、step 没有被该过滤 schema 破坏。
    """

    # 解析 env cfg 时把 2048-env 训练规模压到 smoke 规模，避免把测试变成压力测试。
    env_cfg = parse_env_cfg(TASK_ID, device="cuda:0", num_envs=SMOKE_NUM_ENVS)
    env = None
    try:
        # `gym.make(...)` 会触发 scene creation、prestartup event、sim.reset 与 manager 初始化。
        env = gym.make(TASK_ID, cfg=env_cfg)
        runtime_env = env.unwrapped  # ManagerBasedRLEnv；prestartup stats 挂在该对象上
        runtime_env.sim._app_control_on_stop_handle = None  # smoke 退出时不让 Kit timeline 接管关闭流程

        # 先检查 prestartup event 的 debug stats；这是 collision filter 是否执行的最低信号。
        stats = getattr(runtime_env, "_gm_structural_collision_filter_stats", None)
        assert isinstance(stats, Mapping), "missing structural collision filter stats; prestartup event likely did not run"
        _assert_structural_collision_filter_stats(stats)

        # 再检查 USD stage 中每个结构过滤 link pair 的 `physics:filteredPairs` 关系。
        _assert_structural_filtered_pairs_authored(runtime_env)

        # 最后做短 rollout finite 检查，确认 stage-level collision schema 没在连续 PhysX step 中失效。
        obs, _ = env.reset()
        _assert_finite_tree("reset_obs", obs)
        for step_id in range(SMOKE_STEPS):
            actions = 2.0 * torch.rand(env.action_space.shape, device=runtime_env.device) - 1.0  # $a\in[-1,1]^{B\times A}$
            obs, reward, terminated, truncated, _ = env.step(actions)
            _assert_finite_tree(f"step_{step_id}_obs", obs)
            assert torch.isfinite(reward).all(), f"non-finite reward at smoke step {step_id}"
            assert terminated.shape == truncated.shape == reward.shape, "done/reward batch shapes must agree"
    finally:
        if env is not None:
            env.close()


def _assert_structural_collision_filter_stats(stats: Mapping[str, Any]) -> None:
    r"""检查 `prestartup` 写入的结构碰撞过滤统计量。

    Args:
        stats (Mapping[str, Any]): `env._gm_structural_collision_filter_stats`，记录 API 类型、pair 数、
            directed pair edge 数与缺失 link 名称。
    """

    expected_link_names = _expected_structural_filter_link_names()  # 参与 pairwise filter 的 semantic links
    expected_pairs = generated_structural_collision_filter_pairs(
        palm_link_name=GM_SINGLE_ASSET_CONTACT_LAYOUT.palm_link_name,
        finger_link_chains=GM_SINGLE_ASSET_CONTACT_LAYOUT.finger_link_chains,
        filter_palm_finger=True,
        filter_same_finger=True,
    )  # $\mathcal{F}$，无向 link pair 集合

    assert stats["api"] == "FilteredPairsAPI"
    assert stats["link_pairs"] == len(expected_pairs)
    assert stats["directed_edges"] == 2 * len(expected_pairs) * SMOKE_NUM_ENVS
    assert tuple(stats["missing_link_names"]) == ()
    assert len(expected_link_names) > 0  # 防止未来 layout 解析失败但 stats 恰好为空


def _assert_structural_filtered_pairs_authored(runtime_env) -> None:
    r"""检查 USD stage 中 link-level `PhysicsFilteredPairsAPI` relationship 是否存在。

    Args:
        runtime_env: 当前 unwrapped `ManagerBasedRLEnv`；需要读取 stage 与 cloned env prim paths。
    """

    stage = runtime_env.scene.stage  # 当前 USD stage；prestartup event 已在 sim.reset 前完成 authoring
    old_group_root = "/World/anymani_gm_generated_structural_collision_filters"  # 旧 CollisionGroup 实现的 scope
    assert not stage.GetPrimAtPath(old_group_root).IsValid(), "old CollisionGroup filter root should not be authored"

    expected_pairs = generated_structural_collision_filter_pairs(
        palm_link_name=GM_SINGLE_ASSET_CONTACT_LAYOUT.palm_link_name,
        finger_link_chains=GM_SINGLE_ASSET_CONTACT_LAYOUT.finger_link_chains,
        filter_palm_finger=True,
        filter_same_finger=True,
    )  # $\mathcal{F}$，无向 link pair 集合

    for env_prim_path in runtime_env.scene.env_prim_paths:
        robot_path = f"{env_prim_path}/Robot"  # single-asset scene 中 robot prim path 的 resolved 形式
        for link_a, link_b in expected_pairs:
            _assert_filtered_pair_target(stage, f"{robot_path}/{link_a}", f"{robot_path}/{link_b}")
            _assert_filtered_pair_target(stage, f"{robot_path}/{link_b}", f"{robot_path}/{link_a}")


def _assert_filtered_pair_target(stage, source_link_path: str, target_link_path: str) -> None:
    r"""检查一个 directed `source -> target` filtered pair 是否存在。

    Args:
        stage: 当前 IsaacSim USD stage。
        source_link_path (str): 应持有 `PhysicsFilteredPairsAPI` 的 link prim path。
        target_link_path (str): 应出现在 `physics:filteredPairs` relationship target 中的 link prim path。
    """

    source_prim = stage.GetPrimAtPath(source_link_path)
    target_prim = stage.GetPrimAtPath(target_link_path)
    assert source_prim.IsValid(), f"missing source link prim: {source_link_path}"
    assert target_prim.IsValid(), f"missing target link prim: {target_link_path}"
    assert "PhysicsFilteredPairsAPI" in source_prim.GetAppliedSchemas(), (
        f"missing PhysicsFilteredPairsAPI on {source_link_path}; applied={source_prim.GetAppliedSchemas()}"
    )

    filtered_pairs_rel = source_prim.GetRelationship("physics:filteredPairs")
    assert filtered_pairs_rel, f"missing physics:filteredPairs relationship on {source_link_path}"
    assert target_prim.GetPath() in filtered_pairs_rel.GetTargets(), (
        f"missing filtered pair target {target_link_path} on {source_link_path}; "
        f"targets={filtered_pairs_rel.GetTargets()}"
    )


def _expected_structural_filter_link_names() -> tuple[str, ...]:
    r"""返回当前 single-asset 结构过滤应覆盖的 link 名集合。"""

    link_names = {GM_SINGLE_ASSET_CONTACT_LAYOUT.palm_link_name}  # palm 是 palm-finger pair 的固定端点
    for finger_link_chain in GM_SINGLE_ASSET_CONTACT_LAYOUT.finger_link_chains:
        link_names.update(finger_link_chain)  # 所有 finger links 都参与 palm-finger 或 same-finger filter
    return tuple(sorted(link_names))


def _assert_finite_tree(name: str, value: Any) -> None:
    r"""递归检查 observation tree 中所有 tensor 是否 finite。

    Args:
        name (str): 当前树节点名称，用于失败消息定位。
        value (Any): observation 子树；通常是 `dict[str, torch.Tensor]` 或 tensor。
    """

    if isinstance(value, Mapping):
        for child_name, child_value in value.items():
            _assert_finite_tree(f"{name}.{child_name}", child_value)
        return

    if torch.is_tensor(value):
        assert torch.isfinite(value).all(), f"non-finite tensor in {name}"
