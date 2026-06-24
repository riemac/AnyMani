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
`PhysicsCollisionGroup` 已经在 USD stage 中 author，也不能证明 PhysX 初始化后环境可
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
from anymani.tasks.gm.mdp.events import generated_structural_collision_filter_pairs
from anymani.tasks.gm.single_asset_env_cfg import GM_SINGLE_ASSET_CONTACT_LAYOUT

TASK_ID = "AnyMani-GM-SingleAsset-v0"
r"""被验证的单资产 GM 环境；训练侧 MLP alias 复用同一个 env cfg，因此这里测 gm-owned task 即可。"""

COLLISION_GROUP_ROOT = "/World/anymani_gm_generated_structural_collision_filters"
r"""`apply_generated_structural_collision_filter(...)` 默认写入的 external collision group scope。"""

SMOKE_NUM_ENVS = 2
r"""headless smoke 的最小并行规模；两个 env 可同时验证 collection include 覆盖 cloned env roots。"""

SMOKE_STEPS = 3
r"""随机动作步数；目标是验证 reset/step 链路 finite，不把本 smoke 扩展成训练稳定性测试。"""


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
        对应的 `PhysicsCollisionGroup` 是否在 stage 中存在，并用少量随机动作证明
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

        # 再检查 USD stage 中的 external collision group scope 和每个 link-level group。
        _assert_structural_collision_groups_authored(runtime_env.scene.stage)

        # 最后做 reset/step finite 检查，确认 stage-level collision schema 没破坏真实 PhysX rollout。
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
        stats (Mapping[str, Any]): `env._gm_structural_collision_filter_stats`，记录 group 数、pair 数、
            directed edge 数与缺失 link 名称。
    """

    expected_link_names = _expected_structural_filter_link_names()  # 参与 collision group 的 semantic links
    expected_pairs = generated_structural_collision_filter_pairs(
        palm_link_name=GM_SINGLE_ASSET_CONTACT_LAYOUT.palm_link_name,
        finger_link_chains=GM_SINGLE_ASSET_CONTACT_LAYOUT.finger_link_chains,
        filter_palm_finger=True,
        filter_same_finger=True,
    )  # $\mathcal{F}$，无向 link pair 集合

    assert stats["groups"] == len(expected_link_names)
    assert stats["link_pairs"] == len(expected_pairs)
    assert stats["directed_edges"] == 2 * len(expected_pairs)
    assert tuple(stats["missing_link_names"]) == ()


def _assert_structural_collision_groups_authored(stage) -> None:
    r"""检查 USD stage 中 external `PhysicsCollisionGroup` prim 是否存在。

    Args:
        stage: 当前 IsaacSim USD stage；类型来自 `pxr.Usd.Stage`，为避免纯 Python 解析期依赖不写静态类型。
    """

    root_prim = stage.GetPrimAtPath(COLLISION_GROUP_ROOT)
    assert root_prim.IsValid(), f"missing collision group root prim: {COLLISION_GROUP_ROOT}"

    for link_name in _expected_structural_filter_link_names():
        group_path = f"{COLLISION_GROUP_ROOT}/{link_name}"  # 每个 semantic link 对应一个 external group
        group_prim = stage.GetPrimAtPath(group_path)
        assert group_prim.IsValid(), f"missing collision group prim: {group_path}"
        assert group_prim.GetTypeName() == "PhysicsCollisionGroup"


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
