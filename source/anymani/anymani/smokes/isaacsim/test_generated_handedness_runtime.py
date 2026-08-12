r"""IsaacSim smoke for generated left/right same-$q$ runtime handedness.

本文件是显式 runtime smoke，不属于默认 pytest contract suite。模块导入会启动
`AppLauncher(headless=True)`，运行命令为：

```bash
cd /home/hac/isaac/AnyMani
source /home/hac/isaac/env_isaaclab/bin/activate
timeout --kill-after=10s 60s /home/hac/isaac/IsaacLab/isaaclab.sh -p -m pytest \
  source/anymani/anymani/smokes/isaacsim/test_generated_handedness_runtime.py -q -s
```

纯 Python 合同已经证明导出 sidecar 的 POE/FK 满足严格镜像。本 smoke 进一步验证
URDF importer、PhysX articulation、implicit actuator 与 joint-target buffer 没有引入
第二套 handedness 符号。一个 batched `Articulation` 持有两个 prototype：

- env 0：严格证书 generated left；
- env 1：同 morphology generated right。

`MultiAssetSpawnerCfg(random_choice=False)` 按 prototype 列表 round-robin 路由，因此以上
对应关系是确定的。左右手接收完全相同的正目标增量：

$$
q^{target}_{L,i}=q^{target}_{R,i}=+0.25\ \mathrm{rad}.
$$

令 $S=\operatorname{diag}(-1,1,1)$。PhysX importer 可以为左右 body 选择不同但固定的
actor-local gauge，因此 actor frame 的绝对矩阵不必等于 URDF link frame 的绝对矩阵。
物理不变量是 body 原点和从各自 home actor frame 出发的空间旋转增量：

$$
\Delta R(t)=R(t)R(0)^\top.
$$

从 world position 中减去各自 cloned-env origin 后，整个 rollout 必须满足：

$$
q_L(t)=q_R(t),\qquad
\mathbf p_L(t)=S\mathbf p_R(t),\qquad
\Delta R_L(t)=S\Delta R_R(t)S.
$$

这样可区分两类错误：若 action/joint mapping 错，首先破坏 $q_L=q_R$；若 axis、origin、
inertia 或 importer lowering 错，则 joint state 仍可相同，但 body pose 不再严格镜像。
"""

from __future__ import annotations

# ruff: noqa: I001
# IsaacLab runtime 要求先创建 AppLauncher，再 import sim、torch 与 AnyMani runtime 配置。

from pathlib import Path
import traceback

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import pytest
import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass

from anymani.assets.bank import HandBankCfg
from anymani.assets.generator.hand_generator import HandGenerator, HandGeneratorCfg
from anymani.robots.hand_spawn import HandFrameCfg, HandSpawnAdapter, HandSpawnCfg, HandUrdfSpawnCfg
from anymani.tasks.inhand.config.generated_right_t4_i4_m4_r4.generated_right_t4_i4_m4_r4_adr_env_cfg import (
    GENERATED_OFFICIAL_SLOT_JOINT_ORDER,
)

SMOKE_NUM_ENVS = 2
r"""两个 cloned env 分别承载 left/right prototype；数量变化会破坏确定性 routing 语义。"""

SMOKE_ENV_SPACING = 0.75
r"""cloned env 平移间距，单位 m；body 镜像比较前会减去各自 `env_origins`。"""

SMOKE_STEPS = 60
r"""以 $120\,\mathrm{Hz}$ 推进 0.5 秒，足以让 $3\,\mathrm{N\,m/rad}$ PD drive 产生明显非零运动。"""

ZERO_SETTLE_STEPS = 10
r"""zero target 预热步数；隔离 URDF/PhysX 首帧初始化瞬态后再定义 runtime home。"""

POSITIVE_TARGET_RAD = 0.25
r"""每个 policy slot 的共同正位置目标，单位 rad；位于 canonical LEAP 全部 limits 内。"""

JOINT_ATOL = 5.0e-5
r"""左右实际 joint trajectory 的绝对容差，单位 rad；只吸收 batched PhysX 数值舍入。"""

BODY_POSITION_ATOL = 2.0e-4
r"""全部 body actor-origin 镜像位置容差，单位 m。"""

BODY_ROTATION_ATOL = 3.0e-4
r"""全部 body gauge-invariant 空间旋转增量的镜像逐元素容差。"""


def teardown_module() -> None:
    r"""关闭 IsaacSim app，避免显式 smoke 结束后遗留 Kit 进程。"""

    simulation_app.close()


def _full_leap_connectivity_pool() -> dict[str, dict[str, list[str]]]:
    r"""把 pre-made 空间收缩为唯一 full-chain LEAP 左右 pair。"""

    return {
        "single_palm_leap": {
            "thumb": ["leap_thumb_full"],
            "index": ["leap_non_thumb_full"],
            "middle": ["leap_non_thumb_full"],
            "ring": ["leap_non_thumb_full"],
        }
    }


def _generate_runtime_pair(tmp_path: Path) -> tuple[Path, Path]:
    r"""通过正式 generator 生成本次 runtime smoke 的自包含左右 bundle。

    Args:
        tmp_path (Path): pytest 临时目录；新 run 不污染版本库或正式 generated 树。

    Returns:
        tuple[Path, Path]: 严格按 ``(left_bundle, right_bundle)`` 排列的绝对路径。
    """

    cfg = HandGeneratorCfg(
        mode="made",  # 只生成 canonical morphology，不引入随机 post-mutate
        artifact_level="bundle",  # importer 需要真实 URDF、sidecar 与 materialized meshes
        output_dir=tmp_path / "generated_handedness_runtime",
        handedness="all",  # 同一 run 内同时导出 left/right，避免配置漂移
        hand_presets=["single_palm_leap"],
        connectivity_presets=_full_leap_connectivity_pool(),
        mixed=False,
        missing=False,
        Validate=None,  # 本 smoke 验证 importer/PhysX；机械 validator 已由普通测试覆盖
        premade_parallel=False,  # 两个样本顺序生成，使失败诊断确定
    )
    results = list(HandGenerator(cfg).generate_batch())
    assert len(results) == 2
    bundle_by_side = {
        result.hand_cfg.handedness: result.urdf_path.parent.resolve()
        for result in results
        if result.urdf_path is not None
    }  # sidecar/URDF 物理 handedness -> self-contained bundle root
    assert set(bundle_by_side) == {"left", "right"}
    return bundle_by_side["left"], bundle_by_side["right"]  # 顺序决定 env 0/1 prototype routing


def _make_scene_cfg(left_bundle: Path, right_bundle: Path) -> InteractiveSceneCfg:
    r"""构造两个 same-schema prototype 的最小 batched articulation scene。

    Args:
        left_bundle (Path): env 0 使用的严格 generated left bundle。
        right_bundle (Path): env 1 使用的 canonical generated right bundle。

    Returns:
        InteractiveSceneCfg: 无物体、无地面、无接触扰动的固定基座 hand scene。
    """

    spawn_cfg = HandSpawnCfg(
        bank=HandBankCfg(
            source_mode="pre_made",
            selection_mode="explicit",
            containers=(str(left_bundle), str(right_bundle)),  # RR 顺序固定为 left/right
            validate_mesh_relpaths=True,
            parse_visual_rgba=False,  # headless 动力学 smoke 不需要恢复 debug 材质
            allow_legacy_left_handedness=False,  # left 必须凭本轮严格证书通过安全门
        ),
        frame=HandFrameCfg(
            semantic_R_ha=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            semantic_p_ha=(0.0, 0.0, 0.0),
            anchor_R_eh=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            anchor_p_eh=(0.0, 0.0, 0.5),  # palm/hand origin 的 env-frame 高度，单位 m
        ),
        urdf=HandUrdfSpawnCfg(
            force_usd_conversion=True,  # 临时 URDF 每次都重新转换，排除旧 cache 污染
            merge_fixed_joints=False,  # 保留全部 link bodies，才能逐 body 验证 importer pose
            self_collision=False,  # 隔离 joint-axis/FK；接触会引入与命题无关的碰撞脉冲
            activate_contact_sensors=False,
        ),
        asset_routing="round_robin",  # env $i$ 使用 prototype $i\bmod2$
        restore_visual_materials=False,
        validate_same_schema=True,  # handedness-invariant topology + exact joint order gate
    )
    robot_cfg = HandSpawnAdapter(spawn_cfg).build_articulation_cfg(prim_path="{ENV_REGEX_NS}/Robot")

    @configclass
    class GeneratedHandednessSceneCfg(InteractiveSceneCfg):
        r"""仅含一批 fixed-base generated hands 的最小 runtime scene。"""

        robot: ArticulationCfg = robot_cfg
        r"""env 0/1 共享一个 articulation view，但底层 prototype 分别为 left/right。"""

    return GeneratedHandednessSceneCfg(
        num_envs=SMOKE_NUM_ENVS,
        env_spacing=SMOKE_ENV_SPACING,
        replicate_physics=False,  # heterogeneous prototypes 禁止复制首 env 的 physics schema
        clone_in_fabric=False,
    )


@pytest.mark.isaacsim
def test_generated_left_right_share_joint_targets_and_mirrored_physx_trajectory(tmp_path: Path) -> None:
    r"""验证真实 PhysX 中同一正 target 产生 same-$q$ 且严格镜像的 body trajectory。"""

    try:
        _run_generated_handedness_smoke(tmp_path)
    except BaseException:
        # Kit fast shutdown 可能早于 pytest 刷出 traceback；在 teardown 关闭 app 前显式保留失败证据。
        traceback.print_exc()
        raise


def _run_generated_handedness_smoke(tmp_path: Path) -> None:
    r"""执行 generated handedness runtime 主体，异常由 pytest 入口在 app 关闭前打印。"""

    left_bundle, right_bundle = _generate_runtime_pair(tmp_path)  # 运行时输入不依赖历史错误 left
    sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, render_interval=4, device="cuda:0"))
    scene = InteractiveScene(_make_scene_cfg(left_bundle, right_bundle))
    sim.reset()
    robot = scene["robot"]  # 两个 prototype 的单一 batched Articulation handle
    sim_dt = sim.get_physics_dt()

    # 写回同一 zero joint state；root state 加各自 env origin 后，两手的局部 palm frame 完全重合。
    root_state = robot.data.default_root_state.clone()  # `[2,13]`，cfg 中的 env-local anchor state
    root_state[:, :3] += scene.env_origins  # $p_{wa}=p_{we}+p_{ea}$
    zero_joint_pos = torch.zeros_like(robot.data.default_joint_pos)  # `[2,16]`，相同 $q(0)=0$
    zero_joint_vel = torch.zeros_like(robot.data.default_joint_vel)  # `[2,16]`，相同 $\dot q(0)=0$
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(zero_joint_pos, zero_joint_vel)
    scene.reset()
    # 首个 PhysX step 前显式提交 zero target，避免 URDF importer/actuator 的默认 target 成为隐式预热动作。
    robot.set_joint_position_target(zero_joint_pos)  # $q^{target}(0)=0$，两侧共享同一 generalized target
    scene.write_data_to_sim()  # 把 target buffer lower 到 PhysX drive，而不是只停留在 IsaacLab cache
    q_after_write_native = robot.data.joint_pos.clone()  # 写入后、首次 sim step 前的 importer generalized coordinates
    target_after_reset_native = robot.data.joint_pos_target.clone()  # scene.reset 后 actuator target buffer
    for _ in range(ZERO_SETTLE_STEPS):
        robot.set_joint_position_target(zero_joint_pos)  # 持续保持 $q^{target}=0$，不引入动作变量
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    # `joint_names` 是实际 PhysX native DOF order；policy ids 则复现 Manager action term 的 `_joint_ids`。
    expected_native_order = [
        joint_name
        for joint_level in range(4)
        for joint_name in (
            f"thumb_j{joint_level}",
            f"index_j{joint_level}",
            f"middle_j{joint_level}",
            f"ring_j{joint_level}",
        )
    ]  # Isaac Sim 5.1 importer 按 joint level 交织，并在每层使用 thumb/index/middle/ring
    assert robot.joint_names == expected_native_order, f"unexpected PhysX native order: {robot.joint_names}"
    policy_joint_ids, resolved_policy_names = robot.find_joints(
        list(GENERATED_OFFICIAL_SLOT_JOINT_ORDER),
        preserve_order=True,
    )  # official-slot interleaved order -> native PhysX indices，等价 action term `_joint_ids`
    assert tuple(resolved_policy_names) == GENERATED_OFFICIAL_SLOT_JOINT_ORDER
    assert policy_joint_ids == [1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11, 13, 12, 14, 15]

    # 用 find_bodies 锁住 importer 的完整 body identity/order；fixed roots/tips 不得被 merge 消失。
    body_ids, resolved_body_names = robot.find_bodies(list(robot.body_names), preserve_order=True)
    assert body_ids == list(range(robot.num_bodies))
    assert resolved_body_names == robot.body_names
    assert robot.num_bodies == 24  # LEAP: palm + 3 non-thumb×6 links + thumb×5 links

    # 同一 zero state 必须真实写入两侧；否则后续 body 比较可能把不同 q 误归因于 link geometry。
    initial_q_policy = robot.data.joint_pos[:, policy_joint_ids]
    if not torch.allclose(initial_q_policy[0], initial_q_policy[1], atol=JOINT_ATOL, rtol=0.0):
        print("generated handedness initial joint diagnostics")
        print(f"native_joint_names={robot.joint_names}")
        print(f"policy_joint_ids={policy_joint_ids}")
        print(f"q_after_write_native={q_after_write_native.detach().cpu().tolist()}")
        print(f"target_after_reset_native={target_after_reset_native.detach().cpu().tolist()}")
        print(f"q_after_zero_settle_native={robot.data.joint_pos.detach().cpu().tolist()}")
        print(f"joint_limits_native={robot.data.joint_pos_limits.detach().cpu().tolist()}")
    torch.testing.assert_close(initial_q_policy[0], initial_q_policy[1], atol=JOINT_ATOL, rtol=0.0)
    torch.testing.assert_close(initial_q_policy, torch.zeros_like(initial_q_policy), atol=JOINT_ATOL, rtol=0.0)

    # importer 可为 mirrored links 选择不同固定 actor-local gauge；保存各自 home 旋转作为增量参考。
    home_body_rotations_e = math_utils.matrix_from_quat(robot.data.body_quat_w).clone()  # `[2,B,3,3]`

    # reset 后检查 body 原点绝对镜像与 identity 旋转增量，确认 RR prototype 路由和 home geometry。
    initial_position_error, initial_rotation_error = _mirrored_body_pose_errors(
        scene,
        home_body_rotations_e=home_body_rotations_e,
    )
    if initial_position_error > BODY_POSITION_ATOL or initial_rotation_error > BODY_ROTATION_ATOL:
        _print_mirrored_body_pose_diagnostics(
            scene,
            home_body_rotations_e=home_body_rotations_e,
        )  # 在 app teardown 前保留逐 body importer 误差证据
    assert initial_position_error <= BODY_POSITION_ATOL
    assert initial_rotation_error <= BODY_ROTATION_ATOL

    # 同一正 target 直接按 policy slot 写入；不使用 left action sign map，也不依赖 native block order。
    policy_targets = torch.full(
        (SMOKE_NUM_ENVS, len(policy_joint_ids)),
        POSITIVE_TARGET_RAD,
        device=robot.device,
        dtype=robot.data.joint_pos.dtype,
    )  # `[2,16]`，所有功能动作槽均为相同正增量
    robot.set_joint_position_target(policy_targets, joint_ids=policy_joint_ids)
    torch.testing.assert_close(
        robot.data.joint_pos_target[:, policy_joint_ids],
        policy_targets,
        atol=0.0,
        rtol=0.0,
    )  # target buffer 本身不得因 handedness 二次反号

    max_joint_error = 0.0  # 整段 rollout 的 $\max_t\|q_L(t)-q_R(t)\|_\infty$
    max_body_position_error = initial_position_error
    max_body_rotation_error = initial_rotation_error
    for step_index in range(SMOKE_STEPS):
        robot.set_joint_position_target(policy_targets, joint_ids=policy_joint_ids)  # PD hold，不累积 target
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        q_policy = robot.data.joint_pos[:, policy_joint_ids]  # `[left/right,16]`，official policy slot order
        step_joint_error = float(torch.max(torch.abs(q_policy[0] - q_policy[1])).item())
        max_joint_error = max(max_joint_error, step_joint_error)
        assert step_joint_error <= JOINT_ATOL, f"same-q trajectory diverged at step {step_index}: {step_joint_error}"

        # 每 10 步读取一次全部 body pose；高频 joint 检查与较低频 body 检查共同覆盖完整轨迹。
        if step_index % 10 == 0 or step_index == SMOKE_STEPS - 1:
            position_error, rotation_error = _mirrored_body_pose_errors(
                scene,
                home_body_rotations_e=home_body_rotations_e,
            )
            max_body_position_error = max(max_body_position_error, position_error)
            max_body_rotation_error = max(max_body_rotation_error, rotation_error)
            assert position_error <= BODY_POSITION_ATOL, (
                f"mirrored body positions diverged at step {step_index}: {position_error}"
            )
            assert rotation_error <= BODY_ROTATION_ATOL, (
                f"mirrored body rotations diverged at step {step_index}: {rotation_error}"
            )

    # 防止 smoke 虽然数值对称，却因为 drive 未生效而只验证静止构型。
    final_q_policy = robot.data.joint_pos[:, policy_joint_ids]
    minimum_motion = float(torch.min(final_q_policy).item())  # 所有关节都应沿共同正方向明显移动
    assert minimum_motion > 0.02, f"positive PD target did not produce a nontrivial trajectory: min_q={minimum_motion}"

    print("generated handedness runtime summary")
    print(f"left_bundle={left_bundle}")
    print(f"right_bundle={right_bundle}")
    print(f"native_joint_names={robot.joint_names}")
    print(f"policy_joint_ids={policy_joint_ids}")
    print(f"max_joint_error_rad={max_joint_error:.9g}")
    print(f"max_body_position_error_m={max_body_position_error:.9g}")
    print(f"max_body_rotation_error={max_body_rotation_error:.9g}")
    print(f"minimum_final_positive_motion_rad={minimum_motion:.9g}")


def _mirrored_body_pose_errors(
    scene: InteractiveScene,
    *,
    home_body_rotations_e: torch.Tensor,
) -> tuple[float, float]:
    r"""计算 body 原点和 gauge-invariant 空间旋转增量的严格 YZ 镜像误差。

    Args:
        scene (InteractiveScene): 两 env generated handedness scene。
        home_body_rotations_e (torch.Tensor): 左右 importer 各自选择的 home actor-frame
            旋转，形状 ``[2,B,3,3]``；只用于消除固定 local gauge。

    Returns:
        tuple[float, float]: 最大 body actor-origin 位置误差（m）与空间旋转增量逐元素误差。
    """

    robot = scene["robot"]  # env 0=left，env 1=right
    positions_e = robot.data.body_pos_w - scene.env_origins[:, None, :]  # `[2,B,3]`，world -> cloned env frame
    rotations_e = math_utils.matrix_from_quat(robot.data.body_quat_w)  # `[2,B,3,3]`，runtime buffer 为 wxyz
    signs = torch.tensor((-1.0, 1.0, 1.0), device=robot.device, dtype=positions_e.dtype)  # $S$ 对角
    reflection = torch.diag(signs)  # $S=\operatorname{diag}(-1,1,1)$

    expected_left_positions = positions_e[1] * signs  # $\mathbf p_L=S\mathbf p_R$
    rotation_deltas_e = rotations_e @ home_body_rotations_e.transpose(-1, -2)  # $\Delta R=R(t)R(0)^\top$
    expected_left_rotations = reflection @ rotation_deltas_e[1] @ reflection  # $\Delta R_L=S\Delta R_RS$
    position_error = float(torch.max(torch.abs(positions_e[0] - expected_left_positions)).item())
    rotation_error = float(torch.max(torch.abs(rotation_deltas_e[0] - expected_left_rotations)).item())
    return position_error, rotation_error


def _print_mirrored_body_pose_diagnostics(
    scene: InteractiveScene,
    *,
    home_body_rotations_e: torch.Tensor,
) -> None:
    r"""打印逐 body 的位置与 gauge-invariant 空间旋转增量镜像误差。

    Args:
        scene (InteractiveScene): 两 env generated handedness scene。
        home_body_rotations_e (torch.Tensor): 左右各自 home actor-frame 旋转，形状 ``[2,B,3,3]``。
    """

    robot = scene["robot"]  # env 0=left，env 1=right
    positions_e = robot.data.body_pos_w - scene.env_origins[:, None, :]  # world -> env-local actor origins
    rotations_e = math_utils.matrix_from_quat(robot.data.body_quat_w)  # `[2,B,3,3]`，body actor frames
    signs = torch.tensor((-1.0, 1.0, 1.0), device=robot.device, dtype=positions_e.dtype)  # $S$ 对角
    reflection = torch.diag(signs)  # YZ reflection $S$
    position_errors = torch.max(torch.abs(positions_e[0] - positions_e[1] * signs), dim=-1).values
    rotation_deltas_e = rotations_e @ home_body_rotations_e.transpose(-1, -2)  # 各侧相对 home 的空间旋转
    expected_left_rotations = reflection @ rotation_deltas_e[1] @ reflection  # $S\Delta R_RS$
    rotation_errors = torch.amax(torch.abs(rotation_deltas_e[0] - expected_left_rotations), dim=(-2, -1))

    print("generated handedness home-pose diagnostics")
    for body_index, body_name in enumerate(robot.body_names):
        print(
            f"body={body_name} "
            f"position_error_m={float(position_errors[body_index].item()):.9g} "
            f"rotation_error={float(rotation_errors[body_index].item()):.9g}"
        )
    worst_body_index = int(torch.argmax(rotation_errors).item())  # 最大旋转偏差 link，便于核对 USD/URDF frame
    print(f"worst_rotation_body={robot.body_names[worst_body_index]}")
    print(f"left_rotation_delta={rotation_deltas_e[0, worst_body_index].detach().cpu().tolist()}")
    print(f"right_rotation_delta={rotation_deltas_e[1, worst_body_index].detach().cpu().tolist()}")
    print(f"expected_left_rotation_delta={expected_left_rotations[worst_body_index].detach().cpu().tolist()}")
