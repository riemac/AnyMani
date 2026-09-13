r"""FlashSAC训练的场景装配：原任务、冻结N040和独立SAC方法身份。

场景的成员顺序与环境数在Isaac配置导入时固定，所以prepare_scene_route先于AppLauncher。
create_environment在应用启动后执行，使用同一256资产和原20Hz/120Hz控制合同。
训练结果来自逐回合原始记录；本模块不创建额外的策略评价或回放场景。
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import yaml

from .config import FlashSACConfig


def prepare_scene_route(cohort_lock: Path, num_envs: int) -> dict[str, Any]:
    r"""在Isaac导入前冻结成员级cohort和每资产相同副本数。

    Args:
        cohort_lock: schema1.2 canonical-final成员清单，不按策略表现选成员。
        num_envs: N=A*R个并行环境；同一资产的R个副本共享静态形态。
    """
    path = cohort_lock.expanduser().resolve(strict=True)  # 明确实际清单来源。
    document = yaml.safe_load(path.read_text(encoding="utf-8"))  # 仅读取资产metadata。
    if not isinstance(document, dict) or document.get("schema_version") != "1.2.0":
        raise ValueError("FlashSAC requires a schema1.2 canonical cohort lock")
    members = document.get("members")  # 顺序直接定义本次训练的资产轴。
    if not isinstance(members, list) or not members:
        raise ValueError("FlashSAC cohort must have a nonempty members list")
    if num_envs < len(members) or num_envs % len(members):
        raise ValueError("num_envs must contain equal replicas of every selected asset")
    os.environ["ANYMANI_HETERO_COHORT_LOCK"] = str(path)  # 静态scene读取的明确成员清单。
    os.environ["ANYMANI_HETERO_NUM_ENVS"] = str(num_envs)  # 与稍后实际cfg.scene.num_envs一致。
    os.environ.pop("ANYMANI_HETERO_ASSET_ROWS", None)  # member-level路由不同时使用legacy行选择。
    return document  # 启动方据此核对资产总数与记录分族信息。


def create_environment(
    cohort_lock: Path,
    run_dir: Path,
    *,
    config: FlashSACConfig,
    device: str,
    allow_tf32: bool = True,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    r"""创建原指数姿态任务及SAC终止前观测桥接，返回完整方法身份。

    返回(bridge, frozen_geometry_provider, asset_binding, method_identity)。
    Actor只读合法本体与TIP触觉，Critic可以读原任务特权字段；学习器不在此创建。
    120秒为训练有限时域，首30秒统计由任务原有snapshot记录，失败和timeout均结束bootstrap。
    """
    import gymnasium as gym
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import EventTermCfg, TerminationTermCfg
    from isaaclab.utils.io import dump_yaml

    # vec-env的共享transport依赖rl_games接口；先固定backend，仅复用transport而非PPO学习器。
    from anymani.distill.rl.rl_games_backend import prefer_local_rl_games

    backend = prefer_local_rl_games(strict=True)  # 不让安装包版本静默替换项目锁定接口。
    import anymani.tasks.hetero  # noqa: F401  # 注册任务，重型导入发生在AppLauncher之后。
    from anymani.distill.rl.runtime.palm_rotation_geometry import build_palm_rotation_bf16_geometry_provider
    from anymani.distill.rl.runtime.palm_rotation_vecenv import PalmRotationRlGamesVecEnv
    from anymani.tasks.hetero.config.generated.palm_rotation_mvp_env_cfg import (
        GOOD_PREGRASP_RESET_CFG,
        GeneratedPalmRotationMvpEnvCfg,
    )
    from anymani.tasks.hetero.config.generated.scene import ASSET_BINDING
    from anymani.tasks.hetero.mdp.adr import HeterogeneousAdrCfg, ObjectPositionAdrCfg
    from anymani.tasks.hetero.mdp.episode_horizon import planned_time_out, reset_episode_horizon
    from anymani.tasks.hetero.mdp.orientation_goal import OrientationGoalCfg, configure_orientation_goal

    from .environment import FlashSACEnvironment
    from .identity import build_method_identity

    # 所有参数均为本次实际执行值；policy/physics/episode时间彼此分开记录。
    task_id = "AnyMani-Hetero-Generated-PalmRotation-MVP-RLGames-v0"  # 共用MDP别名不定义学习算法。
    cfg = GeneratedPalmRotationMvpEnvCfg()  # typed cfg保留contact/pregrasp frozen dataclasses。
    cfg.scene.num_envs, cfg.sim.device, cfg.seed = config.num_envs, device, config.seed  # 本次执行轴与随机种子。
    cfg.log_dir = str(run_dir)  # 原任务诊断与SAC记录共用run目录。
    for name in ("jnt_current", "jnt_history", "owner_contact"):
        getattr(cfg.observations.policy, name).params["tip_only"] = True  # 三条Actor接触入口同时受约束。
    cfg.episode_length_s = 120.0  # 训练回合最大2400个20Hz步。
    cfg.commands.goal_pose.horizon_s = 120.0  # 任务统计使用同一计划时域。
    cfg.events.episode_horizon = EventTermCfg(
        func=reset_episode_horizon, mode="reset",  # 新回合重新声明有限时域。
        params={"minimum_seconds": 120.0, "maximum_seconds": 120.0},  # 训练时长固定，不随机缩短。
    )
    cfg.terminations.time_out = TerminationTermCfg(func=planned_time_out, time_out=True)  # 保留truncated事实。
    cfg.rewards.rotation_progress.params["clip_rad_per_step"] = .04  # 只截奖励进展，原始净圈不截断。
    cfg.rewards.rotation_progress.weight = 20.0  # reward/rad，与PPO指数核参照一致。
    release = cfg.curriculum.reward_release.params  # 课程状态可记录，但实际reward释放下限为1。
    release.update(release_start_turns=0., release_end_turns=2., release_floor=1., reference_seconds=30.)
    orientation = OrientationGoalCfg(kernel="exponential")  # 单步1/(exp(4*theta)+0.1)，合格事件+250。
    adr = HeterogeneousAdrCfg(object_position=ObjectPositionAdrCfg(enabled=True))  # 原逐环境位置ADR。
    configure_orientation_goal(cfg, orientation, training=True, adr=adr)  # 失败-20，非TIP惩罚/关节初姿项为0。
    if len(ASSET_BINDING.source_member_keys) != config.asset_count:
        raise ValueError("resolved cohort size disagrees with FlashSAC config")

    # 冻结几何只对encoder执行BF16；学习器权重仍FP32，TF32是显式的内部乘法选项。
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32  # 写入方法运行合同。
    torch.backends.cudnn.allow_tf32 = allow_tf32  # TCN与矩阵乘使用相同精度声明。
    run_dir.mkdir(parents=True, exist_ok=True)  # 日志只落本次run目录。
    env = gym.make(task_id, cfg=cfg)  # 按已经冻结的A*R路由创建实际仿真环境。
    bridge = None  # 构造失败时也能准确关闭已经创建的资源。
    try:
        raw = env.unwrapped  # 精确任务运行时，而非Gym包装器的通用接口。
        if not isinstance(raw, ManagerBasedRLEnv):
            raise TypeError("FlashSAC comparison requires the declared manager-based task")
        if not math.isclose(float(raw.step_dt), .05, abs_tol=1e-12):
            raise ValueError("FlashSAC comparison requires 20Hz policy steps")
        if not math.isclose(float(raw.physics_dt), 1 / 120, abs_tol=1e-12):
            raise ValueError("FlashSAC comparison requires 120Hz physics")
        provider = build_palm_rotation_bf16_geometry_provider(ASSET_BINDING, device=device)  # 相同N040。
        indices = torch.tensor(ASSET_BINDING.asset_index_by_env(config.num_envs), dtype=torch.long, device=device)
        transport = PalmRotationRlGamesVecEnv(
            env, geometry_provider=provider, prototype_index=indices,  # 索引只用于证据与静态几何查找。
            rl_device=device, clip_observations=100., clip_actions=1.,  # 与原任务相同的数值/动作边界。
        )
        bridge = FlashSACEnvironment(transport)  # 同时返回post-reset控制状态与pre-reset回放终点。
        task_contract = {
            "object": "DexCube", "object_scale": 1.1,  # 原固定物体与尺寸。
            "policy_dt_s": float(raw.step_dt), "physics_dt_s": float(raw.physics_dt),
            "episode_seconds": 120., "finite_horizon": True,  # timeout在SAC目标中不bootstrap。
            "action_authority_rad_per_policy_step": 1 / 24, "action_mode": "accumulate-target-and-clip-limits",
            "actor_contact": "tip-only-binary", "pregrasp_rank": 0, "pregrasp_strict": True,
            "drop_distance_m": .07, "axis_limit_degrees": 45.,  # 原物理终止，不因算法改变。
            "rotation_axis_h": [0., 0., 1.], "subgoal_degrees": 30., "goal_reference": "previous-goal",
            "orientation_goal": orientation.to_dict(), "goal_advance": "angle-only",
            "rotation_progress_weight": 20., "rotation_progress_clip_rad_per_step": .04,
            "reward_release": {"floor": 1., "start_turns": 0., "end_turns": 2., "reference_seconds": 30.},
            "adr": {"object_position": vars(adr.object_position)},  # 本次实际训练配置，不硬写评价设置。
        }
        run_contract = {
            "task_id": task_id, "seed": config.seed, "num_envs": config.num_envs,  # 实际实例化参数。
            "asset_count": config.asset_count, "cohort_id": ASSET_BINDING.cohort_id,
            "source_member_keys": list(ASSET_BINDING.source_member_keys),  # 明确成员顺序。
            "device": device, "allow_tf32": allow_tf32, "transport_backend_commit": backend.git_commit,
            "assessment": "recorded-training-artifacts-only", "bootstrap_timeouts": False,
        }
        identity = build_method_identity(
            config=config, provider_identity=provider.identity, cohort_lock=cohort_lock,
            pregrasp=GOOD_PREGRASP_RESET_CFG, task_contract=task_contract, run_contract=run_contract,
        )  # SAC自己的源码与科学合同身份，不继承PPO优化器/GAE/PopArt描述。
        dump_yaml(str(run_dir / "params/env.yaml"), cfg)  # 真实解析后的环境配置。
        return bridge, provider, ASSET_BINDING, identity  # 学习器由外层训练入口显式组装。
    except BaseException:
        if bridge is not None:
            bridge.close()  # 恢复本实例reset回调后关闭共享transport。
        else:
            env.close()  # provider或transport尚未构造完成时直接释放场景。
        raise  # 原始错误交由运行证据层记录。
