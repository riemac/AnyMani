"""Online distillation training entry (teacher joint → student SE3).

This is a project-side script (AnyRotate) that launches Isaac Sim, creates a *student*
IsaacLab env, injects teacher labels via a Gym wrapper, and trains with a custom
rl_games PPO agent that adds an imitation loss term.

Key CLI flags:
- --student-task / --teacher-task
- --teacher-checkpoint
- --align se3_to_relative_joint_delta

Notes:
- This script intentionally disables teacher observation corruption.
- Teacher policy inference is compatible with rl_games RNN checkpoints; hidden states
  are zeroed for envs that reset.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from datetime import datetime
from distutils.util import strtobool

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train student with online distillation (rl_games).")
parser.add_argument("--student-task", type=str, required=True, help="Gym task id for the student env.")
parser.add_argument("--teacher-task", type=str, required=True, help="Gym task id for the teacher env config.")
parser.add_argument("--teacher-checkpoint", type=str, required=True, help="Path to the teacher rl_games .pth checkpoint.")
parser.add_argument("--student-checkpoint", type=str, default=None, help="Optional student checkpoint to resume.")
parser.add_argument("--align", type=str, default="se3_to_relative_joint_delta", help="Alignment strategy.")
parser.add_argument("--teacher-obs-group", type=str, default="policy", help="Observation group name for teacher.")
parser.add_argument("--teacher-action-scale", type=float, default=0.1, help="Teacher relative joint delta scale (raw * scale).")
parser.add_argument("--num-envs", type=int, default=None, help="Override number of parallel envs.")
parser.add_argument("--seed", type=int, default=None, help="Seed.")
parser.add_argument(
    "--log-name",
    type=str,
    default=None,
    help=(
        "Logging sub-directory name under AnyRotate/logs/rl_games/. "
        "Defaults to agent_cfg.params.config.name."
    ),
)
parser.add_argument(
    "--track",
    type=lambda x: bool(strtobool(x)),
    default=False,
    nargs="?",
    const=True,
    help="if toggled, this experiment will be tracked with Weights and Biases",
)
parser.add_argument("--distill-coef", type=float, default=1.0, help="Imitation loss weight.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# torch.compile / Inductor / CUDA Graph: keep disabled by default for Isaac Sim runs
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPHS", "0")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _extract_se3_meta(env) -> tuple[list[str], list[dict]]:
    base_env = env.unwrapped
    se3_names = []
    se3_terms = []
    for name in base_env.action_manager.active_terms:
        term = base_env.action_manager.get_term(name)
        # detect se3 action terms via attributes we rely on
        if not hasattr(term, "_compute_jacobian_inverse") or not hasattr(term, "_get_jacobian"):
            continue
        if getattr(term, "action_dim", None) != 6:
            continue

        se3_names.append(name)

        angular_scale = term._angular_scale  # noqa: SLF001
        angular_bias = term._angular_bias  # noqa: SLF001
        linear_scale = term._linear_scale  # noqa: SLF001
        linear_bias = term._linear_bias  # noqa: SLF001

        angular_vel_limits = term._angular_vel_limits  # noqa: SLF001
        linear_vel_limits = term._linear_vel_limits  # noqa: SLF001

        use_ad = bool(term.cfg.is_xform and term.cfg.use_body_frame and (not term.cfg.use_xform_jacobian))
        Ad_b_bprime = None
        if use_ad:
            Ad_b_bprime = term._Ad_b_bprime.detach().cpu().tolist()  # noqa: SLF001

        se3_terms.append(
            {
                "angular_scale": (angular_scale.detach().cpu().tolist() if hasattr(angular_scale, "detach") else angular_scale),
                "angular_bias": (angular_bias.detach().cpu().tolist() if hasattr(angular_bias, "detach") else angular_bias),
                "linear_scale": (linear_scale.detach().cpu().tolist() if hasattr(linear_scale, "detach") else linear_scale),
                "linear_bias": (linear_bias.detach().cpu().tolist() if hasattr(linear_bias, "detach") else linear_bias),
                "angular_vel_limits": (angular_vel_limits.detach().cpu().tolist() if angular_vel_limits is not None else None),
                "linear_vel_limits": (linear_vel_limits.detach().cpu().tolist() if linear_vel_limits is not None else None),
                "use_ad": use_ad,
                "nj": int(len(term._joint_ids)),  # noqa: SLF001
                "joint_names": list(getattr(term.cfg, "joint_names", [])),
            }
        )

    if len(se3_names) == 0:
        raise RuntimeError("No SE(3) action terms detected; cannot use se3 distillation aligner.")

    return se3_names, se3_terms


def _run(env_cfg, agent_cfg):
    """执行在线蒸馏训练的主函数。
    
    该函数负责：
    1. 创建学生环境（SE3动作空间）
    2. 加载教师策略（关节动作空间）
    3. 通过DistillInfoWrapper包装环境，在每步注入教师标签
    4. 配置并启动rl_games训练器，使用自定义的DistillA2CAgent（包含模仿损失）
    
    Args:
        env_cfg: 学生环境配置（通过Hydra从任务注册表加载）
        agent_cfg: rl_games智能体配置字典
    """
    # ========== 导入依赖模块 ==========
    import gymnasium as gym

    from rl_games.common import env_configurations, vecenv
    from rl_games.common.algo_observer import IsaacAlgoObserver
    from rl_games.torch_runner import Runner

    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

    import isaaclab_tasks  # noqa: F401
    import leaphand.tasks.manager_based.leaphand  # noqa: F401

    from leaphand.distillation.label_wrapper import DistillInfoWrapper
    from leaphand.rl_games.distill_a2c_agent import DistillA2CAgent
    import leaphand.rl_games.se3_tcn_network  # noqa: F401

    # ========== 覆盖环境配置参数 ==========
    # 如果命令行指定了环境数量，则覆盖配置文件中的默认值
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    # 如果命令行指定了设备（cuda:0等），则覆盖配置文件中的默认值
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # ========== 随机种子设置 ==========
    # 如果种子为-1，则随机生成一个种子（用于可复现性）
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # 将种子应用到智能体配置中
    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"].get("seed")
    # 确保智能体使用与环境相同的设备
    agent_cfg["params"]["config"]["device"] = env_cfg.sim.device
    agent_cfg["params"]["config"]["device_name"] = env_cfg.sim.device

    # ========== 日志与实验目录设置（避免默认写到 /home/hac/isaac/runs） ==========
    # 从当前文件路径解析AnyRotate项目根目录：
    # AnyRotate/source/leaphand/leaphand/distillation/distill_train.py -> 向上4级 = AnyRotate
    anyrotate_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    
    # 确定配置名称（用于日志子目录）
    config_name = args_cli.log_name if args_cli.log_name is not None else agent_cfg["params"]["config"]["name"]
    
    # 构建日志根路径：AnyRotate/logs/rl_games/<config_name>
    log_root_path = os.path.join(anyrotate_root, "logs", "rl_games", config_name)
    
    # 获取或生成实验子目录名（时间戳格式）
    log_dir = agent_cfg["params"]["config"].get("full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    
    # 将日志路径写入智能体配置
    agent_cfg["params"]["config"]["train_dir"] = os.path.abspath(log_root_path)
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir
    
    # 尝试将日志路径也写入环境配置（某些环境可能需要）
    try:
        env_cfg.log_dir = os.path.join(agent_cfg["params"]["config"]["train_dir"], log_dir)
    except Exception:  # noqa: BLE001
        pass
    
    # 创建必要的日志子目录（神经网络检查点、TensorBoard摘要等）
    os.makedirs(os.path.join(agent_cfg["params"]["config"]["train_dir"], log_dir, "nn"), exist_ok=True)
    os.makedirs(os.path.join(agent_cfg["params"]["config"]["train_dir"], log_dir, "summaries"), exist_ok=True)

    # ========== 加载教师策略配置（不创建环境实例，仅用于获取观测组和rl_games参数） ==========
    teacher_env_cfg = load_cfg_from_registry(args_cli.teacher_task, "env_cfg_entry_point")
    teacher_agent_cfg = load_cfg_from_registry(args_cli.teacher_task, "rl_games_cfg_entry_point")

    # ========== 提取教师关节动作顺序（用于模仿对齐） ==========
    try:
        # 从教师环境配置中读取关节名称列表
        teacher_joint_names = list(teacher_env_cfg.actions.hand_joint_pos.joint_names)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to read teacher joint_names from teacher_env_cfg.actions.hand_joint_pos.joint_names. "
            "Please ensure --teacher-task uses a joint-action task config."
        ) from exc

    # ========== 提取rl-games环境包装器参数 ==========
    rl_device = agent_cfg["params"]["config"]["device"]
    # 观测值裁剪范围（防止异常值）
    clip_obs = agent_cfg["params"].get("env", {}).get("clip_observations", math.inf)
    # 动作值裁剪范围（防止异常值）
    clip_actions = agent_cfg["params"].get("env", {}).get("clip_actions", math.inf)
    # 观测组配置（可能包含多个观测组，如policy、critic等）
    obs_groups = agent_cfg["params"].get("env", {}).get("obs_groups")
    # 是否将多个观测组拼接为单一向量
    concate_obs_group = agent_cfg["params"].get("env", {}).get("concate_obs_group", True)

    # ========== 创建学生环境 ==========
    env = gym.make(args_cli.student_task, cfg=env_cfg)
    # 如果是多智能体环境，转换为单智能体环境
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # ========== 提取SE3动作项元数据（用于蒸馏对齐） ==========
    # 从学生环境中提取所有SE3动作项的名称和配置信息
    se3_term_names, se3_terms = _extract_se3_meta(env)
    # 计算雅可比伪逆矩阵的总维度（每个关节6个自由度）
    jacobian_pinv_dim = int(sum(int(t["nj"]) * 6 for t in se3_terms))
    # 计算伴随矩阵的总维度（每个SE3项36个元素，即6x6矩阵）
    ad_b_bprime_dim = int(len(se3_terms) * 36)

    # ========== 包装环境：注入教师标签（在每步之前） ==========
    env = DistillInfoWrapper(
        env,
        teacher_env_cfg=teacher_env_cfg,  # 教师环境配置（用于构建观测）
        teacher_agent_cfg=teacher_agent_cfg,  # 教师智能体配置（用于加载策略网络）
        teacher_checkpoint=args_cli.teacher_checkpoint,  # 教师检查点路径
        teacher_obs_group=args_cli.teacher_obs_group,  # 教师使用的观测组名称（通常为"policy"）
        teacher_action_scale=float(args_cli.teacher_action_scale),  # 教师原始动作的缩放系数
        se3_term_names=se3_term_names,  # SE3动作项名称列表
        align=args_cli.align,  # 对齐策略（如"se3_to_relative_joint_delta"）
        deterministic_teacher=True,  # 教师策略使用确定性推理（不采样）
    )

    # ========== 包装环境：rl_games向量化环境包装器 ==========
    env = RlGamesVecEnvWrapper(
        env,
        rl_device,  # 设备（cuda:0等）
        clip_obs,  # 观测裁剪范围
        clip_actions,  # 动作裁剪范围
        obs_groups=obs_groups,  # 观测组配置
        concate_obs_group=concate_obs_group,  # 是否拼接观测组
    )

    # ========== 注册环境到rl_games ==========
    # 注册自定义的向量化环境类型
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    # 注册环境配置（rl_games通过此配置创建环境实例）
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # ========== 设置智能体配置中的环境数量 ==========
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    # ========== 配置蒸馏参数 ==========
    dt = float(env.unwrapped.step_dt)  # 仿真时间步长
    # 指定使用自定义的蒸馏A2C算法
    agent_cfg["params"]["algo"]["name"] = "a2c_continuous_distill"
    # 添加蒸馏配置字典
    agent_cfg["params"]["config"]["distill"] = {
        "imit_coef": float(args_cli.distill_coef),  # 模仿损失权重系数
        "teacher_action_dim": 16,  # 教师动作维度（16个关节）
        "jacobian_pinv_dim": jacobian_pinv_dim,  # 雅可比伪逆矩阵维度
        "ad_b_bprime_dim": ad_b_bprime_dim,  # 伴随矩阵维度
        "dt": dt,  # 时间步长
        "se3_terms": se3_terms,  # SE3动作项配置列表
        "teacher_joint_names": teacher_joint_names,  # 教师关节名称列表
    }

    # ========== 如果指定了学生检查点，则配置恢复训练 ==========
    if args_cli.student_checkpoint is not None:
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = args_cli.student_checkpoint

    # ========== 创建并配置rl_games训练器 ==========
    runner = Runner(IsaacAlgoObserver())  # 创建训练器（带Isaac Sim观察器）
    # 注册自定义的蒸馏A2C智能体构建器
    runner.algo_factory.register_builder("a2c_continuous_distill", lambda **kwargs: DistillA2CAgent(**kwargs))
    # 加载智能体配置
    runner.load(agent_cfg)
    # 重置训练器状态
    runner.reset()

    # ========== 启动训练 ==========
    if args_cli.student_checkpoint is not None:
        # 从检查点恢复训练
        runner.run({"train": True, "play": False, "checkpoint": args_cli.student_checkpoint})
    else:
        # 从头开始训练
        runner.run({"train": True, "play": False})

    # ========== 清理资源 ==========
    env.close()


if __name__ == "__main__":
    # 注意：hydra_task_config() 通过 gym.spec() 解析配置入口点。
    # 因此，必须在调用 hydra_task_config 之前导入任务包（并执行 gym.register）。
    import isaaclab_tasks  # noqa: F401
    import leaphand.tasks.manager_based.leaphand  # noqa: F401

    from isaaclab_tasks.utils.hydra import hydra_task_config

    # 使用Hydra装饰器加载学生任务的环境配置和智能体配置
    @hydra_task_config(args_cli.student_task, "rl_games_cfg_entry_point")
    def main(env_cfg, agent_cfg):
        """主入口函数，由Hydra装饰器注入配置后调用_run执行训练。"""
        _run(env_cfg, agent_cfg)

    # 执行主函数
    main()
    # 关闭Isaac Sim应用
    simulation_app.close()
