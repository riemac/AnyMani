r"""MVP80 task到rl_games的structured rollout transport。

一个rl_games observation同时携带三类数据，但只保存一份：

- actor公开的current/History30/limits/contact-bit与valid masks；
- critic专用的joint dynamics、all-owner force、object/task与reward-release；
- rollout时一次计算的FP32 frozen N040 $Z^e$及静态graph buckets。

该transport不把privileged critic tensor拼入actor输入。分流发生在custom network中，且
``prototype_index``只作为asset routing certificate与stratified-minibatch标签，不参与网络数值图。
N040 graph bucket使用``int16``存储；这不是量化学习特征，因为网络前向会无损恢复``long``索引。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import torch
from gym import spaces
from rl_games.common.vecenv import IVecEnv

from anymani.distill.models.palm_rotation_policy import PalmRotationActorObservation
from anymani.tasks.hetero.mdp.curriculum_state import (
    HETERO_REWARD_RELEASE_STATE_ATTR,
    HeterogeneousRewardReleaseState,
)

if TYPE_CHECKING:
    from anymani.distill.models.palm_rotation_policy import PalmRotationGeometry


class PalmRotationGeometryProvider(Protocol):
    r"""Transport消费的最窄N040 provider合同；避免纯测试加载Isaac/USD模块。"""

    resolve_call_count: int  # 每个state encoder调用次数

    def to(self, device: torch.device | str) -> PalmRotationGeometryProvider:
        r"""把provider evidence/master weights移动到policy device。"""

        ...

    def resolve(
        self,
        prototype_index: torch.Tensor,
        actor_observation: PalmRotationActorObservation,
    ) -> PalmRotationGeometry:
        r"""返回当前$q$对应的FP32 tokens、masks与graph buckets。"""

        ...

# 每个key的sample shape是experience buffer的公开ABI；第一轴$B$由rl_games在运行期添加。
PALM_ROTATION_FLOAT_SHAPES: dict[str, tuple[int, ...]] = {
    "actor_jnt_current": (16, 5),  # $O^a_{t,\mathrm{jnt}}$，当前归一化状态与两类contact bit
    "actor_jnt_history": (30, 16, 5),  # 1.5 s oldest-to-latest History30，20 Hz
    "actor_jnt_limits": (16, 2),  # physical joint limits除以$\pi$
    "actor_owner_contact": (21, 1),  # PALM/JOINT/TIP当前binary contact
    "critic_jnt_state": (16, 4),  # $[q/\pi,\dot q,u/\pi,a]$
    "critic_owner_contact": (21, 2),  # all-owner $[F/0.25\mathrm N,\mathbf1(F\ge0.25\mathrm N)]$
    "critic_obj": (1, 15),  # privileged object pose与twist
    "critic_task": (1, 8),  # privileged command/progress state
    "critic_reward_release": (1,),  # 当前cell级$\lambda_{rew}$
    "geometry_tokens": (21, 128),  # BF16 encoder计算后立即恢复的FP32 $Z^e$
}

# Bool masks保持storage cardinality，不转成float，ghost从attention/action/probability中严格退出。
PALM_ROTATION_BOOL_SHAPES: dict[str, tuple[int, ...]] = {
    "jnt_valid": (16,),  # canonical JOINT有效集合$\mathcal J_{\mathfrak m}$
    "tip_valid": (4,),  # index/middle/ring/thumb TIP有效集合
    "owner_valid": (21,),  # PALM/JOINT/TIP owner有效集合$\mathcal E_{\mathfrak m}$
}

# Graph bucket与asset row都是离散certificate；int16覆盖21-owner距离和80个prototype且节省rollout显存。
PALM_ROTATION_INT16_SHAPES: dict[str, tuple[int, ...]] = {
    "shortest_path": (21, 21),  # owner graph无向最短路bucket
    "parent_direction": (21, 21),  # parent-directed relation bucket
    "child_direction": (21, 21),  # child-directed relation bucket
    "prototype_index": (1,),  # selection-local$k_e\in[0,79]$，只用于routing/stratification
}


def palm_rotation_observation_space(clip_observations: float) -> spaces.Dict:
    r"""构造与structured transport逐key一致的rl_games Dict space。

    Args:
        clip_observations (float): raw actor/critic tensor的对称数值截断；$Z^e$不做截断。

    Returns:
        gym.spaces.Dict: sample-level named observation ABI。
    """

    clip = float(clip_observations)  # raw dynamic observation的有限保护界
    observation_spaces: dict[str, spaces.Space] = {}  # 键名即network adapter读取名

    # N040 token保留完整FP32幅值；其它raw float由任务归一化后再做宽松clip。
    for name, shape in PALM_ROTATION_FLOAT_SHAPES.items():
        bound = np.inf if name == "geometry_tokens" else clip  # encoder输出不被transport静默改写
        observation_spaces[name] = spaces.Box(-bound, bound, shape=shape, dtype=np.float32)

    # Validity是逻辑集合，而不是连续0/1网络特征；storage保持bool可避免重复浮点显存。
    for name, shape in PALM_ROTATION_BOOL_SHAPES.items():
        observation_spaces[name] = spaces.Box(False, True, shape=shape, dtype=np.bool_)

    # 所有graph relation与prototype均非负；网络边界恢复为torch.long embedding index。
    for name, shape in PALM_ROTATION_INT16_SHAPES.items():
        observation_spaces[name] = spaces.Box(0, np.iinfo(np.int16).max, shape=shape, dtype=np.int16)
    return spaces.Dict(observation_spaces)


class PalmRotationRlGamesVecEnv(IVecEnv):
    r"""在env step边界一次解析N040并形成可缓存structured observation。

    若rollout horizon为$H=30$，一次update会解析初态及每个next state；用于PPO update的前$H$个
    states各自只有一次encoder调用，所有5个mini-epochs仅重放experience buffer中的FP32 $Z^e$。
    """

    def __init__(
        self,
        env: Any,
        *,
        geometry_provider: PalmRotationGeometryProvider,
        prototype_index: torch.Tensor,
        rl_device: torch.device | str,
        clip_observations: float,
        clip_actions: float,
    ) -> None:
        r"""绑定Isaac env、80-row round-robin routing与冻结N040 provider。"""

        self.env = env  # ManagerBasedRLEnv/Gym wrapper，拥有物理step与raw named observations
        self._rl_device = torch.device(rl_device)  # actor、critic、buffer与N040共同device
        self._sim_device = torch.device(env.unwrapped.device)  # action写回物理环境的device
        self._clip_observations = float(clip_observations)  # raw dynamic observation clip
        self._clip_actions = float(clip_actions)  # canonical action authority前的$[-1,1]$ transport clip
        self.geometry_provider = geometry_provider.to(self._rl_device)  # FP32 master + scoped BF16 encoder
        self.prototype_index = prototype_index.to(self._rl_device, dtype=torch.long)  # `[N]` fixed routing
        if self.prototype_index.shape != (env.unwrapped.num_envs,):
            raise ValueError("prototype_index must align one-to-one with vectorized environments")
        self._observation_space = palm_rotation_observation_space(self._clip_observations)  # fixed Dict ABI
        self._asset_count = int(self.prototype_index.max().item()) + 1  # selection-local asset cardinality$A$
        self._rollout_count = torch.zeros(self._asset_count, device=self._rl_device)  # samples per asset/update
        self._rollout_sums = {
            name: torch.zeros(self._asset_count, device=self._rl_device)
            for name in (
                "reward_mean",
                "goal_count_mean",
                "net_turns_mean",
                "drop_rate",
                "axis_failure_rate",
                "tip_contact_mean",
                "palm_contact_rate",
                "non_tip_contact_rate",
                "action_clamp_fraction",
                "physical_action_rms",
            )
        }  # post-physics/pre-reset充分统计量之和
        self._current_joint_valid: torch.Tensor | None = None  # 当前state对应的逻辑动作集合`[N,16]`
        self._last_action_clamp_fraction = torch.zeros(self.num_envs, device=self._rl_device)  # active裁剪率`[N]`
        self._last_physical_action_rms = torch.zeros(self.num_envs, device=self._rl_device)  # clamp后RMS`[N]`
        self._terminal_count = torch.zeros(self._asset_count, device=self._rl_device)  # completed episodes per asset/update
        self._terminal_sums = {
            name: torch.zeros(self._asset_count, device=self._rl_device)
            for name in (
                "terminal_goal_count_mean",
                "terminal_net_turns_mean",
                "terminal_absolute_path_turns_mean",
                "terminal_directional_consistency_mean",
                "terminal_timeout_rate",
                "terminal_drop_rate",
                "terminal_axis_failure_rate",
            )
        }  # 只在first-trajectory terminal frame累计，分母与rollout samples分离

    @property
    def unwrapped(self) -> Any:
        r"""返回底层Isaac环境，用于训练入口读取num_envs/device。"""

        return self.env.unwrapped

    @property
    def num_envs(self) -> int:
        r"""返回并行environment数量$N$。"""

        return int(self.unwrapped.num_envs)

    @property
    def observation_space(self) -> spaces.Dict:
        r"""返回单环境structured experience space。"""

        return self._observation_space

    @property
    def action_space(self) -> spaces.Box:
        r"""返回canonical 16-slot action space；ghost概率由custom model排除。"""

        shape = tuple(self.unwrapped.single_action_space.shape)  # sample-level canonical action shape
        return spaces.Box(-self._clip_actions, self._clip_actions, shape=shape, dtype=np.float32)

    def get_number_of_agents(self) -> int:
        r"""MVP每只手为一个single agent。"""

        return 1

    def get_env_info(self) -> dict[str, Any]:
        r"""交付rl_games allocation所需spaces；privileged critic在custom model内分流。"""

        return {
            "observation_space": self.observation_space,  # 单份actor/critic/Z structured buffer
            "action_space": self.action_space,  # `[16]` canonical transport
            "state_space": None,  # 不复制$Z^e$到rl_games第二套central-value buffer
            "value_size": 1,  # 每个hand/environment一个scalar value
            "agents": 1,
        }

    def seed(self, seed: int = -1) -> int:
        r"""把训练seed交给底层Isaac环境。"""

        return int(self.unwrapped.seed(seed))

    def reset(self) -> dict[str, dict[str, torch.Tensor]]:
        r"""执行cold reset并为初始state计算一次N040。"""

        observation, _ = self.env.reset()  # Isaac Gymnasium API返回`(obs, info)`
        return {"obs": self._transport(observation)}  # rl_games顶层必须含`obs`

    def step(
        self, actions: torch.Tensor
    ) -> tuple[dict[str, dict[str, torch.Tensor]], torch.Tensor, torch.Tensor, dict[str, Any]]:
        r"""裁剪action、推进20 Hz policy step并解析唯一next-state N040。"""

        if self._current_joint_valid is None:
            raise RuntimeError("palm-rotation action step requires a preceding structured observation")
        sampled_actions = actions.detach().to(self._rl_device, dtype=torch.float32)  # PPO采样`[N,16]`
        if sampled_actions.shape != self._current_joint_valid.shape:
            raise RuntimeError("sampled actions and active-joint mask disagree")
        active_float = self._current_joint_valid.to(dtype=sampled_actions.dtype)  # ghost不进统计分母
        active_count = active_float.sum(dim=-1).clamp_min(1.0)  # $n_i$，每env真实动作维数
        clipped_actions = sampled_actions.clamp(-self._clip_actions, self._clip_actions)  # 环境实际消费值
        changed = (clipped_actions != sampled_actions).to(dtype=sampled_actions.dtype) * active_float
        self._last_action_clamp_fraction = changed.sum(dim=-1) / active_count  # active DoF被截断比例
        self._last_physical_action_rms = torch.sqrt(
            (clipped_actions.square() * active_float).sum(dim=-1) / active_count
        )  # clamp后无量纲动作RMS
        physical_actions = clipped_actions.to(self._sim_device)  # 写回Isaac action term
        observation, reward, terminated, truncated, extras = self.env.step(physical_actions)  # 6个120 Hz substeps
        extras = {
            key: value.to(self._rl_device, non_blocking=True) if hasattr(value, "to") else value
            for key, value in extras.items()
        }  # logging/timeouts与policy buffer同device
        if "log" in extras:
            extras["episode"] = extras.pop("log")  # IsaacLab -> rl_games observer命名
        done = (terminated | truncated).to(self._rl_device)  # failure或120 s timeout
        self._record_rollout_step(reward.to(self._rl_device))  # 读取command冻结的post-physics/pre-reset snapshot
        return {"obs": self._transport(observation)}, reward.to(self._rl_device), done, extras

    def close(self) -> None:
        r"""关闭底层Isaac environment与simulation resources。"""

        self.env.close()

    def _float(self, value: torch.Tensor, *, clip: bool = True) -> torch.Tensor:
        r"""把任务tensor移动到RL device并保持FP32物理值。"""

        result = value.to(self._rl_device, dtype=torch.float32)  # raw task tensor -> FP32 policy side
        return result.clamp(-self._clip_observations, self._clip_observations) if clip else result

    def _transport(self, observation: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        r"""按信息边界重命名raw groups，并只在此处解析一次$Z^e$。"""

        policy = observation.get("policy")  # actor-only named raw observation
        critic = observation.get("critic")  # privileged named raw observation
        if not isinstance(policy, Mapping) or not isinstance(critic, Mapping):
            raise TypeError("palm-rotation task must expose named policy and critic observation groups")

        # 三套mask必须逐位一致；network只缓存一份，避免actor/critic语义漂移与重复显存。
        joint_valid = policy["jnt_valid"].to(self._rl_device, dtype=torch.bool)  # `[N,16]`
        tip_valid = policy["tip_valid"].to(self._rl_device, dtype=torch.bool)  # `[N,4]`
        owner_valid = policy["owner_valid"].to(self._rl_device, dtype=torch.bool)  # `[N,21]`
        self._current_joint_valid = joint_valid  # 该state采样的下一action沿相同有效集合裁剪/统计
        for name, expected in (("jnt_valid", joint_valid), ("tip_valid", tip_valid), ("owner_valid", owner_valid)):
            actual = critic[name].to(self._rl_device, dtype=torch.bool)  # critic同名mask
            if not torch.equal(actual, expected):
                raise RuntimeError(f"actor/critic {name} masks disagree at rollout transport")

        # N040只读取actor current physical q和masks；contact/history/limits不进入geometry encoder。
        actor_observation = PalmRotationActorObservation(
            jnt_current=self._float(policy["jnt_current"]),  # `[N,16,5]`
            jnt_history=self._float(policy["jnt_history"]),  # `[N,30,16,5]`
            jnt_limits=self._float(policy["jnt_limits"]),  # `[N,16,2]`
            owner_contact=self._float(policy["owner_contact"]),  # `[N,21,1]` binary
            jnt_valid=joint_valid,
            tip_valid=tip_valid,
            owner_valid=owner_valid,
        )
        geometry = self.geometry_provider.resolve(self.prototype_index, actor_observation)  # one BF16 encoder call

        # Transport mapping保留原始axes；network内部dataclass重建是唯一信息分流点。
        return {
            "actor_jnt_current": actor_observation.jnt_current,
            "actor_jnt_history": actor_observation.jnt_history,
            "actor_jnt_limits": actor_observation.jnt_limits,
            "actor_owner_contact": actor_observation.owner_contact,
            "critic_jnt_state": self._float(critic["jnt_state"]),
            "critic_owner_contact": self._float(critic["owner_contact"]),
            "critic_obj": self._float(critic["obj"]),
            "critic_task": self._float(critic["task"]),
            "critic_reward_release": self._float(critic["reward_release"]),
            "jnt_valid": joint_valid,
            "tip_valid": tip_valid,
            "owner_valid": owner_valid,
            "geometry_tokens": geometry.tokens,  # FP32 `[N,21,128]`，mini-epochs直接复用
            "shortest_path": geometry.shortest_path.to(torch.int16),  # exact graph bucket，storage compact
            "parent_direction": geometry.parent_direction.to(torch.int16),
            "child_direction": geometry.child_direction.to(torch.int16),
            "prototype_index": self.prototype_index.to(torch.int16).unsqueeze(-1),  # `[N,1]` sampling label
        }

    def _record_rollout_step(self, reward: torch.Tensor) -> None:
        r"""按asset聚合当前policy step的post-physics/pre-reset任务事实。

        RewardManager最后一个term在automatic reset前冻结command snapshot，因此done rows仍表示terminal
        trajectory，而不是rank-0重置后的零进度。这里不调用reward/command更新，只读取已形成事实。
        """

        command = self.unwrapped.command_manager.get_term("goal_pose")  # N000 moving-subgoal command
        snapshot = getattr(command, "post_physics_evaluation_snapshot", None)
        if not isinstance(snapshot, dict) or not bool(snapshot.get("valid", torch.tensor(False)).all().item()):
            raise RuntimeError("palm-rotation rollout requires a valid post-physics evaluation snapshot")
        two_pi = 2.0 * torch.pi  # signed net rad -> physical turns
        values = {
            "reward_mean": reward.reshape(-1).float(),
            "goal_count_mean": snapshot["completed_subgoals"].to(self._rl_device).float(),
            "net_turns_mean": snapshot["net_rotation_rad"].to(self._rl_device).float() / two_pi,
            "drop_rate": snapshot["termination_object_out_of_anchor"].to(self._rl_device).float(),
            "axis_failure_rate": snapshot["termination_goal_axis_misaligned"].to(self._rl_device).float(),
            "tip_contact_mean": snapshot["tip_active_count"].to(self._rl_device).float(),
            "palm_contact_rate": snapshot["palm_contact"].to(self._rl_device).float(),
            "non_tip_contact_rate": snapshot["finger_non_tip_contact"].to(self._rl_device).float(),
            "action_clamp_fraction": self._last_action_clamp_fraction,
            "physical_action_rms": self._last_physical_action_rms,
        }  # 每项`[N]`与prototype routing一一对齐
        ones = torch.ones_like(self.prototype_index, dtype=torch.float32)  # 每env当前step贡献一个sample
        self._rollout_count.scatter_add_(0, self.prototype_index, ones)
        for name, value in values.items():
            if value.shape != self.prototype_index.shape:
                raise RuntimeError(f"rollout metric {name} does not align with environment axis")
            self._rollout_sums[name].scatter_add_(0, self.prototype_index, value)

        # 完整episode统计只消费terminal rows；snapshot在automatic reset前冻结，故不会读到新pregrasp零值。
        terminal_drop = snapshot["termination_object_out_of_anchor"].to(self._rl_device).bool()
        terminal_axis = snapshot["termination_goal_axis_misaligned"].to(self._rl_device).bool()
        terminal_timeout = snapshot["termination_time_out"].to(self._rl_device).bool()
        terminal = terminal_drop | terminal_axis | terminal_timeout
        if bool(terminal.any().item()):
            labels = self.prototype_index[terminal]  # completed trajectory对应selection-local asset
            net_turns = snapshot["net_rotation_rad"].to(self._rl_device).float()[terminal] / two_pi
            path_turns = snapshot["absolute_path_rotation_rad"].to(self._rl_device).float()[terminal] / two_pi
            directional = torch.clamp(net_turns, min=0.0) / path_turns.clamp_min(torch.finfo(torch.float32).eps)
            terminal_values = {
                "terminal_goal_count_mean": snapshot["completed_subgoals"].to(self._rl_device).float()[terminal],
                "terminal_net_turns_mean": net_turns,
                "terminal_absolute_path_turns_mean": path_turns,
                "terminal_directional_consistency_mean": directional.clamp(max=1.0),
                "terminal_timeout_rate": terminal_timeout[terminal].float(),
                "terminal_drop_rate": terminal_drop[terminal].float(),
                "terminal_axis_failure_rate": terminal_axis[terminal].float(),
            }
            self._terminal_count.scatter_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
            for name, value in terminal_values.items():
                self._terminal_sums[name].scatter_add_(0, labels, value)

    def drain_rollout_metrics(self) -> dict[str, torch.Tensor]:
        r"""返回上个PPO rollout的per-asset均值并清零accumulator。

        Returns:
            dict[str, torch.Tensor]: 每项FP32 CPU `[A]`；每资产样本数必须正且完全相等。
        """

        if bool((self._rollout_count <= 0).any().item()):
            raise RuntimeError("cannot drain incomplete per-asset rollout metrics")
        if bool((self._rollout_count != self._rollout_count[0]).any().item()):
            raise RuntimeError("rollout metric counts are not equal across assets")
        result = {
            name: (total / self._rollout_count).detach().cpu()
            for name, total in self._rollout_sums.items()
        }
        result["rollout_sample_count"] = self._rollout_count.detach().cpu().clone()
        result["completed_episode_count"] = self._terminal_count.detach().cpu().clone()
        terminal_denominator = self._terminal_count.clamp_min(1.0)  # count=0时值为0且由显式分母判别
        for name, total in self._terminal_sums.items():
            result[name] = (total / terminal_denominator).detach().cpu()
        self._rollout_count.zero_()
        for total in self._rollout_sums.values():
            total.zero_()
        self._terminal_count.zero_()
        for total in self._terminal_sums.values():
            total.zero_()
        return result

    def get_env_state(self) -> dict[str, Any]:
        r"""保存per-asset/cell curriculum与N040 call counter，供完整checkpoint恢复。"""

        result: dict[str, Any] = {
            "schema_version": "1.0.0",
            "prototype_index": self.prototype_index.detach().cpu(),  # environment-to-asset routing certificate
            "n040_resolve_call_count": int(self.geometry_provider.resolve_call_count),  # diagnostic continuity
        }
        curriculum = getattr(self.unwrapped, HETERO_REWARD_RELEASE_STATE_ATTR, None)
        if isinstance(curriculum, HeterogeneousRewardReleaseState):
            result["reward_release"] = curriculum.state_dict()  # 80 EMA、8 median/lambda与update counts
        return result

    def set_env_state(self, state: object) -> None:
        r"""在checkpoint model/optimizer恢复后逐值恢复课程与diagnostic counter。"""

        if state is None:
            return
        if not isinstance(state, Mapping) or state.get("schema_version") != "1.0.0":
            raise RuntimeError("palm-rotation checkpoint environment state is missing or incompatible")
        restored_routing = torch.as_tensor(state["prototype_index"], dtype=torch.long, device=self._rl_device)
        if not torch.equal(restored_routing, self.prototype_index):
            raise RuntimeError("checkpoint prototype routing disagrees with current 80-asset environment")
        self.geometry_provider.resolve_call_count = int(state.get("n040_resolve_call_count", 0))
        curriculum = getattr(self.unwrapped, HETERO_REWARD_RELEASE_STATE_ATTR, None)
        if isinstance(curriculum, HeterogeneousRewardReleaseState):
            curriculum.load_state_dict(state.get("reward_release"))  # fail closed on rows/cells/tensor shapes

    def set_train_info(self, env_frames: int, *args: Any, **kwargs: Any) -> None:
        r"""保留rl_games进度hook；当前ADR-0环境不以frame改变物理分布。"""

        _ = (env_frames, args, kwargs)  # curriculum仅由episode terminal net turns驱动


class PalmRotationRlGamesGpuEnv(IVecEnv):
    r"""让Runner factory完整转发step、spaces与checkpoint environment state。"""

    def __init__(self, config_name: str, num_actors: int, *, env: PalmRotationRlGamesVecEnv) -> None:
        r"""持有已构造的单一Isaac vec env；Runner不得重复实例化simulation。"""

        _ = (config_name, num_actors)  # 名称/数量已由训练入口与env自身严格验证
        self.env = env

    def step(self, actions: torch.Tensor):
        r"""转发vectorized policy step。"""

        return self.env.step(actions)

    def reset(self):
        r"""转发cold reset。"""

        return self.env.reset()

    def get_number_of_agents(self) -> int:
        r"""返回single-agent cardinality。"""

        return self.env.get_number_of_agents()

    def get_env_info(self) -> dict[str, Any]:
        r"""返回structured Dict observation与canonical action spaces。"""

        return self.env.get_env_info()

    def set_train_info(self, env_frames: int, *args: Any, **kwargs: Any) -> None:
        r"""把algorithm frame信息转发给MVP wrapper。"""

        self.env.set_train_info(env_frames, *args, **kwargs)

    def get_env_state(self) -> dict[str, Any]:
        r"""返回可序列化课程/diagnostic state。"""

        return self.env.get_env_state()

    def set_env_state(self, state: object) -> None:
        r"""恢复课程/diagnostic state。"""

        self.env.set_env_state(state)

    def drain_rollout_metrics(self) -> dict[str, torch.Tensor]:
        r"""返回最近rollout的per-asset task/contact统计。"""

        return self.env.drain_rollout_metrics()


__all__ = [
    "PALM_ROTATION_BOOL_SHAPES",
    "PALM_ROTATION_FLOAT_SHAPES",
    "PALM_ROTATION_INT16_SHAPES",
    "PalmRotationRlGamesGpuEnv",
    "PalmRotationRlGamesVecEnv",
    "palm_rotation_observation_space",
]
