r"""冻结Actor的均值/随机动作对照，不运行Critic或优化器。

Actor给出有界中心$\bar a$及共享$\log\sigma$；沿用生产PPO的边界映射和采样律：
$$
m=\operatorname{atanh}(\operatorname{clip}(\bar a,-1+\epsilon,1-\epsilon)),\quad
z\sim\mathcal N(m,\sigma^2),\quad a=M\odot\tanh z.
$$
epsilon直接读取生产分布实现。torch.normal与Normal.sample调用相同采样算子；独立Generator只隔离随机流。
所有16槽都采样后再应用ghost mask，与训练的随机数消费形状一致。
默认均值路径直接交付原Tensor，不执行atanh/tanh往返，也不消费随机数。
"""

from __future__ import annotations

from typing import Literal

import torch

from anymani.distill.models.palm_rotation_policy import expanded_policy_log_std

from .palm_rotation_network import PalmRotationMaskedContinuousModel

LATENT_ACTION_EPSILON = PalmRotationMaskedContinuousModel.Network._ACTION_EPS  # 与生成checkpoint的PPO同一atanh边界


def validate_frozen_action_mode(mode: str, seed: int | None) -> None:
    r"""随机模式需要显式63-bit非负seed；均值模式没有动作随机流。"""
    if mode not in {"mean", "sample"}:
        raise ValueError("action mode must be mean or sample")  # 不将未知字符串回退为均值
    if mode == "mean" and seed is not None:
        raise ValueError("mean action mode does not consume an action seed")  # 声明必须对应实际执行
    if mode == "sample" and (type(seed) is not int or not 0 <= seed < 2**63):
        raise ValueError("sample action mode requires an explicit non-negative 63-bit action seed")  # bool不作seed


def select_frozen_actor_actions(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    joint_valid: torch.Tensor,
    *,
    mode: Literal["mean", "sample"] = "mean",  # 均值正式路径与随机诊断路径
    generator: torch.Generator | None = None,  # sample独占设备随机流；mean保持None
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    r"""按声明动作模式输出[B,16]动作及随机诊断所需的分布事实。

    Args:
        mean: Actor的FP32有界中心，ghost已经由Actor置零。
        log_std: checkpoint的共享FP32 latent log-standard-deviation。
        joint_valid: 同形bool活动关节掩码。
        mode: mean保持正式评价路径；sample用于冻结分布诊断。
        generator: sample独占的设备随机流，整个回放持续推进。

    Returns:
        动作和可逐步重建采样关系的trace字段；mean模式的附加字段为空。
    """
    if mode == "mean":
        if generator is not None:
            raise ValueError("mean action mode must not receive an action generator")  # 不静默忽略随机流
        return mean, {}  # 原均值Tensor逐值、逐边界保持；无额外随机数或张量运算
    if mode != "sample" or generator is None:
        raise ValueError("sample action mode requires its own generator")  # 拒绝隐含全局RNG
    if mean.ndim != 2 or mean.shape[-1] != 16 or joint_valid.shape != mean.shape or joint_valid.dtype != torch.bool:
        raise ValueError("frozen sampling expects aligned [B,16] mean and boolean joint mask")  # canonical ABI
    if mean.dtype != torch.float32 or log_std.dtype != torch.float32:
        raise ValueError("frozen sampling requires FP32 center and log-standard-deviation")

    # 与生产rollout保持同一latent location、sigma广播与采样算子；不对有界动作直接加噪声。
    sigma = torch.exp(expanded_policy_log_std(log_std, mean, joint_valid))  # 全局或条件尺度，ghost先中和。
    latent_mean = PalmRotationMaskedContinuousModel.Network._action_to_latent(mean)  # 共享生产atanh/epsilon
    torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
        torch.all(torch.isfinite(sigma) & (sigma > 0)), "invalid frozen sigma"
    )  # 有效非退化Normal；CUDA断言不强制每个policy step同步到CPU
    torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
        torch.all(mean.abs() <= 1 + 1e-6), "frozen action center escaped bounds"
    )  # NaN同时被拒绝，符合生产rollout的动作中心界
    latent_sample = torch.normal(latent_mean, sigma, generator=generator)  # Normal.sample同算子，独立RNG
    action = torch.tanh(latent_sample) * joint_valid.to(dtype=mean.dtype)  # [B,16]，ghost严格为零
    return action, {  # 保存已执行的分布参数与样本，不事后估计噪声或重跑网络
        "policy_action_mean": mean,  # 当前执行动作之前的确定性中心
        "policy_latent_sigma": sigma,  # 与latent样本同坐标
        "policy_latent_sample": latent_sample,  # a=tanh(z)*M可直接重建
        "policy_joint_valid": joint_valid,  # 行内概率与ghost边界
    }
