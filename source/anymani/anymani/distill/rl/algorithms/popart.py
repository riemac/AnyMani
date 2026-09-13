r"""全局单输出PopArt：显式returns统计更新与物理value保持。

forward不随train/eval模式更新统计；一次rollout只调用一次update_from_returns。
预测反归一化没有±5截断，否则统计坐标变化会移动饱和边界，破坏输出保持。
PPO自己的value clipping仍属于损失函数，与这里的仿射坐标变换分离。
"""

from __future__ import annotations

import torch
from torch import nn


class PopArtValueNormalizer(nn.Module):
    r"""维护全局return moments，并补偿Critic末层W/b；不持有或重复注册该head。

    V = s*(Wh+b)+mu，s=sqrt(var+epsilon)。从旧统计o到新统计n时：
    W_n=(s_o/s_n)W_o，b_n=(s_o*b_o+mu_o-mu_n)/s_n。
    统计使用FP64，head维持原dtype；不声称Adam优化轨迹因坐标变换而保持不变。
    """

    running_mean: torch.Tensor
    running_var: torch.Tensor
    count: torch.Tensor

    def __init__(self, epsilon: float = 1.0e-5) -> None:
        r"""保持现有RMS的初始mean=0、var=1、count=1及state-dict key。"""
        super().__init__()
        if not 0 < epsilon < float("inf"):
            raise ValueError("PopArt epsilon must be finite and positive")
        self.epsilon = float(epsilon)
        self.register_buffer("running_mean", torch.zeros(1, dtype=torch.float64))
        self.register_buffer("running_var", torch.ones(1, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))

    def forward(self, value: torch.Tensor, denorm: bool = False, mask=None) -> torch.Tensor:
        r"""按同一冻结快照执行仿射变换；train模式下也只读，不隐式更新或截断。"""
        if mask is not None:
            raise ValueError("PopArt statistics require explicit complete rollout returns, not forward masks")
        mean = self.running_mean.to(dtype=value.dtype)
        scale = (self.running_var + self.epsilon).sqrt().to(dtype=value.dtype)
        return value * scale + mean if denorm else (value - mean) / scale

    @torch.no_grad()
    def update_from_returns(self, returns: torch.Tensor, head: nn.Linear) -> dict[str, torch.Tensor]:
        r"""一次更新[B,1]物理return总体，原子地应用新moments与head补偿。

        只更新统计和head参数，不更新optimizer。调用方必须在任何minibatch开始前调用，
        随后的old values和return targets使用同一个新快照归一化。
        返回仿射系数残差用于低开销检查，不额外运行Critic forward。
        """
        if returns.ndim != 2 or returns.shape[1] != 1 or returns.shape[0] == 0:
            raise ValueError("PopArt returns must be nonempty [B,1]")
        if head.out_features != 1 or head.bias is None:
            raise ValueError("global PopArt requires a scalar Linear head with bias")
        if returns.device != self.running_mean.device or head.weight.device != returns.device:
            raise ValueError("PopArt returns, moments and head must share one device")
        torch._assert_async(torch.isfinite(returns).all(), "PopArt returns must be finite")  # pyright: ignore[reportPrivateImportUsage]
        target = returns.detach().to(dtype=torch.float64)
        old_mean = self.running_mean.clone()
        old_scale = (self.running_var + self.epsilon).sqrt()
        batch_mean = target.mean(dim=0)
        batch_var = target.var(dim=0, unbiased=False)
        batch_count = target.shape[0]
        new_count = self.count + batch_count
        delta = batch_mean - old_mean
        new_mean = old_mean + delta * batch_count / new_count
        new_var = (
            self.running_var * self.count
            + batch_var * batch_count
            + delta.square() * self.count * batch_count / new_count
        ) / new_count
        new_scale = (new_var + self.epsilon).sqrt()

        # 先形成全部候选值并检查，再原位copy；head参数对象不替换，optimizer引用仍有效。
        weight = head.weight.detach().to(dtype=torch.float64)
        bias = head.bias.detach().to(dtype=torch.float64)
        new_weight = (weight * (old_scale / new_scale)).to(dtype=head.weight.dtype)
        new_bias = ((old_scale * bias + old_mean - new_mean) / new_scale).to(dtype=head.bias.dtype)
        torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
            torch.isfinite(new_weight).all() & torch.isfinite(new_bias).all() & torch.isfinite(new_var).all(),
            "PopArt compensation must remain finite",
        )
        weight_error = (new_scale * new_weight.double() - old_scale * weight).abs().max()
        bias_error = (new_scale * new_bias.double() + new_mean - old_scale * bias - old_mean).abs().max()
        head.weight.copy_(new_weight)
        head.bias.copy_(new_bias)
        self.running_mean.copy_(new_mean)
        self.running_var.copy_(new_var)
        self.count.copy_(new_count)
        return {"weight_error": weight_error, "bias_error": bias_error, "count": new_count.detach()}
