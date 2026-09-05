r"""MVP80 actor/critic的进程级FP32/TF32 backend合同。

冻结N040只在provider内部使用局部BF16 autocast；actor、critic、PPO loss与optimizers使用FP32。
显式TF32候选只允许Linear、attention与Conv内部使用TensorFloat-32乘法，参数、activation、loss、GAE与Adam
state仍为FP32。rl_games Runner会修改PyTorch global TF32 flags，因此precision必须在Runner构造后重新执行并验证。
"""

from __future__ import annotations

import torch


def enforce_palm_rotation_precision(*, allow_tf32: bool = False) -> dict[str, bool]:
    r"""恢复声明的FP32/TF32模式并返回可审计的实际flags。

    本函数在Runner构造后、model build前调用，并在``runner.reset``后复验。N040的BF16 autocast是provider
    内部局部scope，不受这两个global flags影响。默认``False``保持既有严格FP32实验；TF32必须由训练CLI
    显式请求并进入run identity。

    Args:
        allow_tf32 (bool): 是否允许CUDA matmul与cuDNN使用TF32内部乘法。

    Returns:
        dict[str, bool]: 实际CUDA matmul/cuDNN TF32开关。
    """

    torch.backends.cuda.matmul.allow_tf32 = allow_tf32  # Linear/attention内部乘法模式；tensor dtype仍是FP32
    torch.backends.cudnn.allow_tf32 = allow_tf32  # TCN convolution内部模式；raw-stack arm无Conv
    flags = {
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }  # 读取真实backend状态，不以配置声明代替执行证据
    if any(value != allow_tf32 for value in flags.values()):
        raise RuntimeError(f"palm-rotation precision contract was not enforced: requested={allow_tf32}, actual={flags}")
    return flags


__all__ = ["enforce_palm_rotation_precision"]
