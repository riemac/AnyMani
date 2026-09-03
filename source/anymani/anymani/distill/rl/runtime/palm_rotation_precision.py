r"""MVP80 actor/critic的进程级FP32 backend合同。

冻结N040只在provider内部使用局部BF16 autocast；actor、critic、PPO loss与optimizers使用FP32。
rl_games Runner会修改PyTorch global TF32 flags，因此precision必须在Runner构造后重新执行并验证。
"""

from __future__ import annotations

import torch


def enforce_palm_rotation_precision() -> dict[str, bool]:
    r"""关闭Runner可能重新开启的TF32并返回可审计的实际flags。

    本函数在Runner构造后、model build前调用，并在``runner.reset``后复验。N040的BF16 autocast是provider
    内部局部scope，不受这两个global flags影响。

    Returns:
        dict[str, bool]: 实际CUDA matmul/cuDNN TF32开关；两项均固定为``False``。
    """

    torch.backends.cuda.matmul.allow_tf32 = False  # actor/critic Linear与attention使用严格FP32 matmul
    torch.backends.cudnn.allow_tf32 = False  # History30 temporal convolution保持FP32
    flags = {
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }  # 读取真实backend状态，不以配置声明代替执行证据
    if any(flags.values()):
        raise RuntimeError(f"palm-rotation FP32 precision contract was not enforced: {flags}")
    return flags


__all__ = ["enforce_palm_rotation_precision"]
