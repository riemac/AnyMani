'Set explicit FP32 and TF32 backend flags for Actor, Critic, and geometry computation.'

from __future__ import annotations

import torch


def enforce_palm_rotation_precision(*, allow_tf32: bool = False) -> dict[str, bool]:
    'Handle enforce PALM rotation precision; shapes [str,bool].'

    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    flags = {
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
    if any(value != allow_tf32 for value in flags.values()):
        raise RuntimeError(f"palm-rotation precision contract was not enforced: requested={allow_tf32}, actual={flags}")
    return flags


__all__ = ["enforce_palm_rotation_precision"]
