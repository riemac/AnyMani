#!/usr/bin/env python3
r"""分解MVP80单个activation slice的actor/critic CUDA代价。

该probe不启动Isaac Sim，也不生成学习结论。它构造与正式PPO相同的张量形状：History30、16 JOINT、
21 owner、128D $Z^e$和三张图关系矩阵，并分别测量：

1. 逐JOINT History30 encoder；
2. 完整actor；
3. 完整privileged critic；
4. actor与critic同一microbatch的两次backward。

正式1280-env配置的activation slice为$M=2400$，2560-env配置为$M=4800$。CUDA Event只包围
forward/backward；输入构造、JSON发布和可选Kineto trace不进入计时。Synthetic数值不代表真实学习质量，
但网络shape、mask cardinality与autograd路径和生产代码相同，可用于选择下一项infra优化。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch
from anymani.distill.models.palm_rotation_policy import (
    PalmRotationActorCritic,
    PalmRotationActorObservation,
    PalmRotationCriticObservation,
    PalmRotationGeometry,
)
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function


def _arguments() -> argparse.Namespace:
    r"""解析microbatch、重复次数与可选trace输出位置。"""

    parser = argparse.ArgumentParser(description="Profile one synthetic MVP80 PPO activation slice.")
    parser.add_argument("--batch_size", type=int, default=2400, help="1280-env=2400; 2560-env=4800.")
    parser.add_argument("--warmups", type=int, default=2, help="Untimed CUDA warmups per measured path.")
    parser.add_argument("--repeats", type=int, default=5, help="CUDA-event samples per measured path.")
    parser.add_argument("--seed", type=int, default=20260903, help="Synthetic input seed.")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device used by the formal run.")
    parser.add_argument(
        "--history_encoder",
        choices=("tcn", "raw_stack"),
        default="tcn",
        help="Matched History30 architecture to profile.",
    )
    parser.add_argument("--tf32", action="store_true", help="Enable TF32 matmul/cuDNN only for this numeric-speed probe.")
    parser.add_argument("--output", type=Path, default=None, help="Optional atomic JSON result path.")
    parser.add_argument("--trace", type=Path, default=None, help="Optional one-step Kineto Chrome trace.")
    args = parser.parse_args()
    if args.batch_size < 80 or args.batch_size % 80 != 0:
        parser.error("--batch_size must be a positive multiple of the 80-asset support axis")
    if args.warmups < 1 or args.repeats < 1:
        parser.error("--warmups and --repeats must be positive")
    return args


def _synthetic_batch(
    batch_size: int,
    device: torch.device,
) -> tuple[PalmRotationActorObservation, PalmRotationCriticObservation, PalmRotationGeometry]:
    r"""构造80种cardinality循环重复的生产shape synthetic batch。

    Active DoF在$7\ldots16$之间循环，TIP数在3/4之间循环；每个microbatch恰好含相同数量的80个
    prototypes。图关系桶取$0\ldots8$，使one-hot graph bias走完整生产路径而非全零特例。

    Args:
        batch_size (int): activation slice样本数$M$。
        device (torch.device): 正式CUDA device。

    Returns:
        tuple: actor observation、critic observation与共享geometry tensors。
    """

    prototype = torch.arange(batch_size, device=device) % 80  # `[M]`，严格80资产等频
    slot = torch.arange(16, device=device).unsqueeze(0)  # `[1,16]` canonical JOINT slots
    active_count = 7 + prototype % 10  # `[M]`，覆盖7..16 active DoF
    joint_valid = slot < active_count.unsqueeze(1)  # bool`[M,16]`
    tip_slot = torch.arange(4, device=device).unsqueeze(0)  # `[1,4]` canonical TIP slots
    tip_valid = tip_slot < (3 + prototype % 2).unsqueeze(1)  # bool`[M,4]`，3/4 TIP
    owner_valid = torch.cat(
        (torch.ones(batch_size, 1, dtype=torch.bool, device=device), joint_valid, tip_valid),
        dim=1,
    )  # bool`[M,21]`，PALM+JOINT+TIP
    current = torch.randn(batch_size, 16, 5, device=device) * joint_valid.unsqueeze(-1)  # `[M,16,5]`
    history = torch.randn(batch_size, 30, 16, 5, device=device) * joint_valid[:, None, :, None]  # History30
    limits = torch.stack((-torch.ones(batch_size, 16, device=device), torch.ones(batch_size, 16, device=device)), dim=-1)
    limits = limits * joint_valid.unsqueeze(-1)  # normalized joint limits`[M,16,2]`
    owner_contact = torch.randint(0, 2, (batch_size, 21, 1), device=device).float()
    owner_contact = owner_contact * owner_valid.unsqueeze(-1)  # binary all-owner contact`[M,21,1]`
    actor = PalmRotationActorObservation(
        jnt_current=current,
        jnt_history=history,
        jnt_limits=limits,
        owner_contact=owner_contact,
        jnt_valid=joint_valid,
        tip_valid=tip_valid,
        owner_valid=owner_valid,
    )

    # Critic使用独立privileged state；随机值只激活完整计算图，不承担物理解释。
    critic = PalmRotationCriticObservation(
        jnt_state=torch.randn(batch_size, 16, 4, device=device) * joint_valid.unsqueeze(-1),
        owner_contact=torch.rand(batch_size, 21, 2, device=device) * owner_valid.unsqueeze(-1),
        obj=torch.randn(batch_size, 1, 15, device=device),
        task=torch.randn(batch_size, 1, 8, device=device),
        reward_release=torch.rand(batch_size, 1, device=device),
        jnt_valid=joint_valid,
        tip_valid=tip_valid,
        owner_valid=owner_valid,
    )
    graph = torch.randint(0, 9, (batch_size, 21, 21), dtype=torch.long, device=device)  # graph buckets`[M,21,21]`
    geometry = PalmRotationGeometry(
        tokens=torch.randn(batch_size, 21, 128, device=device),  # frozen-provider output$Z^e$
        owner_valid=owner_valid,
        shortest_path=graph,
        parent_direction=graph.roll(1, dims=-1),
        child_direction=graph.roll(1, dims=-2),
    )
    return actor, critic, geometry


def _zero_grad(parameters: Iterable[nn.Parameter]) -> None:
    r"""把被测参数梯度置None，匹配生产optimizer的``set_to_none=True``边界。"""

    for parameter in parameters:
        parameter.grad = None  # 不计入CUDA Event；测量聚焦forward/backward网络路径


def _measure(
    name: str,
    closure: Callable[[], tuple[torch.Tensor, ...]],
    parameters: Iterable[nn.Parameter],
    *,
    device: torch.device,
    warmups: int,
    repeats: int,
    backward: bool = True,
) -> dict[str, float | int | str]:
    r"""用CUDA Events测量一个closure的p50/p95与增量峰值显存。

    ``raw_stack``只是无参数view/mask，单独输出不具有autograd graph，因此其history-only条目明确计
    forward-only；完整actor条目仍对local MLP执行真实backward。
    """

    parameter_list = tuple(parameters)  # 多次warmup/repeat复用相同参数集合

    # Warmup覆盖cuBLAS/cuDNN选择与allocator初次扩容；每次都执行真实两段backward。
    for _ in range(warmups):
        _zero_grad(parameter_list)
        losses = closure()  # actor/critic各自返回一个scalar loss
        if backward:
            for loss in losses:
                loss.backward()  # 与生产完全分参的两次backward顺序一致
    torch.cuda.synchronize(device)

    baseline_allocated = int(torch.cuda.memory_allocated(device))  # persistent model+input footprint
    torch.cuda.reset_peak_memory_stats(device)
    timings: list[float] = []
    for _ in range(repeats):
        _zero_grad(parameter_list)
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()  # pyright: ignore[reportCallIssue]
        losses = closure()
        if backward:
            for loss in losses:
                loss.backward()
        stop.record()  # pyright: ignore[reportCallIssue]
        stop.synchronize()
        timings.append(float(start.elapsed_time(stop)))
    timing = torch.tensor(timings, dtype=torch.float64)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    return {
        "name": name,
        "timing_scope": "forward_backward" if backward else "forward_only",
        "p50_ms": float(torch.quantile(timing, 0.50).item()),
        "p95_ms": float(torch.quantile(timing, 0.95).item()),
        "minimum_ms": float(timing.min().item()),
        "maximum_ms": float(timing.max().item()),
        "baseline_allocated_bytes": baseline_allocated,
        "peak_allocated_bytes": peak_allocated,
        "incremental_peak_bytes": max(0, peak_allocated - baseline_allocated),
    }


def _profile_one_combined_step(
    closure: Callable[[], tuple[torch.Tensor, ...]],
    parameters: Iterable[nn.Parameter],
    *,
    device: torch.device,
    trace_path: Path,
) -> list[dict[str, float | str]]:
    r"""导出一次联合step的Kineto trace并返回self-device-time最高的20个operators。"""

    parameter_list = tuple(parameters)
    _zero_grad(parameter_list)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with profile(
        activities=(ProfilerActivity.CPU, ProfilerActivity.CUDA),
        record_shapes=True,
        profile_memory=True,
    ) as profiler:
        with record_function("mvp80_actor_critic_forward_backward"):
            losses = closure()
            for loss in losses:
                loss.backward()
        torch.cuda.synchronize(device)
    profiler.export_chrome_trace(str(trace_path))
    events = sorted(profiler.key_averages(), key=lambda event: event.self_device_time_total, reverse=True)
    return [
        {
            "operator": event.key,
            "self_device_time_ms": float(event.self_device_time_total / 1000.0),
            "device_time_ms": float(event.device_time_total / 1000.0),
            "calls": float(event.count),
        }
        for event in events[:20]
    ]


def main() -> dict[str, Any]:
    r"""运行四条matched CUDA路径并发布可选JSON/trace。"""

    args = _arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("palm-rotation update profile requires CUDA")
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)  # 默认保持正式FP32；显式flag只形成候选证据
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)  # TCN convolution与matmul精度开关共同记录
    package = PalmRotationActorCritic(
        residual_enabled=True,
        history_encoder=args.history_encoder,
    ).to(device).train()  # 当前v5 actor/critic，仅切换声明的History30路径
    actor_observation, critic_observation, geometry = _synthetic_batch(int(args.batch_size), device)

    # 每条closure返回独立scalar tuple；联合路径保留生产actor→critic与两次backward语义。
    def history_closure() -> tuple[torch.Tensor]:
        r"""返回逐JOINT TCN输出能量，激活完整history backward路径。"""

        history = package.actor.history_encoder(actor_observation.jnt_history, actor_observation.jnt_valid)
        return (history.square().mean(),)  # scalar synthetic objective

    def actor_closure() -> tuple[torch.Tensor]:
        r"""返回完整actor mean能量，覆盖TCN、FiLM、pooling与global residual。"""

        return (package.actor(actor_observation, geometry).mean.square().mean(),)  # scalar actor proxy

    def critic_closure() -> tuple[torch.Tensor]:
        r"""返回完整privileged graph critic value能量。"""

        return (package.critic(critic_observation, geometry).square().mean(),)  # scalar critic proxy

    def combined_closure() -> tuple[torch.Tensor, torch.Tensor]:
        r"""返回完全分参的actor/critic synthetic objectives。"""

        actor_loss = package.actor(actor_observation, geometry).mean.square().mean()  # scalar actor proxy
        critic_loss = package.critic(critic_observation, geometry).square().mean()  # scalar critic proxy
        return actor_loss, critic_loss

    measurements = [
        _measure(
            "history_tcn" if args.history_encoder == "tcn" else "history_raw_stack",
            history_closure,
            package.actor.history_encoder.parameters(),
            device=device,
            warmups=int(args.warmups),
            repeats=int(args.repeats),
            backward=args.history_encoder == "tcn",
        ),
        _measure(
            "full_actor_forward_backward",
            actor_closure,
            package.actor.parameters(),
            device=device,
            warmups=int(args.warmups),
            repeats=int(args.repeats),
        ),
        _measure(
            "full_critic_forward_backward",
            critic_closure,
            package.critic.parameters(),
            device=device,
            warmups=int(args.warmups),
            repeats=int(args.repeats),
        ),
        _measure(
            "combined_actor_critic_forward_backward",
            combined_closure,
            package.parameters(),
            device=device,
            warmups=int(args.warmups),
            repeats=int(args.repeats),
        ),
    ]
    result: dict[str, Any] = {
        "artifact_type": "anymani.palm_rotation_update_profile",
        "schema_version": "1.0.0",
        "scope": "synthetic-shape-only-not-learning-evidence",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "batch_size": int(args.batch_size),
        "history_encoder": str(args.history_encoder),
        "tf32": bool(args.tf32),
        "warmups": int(args.warmups),
        "repeats": int(args.repeats),
        "actor_parameters": sum(parameter.numel() for parameter in package.actor.parameters()),
        "critic_parameters": sum(parameter.numel() for parameter in package.critic.parameters()),
        "measurements": measurements,
    }
    if args.trace is not None:
        trace_path = args.trace.expanduser().resolve()
        result["trace"] = str(trace_path)
        result["top_device_operators"] = _profile_one_combined_step(
            combined_closure,
            package.parameters(),
            device=device,
            trace_path=trace_path,
        )
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(output_path)
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
