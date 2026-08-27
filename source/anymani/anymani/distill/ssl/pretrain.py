r"""Schema 8 embodiment pure-pretraining 的普通命令行入口。

Python preset 定义方法、表示、损失和默认训练数值；shell 使用 ``--flag value`` 声明本次运行。
入口把显式 flags 转成内部 Hydra overrides，再恢复完整冻结配置。Hydra 字段路径不暴露给日常运行命令。

运行入口：``python -m anymani.distill.ssl.pretrain``。
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from anymani.distill.ssl.config_store import compose_pretrain_cfg
from anymani.distill.ssl.experiment import EmbodimentPretrain
from anymani.distill.ssl.experiments import DEFAULT_EXPERIMENT_NAME, load_experiment


def _build_parser() -> argparse.ArgumentParser:
    r"""构造面向一次预实验或正式训练的平坦 CLI。

    所有参数默认 ``None``，表示沿用 Python preset。运行者只需写出本次实验主动改变的轴；
    parser 不复制方法、representation 或 objective 的深层结构。
    """

    parser = argparse.ArgumentParser(description="Run AnyMani embodiment geometry pretraining.")
    parser.add_argument(
        "--config",
        default=DEFAULT_EXPERIMENT_NAME,
        help="registered experiment name or path to a Python snapshot exporting EXPERIMENT",
    )
    # 运行身份决定产物位置；teacher baseline 在本次训练内累计，不再有 calibration phase。
    parser.add_argument("--run_name", "--experiment_name", dest="experiment_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--source_cache_root", type=str, default=None)
    parser.add_argument("--source_cache_mode", choices=("auto", "readonly", "read-write", "off"), default=None)
    # 显式数据预算直接决定本次生成多少资产/q 样本以及循环利用几次。
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--num_minibatches", type=int, default=None)
    parser.add_argument("--assets_per_minibatch", type=int, default=None)
    parser.add_argument("--q_per_asset_per_minibatch", type=int, default=None)
    parser.add_argument("--mini_epochs", type=int, default=None)
    parser.add_argument("--microbatch_size", type=int, default=None)
    parser.add_argument("--max_resident_assets", type=int, default=None)
    # optimizer 与运行 cadence 属于一次执行，可从 shell 明确覆盖 preset。
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--max_gradient_norm_per_group", type=float, default=None)
    parser.add_argument("--checkpoint_every_epochs", type=int, default=None)
    # ``--seed`` 是用户看到的统一根 seed；sampling_seed 只服务显式消融。
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sampling_seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--shuffle_assets", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--deterministic_algorithms", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resource_profile", action=argparse.BooleanOptionalAction, default=None)
    return parser


def _config_overrides(args: argparse.Namespace) -> tuple[str, ...]:
    r"""把平坦运行参数映射到内部 structured-config 路径。

    Returns:
        tuple[str, ...]: 交给 ``compose_pretrain_cfg`` 的 Hydra overrides。顺序保持稳定，
            ``sampling_seed`` 位于统一 ``seed`` 之后，因此可主动覆盖训练采样随机域。
    """

    field_paths = (
        ("experiment_name", "run.experiment_name"),
        ("output_dir", "run.output_dir"),
        ("resume_checkpoint", "run.resume_checkpoint"),
        ("source_cache_root", "run.source_cache_root"),
        ("source_cache_mode", "run.source_cache_mode"),
        ("max_epochs", "trainer.max_epochs"),
        ("num_minibatches", "trainer.num_minibatches"),
        ("assets_per_minibatch", "trainer.sampling.assets_per_minibatch"),
        ("q_per_asset_per_minibatch", "trainer.sampling.q_per_asset_per_minibatch"),
        ("mini_epochs", "trainer.mini_epochs"),
        ("microbatch_size", "trainer.microbatch_size"),
        ("max_resident_assets", "trainer.max_resident_assets"),
        ("learning_rate", "trainer.optimizer.learning_rate"),
        ("weight_decay", "trainer.optimizer.weight_decay"),
        ("max_gradient_norm_per_group", "trainer.max_gradient_norm_per_group"),
        ("checkpoint_every_epochs", "trainer.checkpoint_every_epochs"),
        ("device", "trainer.device"),
        ("shuffle_assets", "trainer.sampling.shuffle_assets"),
        ("deterministic_algorithms", "run.deterministic_algorithms"),
        ("resource_profile", "trainer.resource_profile"),
    )
    overrides = [
        f"{path}={getattr(args, field)}"
        for field, path in field_paths
        if getattr(args, field) is not None
    ]
    # 一个 shell seed 同时锁定模型/评估与训练资产/q 顺序，符合单次实验的直观复现语义。
    if args.seed is not None:
        overrides.extend((f"run.seed={args.seed}", f"trainer.sampling.seed={args.seed}"))
    if args.sampling_seed is not None:
        overrides.append(f"trainer.sampling.seed={args.sampling_seed}")
    return tuple(overrides)


def main(argv: Sequence[str] | None = None) -> Path:
    r"""解析平坦 CLI、恢复完整 preset、执行一次生命周期并打印产物目录。

    Args:
        argv (Sequence[str] | None): 测试可传入显式参数；``None`` 读取进程级 ``sys.argv``。

    Returns:
        Path: 本次运行唯一 artifact 根目录。
    """

    args = _build_parser().parse_args(argv)
    preset = load_experiment(args.config)
    config = compose_pretrain_cfg(_config_overrides(args), config_ref=args.config)
    config.validate_composed()
    output_dir = EmbodimentPretrain(
        config,
        config_identity={
            "name": preset.name,
            "module": preset.module_name,
            "path": str(preset.path),
            "sha256": preset.config_sha256,
        },
    ).run()
    print(output_dir)  # shell/调度器唯一 stdout 结果：artifact root
    return output_dir


if __name__ == "__main__":  # ``python -m anymani.distill.ssl.pretrain``
    main()


__all__ = ["main"]
