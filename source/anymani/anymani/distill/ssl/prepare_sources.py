r"""预构建 Geometry SSL 静态 source base 与 selected anchor shards。

默认覆盖 train 的 8 个 bank，以及 validation/evaluation 每条 suite 的 bank 0。该进程不构造模型、
optimizer、query 或 teacher target；cold preparation 时间与后续 warm training 墙钟分开记录。
"""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Sequence
from pathlib import Path

import torch
import yaml

from anymani.distill.ssl.config_store import compose_pretrain_cfg
from anymani.distill.ssl.contracts import build_runtime
from anymani.distill.ssl.experiments import DEFAULT_EXPERIMENT_NAME


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare AnyMani Geometry SSL source artifacts.")
    parser.add_argument("--config", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--source_cache_root", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--partition", action="append", default=[])
    parser.add_argument("--minimum_free_gib", type=float, default=10.0)
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    """解析完整实验，构建 source artifacts 并写 preparation summary。"""

    args = _build_parser().parse_args(argv)
    config = compose_pretrain_cfg(config_ref=args.config)
    root = Path(args.source_cache_root or config.run.source_cache_root).expanduser()
    device = torch.device(args.device or config.trainer.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"source preparation requires an available CUDA device, got {device}")
    root.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root.parent).free
    minimum = int(args.minimum_free_gib * 1024**3)
    if free < minimum:
        raise RuntimeError(f"source preparation requires at least {minimum} free bytes, found {free}")

    data = build_runtime(config.data)
    method = build_runtime(config.method)
    catalog = data.resolve()
    method.configure_source_artifacts(
        root=str(root),
        mode="read-write",
        dataset_manifest_sha256=str(catalog.dataset.source_sha256),
        producer_device=str(device),
    )
    try:
        method.prepare(catalog, device=device, dtype=torch.float32)
        summary = method.prepare_source_artifacts(
            device=device,
            dtype=torch.float32,
            partitions=tuple(args.partition),
        )
        summary_path = root / "preparation_summary.yaml"
        temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
        temporary.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
        temporary.replace(summary_path)
        print(summary_path)
        return summary_path
    finally:
        method.close()


if __name__ == "__main__":
    main()


__all__ = ["main"]
