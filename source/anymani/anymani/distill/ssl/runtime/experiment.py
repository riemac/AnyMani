r"""Geometry SSL 声明配置到唯一执行入口的 runtime 边界。"""

from __future__ import annotations

from pathlib import Path

from anymani.distill.ssl.config import GeometrySSLExperimentCfg


class GeometrySSLExperiment:
    r"""拥有一次 Geometry SSL run 生命周期，不在构造阶段执行 IO 或初始化 CUDA。"""

    def __init__(self, config: GeometrySSLExperimentCfg, *, output_dir: Path | None = None) -> None:
        r"""保存已验证配置与可选测试输出目录；实际副作用只允许发生在 :meth:`run`。"""

        self.config = config
        self.output_dir = output_dir

    def run(self) -> Path:
        r"""执行 resolve、materialize、calibration、train、validation、checkpoint 与 release。"""

        from .trainer import _run_geometry_ssl_lifecycle

        return _run_geometry_ssl_lifecycle(self.config, output_dir_override=self.output_dir)


__all__ = ["GeometrySSLExperiment"]
