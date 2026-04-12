r"""批量生成运行器：从 YAML recipe 到资产产物的端到端入口。

本模块是整个生成管线对外最薄的一层，它的职责只有一件：

    加载 YAML recipe → 构建 HandGenerator → 运行 generate_batch → 按需补落盘

用户可以通过命令行直接调用（待 CLI 入口实现后），也可以在 Python 脚本中
实例化 `GeneratorRunner` 做更细粒度的控制（如接入 progress bar、分批消费等）。

设计说明
--------

### 为什么需要 runner

`HandGenerator.generate_batch()` 是一个迭代器，本身不关心 recipe 从哪里来、
也不关心 `artifact_level="hand_cfg"` 时是否还要给调试树和 sidecar 留磁盘痕迹。
`GeneratorRunner` 把这些“运行期关切”收拢到一处，让 `HandGenerator`
保持纯粹的生成语义。

### 当前与 `assets/generated` 的关系

若调用方没有额外覆盖 `output_dir`，真正的默认落盘目录仍由 `HandGeneratorCfg`
控制，也就是当前项目约定的 `assets/generated/`。runner 只负责：

1. 接受 `cfg/dict/path` 三种入口
2. 可选覆写 `output_dir`
3. 在 `hand_cfg` 轻量模式下补一份 sidecar + tree 文件，方便人工查看
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from ..exporter.sidecar import SidecarCfg, SidecarExporter
from ..generator.hand_generator import HandGenerationResult, HandGenerator, HandGeneratorCfg
from .recipe_loader import RecipeLoader


# ============================================================================
#  GeneratorRunner
# ============================================================================


class GeneratorRunner:
    r"""批量生成运行器。

    从 YAML recipe（或 Python dict / `HandGeneratorCfg` 对象）加载配置，
    驱动 `HandGenerator` 批量生成资产，并在需要时补齐轻量落盘语义。
    """

    cfg: HandGeneratorCfg
    output_dir: Path | None

    def __init__(
        self,
        cfg: HandGeneratorCfg | dict[str, Any] | str | Path,
        output_dir: str | Path | None = None,
    ):
        r"""初始化运行器。

        Args:
            cfg (HandGeneratorCfg | dict | str | Path):
                生成器配置，支持三种形式：

                - ``HandGeneratorCfg`` 对象：直接使用；
                - ``dict``：通过 ``RecipeLoader.load_dict()`` 解析；
                - ``str`` / ``Path``：视为 YAML 文件路径，通过 ``RecipeLoader.load()`` 解析。

            output_dir (str | Path | None):
                可选的产物落盘目录覆写。若不为 ``None``，会同步覆盖到
                `HandGeneratorCfg.output_dir`，确保 generator / exporter / runner
                三层对目录语义保持一致。
        """

        if isinstance(cfg, HandGeneratorCfg):
            self.cfg = cfg
        elif isinstance(cfg, dict):
            self.cfg = RecipeLoader.load_dict(cfg)
        else:
            self.cfg = RecipeLoader.load(cfg)

        self.output_dir = Path(output_dir) if output_dir is not None else None
        if self.output_dir is not None:
            self.cfg = self.cfg.replace(output_dir=self.output_dir)

    def run(self) -> list[HandGenerationResult]:
        r"""运行批量生成并返回所有结果。

        Returns:
            list[HandGenerationResult]: 所有成功生成的结果列表。
        """

        return list(self.stream())

    def stream(self) -> Iterator[HandGenerationResult]:
        r"""以迭代器方式运行批量生成（lazy 消费）。

        Yields:
            HandGenerationResult: 每次成功生成的结果。
        """

        generator = HandGenerator(self.cfg)
        for result in generator.generate_batch():
            # `bundle/urdf` 模式已经由 HandExporter 完成落盘；只有 `hand_cfg`
            # 轻量模式需要 runner 在这里补树文件和 sidecar，方便人工巡检。
            if self.output_dir is not None and self.cfg.artifact_level == "hand_cfg":
                self._save_result(result)
            yield result

    def _save_result(self, result: HandGenerationResult) -> None:
        r"""把单个生成结果落盘到 output_dir。

        这里只补 `artifact_level="hand_cfg"` 时 generator 没有落盘的那部分：

        - `tree.txt`
        - `tree.mmd`
        - `hand.yaml`（轻量 sidecar）
        """

        if self.output_dir is None or result.hand_cfg is None:
            return

        sample_id = str(result.metadata.get("id") or uuid4().hex[:8])
        result.metadata.setdefault("id", sample_id)
        out_dir = self.output_dir / sample_id
        out_dir.mkdir(parents=True, exist_ok=True)

        result.render_trees()
        if result.tree_txt is not None:
            (out_dir / "tree.txt").write_text(result.tree_txt, encoding="utf-8")
        if result.tree_mermaid is not None:
            (out_dir / "tree.mmd").write_text(result.tree_mermaid, encoding="utf-8")

        # 这里复用 Export 配置中的 Sidecar 子配置，而不是偷偷写死一份默认值，
        # 避免 experiment_tag / overwrite 等导出语义在 runner 层被吃掉。
        sidecar_cfg = self.cfg.Export.Sidecar if hasattr(self.cfg.Export, "Sidecar") else SidecarCfg()
        sidecar_result = SidecarExporter(sidecar_cfg).export(
            result.hand_cfg,
            out_dir,
            extra=result.metadata,
        )
        if sidecar_result.written:
            result.sidecar_path = sidecar_result.written[0]


__all__ = ["GeneratorRunner"]
