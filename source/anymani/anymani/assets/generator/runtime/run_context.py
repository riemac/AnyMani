"""`HandGenerator` 的 run 生命周期与 summary 维护。

这个模块只处理“同一轮生成运行”的公共状态，而不处理具体的 hand 生成逻辑。

职责边界固定为：

1. 分配 run 根目录；
2. 初始化 run-level `summary.yaml` 文档；
3. 记录 success / rejection 统计；
4. 把内存中的 summary 刷回磁盘。

这样 `hand_generator.py` 就不必继续混着写：

- 目录分配策略
- summary 数据结构
- 结果统计
- 具体的 build / mutate / export 流水线
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from ..result import HandGenerationResult


@dataclass
class GenerationRunContext:
    r"""一次 `HandGenerator` 运行的共享状态壳。

    Attributes:
        root_dir: 当前 run 的根目录。
        summary: 当前 run 的 summary 文档内存态。
        last_rejection_stage: 最近一次被拒绝的阶段名；成功样本后会清空。
        last_rejection_error_codes: 最近一次拒绝命中的稳定规则代码集合。
    """

    root_dir: Path
    summary: dict[str, Any]
    last_rejection_stage: str | None = None
    last_rejection_error_codes: tuple[str, ...] = ()

    @classmethod
    def create(
        cls,
        cfg: Any,
        *,
        config_dump: dict[str, Any],
    ) -> GenerationRunContext:
        r"""按当前 `HandGeneratorCfg` 分配 run 根目录并初始化 summary。

        Args:
            cfg (Any): 生成器配置。
            config_dump (dict[str, Any]): 已序列化的 cfg 快照。

        Returns:
            GenerationRunContext: 已完成目录分配与 summary 初始化的运行态对象。
        """

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # 所有 run 都统一使用同一时间戳格式
        run_root = _allocate_run_root(cfg, timestamp=timestamp)  # run 根分配与 mode 相关，但与单样本流水线无关
        pre_made_enabled = cfg.mode == "made"  # 当前迁移后只剩 made / mutate 两条正式路径
        post_mutate_enabled = cfg.mode == "mutate" and bool(cfg.Mutate.has_terms())
        summary = {
            "run": {
                "timestamp": timestamp,
                "root_dir": str(run_root),
                "mode": cfg.mode,
                "artifact_level": cfg.artifact_level,
                "phases": {
                    "pre_made": pre_made_enabled,
                    "post_mutate": post_mutate_enabled,
                    "combined": False,
                },
            },
            "config": config_dump,
            "stats": {
                "attempted": 0,
                "succeeded": 0,
                "rejected": 0,
                "rejected_by_stage": {},
                "rejected_by_reason": {},
                "by_topology": {},
            },
        }
        context = cls(root_dir=run_root, summary=summary)
        context.write_summary()  # run 建立后立刻落一份 summary，便于中途失败时也能看见目录语义
        return context

    def write_summary(self) -> None:
        r"""把当前 summary 刷到 `<run_root>/summary.yaml`。"""

        stats = self.summary["stats"]
        stats["topology_count"] = len(stats["by_topology"])  # 派生字段在写回时统一刷新，避免多处重复维护
        summary_path = self.root_dir / "summary.yaml"
        summary_path.write_text(
            yaml.safe_dump(self.summary, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def record_rejection(
        self,
        *,
        stage: str,
        error_codes: tuple[str, ...] = (),
        write_summary: bool = True,
    ) -> None:
        r"""记录一次被拒绝的样本尝试及其稳定原因组合。

        ``rejected_by_reason`` 统计的是 canonical 原因集合，而不是单个规则命中次数。
        因此一个样本同时违反两条规则时只计入一个 ``code_a+code_b`` 键，所有键的
        count 之和始终等于 ``stats.rejected``。
        """

        self.last_rejection_stage = stage
        self.last_rejection_error_codes = canonical_rejection_error_codes(error_codes)
        stats = self.summary["stats"]
        stats["attempted"] += 1
        stats["rejected"] += 1
        rejected_by_stage = dict(stats.get("rejected_by_stage") or {})
        rejected_by_stage[stage] = int(rejected_by_stage.get(stage, 0)) + 1
        stats["rejected_by_stage"] = rejected_by_stage
        reason_key = rejection_reason_key(self.last_rejection_error_codes)
        rejected_by_reason = dict(stats.get("rejected_by_reason") or {})
        rejected_by_reason[reason_key] = int(rejected_by_reason.get(reason_key, 0)) + 1
        stats["rejected_by_reason"] = rejected_by_reason
        if write_summary:
            self.write_summary()

    def record_success(self, result: HandGenerationResult, *, write_summary: bool = True) -> None:
        r"""记录一次成功样本。"""

        self.last_rejection_stage = None
        self.last_rejection_error_codes = ()
        stats = self.summary["stats"]
        stats["attempted"] += 1
        stats["succeeded"] += 1
        topology_key = result_topology_key(result)
        by_topology = dict(stats.get("by_topology") or {})
        by_topology[topology_key] = int(by_topology.get(topology_key, 0)) + 1
        stats["by_topology"] = by_topology
        if write_summary:
            self.write_summary()


def result_topology_key(result: HandGenerationResult) -> str:
    r"""把单个结果映射成 summary 里的 topology 路径键。"""

    topology_name = str(result.metadata.get("topology_name") or result.metadata.get("family") or "unknown_topology")
    topology_group_name = str(
        result.metadata.get("topology_group_name")
        or result.metadata.get("base_hand_preset")
        or result.metadata.get("family")
        or "ungrouped"
    )
    topology_kind = str(result.metadata.get("topology_kind") or "single_family")
    if topology_kind == "mixed":
        return f"mixed/{topology_group_name}/{topology_name}"
    return f"{topology_group_name}/{topology_name}"


def canonical_rejection_error_codes(error_codes: tuple[str, ...]) -> tuple[str, ...]:
    r"""把一次拒绝的规则代码规范成排序、去重后的稳定集合。"""

    normalized = {str(code).strip() for code in error_codes if str(code).strip()}
    return tuple(sorted(normalized))


def rejection_reason_key(error_codes: tuple[str, ...]) -> str:
    r"""把稳定规则代码集合编码成 summary 中可读、可加和的原因键。"""

    canonical_codes = canonical_rejection_error_codes(error_codes)
    return "+".join(canonical_codes) if canonical_codes else "unclassified"


def _allocate_run_root(cfg: Any, *, timestamp: str) -> Path:
    r"""为 made / mutate 两条正式路径分配 run 根目录。"""

    if cfg.mode == "full":
        raise NotImplementedError(
            "mode='full' is temporarily unsupported. "
            "This migration only covers mode='made' and independent mode='mutate'; "
            "the full pipeline has not been adapted to topology-root export semantics yet."
        )

    if cfg.mode == "mutate":
        if cfg.source_topology_dir is None:
            raise ValueError("mode='mutate' requires 'source_topology_dir'")
        base_root = Path(cfg.source_topology_dir)  # mutate-only 的 run 根追加在来源 topology 根下
    else:
        base_root = Path(cfg.output_dir)  # pre-made 的 run 根仍追加在统一 output_dir 下

    run_root = base_root / timestamp
    collision_index = 2
    while run_root.exists():
        run_root = base_root / f"{timestamp}_{collision_index:02d}"  # 同秒重跑时继续追加后缀
        collision_index += 1

    run_root.mkdir(parents=True, exist_ok=False)
    return run_root


__all__ = [
    "GenerationRunContext",
    "canonical_rejection_error_codes",
    "rejection_reason_key",
    "result_topology_key",
]
