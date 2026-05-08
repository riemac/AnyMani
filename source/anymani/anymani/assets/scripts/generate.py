r"""统一资产生成 runner。

本文件位于 `assets/` 子项目内部，刻意遵守 `assets/AGENTS.md` 里的
“自包含性”约束：资产生成相关的声明式配置、runner helper、CLI 入口
都收拢在 `AnyMani/source/anymani/anymani/assets/` 下，而不是散落到仓库根
目录的通用脚本层。

当前 runner 的科研语义是把“生成配置”和“执行编排”分开：

1. `assets/config/*.py`
   只声明 `HandGeneratorCfg`、路径常量和少量 runner 级策略；
2. `assets/scripts/generate.py`
   只负责 CLI 参数解析、配置模块装载、阶段分发与最外层校验；
3. `assets/scripts/_asset_generate_runner.py`
   负责 pre-made / post-mutate 两段的共用 helper。

这套分层对齐 Isaac Lab 的常见使用方式：

- 类似 `tasks/.../config/*.py` 的声明式配置模块；
- 类似 `scripts/train.py` 的执行入口脚本；
- 但又不把资产生产逻辑丢到 `assets/` 子项目外面，避免破坏局部自包含。
"""

from __future__ import annotations

import argparse
import importlib

from ..config import AssetRunStrategyCfg
from ..generator.hand_generator import HandGeneratorCfg
from ._asset_generate_runner import (
    enumerate_post_mutate_bundles,
    enumerate_premade_bundles,
    prepare_post_mutate_run_cfg,
    print_post_mutate_result_summary,
    print_post_mutate_summary,
    print_premade_registry_summary,
    print_premade_result_summary,
)


def _load_config_module(module_name: str):
    r"""按模块路径装载资产生成配置模块。

    这里显式保留“配置模块路径是字符串”的接口，而不是把配置对象写死进 runner，
    原因是科研调参常常需要：

    - 在不同实验文件之间切换；
    - 用 monkeypatch / notebook 临时覆盖模块常量；
    - 后续扩展为 hand-family 分组配置模块。

    Args:
        module_name (str): Python 模块路径，例如
            `anymani.assets.config.asset_gen_cfg`。

    Returns:
        module: 已导入的 Python 模块对象。
    """

    return importlib.import_module(module_name)  # 模块级配置在导入时完成常量实例化


def _build_parser() -> argparse.ArgumentParser:
    r"""构造统一 runner 的 CLI 解析器。

    当前 CLI 只暴露“阶段切换”和“少量高频覆盖项”，不把全部
    `HandGeneratorCfg` 字段平铺成命令行参数，原因是：

    - `HandGeneratorCfg` 仍是最高 façade；
    - 大多数研究配置更适合写在 Python 配置模块里；
    - CLI 只负责覆盖最常调整、且不会破坏整体结构假设的字段。

    Returns:
        argparse.ArgumentParser: 已填充参数定义的解析器。
    """

    parser = argparse.ArgumentParser(description="Unified asset generation runner.")
    parser.add_argument(
        "--stage",
        choices=("pre-made", "post-mutate"),
        required=True,
        help="Which asset generation stage to run.",
    )
    parser.add_argument(
        "--config-module",
        default="anymani.assets.config.asset_gen_cfg",
        help="Python module path containing PRE_MADE_CFG / POST_MUTATE_CFG.",
    )
    parser.add_argument("--source-path", default=None, help="Override post-mutate source topology path.")
    parser.add_argument("--n-samples", type=int, default=None, help="Override HandGeneratorCfg.n_samples.")
    parser.add_argument("--max-enumerate", type=int, default=None, help="Override pre-made max_enumerate.")
    return parser


def _validate_strategy(strategy: AssetRunStrategyCfg) -> None:
    r"""校验当前 runner 级策略占位字段。

    当前只允许 `topology_selection_mode="all"`，不是因为其它策略概念错误，
    而是因为用户明确要求这轮重构先把结构收拢、把未来策略占位声明出来，
    暂时不要把随机 topology 选择逻辑也一起实现。

    Args:
        strategy (AssetRunStrategyCfg): 配置模块中声明的 runner 级策略。

    Raises:
        NotImplementedError: 当用户启用了尚未实现的策略字段时抛出。
    """

    # NOTE:
    # 这里采用“显式拒绝未实现策略”的方式，而不是悄悄忽略字段，
    # 避免未来你在配置里写了随机拓扑采样，却被 runner 静默当成 `all`。
    if strategy.topology_selection_mode != "all":
        raise NotImplementedError(
            "Runner strategy extensions are declared but not implemented yet; "
            "current runner only supports topology_selection_mode='all'."
        )

    # NOTE:
    # `topology_selection_count` 只有在未来随机子集策略里才有语义，
    # 因此当前要求它必须保持 `None`，避免形成伪配置接口。
    if strategy.topology_selection_count is not None:
        raise NotImplementedError(
            "Runner strategy extensions are declared but not implemented yet; "
            "topology_selection_count must stay None for now."
        )


def _run_premade(module, *, max_enumerate: int | None) -> int:
    r"""执行 pre-made 阶段。

    Args:
        module: 已导入的配置模块；要求至少暴露 `PRE_MADE_CFG`。
        max_enumerate (int | None): CLI 临时覆盖的 pre-made 枚举上限。

    Returns:
        int: 进程退出码；成功时返回 `0`。
    """

    cfg: HandGeneratorCfg = module.PRE_MADE_CFG  # 最高 façade 仍然是正式 `HandGeneratorCfg`

    # CLI 覆盖只产生一个临时副本，不回写配置模块，避免污染 notebook / 测试上下文。
    if max_enumerate is not None:
        cfg = cfg.replace(max_enumerate=max_enumerate)  # 只覆盖离散枚举预算，不动其它科研语义

    # registry summary 对调 connectivity preset 很有用，因此保留成显式可开关的 runner 行为。
    if getattr(module, "PRE_MADE_SHOW_REGISTRY", False):
        print_premade_registry_summary(cfg)  # 打印 finger-level recipe registry，便于人工核对

    # 真正的 pre-made 枚举仍由 `HandGenerator(mode="made")` 正式路径完成。
    results = enumerate_premade_bundles(cfg)  # 返回 bundle / urdf / hand_cfg 结果包列表
    print_premade_result_summary(
        results,
        cfg,
        print_limit=getattr(module, "PRE_MADE_PRINT_RESULT_LIMIT", None),
    )
    return 0


def _run_post_mutate(
    module,
    *,
    source_path: str | None,
    n_samples: int | None,
) -> int:
    r"""执行独立 post-mutate 阶段。

    Args:
        module: 已导入的配置模块；要求暴露 `POST_MUTATE_CFG` 与来源路径常量。
        source_path (str | None): CLI 临时覆盖的 pre-made topology 根路径。
        n_samples (int | None): CLI 临时覆盖的 Monte Carlo 采样数。

    Returns:
        int: 进程退出码；成功时返回 `0`。
    """

    cfg: HandGeneratorCfg = module.POST_MUTATE_CFG  # mutate-only 入口仍然只接受正式 `HandGeneratorCfg`

    # 采样预算属于最常见的实验 override，因此允许用 CLI 覆盖；其它复杂 term 仍回到 Python cfg。
    if n_samples is not None:
        cfg = cfg.replace(n_samples=n_samples)  # 只覆盖后变异样本数，不重写 term container

    # 独立 post-mutate 现在只接受 topology 根；run 时间戳由 HandGenerator 运行时自动生成。
    resolved_source_path = source_path or module.POST_MUTATE_SOURCE_TOPOLOGY_PATH

    # helper 会把“用户直观填写的 topology 路径”lower 成正式 mutate-only 所需的 topology-root cfg。
    prepared_cfg, source_topology_dir, planned_run_dir = prepare_post_mutate_run_cfg(
        cfg,
        source_path=resolved_source_path,
    )

    # 先打印 summary，再真正执行 mutate，便于人工在出错前先看到来源 topology 与本轮计划 run 根。
    print_post_mutate_summary(
        prepared_cfg,
        source_path=resolved_source_path,
        source_topology_dir=source_topology_dir,
        planned_run_dir=planned_run_dir,
    )

    # mutate-only 正式执行路径仍收口到 `HandGenerator(mode="mutate")`。
    results = enumerate_post_mutate_bundles(prepared_cfg)  # 基于同一个 source topology 生成一批后变异样本
    print_post_mutate_result_summary(
        results,
        print_limit=getattr(module, "POST_MUTATE_PRINT_RESULT_LIMIT", None),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    r"""统一资产生成 runner 的主入口。

    Args:
        argv (list[str] | None): 可选的命令行参数列表。为 `None` 时使用
            `argparse` 的默认行为，即读取进程级 `sys.argv`。

    Returns:
        int: 退出码；成功时返回 `0`。
    """

    parser = _build_parser()  # 先构造 CLI 语义，再解析当前运行请求
    args = parser.parse_args(argv)  # `argv=None` 时自动读取当前进程命令行

    # 配置模块在这里才动态装载，避免 `--help` 这类只看参数的人机交互也强依赖完整配置导入成功。
    module = _load_config_module(args.config_module)  # 支持科研上切换不同配置模块
    strategy = getattr(module, "ASSET_RUN_STRATEGY", AssetRunStrategyCfg())  # 未显式声明时回退到保守默认策略
    _validate_strategy(strategy)  # 先拒绝未实现策略，避免静默跑错实验

    # pre-made 与 post-mutate 的运行语义已经分离，因此这里做显式阶段分发。
    if args.stage == "pre-made":
        return _run_premade(module, max_enumerate=args.max_enumerate)  # 离散枚举型生成

    return _run_post_mutate(
        module,
        source_path=args.source_path,
        n_samples=args.n_samples,
    )


if __name__ == "__main__":
    raise SystemExit(main())  # CLI 进程入口：把返回码交给 shell
