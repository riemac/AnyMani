r"""generator 运行时子系统。

本包故意保持轻量，不在 `__init__` 阶段主动导入子模块。
原因是 `recipe_loader.py` 需要反向引用 `HandGeneratorCfg`，若在包初始化时抢先导入，
就会把 `hand_generator.py` 和 `runtime/` 拉进循环依赖。

使用方应显式导入需要的子模块，例如：

- `assets.generator.runtime.recipe_loader`
- `assets.generator.runtime.restore`
- `assets.generator.runtime.run_context`
- `assets.generator.runtime.mutate_quota`
"""

__all__: list[str] = []
