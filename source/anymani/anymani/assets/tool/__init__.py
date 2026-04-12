r"""手部资产生成工具层：YAML recipe 解析与批量运行入口。

本子包的定位是"面向用户的使用层"，解决 `HandGeneratorCfg` 直接用 Python
构造时的繁琐问题，让用户既可以用 YAML 文件驱动完整生成流程，也可以把
YAML 作为基础配置后在 Python 中做局部字段覆盖（混合使用）。

模块分工
--------

- ``recipe_loader`` → YAML ↔ HandGeneratorCfg 双向转换，提供 schema 校验
- ``runner``        → 从 YAML 路径到批量生成结果的端到端入口

使用示例
--------

纯 YAML 驱动::

    # 直接在 shell 中运行（待 runner 实现后支持）
    python -m anymani.assets.tool.runner --recipe leap_variant.yaml

纯 Python 驱动::

    from anymani.assets.generator import HandGeneratorCfg, HandGenerator
    cfg = HandGeneratorCfg(sampling_strategy="sample", n_samples=100)
    gen = HandGenerator(cfg)

混合使用（YAML 作为基础，Python 覆盖）::

    from anymani.assets.tool import RecipeLoader
    cfg = RecipeLoader.load("leap_variant.yaml")
    cfg.n_samples = 50  # Python 侧覆盖
    gen = HandGenerator(cfg)
"""

from .recipe_loader import RecipeLoader
from .runner import GeneratorRunner

__all__ = ["RecipeLoader", "GeneratorRunner"]
