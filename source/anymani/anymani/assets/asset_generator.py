'''TODO:用于生成形态各异的多元化手型资产，并确保物理合理性

该文件负责：
- 设计资产生成算法，并确保物理合理性
- 生成 HandCfg 对象，并确保物理合理性
- 生成 URDF 文件、yaml等配置文件（从HandCfg）导出，并确保物理合理性

宏观上，预设 
- Builder: HandCfg 构建器,亦分不同的层级构建，如关节级、手指级、掌级、手级
- Exporter: 从 HandCfg 导出 URDF 文件、yaml等配置文件;从 Joint-level, Finger-level/Palm-level, Hand-level 导出自包含 URDF进行快速检验（可选）
- Validator: 规则验证器。预设人工判断经验与规则，确保所生成手型的物理合理性
- AssetGenerator: 规模化资产生成器。集合了 Builder，Exporter 和 Validator
- AssetRandomizer(可选，先不实现): 资产随机化器。对urdf / HandCfg 进行随机化处理，增加多样性
'''
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import MISSING, Field, dataclass, field, fields, is_dataclass, replace
import math
from pathlib import Path
from typing import Any, ClassVar, Literal, cast, overload

from anymani.assets.asset_base import AssetCfgBase, JointCfg, PalmCfg, FingerCfg, HandCfg

# =======================
# Builder 资产构建逻辑与算法
# =======================
r''' TODO:计划采用 “声明式配置对象 + 运行时对象“ 的设计模式，来实现资产构建的灵活性和可扩展性
依然分 joint-level, finger-level/palm-level, hand-level 三个层级的构建器，但每个层级的构建器都包含两部分：
- 声明式配置对象：包含生成算法所需的参数
- 运行时对象：包含生成算法的实现逻辑，接受声明式配置对象作为输入，并输出 HandCfg 对象 
'''

# --- 配置类 --- #

@dataclass
class BuilderCfg:
    """构建器配置类"""

    class_type: type[Builder] = MISSING
    """关联的构建器类"""

    pass


@dataclass
class JointBuilderCfg(BuilderCfg):
    """关节级构建器配置"""

    class_type: type[JointBuilder] = MISSING
    """关联的关节级构建器类"""

    pass


@dataclass
class FingerBuilderCfg(BuilderCfg):
    """手指级构建器配置"""

    class_type: type[FingerBuilder] = MISSING
    """关联的手指级构建器类"""

    pass


@dataclass
class PalmBuilderCfg(BuilderCfg):
    """掌级构建器配置"""

    class_type: type[PalmBuilder] = MISSING
    """关联的掌级构建器类"""

    pass


@dataclass
class HandBuilderCfg(BuilderCfg):
    """手级构建器配置"""

    class_type: type[HandBuilder] = MISSING
    """关联的手级构建器类"""

    pass

# --- 运行时类 --- #
# 构建的 "算法" 位于该层

class Builder:
    """构建器基类"""

    def __init__(self, cfg: BuilderCfg):
        self.cfg = cfg

    def build(self) -> AssetCfgBase:
        """构建 HandCfg 对象"""
        raise NotImplementedError


class JointBuilder(Builder):
    """关节级构建器"""

    def __init__(self, cfg: JointBuilderCfg):
        super().__init__(cfg)

    def build(self) -> JointCfg:
        """构建 JointCfg 对象"""
        raise NotImplementedError


class FingerBuilder(Builder):
    """手指级构建器"""

    def __init__(self, cfg: FingerBuilderCfg):
        super().__init__(cfg)

    def build(self) -> FingerCfg:
        """构建 FingerCfg 对象"""
        raise NotImplementedError


class PalmBuilder(Builder):
    """掌级构建器"""

    def __init__(self, cfg: PalmBuilderCfg):
        super().__init__(cfg)

    def build(self) -> PalmCfg:
        """构建 PalmCfg 对象"""
        raise NotImplementedError


class HandBuilder(Builder):
    """手级构建器"""

    def __init__(self, cfg: HandBuilderCfg):
        super().__init__(cfg)

    def build(self) -> HandCfg:
        """构建 HandCfg 对象"""
        raise NotImplementedError

# custom 设定。自己构建一组人为的形状，暴露一组可调整的参数

# allegro类，leap类和mixed类；
# allegro和leap类的主要区别在与是掌根关节还是从掌根开始的第二个关节构成“侧摆”的运动，其余的关节则是向掌心的收敛-展开运动
# 次要区别在于mesh类别的差异，以及关节旋转轴、关节限位，还有掌型的细小差别
# mixed类手则允许同时拥有allegro-like和leap-like的手指

# =======================
# Validator 规则验证器
# =======================
r''' TODO:计划采用 “声明式配置对象 + 运行时对象“ 的设计模式，来实现资产验证的灵活性和可扩展性
依然分 joint-level, finger-level/palm-level, hand-level 三个层级的验证器，但每个层级的验证器都包含两部分：
- 声明式配置对象：包含验证算法所需的参数
- 运行时对象：包含验证算法的实现逻辑，接受声明式配置对象作为输入，并输出 HandCfg 对象 
'''

# --- 配置类 --- #

@dataclass
class ValidatorCfg:
    """验证器配置类"""

    class_type: type[Validator] = MISSING
    """关联的验证器类"""

    pass


@dataclass
class JointValidatorCfg(ValidatorCfg):
    """关节级验证器配置"""

    class_type: type[JointValidator] = MISSING
    """关联的关节级验证器类"""

    pass


@dataclass
class FingerValidatorCfg(ValidatorCfg):
    """手指级验证器配置"""

    class_type: type[FingerValidator] = MISSING
    """关联的手指级验证器类"""

    pass


@dataclass
class PalmValidatorCfg(ValidatorCfg):
    """掌级验证器配置"""

    class_type: type[PalmValidator] = MISSING
    """关联的掌级验证器类"""

    pass


@dataclass
class HandValidatorCfg(ValidatorCfg):
    """手级验证器配置"""

    class_type: type[HandValidator] = MISSING
    """关联的手级验证器类"""

# --- 运行时类 --- #
# 验证的 "算法" 位于该层

class Validator:
    """验证器基类"""

    def __init__(self, cfg: ValidatorCfg):
        self.cfg = cfg

    def validate(self) -> None:
        """验证 HandCfg 对象"""
        raise NotImplementedError


class JointValidator(Validator):
    """关节级验证器"""

    def __init__(self, cfg: JointValidatorCfg):
        super().__init__(cfg)

    def validate(self) -> None:
        """验证 JointCfg 对象"""
        raise NotImplementedError


class FingerValidator(Validator):
    """手指级验证器"""

    def __init__(self, cfg: FingerValidatorCfg):
        super().__init__(cfg)

    def validate(self) -> None:
        """验证 FingerCfg 对象"""
        raise NotImplementedError


class PalmValidator(Validator):
    """掌级验证器"""

    def __init__(self, cfg: PalmValidatorCfg):
        super().__init__(cfg)

    def validate(self) -> None:
        """验证 PalmCfg 对象"""
        raise NotImplementedError


class HandValidator(Validator):
    """手级验证器"""

    def __init__(self, cfg: HandValidatorCfg):
        super().__init__(cfg)

    def validate(self) -> None:
        """验证 HandCfg 对象"""
        raise NotImplementedError


# =======================
# Exporter 资产导出器
# =======================
r''' TODO:计划采用单独类直接导出
依然分 joint-level, finger-level/palm-level, hand-level 三个层级的导出器
'''

class Exporter:
    """导出器基类"""

    def __init__(self):
        pass

    def export(self) -> None:
        """导出 HandCfg 对象为 URDF 文件、yaml等配置文件、附带资产（例如自定义的 mesh）"""
        raise NotImplementedError
    

class JointExporter(Exporter):
    """关节级导出器"""

    def __init__(self):
        super().__init__()

    def export(self) -> None:
        """导出 JointCfg 对象为 URDF 文件、yaml等配置文件、附带资产（例如自定义的 mesh）"""
        raise NotImplementedError
    

class FingerExporter(Exporter):
    """手指级导出器"""

    def __init__(self):
        super().__init__()

    def export(self) -> None:
        """导出 FingerCfg 对象为 URDF 文件、yaml等配置文件、附带资产（例如自定义的 mesh）"""
        raise NotImplementedError
    

class PalmExporter(Exporter):
    """掌级导出器"""

    def __init__(self):
        super().__init__()

    def export(self) -> None:
        """导出 PalmCfg 对象为 URDF 文件、yaml等配置文件、附带资产（例如自定义的 mesh）"""
        raise NotImplementedError
    

class HandExporter(Exporter):
    """手级导出器"""

    def __init__(self):
        super().__init__()

    def export(self) -> None:
        """导出 HandCfg 对象为 URDF 文件、yaml等配置文件、附带资产（例如自定义的 mesh）"""
        raise NotImplementedError
    

# =======================
# AssetGenerator 资产生成器
# =======================
r''' TODO:计划采用 “声明式配置对象 + 运行时对象“ 的设计模式，来实现资产生成的灵活性和可扩展性
生成器包含两部分：
- 声明式配置对象：包含生成算法所需的参数
- 运行时对象：包含生成算法的实现逻辑，接受声明式配置对象作为输入，并输出一组 HandCfg 对象及 urdf 
'''

@dataclass
class AssetGeneratorCfg:
    """资产生成器配置类"""

    class_type: type[AssetGenerator] = MISSING
    """关联的资产生成器类"""

    Build: HandBuilderCfg = field(default_factory=HandBuilderCfg)

    

    pass


class AssetGenerator:
    """资产生成器"""

    def __init__(self, cfg: AssetGeneratorCfg):
        self.cfg = cfg

    def generate(self) -> None:
        """生成一组 HandCfg 对象"""
        raise NotImplementedError
