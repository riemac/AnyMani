# TODO:后续还需要将该文件继续拆分，这里仅保留hand,palm,finger,joint,另一文件保留更基层配置类，目前已超过1000行，比较庞大
# 用声明式配置把结构先立起来，用清晰的层次把“对象是什么”和“系统怎么运行”分开，让人可以边想边搭，而不是一开始就陷进一大堆过程式细节
"""资产基类

定位是声明式资产规范 + 规范化解析层。

该文件不负责：
- 组合枚举不同手资产
- URDF / XML 序列化

该文件负责：
- 定义 joint / finger / palm / hand 的声明式配置类
- 接收 asset_generator.py 给出的参数
- 自动做字段校验、默认值补全和规范化解析
- 产出后续生成器、导出器、特征提取器都能稳定消费的 canonical object

注释约定：
- 方法级文档字符串优先采用 Google Doc String 风格
- 数学表达统一使用 LaTeX 定界符 `$...$` / `$$...$$`
- 不使用 `:math:` 作为主要数学注释形式，以便在当前 VSCode 工作流中直接渲染阅读
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import MISSING, Field, dataclass, field, fields, is_dataclass, replace
import math
from pathlib import Path
from typing import Any, ClassVar, Literal, cast, overload

# 常用小向量类型别名。这里不引入 numpy / torch 类型，是为了保持这层
# 资产声明模块足够轻量，便于独立运行、测试和后续脚本化处理。

def _class_to_dict(value: Any) -> Any:
    r"""递归地将 dataclass 资产配置转换为原生 Python 容器。

    Args:
        value (Any): 待转换对象，可以是 dataclass、list、tuple、dict 或标量。

    Returns:
        Any: 与输入同构、但仅由原生 Python 容器组成的对象。
    """

    if is_dataclass(value):
        # dataclass 节点递归展开为 dict，保持字段名与声明层完全一致。
        return {obj_field.name: _class_to_dict(getattr(value, obj_field.name)) for obj_field in fields(value)}
    if isinstance(value, list):
        return [_class_to_dict(item) for item in value]  # list 保持顺序
    if isinstance(value, tuple):
        return tuple(_class_to_dict(item) for item in value)  # tuple 保持不可变语义
    if isinstance(value, dict):
        return {key: _class_to_dict(item) for key, item in value.items()}
    return value


def _update_from_dict(obj: Any, data: dict[str, Any]) -> None:
    r"""就地更新 dataclass 字段，然后重新运行规范化逻辑。

    Args:
        obj (Any): 待更新的 dataclass 对象。
        data (dict[str, Any]): 用于覆盖字段的新字典。

    Raises:
        KeyError: 当 `data` 中包含对象不存在的字段时抛出。
    """

    for key, value in data.items():
        if not hasattr(obj, key):
            raise KeyError(f"Unknown config field: {key}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, Mapping):
            _update_from_dict(current, dict(value))  # 嵌套 dataclass 做递归更新
        else:
            setattr(obj, key, value)  # 叶子字段直接覆盖
    if hasattr(obj, "__post_init__"):
        obj.__post_init__()  # 更新后重新跑规范化与一致性检查


def _validate_missing(obj: Any, prefix: str = "") -> list[str]:
    r"""检查 dataclass 资产配置中是否仍残留 `MISSING` 占位。

    Args:
        obj (Any): 待检查对象。
        prefix (str): 当前递归路径前缀。

    Returns:
        list[str]: 所有仍为 `MISSING` 的字段路径。
    """

    missing: list[str] = []
    for obj_field in fields(obj):
        value = getattr(obj, obj_field.name)
        key = f"{prefix}.{obj_field.name}" if prefix else obj_field.name
        if value is MISSING:
            missing.append(key)  # 记录仍未被实际值替换的占位字段
        elif is_dataclass(value):
            missing.extend(_validate_missing(value, key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if is_dataclass(item):
                    missing.extend(_validate_missing(item, f"{key}[{index}]"))
    return missing


class AssetCfgBase:
    r"""资产声明层的通用辅助方法集合。

    这里保持纯 `dataclass` 语义，仅补充序列化、拷贝和校验这些
    与 Isaac Lab runtime 无关、但对前序资产工程很实用的能力。
    """

    def to_dict(self) -> dict[str, Any]:
        r"""将当前配置对象递归展开为字典。

        Returns:
            dict[str, Any]: 由原生 Python 容器组成的递归字典表示。
        """

        return _class_to_dict(self)

    def from_dict(self, data: dict[str, Any]) -> None:
        r"""就地读取字典并更新当前对象。

        Args:
            data (dict[str, Any]): 用于更新当前对象的字典。
        """

        _update_from_dict(self, data)

    def copy(self):
        r"""返回当前对象的深拷贝。

        Returns:
            Any: 当前对象的深拷贝。
        """

        return deepcopy(self)

    def replace(self, **kwargs):
        r"""返回带局部字段替换的新对象。

        Args:
            **kwargs: 需要替换的新字段。

        Returns:
            Any: 替换字段后的新对象。
        """

        return replace(cast(Any, self), **kwargs)

    def validate(self) -> list[str]:
        r"""返回所有仍未解析完成的 `MISSING` 字段路径。

        Returns:
            list[str]: 尚未被具体值填充的字段路径列表。
        """

        return _validate_missing(self)


Vector2 = tuple[float, float]
Vector3 = tuple[float, float, float]
Vector4 = tuple[float, float, float, float]
Vector6 = tuple[float, float, float, float, float, float]
JointType = Literal["revolute", "fixed"]
Handedness = Literal["left", "right", "unknown"]  # unknown 未来留作 “夹爪手” 等非典型手型的占位
PrimitiveGeometryType = Literal["box", "cylinder", "sphere"]

_FLOAT_TOLERANCE = 1e-12  # 统一的近零容差，用于轴向量与数值稳定性判断


def _sanitize_identifier(name: str, *, field_name: str) -> str:  # field_name为“防御性编程” 的一种手段
    r"""规范化名称字段。

    这里保留你之前的工程约束：若名字以数字开头，则自动加前缀 `a_`，
    避免后续转 USD 或做下游索引时触发命名兼容性问题。

    Args:
        name (str): 原始名称。
        field_name (str): 调用方字段名，用于生成更清晰的错误提示。

    Returns:
        str: 规范化后的名称。

    Raises:
        ValueError: 当名称为空字符串或仅包含空白字符时抛出。
    """

    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    name = name.strip()  # 去除首尾空格，避免出现隐蔽路径错误
    if name[0].isdigit():
        name = f"a_{name}"  # 数字开头的名字自动加前缀 `a_`
    return name

# Python 的函数重载（function overloading）主要服务静态类型检查。
# 这里显式告诉类型检查器：`_ensure_tuple()` 在不同 `length` 下会返回不同定长 tuple 类型。
# 每个签名末尾的 `...` 是类型 stub 写法，不是运行时可变参数。
@overload
def _ensure_tuple(value: Any, *, length: Literal[2], field_name: str) -> Vector2: ...


@overload
def _ensure_tuple(value: Any, *, length: Literal[3], field_name: str) -> Vector3: ...


@overload
def _ensure_tuple(value: Any, *, length: Literal[4], field_name: str) -> Vector4: ...


@overload
def _ensure_tuple(value: Any, *, length: Literal[6], field_name: str) -> Vector6: ...


@overload
def _ensure_tuple(value: Any, *, length: int, field_name: str) -> tuple[float, ...]: ...


def _ensure_tuple(value: Any, *, length: int, field_name: str) -> tuple[float, ...]:
    r"""将输入强制转换为定长浮点 tuple。

    资产生成器可能给出 list / tuple / numpy-like 序列；这里统一为 tuple，
    让配置对象在序列化、比较和哈希语义上更稳定。

    Args:
        value (Any): 待规范化输入。
        length (int): 期望长度。
        field_name (str): 调用方字段名，用于错误提示。

    Returns:
        tuple[float, ...]: 定长浮点 tuple。

    Raises:
        TypeError: 当输入不是合法序列时抛出。
        ValueError: 当输入长度与 `length` 不一致时抛出。
    """

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence with {length} floats, got {value!r}")
    if len(value) != length:
        raise ValueError(f"{field_name} must have length {length}, got {len(value)}")
    return tuple(float(item) for item in value)


def _normalize_axis(axis: Vector3) -> Vector3:
    r"""将关节轴规范化为单位向量。

    Args:
        axis (Vector3): 原始轴向量。

    Returns:
        Vector3: 归一化后的单位向量。

    Raises:
        ValueError: 当输入为零向量时抛出。
    """

    x, y, z = _ensure_tuple(axis, length=3, field_name="axis")
    norm = math.sqrt(x * x + y * y + z * z)
    if norm <= _FLOAT_TOLERANCE:
        raise ValueError("axis cannot be zero vector")
    return (x / norm, y / norm, z / norm)


def _ensure_list(value: Any, *, field_name: str) -> list[Any]:
    r"""将单个对象或 tuple 统一包装为 list。

    Args:
        value (Any): 单个对象、tuple、list 或 `None`。
        field_name (str): 预留字段名参数，便于后续扩展错误提示。

    Returns:
        list[Any]: 统一后的 list。
    """

    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


@dataclass
class PoseCfg(AssetCfgBase):
    r"""局部位姿描述，采用位置 `pos` + 欧拉角 `rpy` 形式。

    该表示与 URDF 的 `<origin xyz="" rpy="">` 语义直接对应，
    便于在“声明式资产层”和后续 URDF 导出层之间保持同构。
    """

    pos: Vector3 = (0.0, 0.0, 0.0)
    """局部平移 $(x, y, z)$。"""

    rpy: Vector3 = (0.0, 0.0, 0.0)
    """局部欧拉角 $(roll, pitch, yaw)$。"""

    def __post_init__(self):
        self.pos = _ensure_tuple(self.pos, length=3, field_name="pos")
        self.rpy = _ensure_tuple(self.rpy, length=3, field_name="rpy")

    @classmethod  # 类方法，支持多种输入形式构造 PoseCfg
    def from_value(cls, value: PoseCfg | Sequence[float] | Mapping[str, Any] | None) -> PoseCfg:
        r"""从常见输入形式构造 `PoseCfg`。

        支持：
        - `PoseCfg`
        - `(x, y, z)`
        - `(x, y, z, roll, pitch, yaw)`
        - `{"pos"/"xyz", "rpy"/"rot"}` 风格字典

        Args:
            value (PoseCfg | Sequence[float] | Mapping[str, Any] | None): 输入位姿。

        Returns:
            PoseCfg: 规范化后的位姿对象。

        Raises:
            TypeError: 当输入形式不受支持时抛出。
        """

        if value is None:
            return cls()
        if isinstance(value, cls):
            return value.copy()
        if isinstance(value, Mapping):
            pos = value.get("pos", value.get("xyz", value.get("position", (0.0, 0.0, 0.0))))
            rpy = value.get("rpy", value.get("rot", value.get("rotation", (0.0, 0.0, 0.0))))
            return cls(pos=pos, rpy=rpy)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) == 3:
                return cls(pos=_ensure_tuple(value, length=3, field_name="pose.pos"))
            if len(value) == 6:
                packed = _ensure_tuple(value, length=6, field_name="pose")
                x, y, z, roll, pitch, yaw = packed
                return cls(pos=(x, y, z), rpy=(roll, pitch, yaw))
        raise TypeError(f"Unsupported pose value: {value!r}")

    @property
    def packed(self) -> Vector6:
        r"""将位置与姿态拼接为 6 维 tuple。

        Returns:
            Vector6: `(*pos, *rpy)` 的拼接结果。
        """

        return (*self.pos, *self.rpy)


@dataclass
class MaterialCfg(AssetCfgBase):
    r"""可选材质/颜色描述，后续可用于 recolored URDF。"""

    name: str | None = None
    """材质名，主要供 visual/recolored 导出层引用。"""

    rgba: Vector4 = (0.7, 0.7, 0.7, 1.0)
    """RGBA 颜色与透明度。"""

    def __post_init__(self):
        self.rgba = _ensure_tuple(self.rgba, length=4, field_name="rgba")  # RGBA 统一写成 4 元浮点 tuple
        if self.name is not None:
            self.name = _sanitize_identifier(self.name, field_name="material.name")  # 材质名也保持下游可序列化


@dataclass
class GeometryCfg(AssetCfgBase):
    r"""几何描述基类。

    这里故意把“几何类型”和“几何实例”拆开：
    - `GeometryCfg` 只描述形状参数
    - `GeometryElementCfg` 再补上 origin / material 等实例化信息

    这样更贴合 URDF 中一个 link 含多个 collision / visual element 的现实结构。
    """

    geometry_type: ClassVar[str] = "geometry"
    """几何类型标识符，用于派生类分发。"""

    @property
    def kind(self) -> str:
        return self.geometry_type

    @property
    def is_primitive(self) -> bool:
        return self.geometry_type in {"box", "cylinder", "sphere"}


@dataclass
class BoxGeometryCfg(GeometryCfg):
    geometry_type: ClassVar[str] = "box"
    size: Vector3
    """box 边长 $(s_x, s_y, s_z)$。"""

    def __post_init__(self):
        self.size = _ensure_tuple(self.size, length=3, field_name="box.size")  # box 边长按 $(s_x, s_y, s_z)$ 存储
        if any(edge <= 0.0 for edge in self.size):
            raise ValueError(f"box.size must be positive, got {self.size}")


@dataclass
class CylinderGeometryCfg(GeometryCfg):
    geometry_type: ClassVar[str] = "cylinder"
    radius: float
    """圆柱半径 $r$。"""

    length: float
    """圆柱长度 $l$。"""

    def __post_init__(self):
        self.radius = float(self.radius)  # 圆柱半径 $r$
        self.length = float(self.length)  # 圆柱长度 $l$
        if self.radius <= 0.0 or self.length <= 0.0:
            raise ValueError(f"cylinder radius/length must be positive, got {(self.radius, self.length)}")


@dataclass
class SphereGeometryCfg(GeometryCfg):
    geometry_type: ClassVar[str] = "sphere"
    radius: float
    """球半径 $r$。"""

    def __post_init__(self):
        self.radius = float(self.radius)  # 球半径 $r$
        if self.radius <= 0.0:
            raise ValueError(f"sphere.radius must be positive, got {self.radius}")


@dataclass
class MeshGeometryCfg(GeometryCfg):
    geometry_type: ClassVar[str] = "mesh"
    file_path: str
    """mesh 文件路径；后续导出层可决定解释为相对或绝对路径。"""

    scale: Vector3 = (1.0, 1.0, 1.0)
    """mesh 局部缩放 $(s_x, s_y, s_z)$。"""

    def __post_init__(self):
        if not isinstance(self.file_path, str) or not self.file_path.strip():
            raise ValueError("mesh.file_path must be a non-empty string")
        self.file_path = self.file_path.strip()  # 去除首尾空格，避免出现隐蔽路径错误
        self.scale = _ensure_tuple(self.scale, length=3, field_name="mesh.scale")  # mesh 局部缩放 $(s_x, s_y, s_z)$
        if any(scale <= 0.0 for scale in self.scale):
            raise ValueError(f"mesh.scale must be positive, got {self.scale}")

    @property
    def suffix(self) -> str:
        return Path(self.file_path).suffix.lower()  # 专门用来提取文件扩展名并转换为小写，便于导出层区分 stl/obj/dae 等 mesh 类型


GeometryValue = GeometryCfg | str | Mapping[str, Any]
# `GeometryValue` 表示 generator 侧允许传入的“尚未完全规范化”的几何输入类型。
# `make_geometry_cfg` 的职责就是把这类松散输入压成统一的 `GeometryCfg` 对象。


def make_geometry_cfg(value: GeometryValue) -> GeometryCfg:
    r"""将多种输入规范化为几何配置对象。

    支持：
    - 直接传入 GeometryCfg 子类实例
    - 直接传入 mesh 路径字符串
    - 传入 dict，使用 type/kind 指示几何类别

    Args:
        value (GeometryValue): 松散几何输入。

    Returns:
        GeometryCfg: 规范化后的几何对象。

    Raises:
        TypeError: 当输入类型不受支持时抛出。
        KeyError: 当字典输入缺少几何类型键时抛出。
        ValueError: 当几何类型值不受支持时抛出。
    """

    if isinstance(value, GeometryCfg):
        return value.copy()  # 已经是规范对象时直接拷贝，避免上游对象被原地共享修改
    if isinstance(value, str):
        return MeshGeometryCfg(file_path=value)  # 裸字符串默认解释为 mesh 路径
    if not isinstance(value, Mapping):
        raise TypeError(f"Unsupported geometry value: {value!r}")

    # 兼容 generator 侧不同命名习惯：`type` 与 `kind` 在这里视为同义字段。
    geometry_type = value.get("type", value.get("kind"))
    if geometry_type is None:
        raise KeyError("Geometry dict must contain 'type' or 'kind'")

    geometry_type = str(geometry_type).lower()
    if geometry_type == "box":
        return BoxGeometryCfg(size=value["size"])  # box 直接读取边长三元组 $(s_x, s_y, s_z)$
    if geometry_type == "cylinder":
        return CylinderGeometryCfg(radius=value["radius"], length=value["length"])  # 读取半径 $r$ 与长度 $l$
    if geometry_type == "sphere":
        return SphereGeometryCfg(radius=value["radius"])  # 球体仅需要半径 $r$
    if geometry_type == "mesh":
        file_path = value.get("file_path", value.get("path", value.get("mesh")))
        return MeshGeometryCfg(file_path=file_path, scale=value.get("scale", (1.0, 1.0, 1.0)))

    raise ValueError(f"Unsupported geometry type: {geometry_type}")


@dataclass
class GeometryElementCfg(AssetCfgBase):
    r"""一个具体的 geometry 实例，包含局部坐标和可选材质。

    同一种 `GeometryCfg` 可以在不同位置重复出现；因此 element 层负责描述
    “把这个几何体摆到哪里、是否附着材质”，而不是重新定义其形状。
    """

    geometry: GeometryCfg
    """具体形状参数本体。"""

    name: str | None = None
    """该 element 的可选名称，主要供调试/导出时标识。"""

    origin: PoseCfg | Sequence[float] | Mapping[str, Any] | None = field(default_factory=PoseCfg)
    """element 相对所属 joint/child link frame 的局部位姿。"""

    material: MaterialCfg | Mapping[str, Any] | None = None
    """可选材质，仅 visual/recolored 相关流程消费。"""

    def __post_init__(self):
        if self.name is not None:
            self.name = _sanitize_identifier(self.name, field_name="geometry_element.name")
        self.geometry = make_geometry_cfg(self.geometry)  # 将 primitive / mesh / dict 统一提升为 GeometryCfg
        self.origin = PoseCfg.from_value(self.origin)  # 将 3 维 / 6 维 / dict pose 写法统一为 PoseCfg
        if self.material is not None and not isinstance(self.material, MaterialCfg):
            if not isinstance(self.material, Mapping):
                raise TypeError(f"material must be MaterialCfg or mapping, got {self.material!r}")
            self.material = MaterialCfg(**self.material)  # NOTE：待确认这里的含义。材质只在 visual/recolored 导出阶段生效


@dataclass
class CollisionGeometryCfg(GeometryElementCfg):
    r"""碰撞几何实例。

    相比 visual，collision 额外关心接触相关偏移量，因此单独派生为子类。
    """

    contact_offset: float | None = None  # TODO：这里的接触碰撞偏移有必要添加吗？我感觉要取消
    """接触膨胀偏移，常用于接触稳定性调节。"""

    rest_offset: float | None = None
    """静息接触偏移。"""

    def __post_init__(self):
        super().__post_init__()
        if self.contact_offset is not None:
            self.contact_offset = float(self.contact_offset)  # 接触膨胀偏移，通常由导出/仿真层消费
            if self.contact_offset < 0.0:
                raise ValueError("contact_offset must be >= 0")
        if self.rest_offset is not None:
            self.rest_offset = float(self.rest_offset)  # 静息接触偏移
            if self.rest_offset < 0.0:
                raise ValueError("rest_offset must be >= 0")


@dataclass
class VisualGeometryCfg(GeometryElementCfg):
    r"""视觉几何实例。

    目前它与 `GeometryElementCfg` 没有新增字段，但保留这个类型分支，
    是为了让导出层能清楚地区分 visual 和 collision 的消费语义。
    """


@dataclass
class InertiaTensorCfg(AssetCfgBase):
    r"""惯量张量。

    采用 URDF 常见的 6 参数对称惯量写法：
    $$
    \mathbf{I} =
    \begin{bmatrix}
    i_{xx} & i_{xy} & i_{xz} \\
    i_{xy} & i_{yy} & i_{yz} \\
    i_{xz} & i_{yz} & i_{zz}
    \end{bmatrix}.
    $$

    对当前资产生成阶段而言，最关键的是保持主对角项为正，
    从而避免明显不物理的惯量定义进入后续仿真链路。
    """

    ixx: float
    """主对角项 $i_{xx}$。"""

    iyy: float
    """主对角项 $i_{yy}$。"""

    izz: float
    """主对角项 $i_{zz}$。"""

    ixy: float = 0.0
    """非对角项 $i_{xy}$。"""

    ixz: float = 0.0
    """非对角项 $i_{xz}$。"""

    iyz: float = 0.0
    """非对角项 $i_{yz}$。"""

    def __post_init__(self):
        self.ixx = float(self.ixx)
        self.iyy = float(self.iyy)
        self.izz = float(self.izz)
        self.ixy = float(self.ixy)
        self.ixz = float(self.ixz)
        self.iyz = float(self.iyz)
        if self.ixx <= 0.0 or self.iyy <= 0.0 or self.izz <= 0.0:
            raise ValueError("Inertia diagonal entries must be positive")


@dataclass
class InertialCfg(AssetCfgBase):  # TODO:如果是不同形状组合而成的 joint/child link，是否需要支持多个 inertia element的统一计算？目前先保持单一 inertia element 的简化假设
    r"""link 的 inertial 描述。

    这里额外显式支持 inertia padding，便于生成资产时做数值稳定化。
    这对应你在 `requirement.md` 中提到的工程现实：生成资产时，惯量往往
    不是追求“最真实”，而是追求“合理且数值稳定”。
    """

    mass: float
    """link 质量 $m$。"""

    inertia: InertiaTensorCfg | Mapping[str, Any]
    """惯量张量本体。"""

    origin: PoseCfg | Sequence[float] | Mapping[str, Any] | None = field(default_factory=PoseCfg)
    """inertial frame 相对 joint/child link frame 的局部位姿。"""

    inertia_padding: float = 0.0
    """作用于主对角线的工程稳定化 padding。"""

    def __post_init__(self):
        self.mass = float(self.mass)  # 统一为标量质量 $m$
        if self.mass <= 0.0:
            raise ValueError(f"mass must be positive, got {self.mass}")
        self.origin = PoseCfg.from_value(self.origin)  # inertial frame 相对 joint/child link frame 的局部位姿
        if not isinstance(self.inertia, InertiaTensorCfg):
            if not isinstance(self.inertia, Mapping):
                raise TypeError(f"inertia must be InertiaTensorCfg or mapping, got {self.inertia!r}")
            self.inertia = InertiaTensorCfg(**self.inertia)  # 兼容 dict 输入，统一抬升为规范对象
        self.inertia_padding = float(self.inertia_padding)
        if self.inertia_padding < 0.0:
            raise ValueError("inertia_padding must be >= 0")
        if self.inertia_padding > 0.0:
            # 对主对角线加 padding，等价于把惯量椭球整体“抬厚”一点，
            # 以减少过小惯量带来的接触/积分数值不稳定。
            self.inertia = InertiaTensorCfg(
                ixx=self.inertia.ixx + self.inertia_padding,
                iyy=self.inertia.iyy + self.inertia_padding,
                izz=self.inertia.izz + self.inertia_padding,
                ixy=self.inertia.ixy,
                ixz=self.inertia.ixz,
                iyz=self.inertia.iyz,
            )

    @classmethod  # NOTE:笔记。类方法的优雅用法，“备选构造函数” (Alternative Constructor)
    def from_box(
        cls,
        size: Vector3,
        density: float,
        *,
        origin: PoseCfg | Sequence[float] | Mapping[str, Any] | None = None,
        min_mass: float = 1e-4,
        inertia_padding: float = 1e-8,
    ) -> InertialCfg:
        r"""由 box 近似直接构造惯量参数。

        对均匀长方体，质量与惯量满足：
        $$
        m = \rho s_x s_y s_z,
        $$        $$
        i_{xx} = \frac{m}{12}(s_y^2 + s_z^2),\quad
        i_{yy} = \frac{m}{12}(s_x^2 + s_z^2),\quad
        i_{zz} = \frac{m}{12}(s_x^2 + s_y^2).
        $$

        这里的 `min_mass` 与 `inertia_padding` 都属于工程稳定化项，
        不是纯理论刚体公式的一部分。

        Args:
            size (Vector3): box 边长 $(s_x, s_y, s_z)$。
            density (float): 体密度 $\rho$。
            origin (PoseCfg | Sequence[float] | Mapping[str, Any] | None): 惯性系局部位姿。
            min_mass (float): 允许的最小质量下界。
            inertia_padding (float): 作用于主对角线的惯量 padding。

        Returns:
            InertialCfg: box 近似得到的惯量配置。

        Raises:
            ValueError: 当 `density` 非正时抛出。
        """

        sx, sy, sz = _ensure_tuple(size, length=3, field_name="size")
        density = float(density)
        if density <= 0.0:
            raise ValueError("density must be positive")
        mass = max(density * sx * sy * sz, min_mass)  # 过小体素仍抬到最小质量，避免近零质量
        ixx = mass * (sy * sy + sz * sz) / 12.0  # 绕 $x$ 轴转动时，横截面尺寸由 $(s_y, s_z)$ 决定
        iyy = mass * (sx * sx + sz * sz) / 12.0
        izz = mass * (sx * sx + sy * sy) / 12.0
        return cls(
            mass=mass,
            origin=origin,
            inertia=InertiaTensorCfg(ixx=ixx, iyy=iyy, izz=izz),
            inertia_padding=inertia_padding,
        )

    @classmethod
    def from_cylinder(  # TODO：这里并不打算沿z轴，后续考虑怎么实现
        cls,
        radius: float,
        length: float,
        density: float,
        *,
        origin: PoseCfg | Sequence[float] | Mapping[str, Any] | None = None,
        min_mass: float = 1e-4,
        inertia_padding: float = 1e-8,
    ) -> InertialCfg:
        r"""由 cylinder 近似直接构造惯量参数。

        采用均匀圆柱体近似：
        $$
        m = \rho \pi r^2 l,
        $$        $$
        i_{xx} = i_{yy} = \frac{m}{12}(3r^2 + l^2),\quad
        i_{zz} = \frac{1}{2}mr^2.
        $$

        这里默认圆柱主轴沿局部 $z$ 轴；若后续导出层采用其他约定，
        应在那里做坐标变换，而不是在声明层混入旋转补偿。

        Args:
            radius (float): 圆柱半径 $r$。
            length (float): 圆柱长度 $l$。
            density (float): 体密度 $\rho$。
            origin (PoseCfg | Sequence[float] | Mapping[str, Any] | None): 惯性系局部位姿。
            min_mass (float): 允许的最小质量下界。
            inertia_padding (float): 作用于主对角线的惯量 padding。

        Returns:
            InertialCfg: cylinder 近似得到的惯量配置。

        Raises:
            ValueError: 当输入的半径、长度或密度非正时抛出。
        """

        radius = float(radius)
        length = float(length)
        density = float(density)
        if radius <= 0.0 or length <= 0.0 or density <= 0.0:
            raise ValueError("radius, length and density must be positive")
        volume = math.pi * radius * radius * length  # 圆柱体积 $V = \pi r^2 l$
        mass = max(density * volume, min_mass)
        ixx = mass * (3.0 * radius * radius + length * length) / 12.0
        iyy = ixx
        izz = 0.5 * mass * radius * radius  # 主轴方向惯量只与半径相关
        return cls(
            mass=mass,
            origin=origin,
            inertia=InertiaTensorCfg(ixx=ixx, iyy=iyy, izz=izz),
            inertia_padding=inertia_padding,
        )

    @classmethod
    def from_sphere(
        cls,
        radius: float,
        density: float,
        *,
        origin: PoseCfg | Sequence[float] | Mapping[str, Any] | None = None,
        min_mass: float = 1e-4,
        inertia_padding: float = 1e-8,
    ) -> InertialCfg:
        r"""由 sphere 近似直接构造惯量参数。

        对均匀球体：
        $$
        m = \rho \frac{4}{3}\pi r^3,\qquad
        i_{xx}=i_{yy}=i_{zz}=\frac{2}{5}mr^2.
        $$

        Args:
            radius (float): 球半径 $r$。
            density (float): 体密度 $\rho$。
            origin (PoseCfg | Sequence[float] | Mapping[str, Any] | None): 惯性系局部位姿。
            min_mass (float): 允许的最小质量下界。
            inertia_padding (float): 作用于主对角线的惯量 padding。

        Returns:
            InertialCfg: sphere 近似得到的惯量配置。

        Raises:
            ValueError: 当输入半径或密度非正时抛出。
        """

        radius = float(radius)
        density = float(density)
        if radius <= 0.0 or density <= 0.0:
            raise ValueError("radius and density must be positive")
        volume = 4.0 / 3.0 * math.pi * radius**3
        mass = max(density * volume, min_mass)
        diagonal = 0.4 * mass * radius * radius  # $\frac{2}{5}mr^2 = 0.4mr^2$
        return cls(
            mass=mass,
            origin=origin,
            inertia=InertiaTensorCfg(ixx=diagonal, iyy=diagonal, izz=diagonal),
            inertia_padding=inertia_padding,
        )


@dataclass
class JointLimitCfg(AssetCfgBase):
    r"""关节限位与可选驱动上界配置。"""

    lower: float
    r"""关节下限 $q_{\min}$。"""

    upper: float
    r"""关节上限 $q_{\max}$。"""

    effort: float | None = None
    r"""可选驱动扭矩/力上界 $\tau_{\max}$。"""

    velocity: float | None = None
    r"""可选关节速度上界 $\dot{q}_{\max}$。"""

    def __post_init__(self):
        self.lower = float(self.lower)  # 下限 $q_{\min}$
        self.upper = float(self.upper)  # 上限 $q_{\max}$
        if self.upper < self.lower:
            raise ValueError(f"upper limit must be >= lower limit, got {(self.lower, self.upper)}")
        if self.effort is not None:
            self.effort = float(self.effort)  # 可选驱动扭矩/力上界
        if self.velocity is not None:
            self.velocity = float(self.velocity)  # 可选关节速度上界


@dataclass
class MimicCfg(AssetCfgBase):  # TODO:仅保留该配置，本次科研项目暂时不考虑欠驱动关节，仅考虑全驱手
    r"""URDF mimic joint 配置。"""

    joint: str
    """被 mimic 的主关节名。"""

    multiplier: float = 1.0
    """线性比例系数 $\alpha$。"""

    offset: float = 0.0
    """线性偏移项 $\beta$。"""

    def __post_init__(self):
        self.joint = _sanitize_identifier(self.joint, field_name="mimic.joint")  # 被模仿的主关节名
        self.multiplier = float(self.multiplier)  # 线性系数 $\alpha$
        self.offset = float(self.offset)  # 偏移项 $\beta$


def _make_collision_cfg(value: Any) -> CollisionGeometryCfg:
    r"""将松散输入规范化为 `CollisionGeometryCfg`。

    这个函数主要服务 generator 侧的“宽输入接口”：
    用户可以只给 primitive 参数、只给 mesh 路径、或给完整 element dict，
    这里统一兜底收口。

    Args:
        value (Any): 松散碰撞几何输入。

    Returns:
        CollisionGeometryCfg: 规范化后的碰撞几何对象。

    Raises:
        TypeError: 当输入无法被解释为合法碰撞几何时抛出。
    """

    if isinstance(value, CollisionGeometryCfg):
        return value.copy()  # 避免共享可变对象
    if isinstance(value, GeometryCfg) or isinstance(value, str):
        return CollisionGeometryCfg(geometry=make_geometry_cfg(value))  # 只给几何时，用默认 element 参数包起来
    if not isinstance(value, Mapping):
        raise TypeError(f"Unsupported collision geometry value: {value!r}")
    if "geometry" in value:
        return CollisionGeometryCfg(**dict(value))  # 已经接近规范格式时直接透传
    geometry_keys = {"type", "kind", "size", "radius", "length", "file_path", "path", "mesh", "scale"}
    if geometry_keys.intersection(value.keys()):
        element_kwargs = {
            key: value[key]  # TODO：同理，如果和我的idea预想一致，那我不建议在这里暴露 contact/rest_offset 这类仿真调参项，还是再构造生成资产时由外部算法解析计算，这里仅保留最基础且必要的几何描述，所谓“第一性原理”
            for key in ("name", "origin", "material", "contact_offset", "rest_offset")
            if key in value
        }
        element_kwargs["geometry"] = value  # 将“裸 geometry dict”抬升到 element.geometry 字段
        return CollisionGeometryCfg(**element_kwargs)
    raise TypeError(f"Unsupported collision geometry mapping: {value!r}")


def _make_visual_cfg(value: Any) -> VisualGeometryCfg:
    r"""将松散输入规范化为 `VisualGeometryCfg`。

    Args:
        value (Any): 松散视觉几何输入。

    Returns:
        VisualGeometryCfg: 规范化后的视觉几何对象。

    Raises:
        TypeError: 当输入无法被解释为合法视觉几何时抛出。
    """

    if isinstance(value, VisualGeometryCfg):
        return value.copy()
    if isinstance(value, GeometryCfg) or isinstance(value, str):
        return VisualGeometryCfg(geometry=make_geometry_cfg(value))
    if not isinstance(value, Mapping):
        raise TypeError(f"Unsupported visual geometry value: {value!r}")
    if "geometry" in value:
        return VisualGeometryCfg(**dict(value))
    geometry_keys = {"type", "kind", "size", "radius", "length", "file_path", "path", "mesh", "scale"}
    if geometry_keys.intersection(value.keys()):
        element_kwargs = {key: value[key] for key in ("name", "origin", "material") if key in value}
        element_kwargs["geometry"] = value
        return VisualGeometryCfg(**element_kwargs)
    raise TypeError(f"Unsupported visual geometry mapping: {value!r}")


@dataclass
class JointCfg(AssetCfgBase):
    r"""关节 + child link 的 joint-centric 描述。

    这一层沿用你当前的建模哲学：以 joint 作为主语义中心，同时把 child link
    的碰撞体、视觉体和惯量绑定到 joint 上，便于后续图建模直接读取
    “关节控制属性 + 子连杆几何属性”的联合表征。
    """

    name: str
    """当前 joint 名，同时也是图建模与导出层的重要索引键。"""

    parent: str
    """父 link 名。"""

    joint_type: JointType = "revolute"
    """当前支持 `revolute` / `fixed` 两类 joint。"""

    child: str | None = None  # 在urdf中，<joint> 和 <link> 是分开声明的，但实际上 joint 和 child link的坐标系是重合的
    """子 link 名；若缺省则在规范化时自动派生。"""

    axis: Vector3 = (0.0, 0.0, 1.0)
    """关节轴方向 $\vec{a}$。"""

    limit: JointLimitCfg | Mapping[str, Any] | Sequence[float] | None = (-math.pi, math.pi)
    """关节限位，可写为对象 / dict / `(lower, upper)` 简写。"""

    origin: PoseCfg | Sequence[float] | Mapping[str, Any] | None = field(default_factory=PoseCfg)
    """joint frame 相对 parent link frame 的局部位姿。"""

    inertial: InertialCfg | Mapping[str, Any] | None = None
    """child link 的惯量属性。"""

    collisions: list[CollisionGeometryCfg] = field(default_factory=list)
    """child link 的碰撞几何集合。"""

    visuals: list[VisualGeometryCfg] = field(default_factory=list)
    """child link 的视觉几何集合。"""

    mimic: MimicCfg | Mapping[str, Any] | None = None
    """可选 mimic 关系。"""

    # TODO:该字段还不确定。按照我的idea，指尖单独剥离出来，与手指的最后一个joint/child link frame解耦，但joint类型是fixed。
    # 然而在joint-centric网络架构中，是不考虑fixed的，最后一个link是同时包含tip的，这里之所以分出来完全是为了资产生成的组合便利性，所以应该保留
    is_tip: bool = False
    """是否将该 joint 视作 fingertip 相关 joint。"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """预留扩展字段，承接 generator 或分析脚本附加信息。"""

    def __post_init__(self):
        self.name = _sanitize_identifier(self.name, field_name="joint.name")  # 关节名也承担下游索引键角色
        if self.joint_type not in {"revolute", "fixed"}:
            raise ValueError(f"invalid joint_type: {self.joint_type}, must be 'revolute' or 'fixed'")
        self.parent = _sanitize_identifier(self.parent, field_name="joint.parent")  # 父 link 名
        self.child = _sanitize_identifier(self.child or f"{self.name}_link", field_name="joint.child")  # 默认 child link 自动派生
        self.origin = PoseCfg.from_value(self.origin)  # joint frame 相对 parent link frame 的位姿

        axis_tuple = _ensure_tuple(self.axis, length=3, field_name="joint.axis")  # 原始轴输入先压成三维 tuple
        if self.joint_type == "fixed" and all(abs(value) <= _FLOAT_TOLERANCE for value in axis_tuple):
            self.axis = (0.0, 0.0, 1.0)  # fixed joint 的轴通常不会真正参与计算，这里给稳定默认值
        else:
            self.axis = _normalize_axis(axis_tuple)  # revolute joint 轴必须是单位向量

        if self.limit is None:
            if self.joint_type != "fixed":
                raise ValueError("Non-fixed joint must provide limit")
        elif isinstance(self.limit, JointLimitCfg):
            self.limit = self.limit.copy()  # 避免上游复用同一对象导致联动修改
        elif isinstance(self.limit, Mapping):
            self.limit = JointLimitCfg(**self.limit)  # dict 写法提升为规范对象
        elif isinstance(self.limit, Sequence) and not isinstance(self.limit, (str, bytes)):
            packed = _ensure_tuple(self.limit, length=2, field_name="joint.limit")  # 支持 `(lower, upper)` 简写
            self.limit = JointLimitCfg(lower=packed[0], upper=packed[1])
        else:
            raise TypeError(f"Unsupported joint limit: {self.limit!r}")

        if self.inertial is not None and not isinstance(self.inertial, InertialCfg):
            if not isinstance(self.inertial, Mapping):
                raise TypeError(f"inertial must be InertialCfg or mapping, got {self.inertial!r}")
            self.inertial = InertialCfg(**self.inertial)  # 允许 generator 直接给 inertial dict

        # 一个 child link 可能由多个 primitive / mesh 组合而成，这里统一规范化。
        self.collisions = [_make_collision_cfg(item) for item in _ensure_list(self.collisions, field_name="collisions")]
        self.visuals = [_make_visual_cfg(item) for item in _ensure_list(self.visuals, field_name="visuals")]

        if self.mimic is not None and not isinstance(self.mimic, MimicCfg):
            if not isinstance(self.mimic, Mapping):
                raise TypeError(f"mimic must be MimicCfg or mapping, got {self.mimic!r}")
            self.mimic = MimicCfg(**self.mimic)  # mimic 关系也统一转成规范对象

    @property
    def dof_count(self) -> int:
        return 0 if self.joint_type == "fixed" else 1  # 当前声明层默认每个 revolute joint 对应 1 个自由度

    @property
    def uses_only_primitive_collision(self) -> bool:
        return all(collision.geometry.is_primitive for collision in self.collisions)  # 便于筛选“纯 primitive hand”


@dataclass
class FingerCfg(AssetCfgBase):
    r"""逻辑上的手指，由串联关节构成。

    这里显式检查拓扑链连续性，而不是只检查字段是否存在。
    对资产生成而言，最常见的错误不是“缺字段”，而是 parent/child 接错，
    所以 finger 层承担第一道运动学结构验收。
    """

    name: str
    """finger 逻辑名。"""

    parent_link: str = "palm"
    """finger 挂载到哪个 link；当前默认直接挂到 palm。"""

    mount: PoseCfg | Sequence[float] | Mapping[str, Any] | None = field(default_factory=PoseCfg)
    """finger-level 挂载位姿入口。"""

    joints: list[JointCfg] = field(default_factory=list)
    """构成该 finger 的串联 joint 列表。"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """预留扩展字段。"""

    def __post_init__(self):
        self.name = _sanitize_identifier(self.name, field_name="finger.name")
        self.parent_link = _sanitize_identifier(self.parent_link, field_name="finger.parent_link")  # finger 挂载到哪个 link
        self.mount = PoseCfg.from_value(self.mount)  # 主要保留为 hand-level 排列信息入口
        self.joints = [joint if isinstance(joint, JointCfg) else JointCfg(**joint) for joint in self.joints]

        if not self.joints:
            raise ValueError(f"finger '{self.name}' must contain at least one joint")

        first_parent = self.joints[0].parent  # finger 首关节必须直接接到声明的挂载 link 上
        if first_parent != self.parent_link:
            raise ValueError(
                f"finger '{self.name}' first joint parent must be '{self.parent_link}', got '{first_parent}'"
            )

        # 检查串联链连续性：后一关节的 parent 必须等于前一关节的 child。
        for previous, current in zip(self.joints[:-1], self.joints[1:]):
            if current.parent != previous.child:
                raise ValueError(
                    f"finger '{self.name}' chain broken: joint '{current.name}' parent is "
                    f"'{current.parent}', expected '{previous.child}'"
                )

        joint_names = [joint.name for joint in self.joints]  # finger 内部不允许关节重名
        if len(joint_names) != len(set(joint_names)):
            raise ValueError(f"finger '{self.name}' contains duplicated joint names: {joint_names}")

    @property
    def joint_names(self) -> list[str]:
        return [joint.name for joint in self.joints]  # 常用于 generator/debug 阶段快速读取 finger 结构

    @property
    def tip_joint(self) -> JointCfg:
        return self.joints[-1]  # 末端关节天然视为 fingertip 对应 joint

    @property
    def tip_link(self) -> str:
        return cast(str, self.tip_joint.child)  # `child` 在规范化后必为字符串，这里显式告知类型检查器

    @property
    def dof_count(self) -> int:
        return sum(joint.dof_count for joint in self.joints)  # finger 的总自由度等于其所有 joint 的 dof 求和


# TODO：这里的配置实现先存疑。关于palm,它其实和fingertip是同一类概念（异构）。它的主要特点是无revolute关节，通常是多个fixed关节、多个primitive/mesh组合而成的复杂link
# 且作为多数finger的公共挂载基座，因此单独抽出来一个PalmCfg；但从另一个角度看，palm也完全可以视作一种特殊的finger（没有joint的finger），因此也可以直接用FingerCfg来描述。
# FIXME:两种实现各有利弊，我个人倾向于前者。但我认为这里的配置字段和FingerCfg的主要逻辑应是一致的，不应该添加collisions/visuals字段，而是
@dataclass
class PalmCfg(AssetCfgBase):
    r"""手掌 / 根 link 描述。

    palm 既是 hand-level 运动学树的根节点，也是多数 finger 的公共挂载基座。
    """

    name: str = "palm"
    """palm / 根 link 名。"""

    origin: PoseCfg | Sequence[float] | Mapping[str, Any] | None = field(default_factory=PoseCfg)
    """palm frame 相对 hand 根参考系的位姿。通常是env/base坐标系（不是world,按照IsaacLab的约定，world是全局静止坐标系，env/base是每个实例的环境/根坐标系）。"""

    inertial: InertialCfg | Mapping[str, Any] | None = None
    """palm 的惯量属性。"""

    collisions: list[CollisionGeometryCfg] = field(default_factory=list)
    """palm 的碰撞几何集合。"""

    visuals: list[VisualGeometryCfg] = field(default_factory=list)
    """palm 的视觉几何集合。"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """预留扩展字段。"""

    def __post_init__(self):
        self.name = _sanitize_identifier(self.name, field_name="palm.name")
        self.origin = PoseCfg.from_value(self.origin)  # palm frame 相对 hand 根参考系的位姿
        if self.inertial is not None and not isinstance(self.inertial, InertialCfg):
            if not isinstance(self.inertial, Mapping):
                raise TypeError(f"inertial must be InertialCfg or mapping, got {self.inertial!r}")
            self.inertial = InertialCfg(**self.inertial)
        # palm 也允许由多个 primitive / mesh 组合而成，因此这里同样走统一规范化流程。
        self.collisions = [_make_collision_cfg(item) for item in _ensure_list(self.collisions, field_name="collisions")]
        self.visuals = [_make_visual_cfg(item) for item in _ensure_list(self.visuals, field_name="visuals")]


@dataclass
class HandCfg(AssetCfgBase):
    r"""整手资产描述。

    该对象是资产生成前序工程的 canonical output：
    generator 负责枚举和筛选，`HandCfg` 负责承接最终合法结构，
    后续 URDF 导出、SDF 预计算、图构建都应优先消费这一层，而不是原始采样参数。
    """

    name: str
    """hand 资产名。"""

    palm: PalmCfg | Mapping[str, Any] = field(default_factory=PalmCfg)
    """palm / 根 link 配置。"""

    fingers: list[FingerCfg] = field(default_factory=list)
    """该 hand 包含的所有 finger。"""

    family: str = "generic"  # NOTE:这里算是预留的手型家族说明，如 “类人手”/“夹爪手”
    """hand family 标签，如 `leap` / `allegro` / `generic`。"""

    handedness: Handedness = "unknown"
    """左右手语义；`unknown` 预留给非典型或未定义 handedness 的手型。"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """预留扩展字段。"""

    def __post_init__(self):
        self.name = _sanitize_identifier(self.name, field_name="hand.name")
        self.family = _sanitize_identifier(self.family, field_name="hand.family")  # 如 leap/allegro/generic 等 family 标签
        if self.handedness not in {"left", "right", "unknown"}:
            raise ValueError(f"invalid handedness: {self.handedness}")

        if not isinstance(self.palm, PalmCfg):
            if not isinstance(self.palm, Mapping):
                raise TypeError(f"palm must be PalmCfg or mapping, got {self.palm!r}")
            self.palm = PalmCfg(**self.palm)  # 支持 hand dict 内嵌 palm dict 写法

        self.fingers = [finger if isinstance(finger, FingerCfg) else FingerCfg(**finger) for finger in self.fingers]
        if not self.fingers:
            raise ValueError("hand must contain at least one finger")

        finger_names = [finger.name for finger in self.fingers]  # finger 名必须全局唯一，便于枚举与调试
        if len(finger_names) != len(set(finger_names)):
            raise ValueError(f"hand contains duplicated finger names: {finger_names}")

        all_joint_names = [joint.name for joint in self.iter_joints()]  # hand 内所有 joint 名必须唯一
        if len(all_joint_names) != len(set(all_joint_names)):  # 这个语法挺高级
            raise ValueError(f"hand contains duplicated joint names: {all_joint_names}")

        all_link_names = [self.palm.name] + [joint.child for joint in self.iter_joints()]  # URDF link 名同样必须唯一
        if len(all_link_names) != len(set(all_link_names)):
            raise ValueError(f"hand contains duplicated link names: {all_link_names}")

        # 当前 hand 语义默认所有 finger 都直接挂在 palm 上。
        # 若未来要支持 finger 挂在 finger-link 上的复杂结构，可以从这里放宽。
        for finger in self.fingers:
            if finger.parent_link != self.palm.name:
                raise ValueError(
                    f"finger '{finger.name}' is mounted on '{finger.parent_link}', expected palm link '{self.palm.name}'"
                )

    def iter_joints(self) -> list[JointCfg]:
        r"""按 finger 顺序扁平化返回整手 joint 列表。

        Returns:
            list[JointCfg]: 扁平化后的 joint 列表。
        """

        return [joint for finger in self.fingers for joint in finger.joints]  # 按 finger 顺序扁平化 joint 列表

    @property
    def joint_name_to_index(self) -> dict[str, int]:  # FIXME:这里应该需要修改，因为后续图网络架构最好索引无关
        return {joint.name: index for index, joint in enumerate(self.iter_joints())}  # 常用于动作/观测向量索引映射

    @property
    def dof_count(self) -> int:
        return sum(joint.dof_count for joint in self.iter_joints())  # hand 总自由度

    @property
    def fingertip_links(self) -> list[str]:
        return [finger.tip_link for finger in self.fingers]  # 常作为 contact / fingertip marker 的快速入口


# 兼容旧命名习惯，保留小写别名，降低从草稿代码切换到这套配置层的摩擦。
joint = JointCfg
finger = FingerCfg
palm = PalmCfg
hand = HandCfg


__all__ = [
    "PoseCfg",
    "MaterialCfg",
    "GeometryCfg",
    "BoxGeometryCfg",
    "CylinderGeometryCfg",
    "SphereGeometryCfg",
    "MeshGeometryCfg",
    "GeometryElementCfg",
    "CollisionGeometryCfg",
    "VisualGeometryCfg",
    "InertiaTensorCfg",
    "InertialCfg",
    "JointLimitCfg",
    "MimicCfg",
    "JointCfg",
    "FingerCfg",
    "PalmCfg",
    "HandCfg",
    "make_geometry_cfg",
    "joint",
    "finger",
    "palm",
    "hand",
]
