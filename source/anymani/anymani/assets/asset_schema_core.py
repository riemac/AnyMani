"""手部资产声明的底层 schema 与辅助工具。

本模块承载的是更靠近“基础语言”的那一层声明式对象，供 embodiment
层复用。它主要包含：

- 通用 dataclass 辅助方法
- 位姿 / 材质等基础描述
- 几何体 schema
- 惯量与惯性 schema
- 底层规范化辅助函数

它**故意**不定义 `JointCfg` / `FingerCfg` / `PalmCfg` / `HandCfg`。
这些更贴近“手部结构”的对象被放在 `asset_schema_embodiment.py`
中，以保持“基础描述”和“embodiment 结构描述”之间的边界清晰。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass, replace
import math
from pathlib import Path
from typing import Any, ClassVar, Literal, cast, overload


def _class_to_dict(value: Any) -> Any:
    r"""递归地把 dataclass 资产配置转成原生 Python 容器。

    Args:
        value (Any): 待转换对象。

    Returns:
        Any: 递归展开后的 Python 原生对象。
    """

    if is_dataclass(value):
        return {obj_field.name: _class_to_dict(getattr(value, obj_field.name)) for obj_field in fields(value)}
    if isinstance(value, list):
        return [_class_to_dict(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_class_to_dict(item) for item in value)
    if isinstance(value, dict):
        return {key: _class_to_dict(item) for key, item in value.items()}
    return value


def _update_from_dict(obj: Any, data: dict[str, Any]) -> None:
    r"""原地更新 dataclass 字段，并重新执行规范化。

    Args:
        obj (Any): 待更新的 dataclass 实例。
        data (dict[str, Any]): 新字段值。

    Raises:
        KeyError: 当 `data` 中出现未知字段时抛出。
    """

    for key, value in data.items():
        if not hasattr(obj, key):
            raise KeyError(f"Unknown config field: {key}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, Mapping):
            _update_from_dict(current, dict(value))
        else:
            setattr(obj, key, value)
    if hasattr(obj, "__post_init__"):
        obj.__post_init__()


def _validate_missing(obj: Any, prefix: str = "") -> list[str]:
    r"""收集所有尚未解析完成的必填字段路径。

    Args:
        obj (Any): 待检查对象。
        prefix (str): 当前递归前缀。

    Returns:
        list[str]: 所有缺失字段的路径。
    """

    missing: list[str] = []
    for obj_field in fields(obj):
        value = getattr(obj, obj_field.name)
        key = f"{prefix}.{obj_field.name}" if prefix else obj_field.name
        if value is ...:
            missing.append(key)
        elif is_dataclass(value):
            missing.extend(_validate_missing(value, key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if is_dataclass(item):
                    missing.extend(_validate_missing(item, f"{key}[{index}]"))
    return missing


class AssetCfgBase:
    r"""资产声明 dataclass 的通用辅助 mixin。"""

    def to_dict(self) -> dict[str, Any]:
        r"""把配置序列化为原生 Python 容器。

        Returns:
            dict[str, Any]: 递归字典表示。
        """

        return _class_to_dict(self)

    def from_dict(self, data: dict[str, Any]) -> None:
        r"""根据输入字典原地更新配置。

        Args:
            data (dict[str, Any]): 输入映射。
        """

        _update_from_dict(self, data)

    def copy(self):
        r"""创建深拷贝。

        Returns:
            Any: 深拷贝后的实例。
        """

        return deepcopy(self)

    def replace(self, **kwargs):
        r"""返回一个替换了部分字段的新配置实例。

        Args:
            **kwargs: 待替换字段。

        Returns:
            Any: 替换后的新实例。
        """

        return replace(cast(Any, self), **kwargs)

    def validate(self) -> list[str]:
        r"""返回尚未解析完成的必填字段路径。

        Returns:
            list[str]: 缺失字段路径列表。
        """

        return _validate_missing(self)


Vector2 = tuple[float, float]
"""二维浮点 tuple。"""

Vector3 = tuple[float, float, float]
"""三维浮点 tuple。"""

Vector4 = tuple[float, float, float, float]
"""四维浮点 tuple。"""

Vector6 = tuple[float, float, float, float, float, float]
"""六维浮点 tuple。"""

JointType = Literal["revolute", "fixed"]
"""当前项目范围内支持的 URDF joint 类型。"""

Handedness = Literal["left", "right", "unknown"]
"""左右手标签；`unknown` 预留给非典型或暂未决定 handedness 的结构。"""

PrimitiveGeometryType = Literal["box", "cylinder", "sphere"]
"""支持的基础几何 primitive 类型。"""

_FLOAT_TOLERANCE = 1e-12
"""几何与轴向检查统一使用的近零容差。"""


def _sanitize_identifier(name: str, *, field_name: str) -> str:
    r"""规范化 schema 中使用的字符串标识符。

    Args:
        name (str): 原始名称。
        field_name (str): 用于报错的逻辑字段名。

    Returns:
        str: 规范化后的名称。

    Raises:
        ValueError: 当名称为空时抛出。
    """

    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    name = name.strip()
    if name[0].isdigit():
        name = f"a_{name}"
    return name


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
    r"""把类序列输入转成定长浮点 tuple。

    Args:
        value (Any): 输入对象。
        length (int): 期望 tuple 长度。
        field_name (str): 报错时使用的字段名。

    Returns:
        tuple[float, ...]: 定长浮点 tuple。

    Raises:
        TypeError: 当输入不是合法序列时抛出。
        ValueError: 当长度与要求不符时抛出。
    """

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence with {length} floats, got {value!r}")
    if len(value) != length:
        raise ValueError(f"{field_name} must have length {length}, got {len(value)}")
    return tuple(float(item) for item in value)


def _normalize_axis(axis: Vector3) -> Vector3:
    r"""把关节轴向规范化为单位向量。

    Args:
        axis (Vector3): 原始轴向。

    Returns:
        Vector3: 单位轴向。

    Raises:
        ValueError: 当向量范数为零时抛出。
    """

    x, y, z = _ensure_tuple(axis, length=3, field_name="axis")
    norm = math.sqrt(x * x + y * y + z * z)
    if norm <= _FLOAT_TOLERANCE:
        raise ValueError("axis cannot be zero vector")
    return (x / norm, y / norm, z / norm)


def _ensure_list(value: Any, *, field_name: str) -> list[Any]:
    r"""把单对象 / tuple / `None` 统一规范为 list。

    Args:
        value (Any): 单对象、tuple、list 或 `None`。
        field_name (str): 预留给未来更细的诊断信息。

    Returns:
        list[Any]: 规范化后的列表。
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
    r"""由 `pos` + `rpy` 表示的局部位姿。

    这一层直接对应 URDF 中 `<origin xyz="" rpy="">` 的语义，
    方便把“代码中的局部几何参数”与“URDF 中的 link / joint 参考系”
    一一对齐。
    """

    pos: Vector3 = (0.0, 0.0, 0.0)
    """局部平移 $(x, y, z)$。"""

    rpy: Vector3 = (0.0, 0.0, 0.0)
    """局部欧拉角 $(roll, pitch, yaw)$。"""

    def __post_init__(self):
        self.pos = _ensure_tuple(self.pos, length=3, field_name="pos")
        self.rpy = _ensure_tuple(self.rpy, length=3, field_name="rpy")

    @classmethod
    def from_value(cls, value: PoseCfg | Sequence[float] | Mapping[str, Any] | None) -> PoseCfg:
        r"""把常见输入形式收敛成一个 `PoseCfg`。

        Args:
            value (PoseCfg | Sequence[float] | Mapping[str, Any] | None): 输入位姿。

        Returns:
            PoseCfg: 规范化位姿。

        Raises:
            TypeError: 当输入形式不被支持时抛出。
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
        r"""把平移和欧拉角打包为 6D tuple。

        Returns:
            Vector6: `(*pos, *rpy)`。
        """

        return (*self.pos, *self.rpy)


@dataclass
class MaterialCfg(AssetCfgBase):
    r"""可选的材质 / 颜色描述。"""

    name: str | None = None
    """材质名称，主要用于 visual 或重着色导出。"""

    rgba: Vector4 = (0.7, 0.7, 0.7, 1.0)
    """颜色分量 RGBA 与透明度。"""

    def __post_init__(self):
        self.rgba = _ensure_tuple(self.rgba, length=4, field_name="rgba")
        if self.name is not None:
            self.name = _sanitize_identifier(self.name, field_name="material.name")


@dataclass
class GeometryCfg(AssetCfgBase):
    r"""几何描述的基类。

    我们刻意把“几何本体”和“几何实例”分开：
    `GeometryCfg` 描述形状参数，而 `GeometryElementCfg`
    负责叠加位姿与可选材质。
    """

    geometry_type: ClassVar[str] = "geometry"
    """供派生类分发使用的几何类型标签。"""

    @property
    def kind(self) -> str:
        r"""返回当前几何的类型名。"""

        return self.geometry_type

    @property
    def is_primitive(self) -> bool:
        r"""判断当前几何是否属于 primitive 家族。"""

        return self.geometry_type in {"box", "cylinder", "sphere"}


@dataclass
class BoxGeometryCfg(GeometryCfg):
    r"""盒状 primitive 几何。"""

    geometry_type: ClassVar[str] = "box"
    size: Vector3
    """盒子的三边长度 $(s_x, s_y, s_z)$。"""

    def __post_init__(self):
        self.size = _ensure_tuple(self.size, length=3, field_name="box.size")
        if any(edge <= 0.0 for edge in self.size):
            raise ValueError(f"box.size must be positive, got {self.size}")


@dataclass
class CylinderGeometryCfg(GeometryCfg):
    r"""圆柱 primitive 几何。"""

    geometry_type: ClassVar[str] = "cylinder"
    radius: float
    """圆柱半径 $r$。"""

    length: float
    """圆柱长度 $l$。"""

    def __post_init__(self):
        self.radius = float(self.radius)
        self.length = float(self.length)
        if self.radius <= 0.0 or self.length <= 0.0:
            raise ValueError(f"cylinder radius/length must be positive, got {(self.radius, self.length)}")


@dataclass
class SphereGeometryCfg(GeometryCfg):
    r"""球体 primitive 几何。"""

    geometry_type: ClassVar[str] = "sphere"
    radius: float
    """球体半径 $r$。"""

    def __post_init__(self):
        self.radius = float(self.radius)
        if self.radius <= 0.0:
            raise ValueError(f"sphere.radius must be positive, got {self.radius}")


@dataclass
class MeshGeometryCfg(GeometryCfg):
    r"""网格几何。"""

    geometry_type: ClassVar[str] = "mesh"
    file_path: str
    """网格文件路径；相对还是绝对由导出层决定。"""

    scale: Vector3 = (1.0, 1.0, 1.0)
    """网格局部缩放 $(s_x, s_y, s_z)$。"""

    def __post_init__(self):
        if not isinstance(self.file_path, str) or not self.file_path.strip():
            raise ValueError("mesh.file_path must be a non-empty string")
        self.file_path = self.file_path.strip()
        self.scale = _ensure_tuple(self.scale, length=3, field_name="mesh.scale")
        if any(scale <= 0.0 for scale in self.scale):
            raise ValueError(f"mesh.scale must be positive, got {self.scale}")

    @property
    def suffix(self) -> str:
        return Path(self.file_path).suffix.lower()


GeometryValue = GeometryCfg | str | Mapping[str, Any]
"""规范化可接受的宽松几何输入。"""


def make_geometry_cfg(value: GeometryValue) -> GeometryCfg:
    r"""把宽松几何输入规范化成一个 `GeometryCfg`。

    Args:
        value (GeometryValue): 宽松几何输入。

    Returns:
        GeometryCfg: 规范化后的几何对象。

    Raises:
        TypeError: 当输入类型不受支持时抛出。
        KeyError: 当字典输入缺少几何类型字段时抛出。
        ValueError: 当几何类型值不受支持时抛出。
    """

    if isinstance(value, GeometryCfg):
        return value.copy()
    if isinstance(value, str):
        return MeshGeometryCfg(file_path=value)
    if not isinstance(value, Mapping):
        raise TypeError(f"Unsupported geometry value: {value!r}")

    geometry_type = value.get("type", value.get("kind"))
    if geometry_type is None:
        raise KeyError("Geometry dict must contain 'type' or 'kind'")

    geometry_type = str(geometry_type).lower()
    if geometry_type == "box":
        return BoxGeometryCfg(size=value["size"])
    if geometry_type == "cylinder":
        return CylinderGeometryCfg(radius=value["radius"], length=value["length"])
    if geometry_type == "sphere":
        return SphereGeometryCfg(radius=value["radius"])
    if geometry_type == "mesh":
        file_path = value.get("file_path", value.get("path", value.get("mesh")))
        return MeshGeometryCfg(file_path=file_path, scale=value.get("scale", (1.0, 1.0, 1.0)))

    raise ValueError(f"Unsupported geometry type: {geometry_type}")


@dataclass
class GeometryElementCfg(AssetCfgBase):
    r"""带局部位姿与可选材质的具体几何实例。"""

    geometry: GeometryCfg
    """底层几何描述对象。"""

    name: str | None = None
    """可选实例名，主要用于调试 / 导出。"""

    origin: PoseCfg | Sequence[float] | Mapping[str, Any] | None = field(default_factory=PoseCfg)
    """相对于所属 joint / child link 参考系的局部位姿。"""

    material: MaterialCfg | Mapping[str, Any] | None = None
    """可选材质，仅由 visual / 重着色流程消费。"""

    def __post_init__(self):
        if self.name is not None:
            self.name = _sanitize_identifier(self.name, field_name="geometry_element.name")
        self.geometry = make_geometry_cfg(self.geometry)
        self.origin = PoseCfg.from_value(self.origin)
        if self.material is not None and not isinstance(self.material, MaterialCfg):
            if not isinstance(self.material, Mapping):
                raise TypeError(f"material must be MaterialCfg or mapping, got {self.material!r}")
            self.material = MaterialCfg(**self.material)


@dataclass
class CollisionGeometryCfg(GeometryElementCfg):
    r"""用于 collision 的几何实例。"""


@dataclass
class VisualGeometryCfg(GeometryElementCfg):
    r"""用于 visual 的几何实例。"""


@dataclass
class InertiaTensorCfg(AssetCfgBase):
    r"""符合 URDF 风格的对称惯量张量。
    $$
    \mathbf{I} =
    \begin{bmatrix}
    i_{xx} & i_{xy} & i_{xz} \\
    i_{xy} & i_{yy} & i_{yz} \\
    i_{xz} & i_{yz} & i_{zz}
    \end{bmatrix}.
    $$
    """

    ixx: float
    """对角项 $i_{xx}$。"""

    iyy: float
    """对角项 $i_{yy}$。"""

    izz: float
    """对角项 $i_{zz}$。"""

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
class InertialCfg(AssetCfgBase):
    r"""单个 link 级刚体的惯性描述。

    在 URDF 里，惯性通常不是只给一个标量，而是要同时给出质量 $m$、
    惯性参考系的位置，以及在该参考系下表达的对称惯量张量：

    $$
    \mathbf{I} =
    \begin{bmatrix}
    i_{xx} & i_{xy} & i_{xz} \\
    i_{xy} & i_{yy} & i_{yz} \\
    i_{xz} & i_{yz} & i_{zz}
    \end{bmatrix}.
    $$

    这里 `inertia_padding` 不是物理项，而是工程性稳定化项，用来避免
    极薄、极小或近退化几何导致的数值不稳定。
    """

    mass: float
    """刚体质量 $m$。"""

    inertia: InertiaTensorCfg | Mapping[str, Any]
    """惯量张量描述。"""

    origin: PoseCfg | Sequence[float] | Mapping[str, Any] | None = field(default_factory=PoseCfg)
    """惯性参考系相对于 link frame 的位姿。"""

    inertia_padding: float = 0.0
    """为数值稳定性附加到对角项上的工程性 padding。"""

    def __post_init__(self):
        self.mass = float(self.mass)
        if self.mass <= 0.0:
            raise ValueError(f"mass must be positive, got {self.mass}")
        self.origin = PoseCfg.from_value(self.origin)
        if not isinstance(self.inertia, InertiaTensorCfg):
            if not isinstance(self.inertia, Mapping):
                raise TypeError(f"inertia must be InertiaTensorCfg or mapping, got {self.inertia!r}")
            self.inertia = InertiaTensorCfg(**self.inertia)
        self.inertia_padding = float(self.inertia_padding)
        if self.inertia_padding < 0.0:
            raise ValueError("inertia_padding must be >= 0")
        if self.inertia_padding > 0.0:
            # 这里仅对对角项施加稳定化 padding，不修改理论惯量的结构。
            self.inertia = InertiaTensorCfg(
                ixx=self.inertia.ixx + self.inertia_padding,
                iyy=self.inertia.iyy + self.inertia_padding,
                izz=self.inertia.izz + self.inertia_padding,
                ixy=self.inertia.ixy,
                ixz=self.inertia.ixz,
                iyz=self.inertia.iyz,
            )

    @classmethod
    def from_box(
        cls,
        size: Vector3,
        density: float,
        *,
        origin: PoseCfg | Sequence[float] | Mapping[str, Any] | None = None,
        min_mass: float = 1e-4,
        inertia_padding: float = 1e-8,
    ) -> InertialCfg:
        r"""由均匀 box primitive 构造惯性参数。

        理论上，一个均匀长方体的质量与质心惯量为：

        $$
        m = \rho s_x s_y s_z,
        $$

        $$
        \mathbf{I}_C =
        \mathrm{diag}\left(
        \frac{m}{12}(s_y^2 + s_z^2),
        \frac{m}{12}(s_x^2 + s_z^2),
        \frac{m}{12}(s_x^2 + s_y^2)
        \right).
        $$

        这里的 `min_mass` 只是工程下限，不改变几何本身的解析公式；
        `inertia_padding` 则用于给对角项增加数值缓冲。

        Args:
            size (Vector3): box 三边长度 $(s_x, s_y, s_z)$。
            density (float): 密度 $\rho$。
            origin (PoseCfg | Sequence[float] | Mapping[str, Any] | None): 惯性参考系位姿。
            min_mass (float): 质量下限，用于数值稳定。
            inertia_padding (float): 惯量对角项 padding。

        Returns:
            InertialCfg: box 推导出的惯性描述。
        """

        sx, sy, sz = _ensure_tuple(size, length=3, field_name="size")
        density = float(density)
        if density <= 0.0:
            raise ValueError("density must be positive")
        # 理论质量公式 $m = \rho V$；`min_mass` 只用于数值稳定。
        mass = max(density * sx * sy * sz, min_mass)
        # box 在质心坐标系下的主惯量。
        ixx = mass * (sy * sy + sz * sz) / 12.0
        iyy = mass * (sx * sx + sz * sz) / 12.0
        izz = mass * (sx * sx + sy * sy) / 12.0
        return cls(
            mass=mass,
            origin=origin,
            inertia=InertiaTensorCfg(ixx=ixx, iyy=iyy, izz=izz),
            inertia_padding=inertia_padding,
        )

    @classmethod
    def from_cylinder(
        cls,
        radius: float,
        length: float,
        density: float,
        *,
        origin: PoseCfg | Sequence[float] | Mapping[str, Any] | None = None,
        principal_axis: Literal["x", "y", "z"] = "z",
        min_mass: float = 1e-4,
        inertia_padding: float = 1e-8,
    ) -> InertialCfg:
        r"""由均匀 cylinder primitive 构造惯性参数。

        理论上，一个均匀圆柱的体积、质量与主惯量为：

        $$
        V = \pi r^2 l, \qquad m = \rho V,
        $$        $$
        I_{\parallel} = \frac{1}{2}mr^2, \qquad
        I_{\perp} = \frac{1}{12}m(3r^2 + l^2).
        $$

        如果 `principal_axis` 取 `x / y / z`，则只是把主惯量在三个轴向
        之间重新排列；公式本身不变。

        Args:
            radius (float): 圆柱半径 $r$。
            length (float): 圆柱长度 $l$。
            density (float): 密度 $\rho$。
            origin (PoseCfg | Sequence[float] | Mapping[str, Any] | None): 惯性参考系位姿。
            principal_axis (Literal["x", "y", "z"]): 圆柱主轴方向。
            min_mass (float): 质量下限，用于数值稳定。
            inertia_padding (float): 惯量对角项 padding。

        Returns:
            InertialCfg: cylinder 推导出的惯性描述。
        """

        radius = float(radius)
        length = float(length)
        density = float(density)
        if radius <= 0.0 or length <= 0.0 or density <= 0.0:
            raise ValueError("radius, length and density must be positive")
        # 理论体积公式 $V = \pi r^2 l$；`min_mass` 只用于稳定极小体积近似。
        volume = math.pi * radius * radius * length
        mass = max(density * volume, min_mass)
        # 平行主轴方向的惯量 $I_{\parallel}$。
        i_parallel = 0.5 * mass * radius * radius
        # 垂直主轴方向的惯量 $I_{\perp}$。
        i_perp = mass * (3.0 * radius * radius + length * length) / 12.0
        # 这里只是主惯量的轴向重排，不改变圆柱本身的解析公式。
        if principal_axis == "x":
            ixx, iyy, izz = i_parallel, i_perp, i_perp
        elif principal_axis == "y":
            ixx, iyy, izz = i_perp, i_parallel, i_perp
        else:
            ixx, iyy, izz = i_perp, i_perp, i_parallel
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
        r"""由均匀 sphere primitive 构造惯性参数。

        理论上，均匀球体的体积、质量与惯量为：

        $$
        V = \frac{4}{3}\pi r^3, \qquad m = \rho V,
        $$
        $$
        I_C = \frac{2}{5}mr^2 \mathbf{I}.
        $$

        这里 `min_mass` 仍然只是工程下限，不改变球体的解析惯量关系。

        Args:
            radius (float): 球体半径 $r$。
            density (float): 密度 $\rho$。
            origin (PoseCfg | Sequence[float] | Mapping[str, Any] | None): 惯性参考系位姿。
            min_mass (float): 质量下限，用于数值稳定。
            inertia_padding (float): 惯量对角项 padding。

        Returns:
            InertialCfg: sphere 推导出的惯性描述。
        """

        radius = float(radius)
        density = float(density)
        if radius <= 0.0 or density <= 0.0:
            raise ValueError("radius and density must be positive")
        # 理论体积公式 $V = \frac{4}{3}\pi r^3$；`min_mass` 只是数值稳定项。
        volume = 4.0 / 3.0 * math.pi * radius**3
        mass = max(density * volume, min_mass)
        # 均匀球体三轴惯量完全相同，$I = \frac{2}{5}mr^2$。
        diagonal = 0.4 * mass * radius * radius
        return cls(
            mass=mass,
            origin=origin,
            inertia=InertiaTensorCfg(ixx=diagonal, iyy=diagonal, izz=diagonal),
            inertia_padding=inertia_padding,
        )


@dataclass
class JointLimitCfg(AssetCfgBase):
    r"""关节限位及可选驱动边界。"""

    lower: float
    r"""关节下界 $q_{\min}$。"""

    upper: float
    r"""关节上界 $q_{\max}$。"""

    effort: float | None = None
    """可选的力矩 / 力上界。"""

    velocity: float | None = None
    """可选的速度上界。"""

    def __post_init__(self):
        self.lower = float(self.lower)
        self.upper = float(self.upper)
        if self.upper < self.lower:
            raise ValueError(f"upper limit must be >= lower limit, got {(self.lower, self.upper)}")
        if self.effort is not None:
            self.effort = float(self.effort)
        if self.velocity is not None:
            self.velocity = float(self.velocity)


@dataclass
class JointPropertiesCfg(AssetCfgBase):
    r"""URDF joint-level 物理属性。

    这里刻意只收第一轮已经从官方 LEAP / Allegro URDF 中确认需要继承的
    joint 层属性，而不把 link 接触材质、actuator gain 或训练期随机化混进来。
    对 AnyMani 的 pre-made 流程来说：

    - `JointLimitCfg` 描述广义坐标 $q$ 的范围和驱动上界；
    - `JointPropertiesCfg` 描述同一个 revolute joint 的轻量物理附加项；
    - link 的 `mass / inertial` 仍由 canonical geometry 单独决定。

    # NOTE:
    LEAP 官方 URDF 使用的是 `<joint_properties friction="0.0"/>`，不是标准
    URDF 的 `<dynamics friction="..."/>`。本 schema 保留这个来源语义，
    exporter v1 也按 LEAP 风格写出，避免 importer 同时看到两套 friction。
    """

    friction: float | None = None
    r"""关节摩擦系数；`None` 表示该 joint 不写 `<joint_properties>` 标签。"""

    def __post_init__(self):
        if self.friction is not None:
            self.friction = float(self.friction)  # 允许官方 URDF 字符串解析后直接传入


@dataclass
class MimicCfg(AssetCfgBase):
    r"""用于 URDF mimic 关节的 schema。"""

    joint: str
    """父关节，供 mimic 关系引用。"""

    multiplier: float = 1.0
    """线性乘子 $\alpha$。"""

    offset: float = 0.0
    """线性偏移 $\beta$。"""

    def __post_init__(self):
        self.joint = _sanitize_identifier(self.joint, field_name="mimic.joint")
        self.multiplier = float(self.multiplier)
        self.offset = float(self.offset)


def _make_collision_cfg(value: Any) -> CollisionGeometryCfg:
    r"""把宽松输入规范化成 `CollisionGeometryCfg`。

    Args:
        value (Any): 宽松的 collision 几何输入。

    Returns:
        CollisionGeometryCfg: 规范化后的 collision 几何实例。

    Raises:
        TypeError: 当输入无法解释为 collision 几何时抛出。
    """

    if isinstance(value, CollisionGeometryCfg):
        return value.copy()
    if isinstance(value, GeometryCfg) or isinstance(value, str):
        return CollisionGeometryCfg(geometry=make_geometry_cfg(value))
    if not isinstance(value, Mapping):
        raise TypeError(f"Unsupported collision geometry value: {value!r}")
    if "geometry" in value:
        return CollisionGeometryCfg(**dict(value))
    geometry_keys = {"type", "kind", "size", "radius", "length", "file_path", "path", "mesh", "scale"}
    if geometry_keys.intersection(value.keys()):
        element_kwargs = {key: value[key] for key in ("name", "origin", "material") if key in value}
        element_kwargs["geometry"] = value
        return CollisionGeometryCfg(**element_kwargs)
    raise TypeError(f"Unsupported collision geometry mapping: {value!r}")


def _make_visual_cfg(value: Any) -> VisualGeometryCfg:
    r"""把宽松输入规范化成 `VisualGeometryCfg`。

    Args:
        value (Any): 宽松的 visual 几何输入。

    Returns:
        VisualGeometryCfg: 规范化后的 visual 几何实例。

    Raises:
        TypeError: 当输入无法解释为 visual 几何时抛出。
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
class WristJointSpec(AssetCfgBase):
    r"""前溯腕关节的运动学声明。

    用于在 ``PalmBuilderCfg`` 中描述 palm 上方可选的腕关节自由度。
    所有空间量均在 **palm frame** 下表达，builder 负责将其反推为
    URDF 父→子链式 origin。

    前溯关节生成的中间 link 是纯运动学占位（无几何、无碰撞、无惯性），
    仅用于引入旋转自由度。
    """

    axis: Vector3
    r"""旋转轴方向（palm frame 下），会被自动规范化为单位向量。"""

    position: Vector3 = (0.0, 0.0, 0.0)
    r"""关节旋转中心相对于 palm origin 的位置偏移（palm frame 下）。"""

    limits: JointLimitCfg | None = None
    r"""可选的关节限位。若为 ``None``，则 builder 写出时不指定限位。"""

    def __post_init__(self):
        self.axis = _normalize_axis(self.axis)
        self.position = _ensure_tuple(self.position, length=3, field_name="wrist_joint.position")


__all__ = [
    "AssetCfgBase",
    "Vector2",
    "Vector3",
    "Vector4",
    "Vector6",
    "JointType",
    "Handedness",
    "PrimitiveGeometryType",
    "PoseCfg",
    "MaterialCfg",
    "GeometryCfg",
    "BoxGeometryCfg",
    "CylinderGeometryCfg",
    "SphereGeometryCfg",
    "MeshGeometryCfg",
    "GeometryValue",
    "GeometryElementCfg",
    "CollisionGeometryCfg",
    "VisualGeometryCfg",
    "InertiaTensorCfg",
    "InertialCfg",
    "JointLimitCfg",
    "JointPropertiesCfg",
    "MimicCfg",
    "make_geometry_cfg",
    "_FLOAT_TOLERANCE",
    "_sanitize_identifier",
    "_ensure_tuple",
    "_normalize_axis",
    "_ensure_list",
    "_make_collision_cfg",
    "_make_visual_cfg",
    "WristJointSpec",
]
