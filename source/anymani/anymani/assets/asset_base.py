"""资产基类

该文件主要描述灵巧手的资产基类，包括关节、手指和手三个不同层级。手包含多个手指和手掌，手指包含多个关节。每个关节具有不同的属性，如限位、碰撞mesh等。
通过定义这些类，我们可以组合创建各种不同类型的手型资产，为跨手型泛化任务提供多种资产来源
"""

from __future__ import annotations

import argparse
from typing import Literal, Union
import math

import xml.etree.ElementTree as ET  # 核心描述层别用 ET，ET 放到最后的 URDF 序列化层更合适

class joint:  # 构成关节self-contained,后续生成方法放在其他类
	'''关节类

	关节对象，构成手指的最小基本单元。对应到urdf中，joint坐标系和child link坐标系完全重合，因此设定joint属性也包含了child link的属性。
	
	主要属性包括name, type, parent, child, axis, limit等。
	
	基础操作：
	- 计算简单派生量
	- 做字段合法性检查
	'''
	OriginType = Union[
		tuple[float, float, float, float, float, float],
		dict[str, tuple[float, float, float]]
	]

	MeshType = Union[  # TODO：这段相对复杂，可能要考虑其他的处理方法，因为我既可以选用单一的简单形状，如box,这种好处理的多，
		# 但也能选用指定路径（绝对/相对）的stl/obj文件，这种也还好。但还有多种基本形状复合而成的，以及基本形状和指定路径文件复合的
		str,

	]

	InertialType = Union[  # TODO:要包括mass，origin,inertia属性
		tuple[float, float, float, float, float, float],
		dict[str, tuple[float, float, float, float, float, float]]
	]

	def __init__(self, joint_name: str, joint_type: Literal["revolute", "fixed"], joint_parent: str, joint_child: str, 
			  joint_axis: tuple[float, float, float], joint_limit: tuple[float, float], joint_origin: OriginType, inertial: InertialType,
			  mesh_type: MeshType, mesh_origin: OriginType) -> None:
		'''初始化关节对象，设置默认属性值

		Args:
			joint_name (str): 关节名称
			joint_type (str): 关节类型
			joint_parent (str): 父链接名称
			joint_child (str): 子链接名称
			joint_axis (tuple[float, float, float]): 关节轴向量
			joint_limit (tuple[float, float]): 关节限位
			joint_origin (OriginType): 关节坐标系（相对于base节点或上一关节坐标）的原点位置 $p$ 和姿态 $R$
				- 可以是一个包含6个元素的元组，分别表示位置和姿态，也可以是一个字典，包含位置和姿态的键值对
			inertial (InertialType): 惯性属性
			mesh_type (str): 网格类型
			mesh_origin (OriginType): 网格相对于关节坐标系的的原点位置 $p$ 和姿态 $R$

		'''
		# 不允许关节名称以数字开头，避免后续转换为usd处理时出现问题
		if joint_name and joint_name[0].isdigit():
			joint_name = f"a_{joint_name}"
		self.joint_name = joint_name

		# 目前只接受旋转和固定两种类型
		if joint_type not in {"revolute", "fixed"}:  # fixed一般代表指尖虚拟关节
			raise ValueError(f"invalid joint_type: {joint_type}, must be 'revolute' or 'fixed'")
		self.joint_type = joint_type  
		
		self.joint_parent = joint_parent
		self.joint_child = joint_child

		# 进行归一化处理
		joint_axis = self._normalize_axis(joint_axis)
		self.joint_axis = joint_axis

		# 关节上限必须大于下限
		if joint_limit[1] < joint_limit[0]:
			raise ValueError(f"upper limit must be greater than lower limit, got {joint_limit}")
		self.joint_limit = joint_limit

		self.pose

	@staticmethod
	def _normalize_axis(axis: tuple[float, float, float]) -> tuple[float, float, float]:
		'''对轴向量进行归一化处理，确保其长度为1'''
		if len(axis) != 3:
			raise ValueError(f"axis must have 3 components, got {axis}")

		x, y, z = map(float, axis)
		norm = math.sqrt(x * x + y * y + z * z)

		if norm == 0:
			raise ValueError("axis cannot be zero vector")

		return x / norm, y / norm, z / norm

	

class finger:
	'''手指类

	手指对象，由多个关节组成。对应到urdf中，finger是一个逻辑单元，包含多个joint和link。
	主要属性包括name, parent, child等。
	'''
	def __init__(self) -> None:
		pass

class palm:
	pass

class hand:
	pass

j = joint()