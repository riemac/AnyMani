r"""由同一资产语义生成静态关节运动学输入与单点 FK 监督。

每个真实关节的15维输入为 $[t^0/L,\operatorname{vec}R^0,a]$，其中 $L=0.1$m，
$a$ 是关节局部单位轴；$T^0$ 把最近活动父关节的child frame接到当前关节零角frame。
固定根段或spacer完整折入此变换。没有活动父关节时，父frame为统一手语义坐标系{h}。
因此基础输入已经足以确定FK，不要求基线从关节数量或ID猜测尺寸。

角度使用绝对物理$q$；先以既有POE实现求$q=0$的基准变换，再构造
$T_i(q)=T_{parent(i)}(q)T_i^0\operatorname{Rot}(a_i,q_i)$。
这一做法同时处理资产非零$q_{home}$，不把$q_{home}$再次加到物理q。
输出是关节frame原点的位置（m），不是collision centroid或离轴指尖点。
自身角度不改变自身轴上原点，但会改变后继关节位置。

来源是HandContainer交付的typed geometry_semantics；不重读URDF、不构造模型，
也不读取物体、接触、动作或任何任务奖励。padding/slot映射只采用调用方显式routing。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from anymani.assets.asset_schema_geometry import HandGeometrySemanticsCfg

from .kinematics import forward_owner_transforms, lower_hand_geometry_semantics

LENGTH_SCALE_M = 0.1  # 统一长度尺度保留跨手尺寸差异，不按每手尺寸归一。
JOINT_KINEMATICS_WIDTH = 15  # 平移3 + 旋转矩阵9 + 局部单位轴3。


@dataclass(frozen=True)
class JointKinematicsBank:
    r"""固定canonical关节轴上的批量物理运动学；无学习参数。

    features: [A,J,15]，前三维为t/L，其余无量纲。
    parent_slot: [A,J]，根为-1；depth: [A,J]，根0、ghost=-1。
    valid: [A,J]布尔mask；不同手指通过各自父链保持严格独立。
    """

    features: torch.Tensor  # 共享给三组学生的基础运动学输入。
    parent_slot: torch.Tensor  # 最近真实活动父关节；不暴露资产ID为特征。
    depth: torch.Tensor  # 计算次序，非学习表征。
    valid: torch.Tensor  # 唯一真实关节mask。

    def __post_init__(self) -> None:
        """在静态装配时拒绝非法旋转、轴、ghost父和非拓扑排序关系。"""
        if self.features.ndim != 3 or self.features.shape[-1] != JOINT_KINEMATICS_WIDTH:
            raise ValueError("joint kinematics features must have shape [A,J,15]")
        shape = self.features.shape[:2]  # A为资产数，J为canonical关节槽数。
        if any(x.shape != shape for x in (self.parent_slot, self.depth, self.valid)):
            raise ValueError("joint kinematics metadata shape disagrees with features")
        if not shape[0] or not shape[1] or self.valid.dtype != torch.bool:
            raise ValueError("joint kinematics needs nonempty axes and boolean valid mask")
        if self.parent_slot.dtype != torch.long or self.depth.dtype != torch.long:
            raise ValueError("joint parent and depth must be int64")
        if len({x.device for x in (self.features, self.parent_slot, self.depth, self.valid)}) != 1:
            raise ValueError("joint kinematics tensors must share a device")
        if not self.features.dtype.is_floating_point or not torch.isfinite(self.features).all():
            raise ValueError("joint kinematics features must be finite floating point")
        if torch.count_nonzero(self.features[~self.valid]) or (self.depth[~self.valid] != -1).any():
            raise ValueError("ghost joint features must be zero and depth -1")
        parent = self.parent_slot  # 此处仅做一次性static检查，不进入每步GPU路径。
        if ((parent < -1) | (parent >= shape[1])).any():
            raise ValueError("joint parent slot lies outside the canonical axis")
        has_parent = self.valid & (parent >= 0)  # 只有活动关节会参与父链。
        safe_parent = parent.clamp_min(0)  # root暂映射0，随后由has_parent排除。
        if (has_parent & ~self.valid.gather(1, safe_parent)).any():
            raise ValueError("active joint parent must be another active joint")
        parent_depth = self.depth.gather(1, safe_parent)  # 批量读取父关节深度。
        if (has_parent & (self.depth != parent_depth + 1)).any():
            raise ValueError("joint parent must be exactly one depth before its child")
        if (self.valid & ~has_parent & (self.depth != 0)).any():
            raise ValueError("root joint depth must be zero")
        active_features = self.features[self.valid]  # 布尔mask先展平资产/关节轴，再选择特征列。
        rotation = active_features[:, 3:12].reshape(-1, 3, 3).double()  # 静态真值检查独立于全局TF32开关。
        eye = torch.eye(3, dtype=rotation.dtype, device=self.features.device)
        if not torch.allclose(rotation.transpose(-1, -2) @ rotation, eye.expand_as(rotation), atol=2e-5, rtol=0):
            raise ValueError("joint rest rotation is not orthogonal")
        if not torch.allclose(
            torch.linalg.det(rotation),
            torch.ones(rotation.shape[0], device=rotation.device, dtype=rotation.dtype),
            atol=2e-5,
            rtol=0,
        ):
            raise ValueError("joint rest rotation must be proper SO(3)")
        norms = torch.linalg.vector_norm(active_features[:, 12:15], dim=-1)  # 局部轴单位长度。
        if not torch.allclose(norms, torch.ones_like(norms), atol=2e-5, rtol=0):
            raise ValueError("joint local axis must have unit length")

    def to(self, device: torch.device | str) -> JointKinematicsBank:
        """把同一不可学习bank移到目标设备，保持dtype及全部物理数值。"""
        return JointKinematicsBank(
            *(value.to(device) for value in (self.features, self.parent_slot, self.depth, self.valid))
        )

    def joint_origins(self, q_rad: torch.Tensor, asset_index: torch.Tensor) -> torch.Tensor:
        r"""计算[B,J,3]手坐标关节原点；ghost精确为零。

        Rodrigues旋转为 $R=I+\sin(q)[a]_\times+(1-\cos(q))[a]_\times^2$。
        同一深度并行计算，避免按全局joint槽号把不同手指串成一条链。
        asset_index只路由静态物理数据，不进入学生连续输入。
        """
        if q_rad.ndim != 2 or q_rad.shape[1] != self.features.shape[1] or asset_index.shape != q_rad.shape[:1]:
            raise ValueError("joint FK expects q shape [B,J] and asset_index [B]")
        if (
            q_rad.device != self.features.device
            or asset_index.device != q_rad.device
            or asset_index.dtype != torch.long
        ):
            raise ValueError("joint FK inputs must share the bank device and int64 asset indices")
        if q_rad.dtype != self.features.dtype:
            raise ValueError("joint FK q and static features must share a floating dtype")
        feature = self.features[asset_index]  # [B,J,15]，同一批可混合不同父链。
        valid, depth = self.valid[asset_index], self.depth[asset_index]  # 物理mask与计算深度。
        parent = self.parent_slot[asset_index]  # [B,J]，root=-1。
        axis = feature[..., 12:15]  # 位于joint-local frame的单位轴。
        x, y, z = axis.unbind(-1)  # Rodrigues斜对称矩阵分量。
        zero = torch.zeros_like(x)  # 形状[B,J]，不产生固定batch尺寸常量。
        skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(*q_rad.shape, 3, 3)
        eye = torch.eye(3, dtype=q_rad.dtype, device=q_rad.device)  # root初始方向I。
        motion = eye + q_rad.sin()[..., None, None] * skew + (1 - q_rad.cos())[..., None, None] * (skew @ skew)
        local_rotation = feature[..., 3:12].reshape(*q_rad.shape, 3, 3) @ motion  # R0 Rot(a,q)。
        local_translation = feature[..., :3] * LENGTH_SCALE_M  # 恢复米制t0。
        rotation = eye.expand(*q_rad.shape, 3, 3)  # 逐深度建立实际child frame方向。
        position = torch.zeros(*q_rad.shape, 3, dtype=q_rad.dtype, device=q_rad.device)  # 手frame原点为0。
        parent_index = parent.clamp_min(0)  # root通过下面的where恢复单位变换。
        max_depth = int(self.depth.max().item())  # static上限，与动态q无关。
        for level in range(max_depth + 1):
            parent_rotation = rotation.gather(1, parent_index[..., None, None].expand(-1, -1, 3, 3))
            parent_position = position.gather(1, parent_index[..., None].expand(-1, -1, 3))
            parent_rotation = torch.where((parent < 0)[..., None, None], eye, parent_rotation)  # 根接{h}。
            parent_position = torch.where((parent < 0)[..., None], 0.0, parent_position)  # 根平移0。
            next_position = (parent_rotation @ local_translation.unsqueeze(-1)).squeeze(-1) + parent_position
            next_rotation = parent_rotation @ local_rotation  # 父姿态把当前局部轴旋转传给后继。
            at_level = valid & (depth == level)  # 只更新本层的真实关节。
            position = torch.where(at_level[..., None], next_position, position)
            rotation = torch.where(at_level[..., None, None], next_rotation, rotation)
        return position  # ghost从未进入更新，保持精确零。


def build_joint_kinematics_bank(
    semantics: Sequence[HandGeometrySemanticsCfg],
    mappings: Sequence[Mapping[str, int]],
    *,
    joint_count: int = 16,
    dtype: torch.dtype = torch.float32,
) -> JointKinematicsBank:
    r"""从已审计typed语义和显式source-name→canonical-slot映射装配bank。

    最近活动父关节之间的固定链由已有POE源完整折叠。使用float64零角放置来减小
    相邻frame求相对变换时的误差，最终再按声明dtype交付输入与FK真值。
    """
    if not semantics or len(semantics) != len(mappings):
        raise ValueError("joint kinematics needs aligned nonempty semantics and routing mappings")
    count = len(semantics)  # 资产数A。
    feature = torch.zeros(count, joint_count, JOINT_KINEMATICS_WIDTH, dtype=torch.float64)
    parent = torch.full((count, joint_count), -1, dtype=torch.long)  # -1也为ghost占位。
    depth = torch.full_like(parent, -1)  # 仅真实关节获得非负深度。
    valid = torch.zeros(count, joint_count, dtype=torch.bool)  # canonical真实关节轴。
    for asset, (item, mapping) in enumerate(zip(semantics, mappings, strict=True)):
        if set(mapping) != set(item.active_joint_names) or len(set(mapping.values())) != len(mapping):
            raise ValueError("source joint names must map one-to-one to canonical slots")
        if any(not 0 <= slot < joint_count for slot in mapping.values()):
            raise ValueError("canonical joint slot lies outside target axis")
        spec = lower_hand_geometry_semantics(item, dtype=torch.float64)  # 同一物理真源。
        zero_pose = forward_owner_transforms(spec, torch.zeros(1, len(item.active_joint_names), dtype=torch.float64))[0]
        owners = {owner.joint_name: owner.owner_index for owner in item.owners if owner.role == "joint"}
        joints = {joint.joint_name: joint for joint in item.kinematic_joints if joint.joint_type == "revolute"}
        if set(owners) != set(mapping):
            raise ValueError("FK origin target requires one named JOINT reference frame per active joint")
        for index, name in enumerate(item.active_joint_names):
            slot = mapping[name]  # 只依赖显式routing，不推断finger命名顺序。
            ancestors = torch.nonzero(spec.joint_ancestor_mask[index], as_tuple=False).flatten()
            transform = zero_pose[owners[name]]  # 当前关节child frame在q=0下的{h}变换。
            level = int(ancestors.numel())  # 固定关节已折入相对frame，不增加活动深度。
            if level:
                parent_source = int(ancestors[spec.joint_ancestor_mask[ancestors].sum(dim=1).argmax()])
                parent_name = item.active_joint_names[parent_source]  # 最近活动父关节。
                parent[asset, slot] = mapping[parent_name]
                transform = torch.linalg.inv(zero_pose[owners[parent_name]]) @ transform  # Tparent^-1 Tchild。
            feature[asset, slot, :3] = transform[:3, 3] / LENGTH_SCALE_M
            feature[asset, slot, 3:12] = transform[:3, :3].reshape(-1)
            feature[asset, slot, 12:15] = torch.tensor(joints[name].axis_local, dtype=torch.float64)
            valid[asset, slot], depth[asset, slot] = True, level  # 按同一source实体建立mask。
    return JointKinematicsBank(feature.to(dtype=dtype), parent, depth, valid)
