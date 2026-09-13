r"""SAC的紧凑动态观测、静态手型表与合法Actor输入重建。

资产索引只用于采样/查静态表，不拼入神经网络。N040特征在抽样后按实际q重建，
Actor与Critic可以共享只读观测和冻结几何，但不共享可训练参数或特权Actor输入。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from anymani.distill.models.palm_rotation_policy import PalmRotationActorObservation, PalmRotationGeometry

from .replay import DYNAMIC_SHAPES

STATIC_FIELDS = ("actor_jnt_limits", "jnt_valid", "tip_valid", "owner_valid", "shortest_path", "parent_direction", "child_direction")
ACTOR_FLAT_DIM = 6585  # current80+history2400+limits32+contact21+geometry2688+graphs1323+masks41。
CRITIC_FLAT_DIM = ACTOR_FLAT_DIM + 130  # 额外privileged joint64+contact42+object15+task8+release1。
_INDEX_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)  # 资产身份只接受精确整数。


def compact_observation(observation: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    r"""只交付回放允许的动态字段；历史与几何tokens不进入持久回放数组。"""
    return {name: observation[name].detach() for name in DYNAMIC_SHAPES}


def compact_raw_groups(observation: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    r"""从任务原始policy/critic组读取单步帧，供reset之前捕获真实终点。"""
    policy, critic = observation["policy"], observation["critic"]
    return {"actor_jnt_current": policy["jnt_current"], "actor_owner_contact": policy["owner_contact"],
            "critic_jnt_state": critic["jnt_state"], "critic_owner_contact": critic["owner_contact"],
            "critic_obj": critic["obj"], "critic_task": critic["task"], "critic_reward_release": critic["reward_release"]}


def actor_inputs(observation: Mapping[str, torch.Tensor]) -> tuple[PalmRotationActorObservation, PalmRotationGeometry]:
    r"""按信息边界构造Actor数据：强制清除非TIP接触和所有ghost动态/几何槽。

    joint_current/history的前三通道是q/pi、u/pi、上一动作，第3号通道是非TIP自身接触，
    在TIP-only下恒零；第4号通道保留所属手指TIP触觉。所有编号从0开始。
    """
    joint = observation["jnt_valid"].bool()
    tip = observation["tip_valid"].bool()
    owner = observation["owner_valid"].bool()
    current = observation["actor_jnt_current"].float().masked_fill(~joint[..., None], 0).clone()
    history = observation["actor_jnt_history"].float().masked_fill(~joint[:, None, :, None], 0).clone()
    current[..., 3] = 0  # Actor不能读取指节接触，即使caller意外传入该通道。
    history[..., 3] = 0
    contact = observation["actor_owner_contact"].float().masked_fill(~owner[..., None], 0).clone()
    contact[:, :17] = 0  # PALM和JOINT触觉不进入部署策略。
    pair = owner[:, :, None] & owner[:, None, :]
    geometry = PalmRotationGeometry(
        tokens=observation["geometry_tokens"].float().masked_fill(~owner[..., None], 0),
        owner_valid=owner,
        shortest_path=observation["shortest_path"].long().masked_fill(~pair, 0),
        parent_direction=observation["parent_direction"].long().masked_fill(~pair, 0),
        child_direction=observation["child_direction"].long().masked_fill(~pair, 0),
    )  # 图是合法形态证据，ghost对之间的离散值不进入embedding。
    actor = PalmRotationActorObservation(
        jnt_current=current, jnt_history=history,
        jnt_limits=observation["actor_jnt_limits"].float().masked_fill(~joint[..., None], 0),
        owner_contact=contact, jnt_valid=joint, tip_valid=tip, owner_valid=owner,
    )
    return actor, geometry


def actor_flat_features(observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
    r"""供可选FlashSAC MLP Actor使用的完整合法输入；不包含任何critic字段或资产ID。"""
    actor, geometry = actor_inputs(observation)
    batch = actor.jnt_current.shape[0]
    fields = (actor.jnt_current, actor.jnt_history, actor.jnt_limits, actor.owner_contact,
              geometry.tokens, geometry.shortest_path, geometry.parent_direction, geometry.child_direction,
              actor.jnt_valid, actor.tip_valid, actor.owner_valid)
    return torch.cat([value.reshape(batch, -1).float() for value in fields], dim=-1)


def critic_flat_features(observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
    r"""Q的独立特征输入：合法历史/几何加特权当前状态，完全不调用Actor的可训练骨架。"""
    joint, owner = observation["jnt_valid"].bool(), observation["owner_valid"].bool()
    batch = joint.shape[0]
    privileged = (
        observation["critic_jnt_state"].float().masked_fill(~joint[..., None], 0),
        observation["critic_owner_contact"].float().masked_fill(~owner[..., None], 0),
        observation["critic_obj"], observation["critic_task"], observation["critic_reward_release"],
    )
    return torch.cat([actor_flat_features(observation), *[value.reshape(batch, -1).float() for value in privileged]], dim=-1)


def concatenate_observations(first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    r"""current/next沿batch轴合并，保持上游cross-batch归一化的样本配对语义。"""
    if first.keys() != second.keys():
        raise ValueError("current and next observation fields must match")
    return {key: torch.cat((first[key], second[key]), dim=0) for key in first}


class StaticHandBank:
    r"""每资产一份限位、mask与图；只做几何证据查找，不给神经网络传入索引。"""

    def __init__(self, fields: Mapping[str, torch.Tensor]):
        self.fields = {name: fields[name].detach().cpu().clone() for name in STATIC_FIELDS}
        counts = {value.shape[0] for value in self.fields.values()}
        if len(counts) != 1 or next(iter(counts)) < 1:
            raise ValueError("static hand fields must share a nonempty asset axis")
        self.asset_count = next(iter(counts))

    @classmethod
    def from_live(cls, observation: Mapping[str, torch.Tensor], asset_indices: torch.Tensor) -> StaticHandBank:
        r"""用首次合法观测建立静态表，并验证同资产不同副本完全一致。"""
        # 路由是离散身份，不允许浮点截断、bool mask或Python负索引改变所指的手型。
        if asset_indices.dtype not in _INDEX_DTYPES:
            raise TypeError("static asset indices must have an integer dtype")
        if asset_indices.ndim != 1 or asset_indices.numel() != observation["jnt_valid"].shape[0]:
            raise ValueError("static asset indices must match the live environment axis")
        if not asset_indices.numel() or bool((asset_indices < 0).any()):
            raise ValueError("static asset indices must be nonempty and nonnegative")
        labels = asset_indices.detach().to(device="cpu", dtype=torch.long)  # 验证之后转换存储dtype，数值身份不变。
        asset_count = int(labels.max()) + 1  # 选中资产必须完整覆盖0..A-1。
        first = []  # 每资产选择一个静态参照副本，随后核对该资产所有副本。
        for asset in range(asset_count):
            rows = torch.where(labels == asset)[0]
            if not rows.numel():
                raise ValueError("static bank needs every selected asset")
            first.append(int(rows[0]))
        cpu = {name: observation[name].detach().cpu() for name in STATIC_FIELDS}
        table = {name: value[first].clone() for name, value in cpu.items()}
        for name, value in cpu.items():
            if not torch.equal(table[name][labels], value):
                raise ValueError(f"replicas disagree on static {name}")
        return cls(table)

    def assemble(self, dynamic: Mapping[str, torch.Tensor], asset_indices: torch.Tensor, geometry_provider: Any,
                 device: torch.device | str) -> dict[str, torch.Tensor]:
        r"""重建一个SAC样本批次；冻结N040仅按实际q与形态解析当前tokens。

        dynamic包含回放重建的History30。geometry_provider.resolve接受资产查找索引和
        PalmRotationActorObservation，返回只读几何tokens；与现有provider契约一致。
        """
        # 查表前完成离散身份与批轴验证；provider只接收已经证明合法的资产索引。
        if asset_indices.dtype not in _INDEX_DTYPES:
            raise TypeError("replay asset indices must have an integer dtype")
        if asset_indices.ndim != 1 or asset_indices.numel() != dynamic["actor_jnt_current"].shape[0]:
            raise ValueError("replay asset indices must match the dynamic batch axis")
        if not asset_indices.numel():
            raise ValueError("replay geometry batch must be nonempty")
        indices = asset_indices.detach().to(device="cpu", dtype=torch.long)  # 整数变换保持精确资产身份。
        if bool(((indices < 0) | (indices >= self.asset_count)).any()):
            raise ValueError("replay asset index is outside the frozen bank")
        result = {name: value.to(device=device, dtype=torch.float32) for name, value in dynamic.items()}
        result.update({name: value[indices].to(device=device) for name, value in self.fields.items()})
        actor = PalmRotationActorObservation(
            jnt_current=result["actor_jnt_current"], jnt_history=result["actor_jnt_history"],
            jnt_limits=result["actor_jnt_limits"], owner_contact=result["actor_owner_contact"],
            jnt_valid=result["jnt_valid"].bool(), tip_valid=result["tip_valid"].bool(), owner_valid=result["owner_valid"].bool(),
        )
        with torch.no_grad():
            geometry = geometry_provider.resolve(indices.to(device=device), actor)
        result["geometry_tokens"] = geometry.tokens.detach()
        return result

    def state_dict(self) -> dict[str, torch.Tensor]:
        r"""静态表很小，独立保存到模型检查点即可重建输入契约。"""
        return {name: value.clone() for name, value in self.fields.items()}
