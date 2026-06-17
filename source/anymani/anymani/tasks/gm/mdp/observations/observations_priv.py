r"""Privileged observation design notes for GM teacher policies.

TODO(privileged obs): 物体物理属性属于 teacher-only 信息，sim2real 不可得。

本项目主线是手型泛化，但保留物体泛化接口。物体泛化必须在 teacher 训练阶段引入
多元物体资产并通过 privileged info 显式条件化策略。本段定义从仿真器提取哪些 raw
physical values，交给 `distill/models` 侧的 object token encoder 投影为 `[OBJ]` 全局 token。

物体表征路线（已与用户对齐）：当前阶段采用扩展 HORA 路线：

- teacher 显式喂 raw physical properties（mass, scale, friction, COM 等），不做 HORA 风格
  的压缩嵌入，避免嵌入与具体手型策略耦合；
- 策略侧由 `distill/models` 将 raw values 投影为 `[OBJ]` token，并在 self-attention 池中与
  joint tokens 交互；
- `[OBJ]` token 的 projection 模块未来可额外接收手形态特征，使物体表征对当前 hand
  embodiment 有感知；
- 若几何形状多样性成为瓶颈，`[OBJ]` token 可升级为多个静态 mesh token（BPS / 几何描述符，
  离线预计算）。

privileged obs 未来应从仿真器提取约 21 维 raw physical values：

- object mass $m$；
- object scale $(s_x,s_y,s_z)$；
- object friction $\mu$；
- object COM offset $(dx,dy,dz)$；
- object pose $(x,y,z,q_w,q_x,q_y,q_z)$；
- object velocity $(v_x,v_y,v_z,\omega_x,\omega_y,\omega_z)$。

数据来源包括 `RigidObject.root_physx_view`、`object.data.root_pos_w`、
`object.data.root_quat_w`、`object.data.root_lin_vel_w` 与 `object.data.root_ang_vel_w`。
所有值均为 SI 物理量，不依赖手构型，满足跨手型泛化的表征解耦需求。

本文件当前只保留科研规格，不注册 ObsTerm。具体投影维度、是否拆分为多个 token、
是否加 position embedding，由 `distill/models` 的 Specialist Policy Transformer 设计决定。
"""

from __future__ import annotations

__all__: list[str] = []
