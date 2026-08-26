r"""Token 类型与分组语义契约 —— palm / joint / tip 三类异构 token 的定义。

对应 `Research/总体/网络架构.md` §1（基本立场）与 §3（分组投影主干）。

== 为什么分三类，而不是统一成 joint token ==

手的几何与接触语义并不全部属于 revolute joint，强行统一会被迫给无关节状态
的实体填充假的 `q/dq/last_action` 槽位，污染表征：

- `PALM` : 不是可控关节，**没有** proprioceptive state（无 $q,\dot q$，无动作）。
           当前资产 palm 几何暂不做 post-mutate；它更像 hand-level anchor /
           global context。post-mutate 的挂载点扰动也不应被 flatten 成 palm
           token 的固定长度属性，因为手指数可变，且每个 mount 必须和具体
           finger/root joint 保持 binding。更原则的表达是：mount 是
           `PALM -> root JOINT` 的关系 / 边特征。
- `JOINT`: revolute 可控关节，**唯一**同时拥有动态状态（$q,\dot q$, last_action）
           与动作输出的类型。actor action head **只**作用于此类 token。
           为保持 joint token 真正同构，incoming pose / mount pose 不放在
           某些 root joint 的私有特征槽里，而是进入 edge feature。
- `TIP`  : fingertip，通常是 fixed link，**没有**关节状态，但它是接触主场，
           承载指尖几何（mesh 描述符）与 per-fingertip 接触观测。
           参与 attention 通信，但**不**输出关节动作。

== 与参考工作的区别（见 网络架构.md §2）==

- `get-zero` : 纯 revolute joint token，palm/tip 被压扁挂进相邻 joint 的特征槽，
               几何表达弱（只有 origin/length/degree，无 mesh）。
- `tro-grasp`: 所有带几何的 link 都是节点（含 palm/tip），靠几何编码自然区分角色；
               但其输出是每 link 的 SE(3)，节点定义服务于 link-space 动作，
               不能照搬到我们的 joint-space 动作输出。

本项目取折中：**输入按 palm/joint/tip 分组**（借 tro-grasp 的语义分离与几何思想），
**输出仍 joint-centric**（符合 joint-space 动作空间）。

== chirality / handedness 的处理原则 ==

不把 `left/right` 当成 policy 输入的基础 type embedding。所谓左/右手更像是
palm frame 下各 finger root / mount 的连续位姿集合所诱导出来的人类标签，而不是
独立物理量。像 `jinghand.urdf` 这类布局，单独问“它是左手还是右手”本身就有点牵强；
真正可观测的是 palm frame 中各 finger root 的 pose set 与拓扑连接。

因此 chirality 应由 mount / edge geometry 学出来，`left/right` 最多作为数据集
metadata、评估分组或可视化标签，不默认喂给策略网络。

== 形状符号约定（贯穿本包）==

- $B$         : batch size
- $D$         : 统一 token 隐空间维度（投影后）
- $N_E$       : 当前结构模式的 PALM/JOINT/TIP physical entity 数
- $N_J$       : 当前结构模式的 actuated JOINT 数
- $N_T$       : 当前结构模式的 TIP 数
- $T=N_E$     : 当前实体表征序列长度
- 同一次前向固定结构模式，不使用 entity padding；不同结构模式分别前向

TOAGENT:
    本文件为设计契约。TokenType 枚举可先落为稳定结构（类型集合已较明确），
    但各类 token 的**具体特征字段与维度**仍属 `网络架构.md` §10 待定项，
    不在此写死数值。注释不可删，可补充。
"""

from __future__ import annotations

import enum


class TokenType(enum.IntEnum):
    r"""手部 token 的语义角色枚举。

    用整数枚举（IntEnum）便于：① 作为 type embedding 的查表索引；
    ② 在 batched tensor 中以 `int` 标记每个 token 的类型，供输出路由与
    bias 计算使用。

    取值含义见模块 docstring 的「为什么分三类」。顺序约定为
    `PALM < JOINT < TIP`，与聚合时 `concat([palm, joint, tip])` 的
    token 轴顺序一致，便于按区段切片做输出路由。

    NOTE: 这是**语义角色**，不是 URDF link 类型的机械映射。例如一个 fixed
          tip link 与一个 revolute joint 的 child link 都可能携带几何，
          但它们的 token 角色由"是否有关节状态 / 是否为接触主场"决定，
          而非 link 是否 fixed。
    """

    # palm：hand-level anchor / global context，无关节状态，通常仅 1 个
    PALM = 0
    # joint：revolute 可控关节，唯一同时有动态状态与动作输出的类型
    JOINT = 1
    # tip：fingertip，无关节状态，承载指尖几何与接触，不输出动作
    TIP = 2


# NOTE: 当前 geometry encoder 的实体角色固定为 PALM/JOINT/TIP，不增加 `[HAND]` 或
#       `[CMD]` token。command 不属于自身几何输入；若未来策略侧比较全局 token，必须作为
#       下游独立消融，不能改写 retained geometry package 的角色轴。
# NOTE: 不在这里把 `left/right` 作为候选输入。chirality 应优先从 palm-frame mount
#       layout / edge geometry 中学习出来，而非用离散标签直接注入。


def action_bearing_types() -> tuple[TokenType, ...]:
    r"""返回**输出关节动作**的 token 类型集合。

    供 `models/heads/action.py` 的 action head 与输出路由使用：只有这些类型的 token
    会被送入 actor head 产生关节增量动作分布参数。

    当前仅 `JOINT` 出动作（见模块 docstring）。封装为函数而非散落常量，
    是为了让"哪些类型出动作"成为单一事实源，便于未来若引入新可控类型时
    （如某些欠驱手的耦合关节）集中修改。

    Returns:
        tuple[TokenType, ...]: 出动作的 token 类型元组，当前为 `(JOINT,)`。
    """

    # 仅 revolute joint 出动作；palm/tip 仅参与表征通信
    return (TokenType.JOINT,)
