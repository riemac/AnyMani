r"""逐 PALM/JOINT/TIP owner 的 parametric Gaussian-field baseline decoder contract。

decoder 从第 $g$ 个 entity latent 输出固定上限 $R$ 组非负 amplitude、hand-frame center 与
symmetric positive-definite covariance。$G=N_E$，entity、surface owner 与 decoder axis 直接
同索引，角色固定为 PALM/JOINT/TIP。当前不训练独立 whole-hand union prediction；若需要联合
诊断，只能在共同 query grid 上从逐 owner induced field 解析派生。

输出不是 policy action，也不是唯一可监督的 component set。训练时先由
``representations.fields.gaussian_field`` 在 sampled queries 上计算 induced density，再用
query-space field loss 对齐 physical density target；component permutation 不影响 loss。

实现阶段必须显式解决：amplitude positivity、covariance SPD parameterization、最小/最大
axis scale、component collapse、无效 group mask 与 $R$ 的计算预算。该 decoder 默认在
PPO 前删除，backbone latent 不被限制为只能解释成 Gaussian 参数。
"""
