r"""Deployable structured inputs 到统一 token contract 的可学习前端。

input adapter 属于模型并随 SSL checkpoint 一起迁入 PPO。当前隐式主线消费当前 $q$、有序
screw/topology、home geometry 与 physical anchors。解析直接压缩只保留为未来候选 adapter 占位；
若激活，可消费缓存支撑点经当前 FK/刚体位姿得到的 current physical points。pretraining target、
distance/最近点/Jacobian、future state、contact、command、history 与 object state 均不得进入几何 adapter。

route adapter、backbone 与 $z^{(0)}/z_i^{(1)}$ heads 的复合构成 retained geometry encoder。这里不再建立一套与 PPO backbone
平行的“representation encoder”，避免 field target 更换时无意更换网络主干。
"""
