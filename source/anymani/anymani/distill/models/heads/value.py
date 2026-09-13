r"""Shared/asymmetric critic value-head contract。

critic 输出 scalar $V(s)$，但 pooling 仍未裁定：可以从全部有效 token、JOINT-only pool、
PALM/hand-level latent 或独立 privileged critic trunk 读取。任何选择都必须声明 actor 与
critic 各自可见的 observation、normalization 与 mask，不得让 privileged target 泄漏进
deployable actor。

mean/attention/palm/hand-token pooling 都是候选，不在 scaffold 中设默认。异构训练还需要
未来记录 per-variant value error 与 explained variance，判断共享 critic 是否造成 advantage
bias；该诊断属于后续 PPO fairness stage，不在当前 representation 实现。
"""
