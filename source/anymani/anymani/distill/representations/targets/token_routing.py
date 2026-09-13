r"""PALM/JOINT/TIP 表面归属体到同索引实体表征的监督路由契约。

经过审核的结构模式具有 $G=N_E$ 个实体/表面归属体，第 $g$ 个实体、归属体与 SSL 解码轴
直接同索引。routing metadata 负责保留 PALM/JOINT/TIP 角色、official asset mapping、ancestor
relation 与 capability；缺失或不受支持的物理来源必须在资产/样本验收时显式失败或掩码，不能
路由到 pooled latent、猜测字符串顺序或填充虚假零 target。

当前不路由独立 whole-hand union target。该模块只路由 SSL target，不决定 policy action
routing；PALM/TIP 参与整手上下文但不输出动作，JOINT-only action 仍由
``models/heads/action.py`` 拥有。
"""
