r"""PALM / JOINT / TIP unified spatial Transformer 候选契约。

本候选在输入端完成 type-specific projection 后，将全部有效 token 放入同一双向
self-attention 序列：

$$
H^{(0)}\in\mathbb R^{B\times T\times D},
\qquad
H^{(l+1)}
=
\operatorname{EncoderLayer}(H^{(l)},M),
$$

其中 $B$ 是 batch size，$T=N_p+N_j+N_t$ 是 token 数，$D$ 是 hidden width，$M$ 是
padding mask。输入端分组只解决异构物理量投影；进入统一隐空间后继续隔离 PALM/JOINT/TIP
会阻断跨指与 joint-tip 通信，因此候选主干允许全部有效 token 交互，输出端再按 owner/type
路由。

历史保守锚点为 Pre-LN、dropout 0、约 $D=128$、4 layers、4 heads，但这些数值没有被
heterogeneous PPO 证据接受，只能作为未来 preset 起点。no-bias 路线应作为最小 baseline；
dynamic $SE(3)$ relation bias、factorized temporal attention 与 alternating axial blocks 均不
属于该公共 spatial candidate 的默认组成。

实现前必须先验证：valid-mask 语义、padding 不影响有效 token、variable $T$、JOINT-only
action routing、参数量/FLOPs、4096-env rollout throughput 与单策略 forward latency。
"""
