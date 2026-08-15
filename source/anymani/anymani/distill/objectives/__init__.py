r"""跨训练阶段可复用的数学 objective 层。

objective 只定义 prediction、target、mask、权重与 scalar loss 的物理/统计语义，不负责
构造模型、采样数据、执行 optimizer step 或启动 simulator。representation objective 与
未来 RL objective 分目录保存，避免把 advantage estimation、minibatch sampling、ADR 等
算法/环境机制误称为 loss。
"""
