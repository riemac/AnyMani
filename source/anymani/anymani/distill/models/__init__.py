r"""SSL、RL 与 IL 共用的 learnable model layer。

本包只拥有可学习组件，不拥有 physical target generation、loss、trainer 或 IsaacLab MDP：

```text
    input_adapters/  # route-valid deployable evidence -> owner-aware features
    backbones/       # graph-biased encoder-only Transformer 与其他 stage backbones
    decoders/        # pretraining/distillation outputs；representation decoder 默认可丢弃
    heads/           # JOINT action、value 与非 field-specific auxiliary outputs
    policy.py        # stage-independent model assembly contract
```

稳定边界是：PALM/JOINT/TIP physical entity 与 surface owner 直接同索引；只有 JOINT
routing 产生 joint action。Geometry encoder 原生处理可变实体数，并可用显式 entity/joint masks
把不同结构 padding 到同一前向；逐结构独立前向仍是输出与梯度 oracle。
输入 adapter 与 backbone final-norm unified $Z$ 共同构成 SSL 预训练后迁入 PPO 的 retained
geometry encoder，不能为每种 query/field target 复制不同 policy trunk。

当前 Geometry SSL 明确使用全连接 graph-biased encoder-only Transformer：无向最短路径、parent 与
child distance 形成每头可学习加性 bias；不是 hard mask。``temporal_encoder.py`` 仍是
single-asset tactile TCN baseline；``backbones/candidates/spatial_transformer.py`` 不属于 Geometry
canonical。``config.py`` 与 ``tokens.py`` 保留其他 executable draft/API，不是 SSL registry schema。

Physical geometry source、field、query 与 target 位于 ``distill.representations``；所有 loss
位于 ``distill.objectives``。新增实现时必须保留 frame、unit、mask、token routing、latency
与 checkpoint-retention 语义，不得因某个 baseline 已可运行就宣称整个模型栈完成。
"""

# NOTE: 暂不从 package root re-export 具体模型；各 stage 直接 import 稳定模块路径。
__all__: list[str] = []
