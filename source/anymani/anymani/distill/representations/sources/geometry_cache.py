r"""Per-asset 静态 geometry cache 与 provenance contract。

第一版采用“静态缓存 + 在线 query”：每个 asset 只构建一次 local collision carrier、
BVH/加速结构、面积加权 surface evidence、semantic group mapping、home geometry、screw
chain 与 field capability；训练 batch 再采样 $q$ 和 query points，并在缓存几何上生成
标签。这样 field family、BPS layout、density bandwidth 或 query mixture 改变时不必重建
整个离线数据集。

cache identity 至少应由代码/target-generator 版本、asset/mesh 内容哈希、单位、``{a}``
到 ``{h}`` 校准、semantic sidecar 版本、collision source 与数值 backend 决定。缓存产物
必须记录是否为 watertight、surface-only 或 signed-volume capable，避免 SDF/occupancy
路线静默消费不可信符号。

允许缓存的是 embodiment-only、在 episode 内不随 $q$ 变化的物理证据。不得永久缓存：

- 当前 $q$ 下的 posed field；
- contact/history/object-conditioned activation；
- 需要随 PPO mini-epoch 更新梯度的 learned embedding。

若 profile 证明某组固定 target 很昂贵，可以追加受版本控制的 label cache；它是性能层，
不能改变 source/field/query 的数学语义。
"""
