r"""跨 SSL、RL 与 IL 的实验记录、评估和事后分析边界。

``recording`` 把 resolved experiment、运行 provenance、TensorBoard events、结构化指标与
checkpoint lineage 写入 ``AnyMani/logs``；``evaluation`` 定义可复算的固定评估协议；
``analysis`` 只读这些产物并比较 runs。该包不选择实验、不实现 trainer，也不拥有物理 target
或 learnable model。

当前 geometry SSL 已提供 ``recording.geometry_ssl`` 与 ``evaluation.geometry_ssl``；通用跨 stage schema 和只读 analysis 仍待实现。
"""
