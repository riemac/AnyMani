r"""训练后 run 审计、跨 seed/variant 比较与报告边界。

分析层只读 ``AnyMani/logs`` 中的 manifest、TensorBoard、JSONL、NPZ 与 checkpoint metadata；
它不被 model forward 或 trainer import，也不重新定义项目 metric 的数学语义。项目 metric
由 ``diagnostics.evaluation`` 定义，分析层只负责聚合、比较、统计不确定性和生成派生报告。

当前已有 Geometry SSL asset/q 两级 paired bootstrap；它只消费冻结的 validation ablation evidence，
不重新运行模型或修改训练 artifact。
"""
