r"""Run 产物与 lineage 的 stage-neutral 记录边界。

当前 geometry SSL writer 把 resolved config、资产 provenance、TensorBoard events、JSONL
指标、NPZ 密集数组和 checkpoint 写入 resolved run directory。它忠实记录
stage 已决定的实验语义，不选择模型、loss、数据 split 或训练预算。

通用跨 stage manifest、lineage 与 reporter schema 仍待实现。
"""
