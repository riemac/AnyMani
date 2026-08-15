r"""Candidate-neutral SSL target batch 与论文对照目标。

target 层把 source、field 与 query 组合为监督张量，同时保存 frame、单位、semantic
group、mask 与 provenance。它不构造神经网络，也不把 decoder 的非唯一内部参数当成
物理真值。
"""
