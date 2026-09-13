r"""Imitation learning、teacher-student distillation 与 adaptation stage。

IL消费``models``的共享adapter/backbone/head，不复制student architecture。当前可执行面只包含固定
accepted-teacher environment-action mean的replica-isolated tiny-overfit；BC、DAgger、RMA、privileged-feature
distillation与多teacher morphology aggregation仍需逐项建立，不宣称存在统一trainer。
"""
