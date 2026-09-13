r"""Policy action、value 与非 field-specific auxiliary heads。

head 按 token semantic role 路由：JOINT token 是当前唯一 action-bearing 类型；PALM/TIP
参与表征通信但不直接产生 joint action。representation-specific 重建输出属于
``models.decoders.representations``，避免把 disposable SSL decoder 与部署 policy head 混写。
"""
