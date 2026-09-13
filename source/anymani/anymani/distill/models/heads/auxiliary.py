r"""非 physical-field decoder 的通用 auxiliary-head 边界。

本模块只预留未来可与部署 policy 同时存在的轻量 auxiliary outputs，例如 contact-state、
diagnostic latent probe 或蒸馏对齐。BPS/UDF/density/Gaussian/FK reconstruction 已分别属于
``models.decoders.representations`` 与 ``objectives.representations``，其中 FK baseline 也有
独立 ``decoders/representations/fk_points.py``；不能在这里复制一套含义相同的 head。

任何 auxiliary head 必须声明 owner token、target observability、训练阶段、checkpoint
retention 与推理预算。当前不选择具体 head，也不让 PALM/TIP 因“有 token”就自动承担
没有物理依据的预测任务。
"""
