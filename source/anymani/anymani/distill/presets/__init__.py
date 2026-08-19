r"""Hydra YAML runtime preset 的 package 入口。

完整科学选择由具体 experiment dataclass 声明；本目录只保存可替换的硬件/runtime group。
当前 ``ssl/trainer/single_gpu_16gb.yaml`` 声明 AdamW、gradient clipping、minibatch、
accumulation、resident window、device/dtype 与记录 cadence，不拥有 q/sigma 相关结构。
"""
