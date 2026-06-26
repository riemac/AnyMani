r"""GM task configuration variants.

本包只放具体实验构型：single asset、某个真实手、某组 generated hand bank、
或未来 heterogeneous training preset。公共 in-hand MDP assembly 保持在
`anymani.tasks.gm.inhand_env_cfg`，variant 通过继承和局部 override 表达差异。
"""
