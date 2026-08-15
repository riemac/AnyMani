r"""Fixed ordered BPS 与 sampled spatial queries。

query 层只决定“在 hand-frame 空间的哪些位置读取 field”，不决定 field 的物理语义，
也不拥有 neural decoder。所有 query coordinates 都必须声明 frame 与单位，并随 target
batch 保存或可由版本化 layout 确定性重建。
"""
