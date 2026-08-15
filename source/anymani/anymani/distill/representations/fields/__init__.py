r"""从 posed physical geometry 导出的标量场与参数场。

field family 定义“空间 query 应返回什么物理量”，与 query layout 和 decoder 形式正交。
同一个 UDF、SDF、density 或 occupancy field 可以在 fixed BPS 上采样成显式向量，也可由
conditional implicit decoder 在随机 queries 上逐点预测。所有 field 必须注明 domain、
frame、单位、inside/outside convention、归一化与对真实空间变换的 equivariance。
"""
