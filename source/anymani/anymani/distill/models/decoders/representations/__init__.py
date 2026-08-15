r"""Physical-field SSL 的可替换 decoder family。

fixed vector、conditional implicit query、parametric Gaussian 与 FK-point decoder 可以共享
完全相同的 input adapter/backbone，并在统一数据 split 与容量预算下比较。decoder output
形式不定义 backbone latent 的永久语义，也不改变 policy action interface。
"""
