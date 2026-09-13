r"""尚未被实验选择的 backbone 候选。

此包保存仍有研究价值、但尚未获得训练证据的网络设计，避免它们污染公共 backbone
contract。candidate 只有在对应 shape/mask/latency contract、纯 tensor tests 与训练 preset
一并建立后，才可成为可运行路线。
"""
