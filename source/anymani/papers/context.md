本项目是做手内操作任务的，以下收集了我读过的、与之相关、我认为有代表性的文献以及代码，当和 Agent 讨论交流 idea时，Agent 需阅读参考，至少要把正文全读完

# 文献

## 手内操作

位于 `AnyMani/source/anymani/papers/inhand` 下:

1. Qi 等 - 2022 - In-Hand Object Rotation via Rapid Motor Adaptation.pdf (正文1-8页，附录12-16页)
2. Qi 等 - 2023 - General In-Hand Object Rotation with Vision and Touch.pdf (正文1-9页，附录15-16页，选读)
3. Tao 等 - 2023 - A Multi-Agent Approach for Adaptive Finger Cooperation in Learning-based In-Hand Manipulation.pdf (正文1-6页，选读)
4. Yang 等 - 2024 - AnyRotate Gravity-Invariant In-Hand Object Rotation with Sim-to-Real Touch.pdf (正文1-8页，附录13-21页)
5. Patel和Song - 2024 - GET-Zero Graph Embodiment Transformer for Zero-shot Embodiment Generalization.pdf (正文1-6页)
> 现有问题：只能迁移到Leaphand家族的不同embodiment，无法迁移到不同的leaphand以外的手型
6. Liu 等 - 2025 - DexNDM Closing the Reality Gap for Dexterous In-Hand Rotation via Joint-Wise Neural Dynamics Model.pdf (正文1-9页，附录14-47页)

## 灵巧抓取

位于 `AnyMani/source/anymani/papers/grasp` 下:

1. Fei 等 - 2025 - T(R,O) Grasp Efficient Graph Diffusion of Robot-Object Spatial Transformation for Cross-Embodiment.pdf (正文1-8页，附录11-12页)

# 代码

## TRO-Grasp

1. `TRO-Grasp/model/tro_graph.py`
2. `TRO-Grasp/model/denoiser.py`

## Get-zero

1. `get_zero/get_zero/rl/models/embodiment_transformer.py`
2. `get_zero/get_zero/distill/models/embodiment_attention.py`
3. `get_zero/get_zero/distill/models/embodiment_transformer.py`
4. `get_zero/get_zero/distill/models/vis_embodiment_transformer.py`