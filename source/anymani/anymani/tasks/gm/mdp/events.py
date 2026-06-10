r"""Reset and domain-randomization terms for `tasks.gm`.

资产形态随机性主要来自 `assets` 生成的 hand bank。这里的 event 只处理
episode-level dynamics randomization 与 reset，例如 object mass/friction、
robot actuator gain、初始关节偏置、外力扰动。

TODO:
    不要在 reset 中切换 hand topology。若需要“每段训练换一批 assets”，由
    `distill` 重建 env cfg 或分段启动训练。
"""

from __future__ import annotations

__all__: list[str] = []
