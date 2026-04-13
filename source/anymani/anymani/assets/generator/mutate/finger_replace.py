r"""整根手指替换工具：在已有 HandCfg 上把某根手指换成另一个 preset/配置。

对应 `前后序.png` 中的"重合区"操作 `整根手指替换`。图中注记：

    前序：build 时选不同 finger cfg；后序：HandCfg 上换 finger。
    重合区的处理：允许两边都能做，但推荐用后序做"在已有手上做派生"。

设计说明
--------

### 职责边界

`FingerReplaceMutator` 接受一个已有的 `HandCfg`，用一个新的 `FingerCfg`（来自
preset 名或直接传入的对象）替换指定的 finger，并确保：

- 新 finger 的 `parent_link` 与被替换 finger 相同
- 运动学链在全手层面仍然连续
- 替换后能通过 `HandValidator` 的全局检查

### 替换策略

- ``"preset"``：从已注册的 finger preset 库里按名字取出 `FingerCfg`
- ``"cfg"``：用户直接传入一个现成的 `FingerCfg` 对象（`replacement_finger_cfg`）

### 挂载点继承

被替换 finger 的挂载位姿（`mount.pos` / `mount.rpy`）默认**继承**给新 finger，
避免新 finger 出现在错误位置。可通过 `inherit_mount=False` 关闭，使用新 finger
自带的默认挂载位姿。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Literal

from ...asset_base import AssetCfgBase, FingerCfg, HandCfg
from ...presets import FINGER_PRESET_REGISTRY, get_finger_builder_preset
from ._base import MutatorBase


# ============================================================================
#  配置类
# ============================================================================


@dataclass
class FingerReplaceCfg(AssetCfgBase):
    r"""整根手指替换工具配置。"""

    class_type: type["FingerReplaceMutator"] | None = None
    """关联的运行时类。"""

    target_finger: str | None = None
    """要被替换的手指名称；为 ``None`` 时由运行时从全部 finger 中随机选一根。"""

    strategy: Literal["preset", "cfg"] = "preset"
    """替换来源策略。``preset`` 从已注册 finger preset 库按名字取 `FingerCfg`；
    ``cfg`` 直接使用 ``replacement_finger_cfg`` 字段传入的对象。"""

    replacement_preset_name: str | None = None
    """目标 finger preset 名称（仅 ``strategy="preset"`` 时使用）。
    为 ``None`` 时运行时从可用 preset 中随机选一个（排除被替换 finger 的当前 preset）。"""

    replacement_finger_cfg: FingerCfg | None = None
    """直接传入的替换 `FingerCfg`（仅 ``strategy="cfg"`` 时使用）。"""

    inherit_mount: bool = True
    """是否把被替换 finger 的挂载位姿（mount.pos / mount.rpy）继承给新 finger。
    开启（默认）可避免新 finger 出现在不合理位置。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = FingerReplaceMutator
        if self.strategy == "cfg" and self.replacement_finger_cfg is None:
            raise ValueError(
                "strategy='cfg' requires replacement_finger_cfg to be provided"
            )


# ============================================================================
#  运行时壳
# ============================================================================


class FingerReplaceMutator(MutatorBase):
    r"""整根手指替换运行时壳。

    在已有的 `HandCfg` 上把指定 finger 换成另一个 `FingerCfg`，保持挂载点继承和
    全手一致性。
    """

    cfg: FingerReplaceCfg

    def __init__(self, cfg: FingerReplaceCfg):
        self.cfg = cfg

    def mutate(self, target: HandCfg) -> HandCfg | None:
        r"""对已构建的 `HandCfg` 执行整根手指替换。

        Args:
            target (HandCfg): 待变异的整手配置。

        Returns:
            HandCfg | None: 替换手指后的整手配置；若目标 finger 不存在或
            替换后违反全手唯一性约束，则返回 ``None``。
        """

        mutated = target.copy()  # 整根 finger 替换时，整手其余部分应保持完全不动
        if not mutated.fingers:
            return None

        if self.cfg.target_finger is None:
            target_index = random.randrange(len(mutated.fingers))  # 未显式指定时再交给运行时随机选择
        else:
            target_index = next((index for index, finger in enumerate(mutated.fingers) if finger.name == self.cfg.target_finger), -1)
            if target_index < 0:
                return None

        old_finger = mutated.fingers[target_index]  # 被替换的 finger slot
        replacement = self._build_replacement_finger(old_finger)
        if replacement is None:
            return None

        mutated.fingers[target_index] = replacement
        try:
            # 用 `HandCfg.replace(...)` 重新过一遍 schema 层唯一性与链一致性检查。
            return mutated.replace(fingers=mutated.fingers)
        except Exception:
            return None

        # TODO:算法之一（finger replacement）
        # ────────────────────────────────────────
        # 输入
        #   target: 已构建好的 `HandCfg`
        #   cfg.target_finger: 被替换的手指名，None 时随机选
        #   cfg.strategy: "preset" | "cfg"
        #   cfg.replacement_preset_name: preset 名（strategy="preset"）
        #   cfg.replacement_finger_cfg: 直接传入的 FingerCfg（strategy="cfg"）
        #   cfg.inherit_mount: 是否继承原 finger 的挂载位姿
        #
        # 输出：HandCfg | None
        #
        # ── 选取目标 ──
        #   1. 若 target_finger 为 None，从 target.fingers 中随机选一根
        #   2. 定位该 finger（找不到则返回 None）
        #
        # ── 构造新 FingerCfg ──
        #   [strategy="preset"]
        #   3a. 从 finger preset 注册表按 replacement_preset_name 查找 FingerCfg
        #       若 replacement_preset_name 为 None，从注册表随机选一个（排除当前 preset）
        #   [strategy="cfg"]
        #   3b. 直接使用 cfg.replacement_finger_cfg
        #
        # ── 挂载继承 ──
        #   4. 若 inherit_mount=True：
        #      new_finger.mount = deepcopy(old_finger.mount)
        #      同时确保 new_finger.name = old_finger.name（替换不改变 finger slot 名称）
        #      new_finger.parent_link = old_finger.parent_link
        #
        # ── 全局一致性预检 ──
        #   5. 检查替换后全手 joint 名称是否仍全局唯一（若不唯一则返回 None）
        #   6. 检查替换后 link 名称是否仍全局唯一
        #
        # ── 重建 HandCfg ──
        #   7. 深拷贝 target，把目标 finger 替换为新 FingerCfg，返回新对象
        #
        # ── 与 preset 的交叉验证 ──
        #   新 finger 的关节数量和类型应与目标 hand 的 family 约定兼容；
        #   若存在 family-level 的"finger slot 约束"（如 slot 0 必须是 4-DOF finger），
        #   替换操作应在此阶段校验。
        #
        # IDEA：finger replace 是重合区操作，后序语义是"在已有手上做派生"；
        # 它的价值在于能快速枚举"换一根手指的变化量"，而不需要重新跑整套 pre-made 流程。

    def _build_replacement_finger(self, old_finger: FingerCfg) -> FingerCfg | None:
        r"""根据配置构造将要替换进去的新 finger。"""

        if self.cfg.strategy == "cfg":
            candidate = self.cfg.replacement_finger_cfg.copy()
        else:
            preset_name = self.cfg.replacement_preset_name
            if preset_name is None:
                preset_pool = [name for name in FINGER_PRESET_REGISTRY if name != old_finger.metadata.get("builder")]
                preset_name = random.choice(preset_pool or list(FINGER_PRESET_REGISTRY))

            builder_cfg = get_finger_builder_preset(preset_name)
            builder_cfg = builder_cfg.replace(name=old_finger.name, parent_link=old_finger.parent_link)
            candidate = builder_cfg.class_type(builder_cfg).build()

        if self.cfg.inherit_mount:
            candidate = candidate.replace(
                name=old_finger.name,  # slot 名不变，替换的是结构而不是 finger 身份
                parent_link=old_finger.parent_link,
                mount=old_finger.mount.copy(),
            )
        else:
            candidate = candidate.replace(
                name=old_finger.name,
                parent_link=old_finger.parent_link,
            )
        return candidate


__all__ = ["FingerReplaceCfg", "FingerReplaceMutator"]
