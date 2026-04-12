r"""整手级验证规则集和验证流水线。

这是验证层的顶层编排者，整合 joint / finger 级验证后再加上手级全局规则。
对应 `资产生产概略.png` 中 `pre-made → validator → HandCfgs` 和
`post-mutate → HandCfgs` 两处 validator 节点。

当前收纳的手级全局规则
----------------------

1. **全局名称唯一性**：joint / link / finger 名称在全手范围内不得重复
   （schema 已做，但 post-mutate 后可能再违规，需重跑）
2. **DOF 总量范围**：整手 DOF 是否在 ``[dof_min, dof_max]`` 内
3. **手指数量范围**：finger 数量是否在 ``[finger_count_min, finger_count_max]`` 内
4. **所有旁链挂载一致性**：所有 finger 的 ``parent_link`` 均等于 ``palm.name``
   （schema 已做，post-mutate 的 finger_replace 后可能破坏，需重跑）
5. **相邻手指挂载最小间距**：任意两根手指的 mount 位置之间的欧氏距离
   不得低于 ``min_finger_spacing``（meter），防止手指在几何上重叠

设计说明
--------

### 两次校验时机

从 `资产生产概略.png` 可以看到，validator 在流水线里出现了两次：
- 第一次：pre-made 之后（先验产物的基本合法性，再喂给 post-mutate）
- 第二次：post-mutate 之后（验证 mutate 没有破坏全局一致性）

两次使用的实际上是同一个 `HandValidator`，只是调用时机不同。

### strict 模式

``strict=True`` 时，所有 warnings 被升级为 errors；适合用于
pre-made 产物的初次验证（要求更严格），而 post-mutate 后可能需要
宽松一些（warnings 只记录不拒绝）。

### 手指间距计算

间距定义为任意两根手指 ``mount.pos`` 之间的欧氏距离，是最轻量的近似。
更精确的碰撞检测需要几何层支持，当前阶段不纳入。

$$
d_{ij} = \|p_i - p_j\|_2 \geq d_{\min}
$$
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..asset_base import AssetCfgBase, HandCfg
from ._base import ValidatorBase, ValidationResult
from .finger_rules import FingerValidatorCfg, FingerValidator


# ============================================================================
#  配置类
# ============================================================================


@dataclass
class HandValidatorCfg(AssetCfgBase):
    r"""整手级验证规则配置（同时作为验证流水线入口）。"""

    class_type: type["HandValidator"] | None = None
    """关联的运行时类。"""

    dof_min: int | None = 1
    """允许的最小总 DOF；为 ``None`` 时不检查下限。"""

    dof_max: int | None = None
    """允许的最大总 DOF；为 ``None`` 时不检查上限。"""

    finger_count_min: int | None = 1
    """允许的最少手指数；为 ``None`` 时不检查。"""

    finger_count_max: int | None = None
    """允许的最多手指数；为 ``None`` 时不检查。"""


    check_global_uniqueness: bool = True
    """是否重跑全局名称唯一性检查（joint / link / finger 名称全手唯一）。
    建议在 post-mutate（尤其是 joint_delete / finger_replace）后始终开启。"""

    check_mount_consistency: bool = True
    """是否检查所有 finger 的 ``parent_link`` 均等于 ``palm.name``。"""

    check_finger_spacing: bool = True
    """是否检查任意两根手指挂载点之间的欧氏距离不低于 ``min_finger_spacing``。"""

    min_finger_spacing: float = 0.015
    """允许的最小手指间挂载间距（meter）；低于此值记 warning。
    默认 0.015 m（1.5 cm），参考指根 frame 间距减去典型 mesh size（宽度）。因为 mesh frame的约定，还要注意偏移等
    实际使用中可根据 palm 尺寸和手指 collision primitive 宽度调整（建议 0.015~0.02 m）。"""

    finger: FingerValidatorCfg = field(default_factory=FingerValidatorCfg)
    """手指级验证配置；hand 验证器内部对每根 finger 跑此配置。"""

    strict: bool = False
    """是否把所有 warnings 升级为 errors（严格模式）。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = HandValidator


# ============================================================================
#  运行时壳
# ============================================================================


class HandValidator(ValidatorBase):
    r"""整手级验证器（同时是验证流水线）。

    对 `HandCfg` 做层次化验证：先调用 `FingerValidator` 对每根手指跑规则，
    再叠加手级全局规则，最终返回合并后的 `ValidationResult`。
    """

    cfg: HandValidatorCfg

    def __init__(self, cfg: HandValidatorCfg):
        self.cfg = cfg

    def validate(self, target: HandCfg) -> ValidationResult:  # type: ignore[override]
        r"""对 `HandCfg` 执行层次化全手验证。

        Args:
            target (HandCfg): 待验证的整手配置。

        Returns:
            ValidationResult: 含所有层级 errors / warnings 的合并结果。
        """

        result = ValidationResult()
        finger_validator = FingerValidator(self.cfg.finger)

        for finger in target.fingers:
            result.merge(finger_validator.validate(finger))

        if self.cfg.check_global_uniqueness:
            finger_names = [finger.name for finger in target.fingers]
            if len(finger_names) != len(set(finger_names)):
                result.errors.append(f"hand '{target.name}': duplicate finger names")

            joint_names = [joint.name for joint in target.iter_joints()]
            if len(joint_names) != len(set(joint_names)):
                result.errors.append(f"hand '{target.name}': duplicate joint names")

            link_names = [target.palm.name] + [joint.child for joint in target.iter_joints()]
            if len(link_names) != len(set(link_names)):
                result.errors.append(f"hand '{target.name}': duplicate link names")

        dof = target.dof_count
        if self.cfg.dof_min is not None and dof < self.cfg.dof_min:
            result.errors.append(f"hand '{target.name}': dof {dof} < min {self.cfg.dof_min}")
        if self.cfg.dof_max is not None and dof > self.cfg.dof_max:
            result.warnings.append(f"hand '{target.name}': dof {dof} > max {self.cfg.dof_max}")

        finger_count = len(target.fingers)
        if self.cfg.finger_count_min is not None and finger_count < self.cfg.finger_count_min:
            result.errors.append(
                f"hand '{target.name}': finger count {finger_count} < min {self.cfg.finger_count_min}"
            )
        if self.cfg.finger_count_max is not None and finger_count > self.cfg.finger_count_max:
            result.warnings.append(
                f"hand '{target.name}': finger count {finger_count} > max {self.cfg.finger_count_max}"
            )

        if self.cfg.check_mount_consistency:
            for finger in target.fingers:
                if finger.parent_link != target.palm.name:
                    result.errors.append(
                        f"finger '{finger.name}' parent_link '{finger.parent_link}' != palm '{target.palm.name}'"
                    )

        if self.cfg.check_finger_spacing:
            mounts = [(finger.name, finger.mount.pos) for finger in target.fingers]
            for idx in range(len(mounts)):
                for jdx in range(idx + 1, len(mounts)):
                    name_i, pos_i = mounts[idx]
                    name_j, pos_j = mounts[jdx]
                    distance = math.sqrt(sum((lhs - rhs) ** 2 for lhs, rhs in zip(pos_i, pos_j)))
                    if distance < self.cfg.min_finger_spacing:
                        result.warnings.append(
                            f"finger spacing '{name_i}'-'{name_j}': "
                            f"{distance * 100.0:.2f} cm < min {self.cfg.min_finger_spacing * 100.0:.2f} cm"
                        )

        if self.cfg.strict:
            result = result.as_strict()
        result.passed = len(result.errors) == 0
        return result

        # TODO:算法之一（hand-level hierarchical validation）
        # ────────────────────────────────────────
        # 输入
        #   target: HandCfg
        #   cfg: HandValidatorCfg
        #
        # 输出：ValidationResult（含 finger 级 + joint 级的合并）
        #
        # result = ValidationResult()
        # finger_v = FingerValidator(cfg.finger)
        #
        # ── 逐手指校验（下沉到 finger 级，finger 内再下沉到 joint 级）──
        #   for finger in target.fingers:
        #     fresult = finger_v.validate(finger)
        #     result.merge(fresult)
        #
        # ── 规则 1：全局名称唯一性（可选重跑） ──
        #   if cfg.check_global_uniqueness:
        #     finger_names = [f.name for f in target.fingers]
        #     if len(finger_names) != len(set(finger_names)):
        #       result.errors.append(f"hand '{target.name}': duplicate finger names")
        #
        #     all_joint_names = [j.name for j in target.iter_joints()]
        #     if len(all_joint_names) != len(set(all_joint_names)):
        #       result.errors.append(f"hand '{target.name}': duplicate joint names")
        #
        #     all_link_names = [target.palm.name] + [j.child for j in target.iter_joints()]
        #     if len(all_link_names) != len(set(all_link_names)):
        #       result.errors.append(f"hand '{target.name}': duplicate link names")
        #
        # ── 规则 2：DOF 总量范围 ──
        #   dof = target.dof_count
        #   if cfg.dof_min is not None and dof < cfg.dof_min:
        #     result.errors.append(f"hand '{target.name}': dof {dof} < min {cfg.dof_min}")
        #   if cfg.dof_max is not None and dof > cfg.dof_max:
        #     result.warnings.append(f"hand '{target.name}': dof {dof} > max {cfg.dof_max}")
        #
        # ── 规则 3：手指数量范围 ──
        #   fc = len(target.fingers)
        #   if cfg.finger_count_min and fc < cfg.finger_count_min:
        #     result.errors.append(f"hand: finger count {fc} < min")
        #   if cfg.finger_count_max and fc > cfg.finger_count_max:
        #     result.warnings.append(f"hand: finger count {fc} > max")
        #
        # ── 规则 4：挂载一致性 ──
        #   if cfg.check_mount_consistency:
        #     for f in target.fingers:
        #       if f.parent_link != target.palm.name:
        #         result.errors.append(
        #           f"finger '{f.name}' parent_link '{f.parent_link}' != palm '{target.palm.name}'"
        #         )
        #
        # ── 规则 5：相邻手指最小挂载间距 ──
        #   if cfg.check_finger_spacing:
        #     mounts = [(f.name, f.mount.pos) for f in target.fingers if f.mount is not None]
        #     for i in range(len(mounts)):
        #       for j in range(i + 1, len(mounts)):
        #         fi_name, pi = mounts[i]
        #         fj_name, pj = mounts[j]
        #         # 简单近似：|| p_i_mount - p_j_mount || ≥ d_min
        #         # 更精确做法：d = ||p - p'|| - r_i - r_j，其中 r 为指根 collision mesh 半径；
        #         # 当前草案先用 mount 坐标原点之间的欧氏距离，r 修正留给后续实现。
        #         dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(pi, pj)))
        #         if dist < cfg.min_finger_spacing:
        #           result.warnings.append(
        #             f"finger spacing '{fi_name}'-'{fj_name}': "
        #             f"{dist*100:.2f} cm < min {cfg.min_finger_spacing*100:.2f} cm"
        #           )
        #
        # ── strict 升级 ──
        #   if cfg.strict: result = result.as_strict()
        #
        # result.passed = len(result.errors) == 0
        # return result
        #
        # ── 关于两次校验时机 ──
        #   pre-made 后的首次验证：建议 strict=True（要求产物基线正确）
        #   post-mutate 后的二次验证：建议 strict=False（warnings 只记录）
        #   调用方可以用 cfg.strict 或 result.as_strict() 灵活控制。
        #
        # IDEA：handler 设计的核心价值是"层次化合并"——用户只需关心 HandCfg，
        # 不需要手动对每根 finger / 每个 joint 单独跑验证。


__all__ = ["HandValidatorCfg", "HandValidator"]
