# 单资产标定台结构碰撞过滤消融记录

本文记录 2026-06-24 在 AnyMani 单 asset contact-basin / pre-grasp 标定台中，对 generated hand 抖动现象做的一个小消融。目标不是修改训练环境默认物理，而是先判断 generated asset 的结构性自碰撞是否会污染人工调参时的可视反馈。

## 现象

在 `single_asset_grasp_calibrator.py` 中拖动 generated hand 的某个 thumb joint slider 时，期望看到的是单根手指沿关节自由度平滑变化。但实际观察到：

- 被拖动的手指会抖；
- 没有被调的其他手指也会抖；
- object 在没有主动接触变化时也可能被连带扰动；
- 最终姿态仍会到达 slider 指定构型，因此问题更像 PhysX 接触解算过程中的高频修正，而不是 joint target 写错。

作为对照，使用官方 LEAP USD 资产时，单根手指关节变化明显更平滑；只有真正碰到 object 时才出现可理解的接触抖动。因此更可疑的是 generated URDF 的 collision geometry / self-collision pair，而不是 GUI slider 事件或关节映射本身。

## 假设

generated hand 的 palm、finger root、proximal links 和 tip mesh 在装配处可能存在轻微穿插或过近接触。若 articulation 开启 self-collision，PhysX 会在每次写入 joint state / target 后尝试解穿透：

$$
q_{\text{state}} \leftarrow q^\star,\qquad
\text{solver}\big(C_i(q), C_j(q)\big)\rightarrow \Delta v,\Delta x .
$$

当这些 contact pair 属于“机构装配内部”而不是真实任务接触时，solver 的 depenetration 会表现为不相关 link 的可视抖动。该抖动会干扰人工判断 contact basin，也可能误导后续 reset pose / pre-grasp 的选择。

## 本次实现

在 `source/anymani/anymani/tools/single_asset_grasp_calibrator.py` 中加入 generated-only CLI：

```bash
--generated-collision-filter {none,finger_palm,finger_palm_same_finger}
```

语义如下：

- `none`：默认基线，不改变碰撞关系。
- `finger_palm`：过滤 `palm` 与所有 generated finger links 的 collision pair。
- `finger_palm_same_finger`：在 `finger_palm` 基础上，额外过滤同一根 finger 内部 link-link pair。

实现刻意使用 stage-level `PhysicsCollisionGroup`，而不是永久改 URDF / asset generator。每个 generated link 建一个 external collision group，collection include 指向 link prim，并用 `expandPrims` 包含其下 collision descendants。这样可以保留不同 fingers 之间的碰撞，也不影响 hand-object 接触。

## 验证结果

自动 smoke 已通过：

```bash
cd /home/hac/isaac/AnyMani
source /home/hac/isaac/env_isaaclab/bin/activate
/home/hac/isaac/IsaacLab/isaaclab.sh -p source/anymani/anymani/tools/single_asset_grasp_calibrator.py \
  --generated-collision-filter finger_palm_same_finger \
  --smoke-seconds 1
```

启动日志确认写入：

```text
generated collision filter authored mode='finger_palm_same_finger', groups=24, link_pairs=78, directed_edges=156
```

用户随后在 GUI 中复测，主观观察为“好很多”，说明 generated hand 的抖动大概率确实与结构性 self-collision / palm-finger collision 有关。

## 后续建议

- 标定阶段：继续用 `--generated-collision-filter finger_palm_same_finger` 做人工 contact basin / pre-grasp 调参，先拿到稳定可读的 reset seed。
- 训练阶段：不要直接把本工具的过滤结果无脑搬进 MDP；应在 `tasks/gm` 中显式设计一个结构碰撞过滤配置，并做 `none / structural_filter` 消融。
- 资产阶段：后续可用 asset validator 检查 palm-root、same-finger adjacent links、tip mesh 与 proximal link 的 SDF clearance，判断是要靠 collision filter 规避，还是要在生成几何上修正。
- 物理解释：官方 LEAP 在 hand-object 接触时仍会抖，说明真实接触抖动不可完全消除；本次过滤只针对不应参与任务物理的机构内部假接触。
