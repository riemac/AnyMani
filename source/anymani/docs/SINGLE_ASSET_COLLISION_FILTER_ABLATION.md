# 单资产标定台结构碰撞过滤消融记录

本文记录 2026-06-24 在 AnyMani 单 asset contact-basin / pre-grasp 标定台中，对 generated hand 抖动现象做的一个小消融，以及随后把该规则接入 `tasks/gm` 训练环境后的运行时验证。核心目标是区分“任务真实接触”和“generated 机构装配内部假接触”，避免结构性 self-collision 污染人工调参和 RL reset 初态。

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

## 标定台实现

在 `source/anymani/anymani/tools/single_asset_grasp_calibrator.py` 中加入 generated-only CLI：

```bash
--generated-collision-filter {none,finger_palm,finger_palm_same_finger}
```

语义如下：

- `none`：默认基线，不改变碰撞关系。
- `finger_palm`：过滤 `palm` 与所有 generated finger links 的 collision pair。
- `finger_palm_same_finger`：在 `finger_palm` 基础上，额外过滤同一根 finger 内部 link-link pair。

实现刻意使用 stage-level `PhysicsCollisionGroup`，而不是永久改 URDF / asset generator。每个 generated link 建一个 external collision group，collection include 指向 link prim，并用 `expandPrims` 包含其下 collision descendants。这样可以保留不同 fingers 之间的碰撞，也不影响 hand-object 接触。

## 标定台验证结果

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

## 训练环境接入

标定台消融通过后，该规则被提升为 generated asset 的训练物理约定，并接入 `tasks/gm`。训练环境最初复用了标定台的 `PhysicsCollisionGroup` 写法；2026-06-25 在 4096-env 训练启动时观察到 PhysX warning：

```text
Collisions are supported currently only in one collision group.
```

本地 IsaacSim schema 显示 `PhysicsCollisionGroup` 是 coarse filtering，而 `PhysicsFilteredPairsAPI` 是 fine-grained filtering，且后者优先级高于 collision group。由于 IsaacLab / Cloner 也会用 collision group 表达 env 间 collision filtering，训练环境的 generated-hand 内部结构过滤已改为 pairwise `FilteredPairsAPI`，避免同一 collider 同时进入多个 group。

- `GmSingleAssetEventsCfg.apply_structural_collision_filter` 使用 `mode="prestartup"`，在 PhysX 初始化前向 link prim 写入 `PhysicsFilteredPairsAPI` / `physics:filteredPairs`。
- 过滤 palm 与任意 finger link 的 collision pair。
- 过滤同一根 finger 内部任意 link-link collision pair。
- 保留不同 fingers 之间的 collision pair，避免把真实 finger-finger 接触也抹掉。
- link chains 来自 selected hand 的 `hand.yaml` sidecar，而不是在 env cfg 中硬编码四指名称。

这个实现不是 episode reset event 的职责。它必须发生在 scene spawn 之后、`sim.reset()` 之前；否则 PhysX 初始化时不会读取这批 pair filter。

## 训练环境 smoke 验证

纯 contract test 只能证明 link-pair 集合：

$$
\mathcal{F}
  = \{(\text{palm}, l)\mid l \in \cup_f F_f\}
    \cup
    \bigcup_f \{(a,b)\mid a,b\in F_f,\ a\ne b\}
$$

生成正确，不能证明 USD stage 上真的存在 `PhysicsFilteredPairsAPI`，也不能证明 PhysX reset / step 稳定。因此新增显式 IsaacSim smoke：

```bash
cd /home/hac/isaac/AnyMani
source /home/hac/isaac/env_isaaclab/bin/activate
timeout --kill-after=20s 240s /home/hac/isaac/IsaacLab/isaaclab.sh -p -m pytest \
  source/anymani/anymani/smokes/isaacsim/test_gm_single_asset_structural_collision.py -q -s
```

该 smoke 不进入默认 `pytest`，验收信号为：

- `env._gm_structural_collision_filter_stats` 存在；
- `api == "FilteredPairsAPI"`；
- `link_pairs`、`directed_edges` 与 sidecar 推导的结构规则一致；
- `missing_link_names == ()`；
- 每个结构过滤 link pair 都在对应 link prim 的 `physics:filteredPairs` relationship 中双向出现；
- 不再 author `/World/anymani_gm_generated_structural_collision_filters` 旧 group root；
- `AnyMani-GM-SingleAsset-v0` 可 reset，并完成 64 步随机 action 短 rollout，obs / reward 全 finite。

这个 smoke 的边界是“证明 USD / PhysX 运行时看见了 pairwise filter，并且短 rollout 没有基础数值崩溃”。它不能替代长训练回放，也不声称 reward、action space 或 contact basin 已经足够好；这些仍需要通过单资产训练曲线、视频回放和后续 ablation 判断。

## 后续建议

- 标定阶段：继续用 `--generated-collision-filter finger_palm_same_finger` 做人工 contact basin / pre-grasp 调参，先拿到稳定可读的 reset seed。
- 训练阶段：当前默认采用 structural filter 作为 generated asset 的物理稳定化规则；若后续对收敛结果存疑，再做 `none / structural_filter` 消融。
- 资产阶段：后续可用 asset validator 检查 palm-root、same-finger adjacent links、tip mesh 与 proximal link 的 SDF clearance，判断是要靠 collision filter 规避，还是要在生成几何上修正。
- 物理解释：官方 LEAP 在 hand-object 接触时仍会抖，说明真实接触抖动不可完全消除；本次过滤只针对不应参与任务物理的机构内部假接触。
