# Embodiment Geometry Pretraining

直接从任务回报学习跨手型策略，会把手的几何、接触、动作坐标与任务目标同时压进同一表示。即使 reward 上升，也难以判断网络恢复了可迁移的物理结构，还是只记住某一手型和任务的统计捷径。本模块先研究一个更窄、可证伪的问题：不启动 Isaac Sim、不观察物体，只给定手型 $\mathfrak m$、当前物理关节构型 $q$ 和静态几何证据，能否编码整只手的局部空间占据及其对关节运动的一阶响应？

这不是离线生成完毕的数据集训练，也不是 on-policy reinforcement learning。资产 YAML 与静态 catalog 预先存在；训练时在线采样 $q\sim\mu(q\mid\mathfrak m)$ 和方法随机条件 $\xi\sim\nu(\xi\mid\mathfrak m,q)$，再由无梯度 physical oracle 计算 $y=T(\mathfrak m,q,\xi)$。因此它是 online procedural supervised pretraining：生命周期具有 collect/realize、objective、backward 和 update，但没有 transition、reward、return、GAE、importance ratio 或 PPO clipping。

## 物理监督

资产 sidecar 把手组织为 PALM、JOINT 与 TIP owners。对 owner $g$、固定 hand-frame query $x$ 和构型 $q$，teacher 从真实 collision union 计算 unsigned distance $d_g(x;q)$，并以实际米制带宽定义 Gaussian 邻近场：

$$
\rho_{\sigma,g}(x;q)=\exp\!\left[-\frac{d_g(x;q)^2}{2\sigma^2}\right].
$$

一阶监督首先计算距离灵敏度 $\kappa_{g,i}=\partial d_g/\partial q_i$，再由链式法则得到场灵敏度 $g_{\sigma,g,i}=-(d_g/\sigma^2)\rho_{\sigma,g}\kappa_{g,i}$。$d$、$\kappa$ 与 $g$ 的单位分别为 m、m/rad 与 rad$^{-1}$，$\rho$ 无量纲。

`GeometrySource` 每项资产只物化一次 q-independent oracle，并独立生成 8 套 physical anchor bank；同资产 q-block 内共享一套并均衡轮换，validation、独立 q-bank 和 PPO 固定 $A^{(0)}$。每个 owner 使用 64 个 query，workspace / owner-shell / adjacent 的比例为 50:25:25。训练 sigma 中心为 4、16、64 mm，并做 log-space ±10% jitter；validation 关闭 jitter，仍用同一组 4、16、64 mm。一阶边只从 owner-shell 抽取：每个有效 JOINT 在训练期 1 条 active + 1 条 structure-zero，validation 为 4+4。

## 保留表示与三项目标

retained encoder 只读取当前物理 q 与静态 evidence。多锚点前端把 home surface 和空间旋量表示为相对完整 anchor constellation 的关系；graph-biased Transformer 保持全连接 attention，拓扑只作 soft bias。模型输出逐 owner 的 $Z^{(0)}$ 与逐 JOINT 的 $z_i^{(1)}$，其中

$$
z_i^{(1)}=H_1\!\left([z_i^{(0)}\Vert f_i^{screw}]\right)
$$

是整手场 Jacobian 第 $i$ 列的固定宽度表示，不是对自身 $z_i^{(0)}$ 的普通导数。两个 SSL-only readers、query backend 和 target backend 在导出 retained artifact 时删除。

科学方法聚合根位于 `distill/methods/multi_anchor_gaussian_implicit_field/`。三项主损失是 density、κ 与 derived-field，后者使用 $\hat g^{(\kappa)}=-(d/\sigma^2)\hat\rho\hat\kappa$ 对齐物理 teacher $g$；paired parity 作为独立 validation audit。joint-sign rewrite 是 method 专属输入增强：每个 $(asset,q)$ 以 0.20 概率恰好翻一个有效 JOINT 的 $(q,q_{home},\mathcal S)$；density/distance 保持不变，对应 JOINT 的 $\kappa/g$ 翻号。归约按 $(asset,q)$ 等权；一阶 active/zero 先分别平均，再 1:1 合并。owner、query 和 edge 轴由 Method 解释，Trainer 接收归约后的充分统计。

`calibrate_objectives` 与 `pretrain` 共用同一 `EmbodimentPretrainCfg` 类型、正式 `ssl.yaml` 和运行管线。预实验按 `max_epochs × num_minibatches` 生成新数据，强制 `mini_epochs=1`，在 `no_grad` 下累计三项，不执行 backward/update，也不自动改权重；研究者根据 artifact 的新 pair 数、均值与 trace 决定正式权重。artifact 保存代码与工作区 lineage，正式运行引用时核对数据集、公式和 Method 类型。

## 声明式运行架构

schema 6 的 `EmbodimentPretrainCfg` 只组合 `data`、`method`、`trainer` 和 `run`。完整实验由 [`experiments/multi_anchor_gaussion_implicit_field.py`](experiments/multi_anchor_gaussion_implicit_field.py) 用 Python 装配，Hydra 只从 ConfigStore 加载。Trainer 声明 `max_epochs / num_minibatches / mini_epochs / microbatch_size / sampling`，不存在 phase-specific 数据预算字段。

## 在线日程与恢复

canonical 训练执行 32 个 epoch，每轮生成 4 个新 minibatches。每个 minibatch 固定包含 64 个互异资产、每项生成 8 个新 Sobol q，即 512 个 pair；它被切成 8 个 64-pair microbatches 累积梯度，随后独立执行一次 optimizer update。默认 `mini_epochs=1`，因此完整运行生成 65,536 个新 pairs、使用 65,536 次 pairs 并更新 128 次。epoch 只组织在线采样与更新，不表示 catalog 遍历；当前预算恰好在整个运行中使用每项 train asset 一次。

schema 6 full checkpoint 只在完整 epoch 结束后保存，记录 epoch、optimizer update、新 pair、pair use、teacher realization、全局 minibatch cursor、下一轮 permutation、每资产 Sobol cursor、validation selection、CPU/CUDA RNG、resolved config 与 physical manifest。epoch 内中断时从上一个边界确定性重放，teacher buffer 不进入 checkpoint。训练结束后加载 validation-best immutable checkpoint，再导出独立 `retained_artifact.pt`。

Method session 封装 source、Sobol cursor、resident state 和具体 batch。validation 每项资产固定 64 q，以 density、κ 与 derived-field 的初始化归一化分数选择 best checkpoint。训练结束后加载冻结 best，在 evaluation unseen-variant-set/unseen-mother 上运行相同固定测度及六项 ablation；official-zero-shot 为空时显式报告空集。

## 证据边界

当前 contracts 覆盖 Python composition、显式 minibatch 截止、mini-epoch 数据复用、确定性重放/resume、固定评估尾块、三项精确归约、流式预实验、Method session、通用 full checkpoint、retained artifact、validation selection 与 final evaluation。最小正式 fit CUDA double 和 6 项真实 generated-asset Warp smoke 已通过。RTX 5070 Ti 上三项目标 64-sample 普通前向与反向实测为 261.04 samples/s、峰值 10.06 GiB；retained encoder $B=4096$ p95 为 20.13 ms。两项计时均排除 teacher、policy、Isaac Sim 与 `env.step`。

正式 8192-asset pilot、完整 unseen suites 统计、Isaac pose parity 和 PPO transfer 尚未运行。目前只证明执行与物理合同闭合，不支持跨手型泛化已经成立。

## 运行入口

```bash
source /home/hac/isaac/env_isaaclab/bin/activate

# 三项 objective 前向预实验
python -m anymani.distill.ssl.pretrain \
  --phase calibrate_objectives \
  --max_epochs 32 \
  --num_minibatches 4 \
  --assets_per_minibatch 64 \
  --q_per_asset_per_minibatch 8 \
  --mini_epochs 1 \
  --microbatch_size 64 \
  --max_resident_assets 64 \
  --seed 20260813 \
  --device cuda:0 \
  --experiment_name canonical_multi_anchor_gaussian_preexperiment
```

预实验完整覆盖 `32 × 4 × 64 × 8 = 65536` 个新 `(asset,q)` pairs，不执行 optimizer update。CLI flags 描述一次运行，Python preset 定义方法、representation、objective 和未覆盖默认值。SSL 是 task-free PyTorch/Warp 进程，没有 Isaac Sim 窗口或 `--headless`。产物位于 `logs/ssl/<experiment>/<UTC timestamp>/`；JSONL 保存逐 update 预算轴，TensorBoard 的 epoch/validation 曲线以 `new_pairs_seen` 为横轴。
