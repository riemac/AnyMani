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

## 保留表示与五项目标

retained encoder 只读取当前物理 q 与静态 evidence。多锚点前端把 home surface 和空间旋量表示为相对完整 anchor constellation 的关系；graph-biased Transformer 保持全连接 attention，拓扑只作 soft bias。模型输出逐 owner 的 $Z^{(0)}$ 与逐 JOINT 的 $z_i^{(1)}$，其中

$$
z_i^{(1)}=H_1\!\left([z_i^{(0)}\Vert f_i^{screw}]\right)
$$

是整手场 Jacobian 第 $i$ 列的固定宽度表示，不是对自身 $z_i^{(0)}$ 的普通导数。两个 SSL-only readers、query backend 和 target backend 在导出 retained artifact 时删除。

科学方法聚合根位于 `distill/methods/multi_anchor_gaussian_implicit_field/`。五项主损失是 density、κ、derived-field、Sobolev 与 chain。paired parity 不再作为第六项主损失。joint-sign rewrite 是 method 专属输入增强：每个 $(asset,q)$ 以 0.20 概率恰好翻一个有效 JOINT 的 $(q,q_{home},\mathcal S)$；density/distance 不变，对应 JOINT 的 $\kappa/g$ 翻号。validation 另做双前向 parity audit。归约按 $(asset,q)$ 等权；一阶 active/zero 先分别平均，再 1:1 合并。Trainer 不解释 owner、query 或 edge 轴。

`calibrate_objectives` 与 `pretrain` 共用同一 `EmbodimentPretrainCfg` 和正式 `ssl.yaml`。calibration 前向计算全部五项，不更新参数，也不自动改权重；人看 artifact 后修改 `OBJECTIVES_CFG` 的显式权重。

## 声明式运行架构

schema 4 的 `EmbodimentPretrainCfg` 只组合五个 concrete roles：`data`、`method`、`trainer`、`evaluation` 和 `run`。完整实验由 [`experiments/multi_anchor_gaussion_implicit_field.py`](experiments/multi_anchor_gaussion_implicit_field.py) 用 Python 装配；[`canonical_multi_anchor_gaussian.yaml`](../../presets/ssl/canonical_multi_anchor_gaussian.yaml) 只保留旧 CLI 名称。padding 上限由 method 从 resolved 资产和 model graph support 推导，超出则失败。schema 1/2 配置和含 paired 的旧 checkpoint fail-closed。

## 在线日程与恢复

每个 epoch 打乱一次全部 train assets，再按 `max_resident_assets` 切成 GPU window。同一窗内的资产先完成该 epoch 的全部 q coverage，再切下一窗；尾资产组、尾 q block 和尾 accumulation 保持真实较短长度。正式 8192-asset 的 epoch 数和每资产 q 预算尚未拍板；当前 Python façade 里的 `20 × 256` 只是脚手架默认值。

schema 4 full checkpoint 保存完整 model/readers、optimizer、声明权重、calibration artifact hash、dirty worktree 指纹、当前 permutation、window/q-round/group cursor、每资产 Sobol cursor、validation selection history、CPU/CUDA RNG、resolved config 和 expanded physical manifest。resume 后下一资产组与下一 q 必须和不中断运行一致。训练结束后加载 validation-best immutable checkpoint，再导出独立 `retained_artifact.pt`。

validation 每项资产固定 64 q，计算 density、κ 与 derived-field 的 initialization-normalized score。冻结后执行 query-only、same-asset q shuffle、cross-asset shuffle、first-order zero、JOINT shuffle 和 sign flip。training-morphology independent-q bank 通过 method 封闭接口重放完全相同的 q/query/teacher digest。unseen-variant-set、unseen-mother 和 official-zero-shot 当前只完成 asset/physical identity 审计。

## 证据边界

当前 deterministic contracts 覆盖 schema 4 Python composition、window-major coverage/tails/resume、五项 objective 归约、forward-only calibration、source realization、query/sigma replay、padding masks、joint-sign rewrite、full checkpoint 和 standalone artifact。synthetic integration 通过；real CUDA/Warp integration 按环境条件执行。RTX 5070 Ti 上既有 retained encoder $B=4096$ p95 为 20.13 ms，但该计时排除 teacher、readers、policy、Isaac Sim 与 `env.step`。

正式 8192-asset pilot、unseen suites 模型评估、official zero-shot、Isaac pose parity 和 PPO transfer 尚未运行。目前证据支持实现与物理合同闭合，不支持跨手型泛化已经成立。

## 运行入口

```bash
source /home/hac/isaac/env_isaaclab/bin/activate
python -m anymani.distill.ssl.pretrain

# Hydra override 示例
python -m anymani.distill.ssl.pretrain \
  trainer.optimizer.learning_rate=1e-4 \
  method.representation.query.query_count=128 \
  run.phase=calibrate_objectives
```

产物位于 `logs/ssl/<experiment>/<UTC timestamp>/`，包括 resolved config、dataset/expanded manifest、calibration、JSONL metrics、fixed validation、independent-q replay、ablation、selection history、full checkpoints 和 standalone retained artifact。实验统计语义见 [`../diagnostics/README.md`](../diagnostics/README.md)。
