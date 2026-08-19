# AGENTS.md

`ssl` 拥有 task-free embodiment representation pretraining 的声明组合、在线训练生命周期、evaluation、checkpoint 与 evidence。它消费 `assets` 的 typed semantics 和 `representations.sources` 的 simulator-independent oracle；不解析 URDF/hand.yaml，不拥有 task MDP、PPO action 或 Isaac Sim 生命周期。

## 结构与所有权

```text
ssl/experiment.py      schema 3 EmbodimentPretrainCfg 与唯一 façade
ssl/config_store.py    concrete Hydra structured schema 注册，不解析字段
ssl/data/              asset catalog role，只解析固定 dataset partitions
ssl/methods/           representation -> model -> shared nodes -> objective 调用图
ssl/runtime/           sampling、resident lease、train/evaluation/checkpoint lifecycle
ssl/checkpoint.py      full resume checkpoint 与 standalone retained artifact
ssl/pretrain.py        python -m CLI，只 compose concrete config 并调用 façade
distill/presets/ssl/   canonical Hydra YAML recipe 与各 component 数值
```

局部 cfg 与 runtime 由同一 owner module 管理，并用不序列化的 `ClassVar runtime_type` 关联。`EmbodimentPretrainCfg` 只能组合 `data / method / trainer / evaluation / run`；不要恢复集中式字段 parser、递归 `_target_`、Python canonical config 子类、万能 registry 或兼容 alias。正式科学数值属于 packaged YAML，dataclass 默认值只提供最小合同。

## 运行边界

Data 只交付平等 assets、partitions 和 provenance，不采 q、不生成 query/target、不把 family/mother 转为隐藏权重。Method 负责 representation/model/objective 的调用顺序和共享 autograd 节点。Trainer 负责每 q round 全资产 permutation、Sobol q、coverage、minibatch、尾组、gradient accumulation 和 optimizer。Evaluation 负责 fixed bank、selection、q-bank replay 与 ablation。Run 负责路径、resume seed 和 lineage。

GPU resident window 是资源限制，不得改变全资产随机顺序或统计测度。驱逐必须释放 `DeviceGeometrySource` 的 Warp lease。checkpoint 只在 optimizer boundary 保存，并必须覆盖 schedule permutation/cursor、每资产 Sobol cursor、optimizer、selection 与 CPU/CUDA RNG；resume 后下一样本必须逐项一致。

full checkpoint 只服务预训练恢复。IL/PPO 只能读取 standalone retained artifact；artifact 只能包含 retained encoder cfg/state、输入合同、`FeatureSpec` 和 lineage，不得包含 optimizer、runtime state、query/target backend、reader 或 objective。

## 科研语义

保留 50:25:25 query、多带宽 density、sampled κ、derived field、Sobolev、chain 和 paired parity。query、sigma、edge、surface/anchor realization 与 privileged target 属于 representation。模型的一阶 head 固定为 `[z_i^(0) | f_i^screw] -> z_i^(1)`；κ reader读取 owner Z0、query feature、对应 Z1 和 joint selector，不能把两者混成同一个“kinematics decoder”。

paired rewrite 必须同步变换 `(q, q_home, space_screws)`；不要输入裸 sign bit，也不要恢复 even/odd screw 双支。Trainer 按 term 交付的 additive numerator/denominator 合并，不解释 owner/query/edge 轴。calibration 只消费固定 generated train minibatches，结果作为 runtime evidence 冻结，不覆盖 declared term cfg。

official assets 不参与 train、calibration 或 checkpoint selection。所有 train/validation/evaluation roles 在 geometry materialization 后仍须同时按 `content_hash` 和 `physical_geometry_hash` 隔离。未拍板的 surface/anchor/query/sigma 刷新周期、样本复用与新 reduction 规则不得在框架重构中暗自固定。

## 验证

```bash
source /home/hac/isaac/env_isaaclab/bin/activate
pytest source/anymani/anymani/distill/tests/contracts/ssl -q
pytest source/anymani/anymani/distill/tests/contracts -q
pytest source/anymani/anymani/distill/tests/integration -q
ruff check source/anymani/anymani/distill/ssl
pyright source/anymani/anymani/distill/ssl
```

完整训练依赖 CUDA Warp；普通 contracts 不启动 Isaac Sim。没有正式 pilot、unseen-suite evaluation 或 PPO transfer 证据时，不得声明 cross-topology、cross-DOF、official zero-shot 或策略泛化成立。
