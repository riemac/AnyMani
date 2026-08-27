# AGENTS.md

`ssl/` 负责 task-free embodiment pretraining 的实验声明、Trainer 生命周期、checkpoint 和证据入口。
它消费 `assets` 的 typed semantics 与 `distill.methods` 的封闭接口；URDF/`hand.yaml`、物理 teacher、
固定评估测度、task MDP 和 Isaac Sim 生命周期继续由各自模块维护。

## Project Structure

```text
ssl/
├── experiment.py                    schema-8 pure-pretrain 根配置与 façade
├── post_training.py                 独立 validation/evaluation 根配置与 façade
├── config_store.py                  通过 experiments registry 注册配置
├── contracts.py                     runtime type 装配
├── pretrain.py                      配置驱动的训练/resume CLI
├── prepare_sources.py               Geometry Source Artifact Cache 前置准备 CLI
├── validate.py                      显式 checkpoint validation CLI
├── evaluate.py                      显式 checkpoint evaluation CLI
├── checkpoint.py                    通用 method/optimizer/trainer state 容器
├── experiments/
│   ├── __init__.py                  显式实验 registry
│   ├── geometry_ssl_multitask_representation_v0_7_3.py
│   └── multi_anchor_gaussion_implicit_field.py  legacy 快照，文件名保留历史拼写
├── data/                            固定 dataset partitions 的 typed 解析
└── runtime/                         sampling、resident window、lifecycle、checkpoint
```

## 实验快照

每次达成研究共识后，在 `experiments/` 新建一个语义自包含的 Python 快照。文件顶部用 paper flavor
说明研究问题、teacher、输入边界、retained representation、网络、objective、sampling、训练预算和
预期证据；重点是帮助科研人员逐行理解这次实验，不写成防御性免责声明。

快照至少包含 data、method、representation/model、objective、trainer 和 run；如论文需要 validation
或 evaluation，它们也从同一快照导出。`EXPERIMENT` 是训练主配置，事后阶段不属于训练生命周期。
新快照必须在 registry 中显式登记。旧快照按用户要求保留，但新实验不能隐式引用 legacy preset。

agent 负责判断参数变化的科研影响并说明依据。epochs、minibatch 预算、seed、device、output、cache、
microbatch、checkpoint cadence 和普通 optimizer 运行参数通常可以修订；`Z` 维度、token/entity 定义、
backbone 容量或结构、reader、objective/FairGrad、teacher 几何、sampling 或监督语义改变时必须新建
版本化快照。

## CLI 规则

日常训练只使用 `pretrain.py`。CLI 选择实验快照并提供平坦运行覆盖，不复制 method/model/objective
的深层字段，也不接受任意 Hydra `key=value` 路径。`--config` 接受 registry 名称或 Python 快照路径；
`--run_name` 只表示本次输出目录身份，不表示配置版本。

```bash
source /home/hac/isaac/env_isaaclab/bin/activate
/home/hac/isaac/IsaacLab/isaaclab.sh -p -m anymani.distill.ssl.pretrain --config geometry_ssl_multitask_representation_v0_7_3 --device cuda:0 --seed 20260813
```

训练入口不自动调用 validation、evaluation、PCA、best selection 或 retained export。实验快照若将
`source_cache_mode` 设为 `auto`，pretrain runtime 会在同一训练进程内检查并补建缺失的 source artifact，
随后以 readonly 方式训练；用户不需要手工拆分准备和训练命令。训练完成后，agent 必须给出与当前快照完全
匹配、可以直接复制执行的准确命令，不要求用户自行拼接深层配置字段。

## Geometry Source Artifact Cache

source cache 是从资产几何预先构造的静态 source artifact，不是原始资产副本，也不是包含 `q` 和 teacher
target 的离线训练数据集。它保存 collision union、home geometry、anchor bank、surface sampling 数据
以及 source/provenance identity；训练仍在线采样 `q`、query 和 sigma，再用物理 teacher 计算监督。

`prepare_sources.py` 保留为显式维护入口。正式实验快照可使用 `source_cache_mode=auto`，由 pretrain
内部完成“已有则复用、缺失则补建”；训练完成 source preparation 后切换为 readonly。显式 readonly
模式下，cache 缺失、损坏或 identity 不匹配时必须 fail closed。准备和训练必须引用同一个 `--config`：

```bash
/home/hac/isaac/IsaacLab/isaaclab.sh -p -m anymani.distill.ssl.prepare_sources --config geometry_ssl_multitask_representation_v0_7_3 --device cuda:0
```

`backup.sh` 只负责训练，不包含 prepare、status、validate 或 evaluate 子命令；source preparation 是否
发生由实验配置和 pretrain runtime 内部决定。

## Role And Checkpoint Boundaries

Data 交付 assets、partitions 和 provenance。Method/session 拥有 realization、teacher baseline、objective、
固定评估测度、完整 state 和 retained export。Trainer 只拥有 epoch、minibatch、mini-epoch、optimizer
和训练 checkpoint。validation/evaluation runtime 只消费已完成的 schema-8 full checkpoint，不回写训练目录。

训练 catalog、Sobol cursor、resident window 和 source identity 共同定义可复现轨迹；性能重排不得改变它们。
entity permutation 只能作为完整 typed entity-axis 变换执行：token、home evidence、graph、query、target
和 provenance 必须使用同一个双射；joint coordinate axis 不随之重排。相关合同同时覆盖合法 permutation
的 parity 与故意遗漏 graph/routing/target 轴的反例，后者必须 fail closed。
pure pretrain 不生成 best checkpoint 或 retained artifact。full checkpoint 服务 SSL resume/事后评估；RL/IL
只消费 schema-5 standalone retained artifact。

## Verification

配置 registry、CLI、快照 identity、source cache 和训练生命周期改变时，补充最小 contract test，并运行：

```bash
/home/hac/isaac/env_isaaclab/bin/python -m pytest -q source/anymani/anymani/distill/tests/contracts
/home/hac/isaac/env_isaaclab/bin/python -m pytest -q source/anymani/anymani/distill/tests/integration
ruff check source/anymani/anymani/distill/ssl
pyright source/anymani/anymani/distill/ssl
git diff --check
```

完整训练依赖 CUDA Warp。跨手型泛化结论必须由正式 pilot、unseen-suite evaluation 和 PPO transfer
证据支持，配置和合同测试本身不等于学习结果。
