# AGENTS.md

AnyMani 是基于 Isaac Lab 的灵巧操作研究框架，服务手型泛化的手内操作。本文只保存稳定的代码边界、
测试规则和运行入口，不记录某一轮实验进度。


## 项目架构

当前按 “资产生产-机器人适配-任务环境-训练管线-网络架构” 功能职责和边界划分：

| 目录 | 职责 | 边界 |
|------|------|------|
| `source/anymani/anymani/assets/` | 手资产生成：pre-made、post-mutate、validator、exporter、physics closure | 不写任务 reward / policy / 训练逻辑 |
| `source/anymani/anymani/robots/` | embodiment runtime adapter：把 generated hand / 真实手资产 lower 成 Isaac Lab robot cfg | 不写任务 MDP，不拥有 asset bank 生成逻辑，不承载训练算法 |
| `source/anymani/anymani/tasks/` | Isaac Lab 任务环境：scene、obs、action、reward、reset、termination、Gym 注册 | 不处理资产生成细节，不承载训练算法 |
| `source/anymani/anymani/distill/` | 物理表征、共享模型、SSL/IL/RL stage、实验诊断与训练入口 | 消费 `tasks` 暴露的 env contract，不接管 env / spawn 内部实现 |

核心依赖方向：

```text
assets -> robots -> tasks
assets ------------> distill
tasks  ------------> distill
```

`robots` 定义“某个手如何被 Isaac Lab spawn”，`tasks` 定义“手在什么任务里交互”，
`distill` 定义“如何构造学习表征并训练跨手型策略”。`distill.representations.sources` 可直接消费
`assets` 的 typed geometry semantics 来构造 simulator-independent POE/FK/Jacobian 与碰撞场教师；
Isaac spawn/articulation 仍只属于 `robots`。`tasks` 通过 `robots` 消费 asset bank，不拥有 importer 细节。

`Research/` 是独立的下游证据与科研记录仓库：它可以引用 AnyMani 的 commit、配置、日志和公开任务接口，
但 `source/anymani/` 不得 import、解析或要求 Research vault 存在。N052 等实验编号属于 Research，
AnyMani 文件、公共符号与 Gym ID 使用稳定的算法/环境语义。

## 开发约定

### 1. 及时出清，避免臃肿

本项目允许研究过程中的阶段性实现，但一旦新的抽象、接口或实验 contract 稳定，应及时删除已废弃的旧实现、旧字段、旧测试和旧注释。

不用的东西不应作为“历史说明”继续留在代码里；它会污染科研语义，让后续读者误以为旧路线仍是可选建模方案。除非用户明确要求保留迁移期兼容，否则出清优先于 deprecated 壳、兼容包装和长篇历史注释。

原则上，当遇到超过 1k 行的大文件，要考虑是否拆分或重构。

### 2. 注释服务科研语义

开发时遵守 `annotation` skill。

### 3. 代码自查工具

根目录 `pyproject.toml` 配有 `ruff`(lint/format)与 `pyright`(类型检查,basic 模式,已指向 `env_isaaclab` 环境),规则对齐项目既有 pre-commit 工具链(black 行宽 120 + flake8 + isort + pyupgrade)。供 agent 改完代码后做快速自查,非强制流程;正式提交仍以 pre-commit 为准。

### 4. 测试策略：contract TDD + runtime smoke

本项目鼓励对**纯数学、schema、张量 shape、路径/manifest、网络模块和 MDP contract** 采用 TDD：先写最小失败用例，再实现。Isaac Sim / PhysX / 训练闭环不强行 TDD；这类重型路径用少量 headless smoke / integration test 验证能 reset、step、输出合法张量。

优先测试能提前揪出科研语义错误的内容：坐标系变换、SO(3)/SE(3) 公式、reward 单调性、obs/action 量纲、mask 语义、cache key/schema、token routing、attention bias shape 与零初始化。不要为了追覆盖率给研究草稿或纯注释 scaffold 写空洞测试。

测试按“要证伪的命题”选择最低足够层级：若命题是公式、schema、配置声明或张量 shape，就用默认 contract test；若命题依赖 Isaac Sim 运行时状态、USD stage 副作用、PhysX handle、传感器 buffer 或完整 reset/step 生命周期，就用显式 runtime smoke；若命题只能从 rollout 统计判断，例如 reward 量级、成功判据释放、训练入口日志是否闭合，就用短训练 sanity，不把它伪装成普通单元测试。

默认 `pytest` 只运行 `pytest.ini` 中声明的 contract paths，约定为不启动 Isaac Sim / Kit / `AppLauncher`。runtime smoke 统一放在 `source/anymani/anymani/smokes/` 下，并通过显式路径运行。不要主要依赖 pytest marker 排除 IsaacSim 测试，因为 pytest 收集阶段可能已 import 测试模块并启动 Kit。

硬件绑定的 `performance` 与短训练 `training_sanity` 虽可位于 contract tree 下，但必须由默认 marker expression 排除并通过显式路径/marker 运行。`distill/tests/AGENTS.md` 记录 geometry encoder 的具体计时边界；性能 test 不得通过缩小已声明 batch/structure shape 或缓存会随 full fine-tune 失效的 learned activation 通过门槛。

IsaacSim smoke 必须小而硬：少量 env、少量 step、明确 assert 运行时副作用，并用 `timeout --kill-after` 包住。例如：

```bash
cd /home/hac/isaac/AnyMani
source /home/hac/isaac/env_isaaclab/bin/activate
timeout --kill-after=20s 60s /home/hac/isaac/IsaacLab/isaaclab.sh -p -m pytest \
  source/anymani/anymani/smokes/isaacsim/test_gm_single_asset_structural_collision.py -q -s
```

开发新的 smoke test 可在 `AnyMani/source/anymani/anymani/smokes` 下。

### 5. 训练记录分析

本项目 uv 环境内装有 TensorBoard 等分析依赖。运行时结构化证据由 `distill/diagnostics` 的项目合同管理，实际产物位于 `AnyMani/logs/`；TensorBoard只用于在线曲线。SSL以JSONL保存可审计分层指标、NPZ保存dense latent/mask/error arrays；掌旋RL以Parquet保存global/cell/asset逐update指标、HDF5保存固定trajectory，并由父进程单独记录RSS/swap/NVML和PhysX/CUDA fatal evidence。Research可下游引用这些证据并维护实验结论，但source不得依赖Research vault。

### 6. 研究产物与脚本归属

新benchmark统一落在`logs/benchmarks/<topic>/<case>/`：topic是稳定任务主题，如`heterogeneous_rotation`；case标识一次可复盘试验。配置、命令、资源、评价、分析和媒体放在该case内，必要时再按子条件分目录。不要把每个检查点或分析文件继续平铺到`logs/benchmarks`根部。正式训练自己的`logs/distill/rl_games/<task>/<run>/`层级保持不变。

新增或修改的产物生产入口应提供分层默认路径及显式输出覆盖；调用保留平铺默认值的历史入口时，显式指定新case目录。历史日志、checkpoint及其引用路径保持原样，迁移需另行验证身份与引用，不能作为顺手清理。

| 工具职责 | 新入口归属 |
| --- | --- |
| 资产生成、选择与集合发布 | `assets/scripts/` |
| 预抓取搜索、准入与专属物理编排 | `pregrasp/scripts/`；纯数学仍留核心模块 |
| PPO训练辅助、迁移、梯度与策略探针 | `distill/rl/scripts/` |
| 可复用只读指标与证据分析 | `distill/diagnostics/`的对应分析模块 |
| 跨模块研究编排 | `scripts/research/<topic>/` |
| 通用进程/资源benchmark包装 | `scripts/benchmarks/` |

已有入口不为满足目录样式立即搬迁；新目录只在确有工具时创建，不留空包或占位接口。迁移时同步调用方、文档和复现身份，历史命令按当时版本解释。Research内的一次性绘图辅助不成为AnyMani运行时依赖。

## 常用操作

### 环境激活

项目为 uv 环境

```bash
source ~/isaac/env_isaaclab/bin/activate
```

python 脚本使用该环境下的可执行文件和库，而非系统默认的 python3。

### 列出所有环境

```bash
cd /home/hac/isaac/AnyMani
source /home/hac/isaac/env_isaaclab/bin/activate
python scripts/list_envs.py
```

### 运行环境

```bash
# 随机 agent 测试,验证环境可用性
python scripts/random_agent.py --task AnyMani-LeapHand-Joint-v0 --num_envs 1 --headless
```
