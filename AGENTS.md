# AGENTS.md

本文件为 AnyMani 项目说明。它是基于 Isaac Lab 的灵巧操作研究框架，目前处于 embodiment branch，用于研究和开发 "手型泛化的手内操作" 科研任务。具体背景可见 `AnyMani/Research/总体/科研背景说明.md`。


## 项目架构

当前按 “资产生产-训练管线-网络架构” 功能职责和边界划分：

| 目录 | 职责 | 边界 |
|------|------|------|
| `source/anymani/anymani/assets/` | 手资产生成：pre-made、post-mutate、validator、exporter、physics closure | 不写任务 reward / policy / 训练逻辑 |
| `source/anymani/anymani/tasks/` | Isaac Lab 任务环境：scene、obs、action、reward、reset、termination、Gym 注册 | 不处理资产生成细节，不承载训练算法 |
| `source/anymani/anymani/distill/` | 网络架构与训练管线：models、IL / distillation、RL、训练入口 | 消费 `assets` 和 `tasks`，不接管 env 内部实现 |

核心依赖方向：

```text
assets -> tasks -> distill
```

`tasks` 定义“手在什么任务里交互”，`distill` 定义“如何训练跨手型策略”; `distill` 依赖于 `tasks` 定义的环境接口，`distill` 和 `tasks` 都消费 `assets` 导出的 asset bank。

## 开发约定

### 1. 及时出清，避免臃肿

本项目允许研究过程中的阶段性实现，但一旦新的抽象、接口或实验 contract 稳定，应及时删除已废弃的旧实现、旧字段、旧测试和旧注释。

不用的东西不应作为“历史说明”继续留在代码里；它会污染科研语义，让后续读者误以为旧路线仍是可选建模方案。除非用户明确要求保留迁移期兼容，否则出清优先于 deprecated 壳、兼容包装和长篇历史注释。

### 2. 注释服务科研语义

开发时遵守 `annotation` skill。

### 3. 代码自查工具

根目录 `pyproject.toml` 配有 `ruff`(lint/format)与 `pyright`(类型检查,basic 模式,已指向 `env_isaaclab` 环境),规则对齐项目既有 pre-commit 工具链(black 行宽 120 + flake8 + isort + pyupgrade)。供 agent 改完代码后做快速自查,非强制流程;正式提交仍以 pre-commit 为准。

### 4. 测试策略：contract TDD + simulation smoke

本项目鼓励对**纯数学、schema、张量 shape、路径/manifest、网络模块和 MDP contract** 采用 TDD：先写最小失败用例，再实现。Isaac Sim / PhysX / 训练闭环不强行 TDD；这类重型路径用少量 headless smoke / integration test 验证能 reset、step、输出合法张量。

优先测试能提前揪出科研语义错误的内容：坐标系变换、SO(3)/SE(3) 公式、reward 单调性、obs/action 量纲、mask 语义、cache key/schema、token routing、attention bias shape 与零初始化。不要为了追覆盖率给研究草稿或纯注释 scaffold 写空洞测试。

默认 `pytest` 只运行 `pytest.ini` 中声明的 contract paths，约定为不启动 Isaac Sim / Kit / `AppLauncher`。凡验证 URDF/USD import、`PhysicsCollisionGroup`、contact sensor、scene cloning、PhysX step、完整 reset/step 生命周期的测试，放在 `source/anymani/anymani/smokes/` 下，并通过显式路径运行。不要主要依赖 pytest marker 排除 IsaacSim 测试，因为 pytest 收集阶段可能已 import 测试模块并启动 Kit。

IsaacSim smoke 必须小而硬：少量 env、少量 step、明确 assert 运行时副作用，并用 `timeout --kill-after` 包住。例如：

```bash
cd /home/hac/isaac/AnyMani
source /home/hac/isaac/env_isaaclab/bin/activate
timeout --kill-after=20s 240s /home/hac/isaac/IsaacLab/isaaclab.sh -p -m pytest \
  source/anymani/anymani/smokes/isaacsim/test_gm_single_asset_structural_collision.py -q -s
```

### 5. 训练记录分析

本项目 uv 环境内装了 tensorboard 等库，可供 agent 使用 bash 脚本命令来分析训练记录，给出合适建议。

## 常用操作

### 环境激活

项目为 uv 环境

```bash
source ~/isaac/env_isaaclab/bin/activate
```

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

## 参考项目

| 项目 | 说明 | 路径 |
|------|------|------|
| **Isaac Lab** | 上游框架 | `/home/hac/isaac/IsaacLab` |
| **rl_games** | RL 算法库 | `/home/hac/isaac/rl_games` |
| **get-zero** | get-zero 论文项目代码 | `/home/hac/isaac/get_zero` |
| **tro-grasp** | tro-grasp 论文项目代码 | `/home/hac/isaac/TRO-Grasp` |
