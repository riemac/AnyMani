# AGENTS.md

## Project Overview

基于Isaac Lab框架的机器人仿真环境开发规范文档。本项目专注于LeapHand灵巧手的强化学习训练环境构建。

## 核心首要原则

1. **🤖 主动确认，持续循环 (Proactive Confirmation, Continuous Loop)**

   * **操作机制:**

     1. **输出方案:** 在对话中输出完整的分析内容和下一步方案（代码、计划、分析等）。
     2. **请求确认:** 调用 `mcp-feedback-enhanced` 工具，用一句话摘要请求批准。
     3. **执行:** 获得批准后执行该步骤。
   * **循环机制:**
     一旦步骤执行完毕，必须再次调用 `mcp-feedback-enhanced`，准备进入下一步。
   * **终止条件:** 仅当收到“任务结束”、“停止对话”等明确指令时循环终止。

2. **🧐 事实驱动，杜绝猜测**

   * **信息源:** 不分先后，具情况选用

     1. 本地代码（查询IsaacLab 源码、项目文档、示例，都首选 `semantic_search` 对整个codebase进行检索，该方式在信息探知上效率最高）。
     2. 官方文档（`context7`工具或 `githubRepo`，当查询isaacsim, physx, rl_games等不在工作区的第三方库）。
     3. 网络搜索（`github`, `fetch`工具）。
   * **精准定位:** 
     回复问题时，若引用了代码、文档片段等信息，提供精准定位，可让我快速跳转。

注意，用中文回答

## 工作区结构

```bash
AnyRotate/                    # 个人AnyRotate项目根目录（位于~/isaac/AnyRotate）
├── scripts/                 # 脚本目录
│   ├── debug/              # 调试脚本目录
│   ├── demo/               # 演示脚本目录
│   ├── evaluate/           # 评估脚本目录
│   ├── list_envs.py        # 环境列表脚本
│   ├── random_agent.py     # 随机智能体示例
│   ├── zero_agent.py       # 零智能体示例
│   └── rl_games/           # rl_games 强化学习相关脚本
├── source/                  # 源代码目录
│   └── leaphand/           # 主要功能模块
│       ├── assets/         # 资源文件目录 (*.usda)
│       ├── docs/           # 项目文档 (*.md)
│       ├── leaphand/       # 核心功能实现
│       │   ├── robots/     # 机器人定义
│       │   │   └── leap.py
│       │   ├── tasks/      # 任务环境定义
│       │   │   ├── direct/leaphand/   # DirectRLEnv
│       │   │   └── manager_based/leaphand/   # ManagerBasedRLEnv（目前主要开发目录）
│       └── pyproject.toml   # Python 项目配置
├── logs/                    # 日志目录
└── outputs/                 # 输出目录

IsaacLab/                    # IsaacLab官方核心框架目录（位于~/isaac/IsaacLab）

```

### 研究进展

目前正在做Leaphand手内操作方面的研究，主要环境架构为ManagerBasedRLEnv，主要使用的算法库为rl_games。最新科研idea位于 `source/leaphand/leaphand/ideas/idea.ipynb`

## 开发规范

### 脚本开发

* 脚本开发通常是为了验证、调试或评估特定功能。
* Isaaclab项目提供了两个基础脚本：`list_envs.py` 用于列出所有可用环境，`random_agent.py` 该用于验证环境正确性和集成功能。
* 注意脚本复用，一般`random_agent.py`可测试大部分内容，若不满足的情况才需要开发新的脚本。
* 如若开发脚本，应按照性质放在`scripts/`目录下对应的子目录中，`debug/`用于调试功能，`demo/`用于演示功能（含可视化），`evaluate/`用于评估功能。
* 遵循 standalone 开发模式，使用 **appLauncher** 作为核心启动器。
* 注意：部分依赖库如omni,card需在 **IsaacSim 环境启动后** 才能正常导入。

### 文档管理

* 非必要情况不需新增文档。
* 所有项目文档统一存放在 `source/leaphand/docs/` 目录。
  
### 代码风格

* 完成文件开发后，调用 `pylance mcp server` 相关工具进行代码语法检查。

## 注意事项

### 操作要求

* **环境激活:**
  执行终端指令前，必须在 `~/isaac` 目录下激活 uv 环境：

  ```bash
  source env_isaac/bin/activate
  ```
* **路径切换:**
  在什么项目开发或验证，应切换到相应目录下。如在 `AnyRotate` 项目根目录下进行：

  ```bash
  cd /home/hac/isaac/AnyRotate
  ```
* **IsaacSim 模块导入:**
  某些IsaacSim模块（如 `isaacgym`, `omni.isaac` 等）只能在Applauncher启动IsaacSim环境后导入使用，否则会报找不到错误。

* **指令遵循:**
  - `mcp-feedback-enhanced` 调用失败时必须重试。
  - 优先使用 `semantic_search` 工具查询代码库相关内容。深入探索某一主题时，可反复使用 `semantic_search`。当其他本地查询方式搜索不到相关内容，也切换到 `semantic_search` 工具。只有在知道精确定位后，再使用 `file_read` 工具读取整个文件内容。

### 工程问题

* **环境原点偏置:**
  多环境并行训练时，每个环境实例有自己的原点偏置 (`env_origins`)。开发时需要考虑这个偏置对位置、姿态等计算的影响。

* **坐标表示与转换:**
  注意使用的是World坐标系还是Body坐标系，以及如何正确转换。

* **任务环境导入路径:**
  项目使用 `uv pip install -e source/leaphand` 将 Extension 注册到 Python 解析路径，可直接导入：
  ```python
  from leaphand.tasks.direct.leaphand.leaphand_env import LeaphandEnv
  from leaphand.tasks.direct.leaphand.leaphand_env_cfg import LeaphandEnvCfg
  ```
* **环境步数:**
  ManagerBasedRLEnv 的 `common_step_counter` 是针对所有环境的共同步数，不是单独环境步数×环境数。在课程学习中需注意区分。

* **环境与管理器:**
  ManagerBasedRLEnv 环境架构下，环境类及其各管理器已暴露大量可用属性和信息，开发过程中应优先复用这些现有资源，避免重复实现功能。
  ManagerBasedRLENV 的各模块功能实现应self-contained，专注该模块的功能

* **SceneEntityCfg 关节索引顺序:**
  使用 `SceneEntityCfg` 指定 `joint_names` 时，需注意 `preserve_order` 的设置，以决定关节索引顺序与指定顺序是否一致。

* **雅可比矩阵**
  - PhysX 返回的雅可比矩阵顺序为 [线速度 v; 角速度 w]，注意与某些文献中 [w; v] 的顺序不同。本项目遵循《Modern Robotics》里的 [w; v] 约定。
  - 获取雅可比矩阵时注意基座的类型与索引的处理。对于固定基座关节机器人，self._asset.root_physx_view.get_jacobians() 的 body 索引需要减1，因是雅可比矩阵不包含固定基座的刚体。而浮动基座 Jacobian 包含基座，索引不变，但关节索引需要偏移 6 (跳过浮动基座的 6 个自由度)。



### 个人偏好

* **数理回复:**
  解释算法等机理性内容，结合数学公式。简洁美观的经渲染数学公式比大段文字和代码描述更易懂。

* **注释风格:**
  使用与 IsaacLab 官方一致的 Google Docstring Style。对于实现复杂算法的方法，在 docstring 中增加 Notes 部分，采用增强型 ASCII 风格 + 伪代码来描述算法核心思想和数学模型。

* **表格对比:**
  涉及到众多复杂可比项时，使用表格进行总结对比。

## 代码实践

* **代码隔离:** 无明确指示，不修改 IsaacLab 核心代码，开发主要在独立项目中进行。
* **风格一致:** 代码与项目风格与 IsaacLab 保持一致。
* **善用框架:** 优先利用 IsaacLab 现有功能（包括类、方法、属性等信息），避免重复造轮子。