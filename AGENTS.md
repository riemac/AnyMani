# AGENTS.md

## 适用范围
所有问题和任务皆适用且需遵循本文规范。包括简单的问题咨询、代码开发、调试排障、文档编写等，没有例外。

## Project Overview

基于Isaac Lab框架的机器人仿真环境开发规范文档。本项目专注于LeapHand灵巧手的强化学习训练环境构建

## 核心首要原则

1. **🤖 主动确认，持续循环 (Proactive Confirmation, Continuous Loop)**
   
  良好及时的反馈可以显著提升任务完成效率和质量。务必遵循以下准则：

  * **操作机制**
    1. **步骤分析：** 在每个任务步骤开始前进行充分分析，明确目标、输入输出与潜在风险
    2. **输出方案：** 在对话中详细给出下一步方案（代码草案、执行计划、数据需求等），避免仅在反馈摘要中放置关键细节
    3. **请求确认：** 使用 `mcp-feedback-enhanced` 工具提交一条一言以概括的反馈摘要以请求批准；若调用失败，必须重试直至成功
    4. **创建计划：** 通过 `manage_todo_list` 工具将方案规划为具体待办项列表
    5. **执行：** 在得到批准后按待办项逐一执行，并在执行过程中记录关键结果与假设以便复盘
  * **循环机制**
    - 每完成一个步骤后，必须再次调用 `mcp-feedback-enhanced` 汇报进展并请求进入下一步骤的批准；确认后继续下一轮分析与执行
  * **终止条件**
    - 仅在反馈时收到明确指令（例如“任务结束”、“停止对话”等）时终止循环

1. **🧐 事实驱动，杜绝猜测**
  
  精准的上下文信息是高质量回复的基础。在收集信息时，务必遵循以下准则：

   * **信息源:**

     1. **本地代码**（查询IsaacLab 源码、项目文档、示例，都首选对整个codebase进行检索，该方式在信息探知上效率最高）
        - **工具选择:**
          - 如果是IsaacLab源码：优先使用 `mcp_augmentcode_codebase-retrieval`（精度极高，仅限IsaacLab代码库），不可用时改用 `semantic_search`
          - 如果是个人项目及跨项目代码库：使用 `semantic_search`
        - **检索策略:**
          - 优先codebase检索，可反复调用检索工具深入探索，直至精准定位所需信息。如有必要再细化到特定文件
          - 必须使用规范完整的问题描述长句而非简短关键词罗列，例如："What's the aaa... How to implement xxx for yyy in zzz? Including www ..." 或类似语句
          - 尽量避免使用 `grep_search`
          - 确定关键文件后，使用 `read_file` 读取完整内容（startline=1, endline默认1000行起步）。若再 codebase 检索时便已获取所需内容，则无需使用 `read_file`

     2. **官方文档**（查询 isaacsim, physx, rl_games 等不在工作区的第三方库）
        - **工具选择（按优先级）:**
          - `DeepWiki` (`mcp_cognitionai_d_ask_question`): **首选**，适用于概念解释、教程类问题、跨框架对比、功能可用性查询，信息密度最高
          - `Context7` (`mcp_io_github_ups_get-library-docs`): 仅在查询**主流热门库** (transformers, PyTorch, MuJoCo) 时使用，省 token
          - `GitHub Repo` (`github_repo`): 需要**精确源码行号定位**时使用，token 消耗约为 DeepWiki 的 4 倍
        
     3. **网络搜索**（`github`, `fetch`工具）
   * **精准定位:** 
     回复问题时，若引用了代码、文档片段等信息，提供精准定位，可让我快速跳转

注意，用中文回答

## 开发规范

### 脚本开发

* 脚本开发通常是为了验证、调试或评估特定功能。
* 注意脚本复用，一般`random_agent.py`可测试大部分内容，若不满足的情况才需要开发新的脚本
* 如若开发脚本，应按照性质放在`scripts/`目录下对应的子目录中，`debug/`用于调试功能，`demo/`用于演示功能（含可视化），`evaluate/`用于评估功能
* 遵循 standalone 开发模式，使用 **appLauncher** 作为核心启动器

### 文档管理

* 非必要情况不需新增文档。
* 所有项目文档统一存放在 `source/leaphand/docs/` 目录
  
### 代码风格

* 完成文件开发后，调用 `pylance mcp server` 相关工具进行代码语法检查

### 工程回馈

* 在实际项目开发时，若遇到一些通用性较强的工程问题或踩坑，可更新本AGENTS.md

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

### 工程问题

* **环境原点偏置:**
  多环境并行训练时，每个环境实例有自己的原点偏置 (`env_origins`)。开发时需要考虑这个偏置对位置、姿态、雅可比矩阵等计算的影响。

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