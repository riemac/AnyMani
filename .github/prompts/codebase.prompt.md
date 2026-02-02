---
agent: agent
---
## 核心首要原则 (Core Principles)

1. **🤖 主动确认，持续循环 (Proactive Confirmation, Continuous Loop)**
   
  良好及时的反馈可以显著提升任务完成效率和质量。务必遵循以下准则：

  * **操作机制**
    1. **步骤分析：** 在每个任务步骤开始前进行充分分析，明确目标、输入输出与潜在风险，制定可验证的成功标准
    2. **输出方案：** 在对话中详细给出下一步方案（代码草案、执行计划、数据需求等），避免仅在反馈摘要中放置关键细节
    3. **请求确认：** 使用 `mcp-feedback-enhanced` 工具提交一条一言以概括的反馈摘要以请求批准；若调用失败，必须重试直至成功
    4. **创建计划：** 通过 `manage_todo_list` 工具将方案拆解为明确的待办项
    5. **执行：** 在得到批准后按待办项逐一执行，并在执行过程中记录关键结果与假设以便复盘
  * **循环机制**
    - 每完成一个步骤后，必须再次调用 `mcp-feedback-enhanced` 汇报进展并请求进入下一步骤的批准；确认后继续下一轮分析与执行
  * **终止条件**
    - 仅在反馈时收到明确指令（例如“任务结束”、“停止对话”等）时终止循环

1. **🧐 事实驱动，杜绝猜测**
  
  精准的上下文信息是高质量回复的基础。在收集信息时，务必遵循以下准则：

   * **信息源:** 不分先后，具情况选用

     1. **本地代码**（查询IsaacLab 源码、项目文档、示例，都首选对整个codebase进行检索，该方式在信息探知上效率最高）
        - **工具选择:**
          - 如果是IsaacLab源码：优先使用 `mcp_augmentcode_codebase-retrieval`（精度极高，仅限IsaacLab代码库），不可用时改用 `semantic_search`
          - 如果是个人项目及跨项目代码库：使用 `semantic_search`
        - **检索策略:**
          - 优先codebase检索，再细化到特定文件
          - 必须使用规范完整的问题描述长句而非简短关键词罗列，例如："What's the aaa... How to implement xxx for yyy in zzz? Including www ..." 或类似语句
          - 可反复调用检索工具深入探索，直至精准定位所需信息
          - 尽量避免使用 `grep_search`
          - 确定关键文件后，使用 `read_file` 读取完整内容。若再 codebase 检索时便已获取所需内容，则无需使用 `read_file`

     2. **官方文档**（查询 isaacsim, physx, rl_games 等不在工作区的第三方库）
        - **工具选择（按优先级）:**
          - `DeepWiki` (`mcp_cognitionai_d_ask_question`): **首选**，适用于概念解释、教程类问题、跨框架对比、功能可用性查询，信息密度最高
          - `Context7` (`mcp_io_github_ups_get-library-docs`): 仅在查询**主流热门库** (transformers, PyTorch, MuJoCo) 时使用，省 token
          - `GitHub Repo` (`github_repo`): 需要**精确源码行号定位**时使用，token 消耗约为 DeepWiki 的 4 倍
        
     3. **网络搜索**（`github`, `fetch`工具）
   * **精准定位:** 
     回复问题时，若引用了代码、文档片段等信息，提供精准定位，可让我快速跳转