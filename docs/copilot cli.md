# Copilot CLI 实验笔记

以第一手实验结论为准，优先于官方文档描述。

---

## 一、Context 注入机制

### 自动注入的 Instructions 文件

CLI 启动时自动读取以下路径（按优先级从高到低）：

```
AGENTS.md（git root & cwd）
CLAUDE.md / GEMINI.md
.github/copilot-instructions.md
.github/instructions/**/*.instructions.md
$HOME/.copilot/copilot-instructions.md
COPILOT_CUSTOM_INSTRUCTIONS_DIRS（环境变量，逗号分隔多个目录）
```

### 子目录 AGENTS.md 的注入规则

- **根目录 `AGENTS.md`**：自动注入，最高优先级，无需任何配置
- **子目录 `AGENTS.md`**：**不会**因为读了同目录文件就自动注入
- 要注入子目录：切换 cwd，或配置 `COPILOT_CUSTOM_INSTRUCTIONS_DIRS`：
  ```bash
  export COPILOT_CUSTOM_INSTRUCTIONS_DIRS="/path/to/dir1,/path/to/dir2"
  ```

---

## 二、Plugin 系统

```bash
copilot plugin install awesome-copilot@awesome-copilot  # 安装
copilot plugin list                                      # 查看已安装
copilot plugin marketplace browse awesome-copilot       # 浏览市场
```

- awesome-copilot 插件提供额外的 skills 和 agents（如 `suggest-awesome-github-copilot-agents`）
- 插件格式：`<name>@<marketplace>`，当前唯一市场为 `awesome-copilot`

---

## 三、Session 与 Memory

### Session 机制

每个对话窗口对应唯一 UUID，存储在 `~/.copilot/session-state/<uuid>/`：

```
plan.md          # 任务规划（系统 prompt 注入）
memory.md        # 手动维护的会话记忆
checkpoints/     # 检查点存档
files/           # 持久化 artifacts
session.db       # SQLite 状态库
events.jsonl     # 事件日志
```

### Session UUID 生命周期

| 操作 | UUID 变化？ | session-state 文件保留？ |
|---|---|---|
| `/compact`（压缩上下文） | ❌ 不变 | ✅ 保留 |
| `/restart`（重启 CLI） | ❌ 不变 | ✅ 保留 |
| `/resume <id>` | ❌ 恢复指定 session | ✅ 原目录不变 |
| `/clear` / `/new` | ✅ **新建 UUID** | ❌ 旧目录留磁盘，不再注入 |

**结论**：`memory.md` 在 `/compact`、`/restart` 后完全安全；`/clear` 后失效（新 UUID 新目录）。

### 跨 Session 持久化方案

CLI **没有** Chat 那样的平台级 memory 机制：

| 范围 | 方案 |
|---|---|
| 单 session 内 | `session-state/memory.md`（compact 安全） |
| 跨 session（项目级） | 存到 git 仓库，如 `.github/state.md` 或项目 docs |
| 跨对话知识库（实验性） | `/chronicle`，需开启 `--experimental` |

---

## 四、Subagent 系统

### 4.1 内置 Agent 类型

| agent | 默认模型 | 工具集 | MCP | 适用场景 |
|---|---|---|---|---|
| `explore` | Haiku | grep/glob/view/bash | ❌ | 并行只读探索，最轻量 |
| `task` | Haiku | 完整 CLI 工具 | ✅ | 执行命令，只关心成功/失败 |
| `general-purpose` | Sonnet | 完整 CLI 工具 | ✅ | 复杂多步任务，高质量推理 |
| `code-review` | Sonnet | 完整 CLI 工具（只读） | ✅ | 代码审查，**不改代码** |

### 4.2 配额与并发

- **委派 subagent 不消耗主会话额度**（实验确认，`Remaining reqs.` 无变化）
- **并发上限**：实测 5 个 background `explore`（gpt-5.4，含复杂跨目录任务）全部成功，耗时 34-321s 不等（与任务复杂度相关，非并发限制）；实际上限未知，`general-purpose` / `task` 类型并发未系统测试

### 4.3 Background vs Sync

| 模式 | 阻塞主 agent？ | `/tasks` 可见？ | 适用场景 |
|---|---|---|---|
| `background` | ❌ | ✅（按 Enter 查详情，r 移除） | 耗时任务，需并发 |
| `sync` | ✅ | ❌ | 需立即用结果 |

### 4.4 模型选择

**显式指定**（推荐，`/tasks` 显示模型名 + `(override)`）：

```python
# task 工具的 model 参数
model: 'claude-haiku-4.5'    # 快/便宜（内置 explore/task 默认）
model: 'claude-sonnet-4.6'   # 均衡（推荐复杂任务）
model: 'claude-opus-4.5'     # 最强，慢
model: 'gpt-5.4'             # GPT 强推理
model: 'gpt-5.4-mini'        # GPT 快速
model: 'gpt-5.3-codex'       # GPT 代码专用
```

**不指定**：走系统内部默认（UI 不显示），自定义 agent 默认为 Claude Sonnet 4（比主会话旧）。**需要强推理的任务，显式指定模型更可靠。**

### 4.5 嵌套调用（实验确认）

任何 agent（内置或自定义）都可以通过 `task` 工具再派子 agent，无层级限制：

| 调用链 | 结果 |
|---|---|
| 主 agent → 内置 `explore` / `general-purpose` | ✅ |
| 自定义 `IdeaCli` → 内置 `explore` / `general-purpose` | ✅ |
| 自定义 `IdeaCli` → 自定义 `ExploreCli` | ✅ |
| 内置 `general-purpose` → 内置 `explore` | ✅ |

应用场景：MasterCli 作为编排层，按任务性质分发给 ExploreCli（只读调研）和 general-purpose（写入执行）。

### 4.6 权限与工具边界

**两层权限模型**：

```
层 1：CLI 工具权限系统（弹框确认）
    → /allow-all (yolo) 控制
    → background subagent 无法响应弹框，遇权限墙静默失败

层 2：Agent Prompt 约束
    → yolo 无法影响
    → 模型严格遵守 agent 定义里的行为指令（如"只读"）
```

**文件系统**：无沙箱，subagent 以宿主用户权限运行，实际约束来自上述两层。

**自定义 subagent vs 内置 general-purpose 的工具差异**：

| 工具 | 自定义 subagent | 内置 general-purpose |
|---|---|---|
| `create` / `edit` | ❌ 系统层屏蔽 | ✅ |
| `ask_user` | ❌ 系统层屏蔽 | ✅（仅主 agent） |
| `bash` 写文件 | ✅（受 prompt 约束） | ✅ |
| MCP 工具 | ✅（无 tools 限制时） | ✅ |

- **写文件行为由 agent prompt 决定，不由模型决定**（实验确认）
- `ask_user` 屏蔽设计原因：防止多个并发 subagent 同时弹问题

---

## 五、Custom Agent 配置规范

### 5.1 frontmatter 字段速查

```yaml
---
name: MyAgent                     # CLI 里的 agent_type 名称
description: "描述（含冒号必须加引号）"   # ⚠️ 含冒号不加引号会 YAML 解析失败
model: claude-sonnet-4.6          # CLI 格式（可选），vscode 格式会被忽略
# target: vscode                  # ⚠️ 加了 target 在 CLI 里不可见，不写才能在 CLI+Chat 都用
# tools: [bash, view]             # ⚠️ 白名单过滤，不是"添加"；不写 = 继承全部工具（推荐）
---
```

**各字段注意事项**：

| 字段 | 注意 |
|---|---|
| `description` | 含 `:` 必须加引号，否则 YAML 解析失败整个文件 |
| `target` | 不写 = CLI+Chat 通用；写 `vscode`/`github-copilot` 在 CLI 不可见 |
| `model` | CLI 格式：`gpt-5.4-mini`；vscode 格式（`GPT-5.4 mini (copilot)`）被忽略 |
| `tools` | 白名单过滤而非添加；写了反而限制工具集，**推荐不写** |
| `agents: []` | vscode 专用，CLI 会 warning 但不报错 |
| `user-invocable: false` | 阻止用户主动调用该 agent |
| 生效时机 | 修改 frontmatter 后需 `/restart` 才能刷新 agent 列表缓存 |

### 5.2 MCP 工具配置

工具来源（两处合并注入所有 agent）：

| 来源 | 位置 | 内容 |
|---|---|---|
| CLI 内置 | 自带 | `github-mcp-server-*` 系列（20+ 个） |
| 项目级 | `.mcp.json`（项目根目录） | augmentcode、context7、pdf-reader、deepwiki |

`.mcp.json` 示例：
```json
{
  "mcpServers": {
    "augmentcode": { "type": "stdio", "command": "auggie", "args": ["--mcp", "--mcp-auto-workspace"] },
    "deepwiki":    { "type": "http",  "url": "https://mcp.deepwiki.com/mcp" }
  }
}
```

**继承规则**：无 `tools` 限制的自定义 agent 继承全部 MCP；内置 `explore` 被系统层限制，**不继承 MCP**。

### 5.3 完整工具清单（自定义 agent 实验确认，~50 个）

**CLI 内置**：`bash`、`view`、`create`\*、`edit`\*、`grep`、`glob`、`web_fetch`、`web_search`、`sql`、`skill`、`task`、`read_agent`、`list_agents`、`report_intent`、`fetch_copilot_cli_documentation`、`ide-get_selection`、`ide-get_diagnostics`、`ask_user`\*

> \* 自定义 subagent 时，`create`/`edit`/`ask_user` 被系统屏蔽；GPT 模型用 `apply_patch`/`rg` 替代 `edit`/`grep`

**MCP**：`augmentcode-codebase-retrieval`、`context7-*`、`pdf-reader-read_pdf`、`deepwiki-*`、`github-mcp-server-*`

---

## 六、实战对比：explore vs research（触觉传感器调研）

同一 prompt，内置 `explore`（gpt-5.4-mini）vs 自定义 `research`（Sonnet 4 默认）：

| 维度 | explore (gpt-5.4-mini) | research (Sonnet 4) |
|---|---|---|
| 耗时 | **618s** | **272s**（快 2.3x） |
| 准确性 | ✅ 正确找到 demo 脚本 | ❌ 错报"demo 脚本不存在" |
| 代码深度 | 行号 + snippet，找到额外 API | 无行号 |
| 格式 | 详细但略重 | 简洁，有表格 |

**建议**：explore 用系统默认（Haiku），不指定模型；深度调研用 research + 显式指定 Sonnet 4.6。
