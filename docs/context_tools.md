# 外部信息检索工具对比分析

本文档记录了对外部信息检索工具（DeepWiki、Context7、GitHub Repo）的对比测试和分析结果。

## 工具概览

| 工具 | MCP 工具 | 功能描述 |
|------|---------|---------|
| **DeepWiki** | `mcp_cognitionai_d_*` | GitHub 仓库智能问答和文档生成 |
| **Context7** | `mcp_io_github_ups_*` | 库文档检索和代码示例获取 |
| **GitHub Repo** | `github_repo` | 源代码片段搜索 |

## 测试总结（7 轮）

| 测试 | 问题类型 | DeepWiki | Context7 | GitHub Repo |
|------|---------|----------|----------|-------------|
| PhysX Solver | 大型 C++ 仓库概念 | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| rl_games 网络 | 中型 Python 仓库教程 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| transformers ViT | 超大型热门库 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| IsaacLab ViT 集成 | 跨领域综合问题 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 可微物理对比 | 跨框架对比 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| GelSight 触觉 | 专业功能查询 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| GNN 集成 IsaacLab | 跨库整合 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

### 核心发现

- **DeepWiki**: 7/7 满分 - 所有场景稳定优秀
- **Context7**: 仅热门库（transformers, MuJoCo）表现好，中小型库效果差
- **GitHub Repo**: 精准源码定位好，但 token 消耗高、信息密度低

## Token 效率

| 指标 | DeepWiki | Context7 | GitHub Repo |
|------|----------|----------|-------------|
| 平均 Token | ~1,525 | ~1,250 | ~6,125 |
| 相对消耗 | **1x** | 0.82x | **4x** |
| 信息密度 | 极高 | 高 | 中 |

**结论**: GitHub Repo 消耗约为 DeepWiki 的 4 倍，但信息密度更低。

## 工具选择指南

```
┌────────────────────────────────────────────────────────────────┐
│                      问题类型决策树                              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  需要概念解释/教程/跨领域整合？                                   │
│      └── YES → DeepWiki (首选)                                 │
│                                                                │
│  查询主流热门库 (transformers, PyTorch, MuJoCo)?                │
│      └── YES → Context7 (省 token)                             │
│      └── NO  → DeepWiki                                        │
│                                                                │
│  需要精确源码行号定位?                                           │
│      └── YES → GitHub Repo                                     │
│                                                                │
│  查询不存在的功能/实验性功能?                                     │
│      └── DeepWiki (能识别"功能不存在"并给替代方案)               │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 适用场景

| 场景 | 推荐工具 | 原因 |
|------|---------|------|
| "如何做 X" 教程类 | DeepWiki | 结构化步骤指导 |
| 跨框架对比 | DeepWiki | 能整合多仓库信息 |
| 功能可用性查询 | DeepWiki | 能识别"功能不存在" |
| 热门库 API | Context7 | 省 token + 高质量 |
| 源码实现参考 | GitHub Repo | 精确行号定位 |
| 小众/专业库 | DeepWiki | Context7 效果差 |

### 不推荐场景

| 工具 | 不适用场景 |
|------|-----------|
| DeepWiki | 需要精确行号定位 |
| Context7 | 小众库、跨领域问题、深度技术问题 |
| GitHub Repo | 概念理解、"为什么"类问题 |

## 推荐工作流

```
1. 概念探索 → DeepWiki.ask_question
2. 实现参考 → GitHub Repo → read_file 读取完整文件
3. API 查阅 → Context7 (热门库) 或 DeepWiki (小众库)
```

## 测试详情

<details>
<summary>点击展开测试详情</summary>

### 测试 1: PhysX TGS vs PGS Solver
- **仓库**: `NVIDIA-Omniverse/PhysX`
- **DeepWiki**: 完整概念解释，含数学公式、源码引用
- **GitHub Repo**: 50+ 代码片段，精确行号
- **Context7**: 仅 CHANGELOG，无法回答概念问题

### 测试 2: rl_games 自定义网络
- **仓库**: `Denys88/rl_games`
- **DeepWiki**: 完整 5 步教程
- **GitHub Repo**: 覆盖 `network_builder.py` 核心文件
- **Context7**: 缺乏完整实现示例

### 测试 3: transformers ViT
- **仓库**: `huggingface/transformers`
- **DeepWiki**: pipeline + AutoModel 方法
- **Context7**: 10 个高质量代码示例 (热门库表现好)
- **GitHub Repo**: 多为源码定义而非使用示例

### 测试 4: IsaacLab ViT 集成
- **仓库**: `isaac-sim/IsaacLab`
- **DeepWiki**: 完整 4 步教程
- **GitHub Repo**: 精准定位 `observations.py` L553-594 (Theia Transformer)
- **Context7**: 仅表面使用命令

### 测试 5: 可微物理对比
- **仓库**: `isaac-sim/IsaacLab` + `google-deepmind/mujoco`
- **技术结论**: 
  - IsaacLab: Newton Physics (实验性)
  - MuJoCo: MJX (正式发布)

### 测试 6: GelSight 触觉
- **仓库**: `isaac-sim/IsaacLab` + `isaac-sim/IsaacSim`
- **结论**: 两者都无内置 GelSight，需 ContactSensor + Camera 自定义

### 测试 7: rl_games GNN 集成 IsaacLab
- **关键代码模板**:
```python
# 1. 继承 BaseNetwork 定义 GNN
class GNNNetwork(NetworkBuilder.BaseNetwork): ...

# 2. 创建 NetworkBuilder
class GNNBuilder(NetworkBuilder): ...

# 3. 在 train.py 中注册
model_builder.register_network('gnn_network', GNNBuilder)
```
- **源码位置**: `rl_games/envs/test_network.py` (完整示例)

</details>

---

*最后更新: 2025-01*
