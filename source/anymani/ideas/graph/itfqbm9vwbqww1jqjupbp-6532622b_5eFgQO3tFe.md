
# 研究背景说明文档
## 用途：作为后续实施讨论的上下文背景，供 AI Agent 参考

---

## 一、研究目标

构建一个用于**手内操作（In-hand Manipulation）**的机器人学习框架，核心目标是：
- 通过**程序化合成多样化机械手 URDF 资产**（简单几何体组合），训练一个泛化能力强的策略网络
- 训练完成后，能够**零样本（Zero-shot）或少样本迁移**到真实机械手（如 Allegro Hand、LEAP Hand）

---

## 二、核心网络架构设计

### 2.1 总体思路
- 以 **GET-Zero** 和 **TRO-Grasp** 为主要参考
- 核心机制：**图注意力网络（Graph Attention Network / Graph Transformer）**
- 坐标系选择：**Joint-centric**（以关节为中心），在关节动作空间（Joint Action Space）上进行控制
- 控制频率硬性指标：**> 20Hz**（手内操作稳定性要求）

### 2.2 图结构定义

**节点（Node）**：每个节点对应一个 Joint，同时融合其 Child Link 的几何特征

节点特征向量：
```
v_i = Concat(q_i, q_dot_i, axis_i, q_min_i, q_max_i, g_i_BPS)
```
- `q_i`：当前关节角度（动态）
- `q_dot_i`：当前关节角速度（动态）
- `axis_i`：关节旋转轴方向（静态）
- `q_min_i, q_max_i`：关节限位（静态）
- `g_i_SDF`：Child Link 的 SDF 几何特征（静态，离线预计算，推理时查表）

**边（Edge）**：所有节点对之间的边完全**同构（Homogeneous）**

边特征：
```
e_ij = SE(3)^rest_{i→j}
```
- 从 URDF 运动学链中提取的静态相对位姿（rest pose 下）
- 所有边（相邻/非相邻）特征维度和语义完全一致
- 推理时为常数，零计算开销

### 2.3 架构设计的关键决策理由

| 决策 | 理由 |
|------|------|
| Joint-centric（非 Link-centric） | 关节动作空间是手内操作最稳妥的控制空间 |
| Child Link 几何特征绑定到节点 | Joint 原点与 Child Link 原点在 URDF 中几乎重合，语义一致 |
| 边完全同构，只用 SE(3) | 异构边破坏 Attention 可比性，几何信息属于节点而非边 |
| 静态 SE(3)^rest 而非动态 FK | 让网络隐式学习 FK，减少对边特征的过度依赖；FK 计算虽快但设计更干净 |
| 几何特征离线预计算 | 推理时查表 O(1)，满足 >20Hz 频率约束 |

---

## 三、Link 几何特征提取方案

### 3.1 核心方案：球形 SDF 采样

**方法**：
1. 以 Joint i 的坐标原点为球心，构建固定半径 R 的球（建议 R = 5cm 或该 Link 包围球的 1.5 倍）
2. 在球内用 **Fibonacci 螺旋**生成 N 个固定采样点（所有 Link 共享同一套采样点坐标）
3. 对每个采样点，计算到该 Link 所有 Collision Geometry 的 **SDF 并集**（取 smooth minimum 或 hard minimum）
4. 输出 N 维 SDF 特征向量（连续值，负=内部，正=外部）

**关键优势**：
- **统一表征**：无论 Collision 是 1 个 Box、6 个 Box 组合、还是精细 STL，输出维度完全一致
- **无需预训练模块**：SDF 采样是确定性几何计算，直接得到特征向量
- **训练-测试特征空间统一**：训练用简化几何体的 SDF，测试用真实 Mesh 的 SDF，同一管线
- **连续可微**：SDF 值连续，对网络输入友好

**参数建议**：
- 采样点数 N = 64（足够描述手指连杆形状，特征维度不过大）
- 所有 Link 共享固定 Basis Points（Fibonacci 球面分层）
- SDF 值建议做归一化（除以 R）

### 3.2 已验证（代码实验）

用 Allegro Hand 和 LEAP Hand 的真实 URDF 片段验证：
- Allegro Finger（1 个 Box）→ 64 维 SDF 向量 ✅
- Allegro Palm（3 个 Box 组合）→ 64 维 SDF 向量 ✅
- LEAP MCP Joint（6 个 Box 组合）→ 64 维 SDF 向量 ✅
- 三者 SDF pattern 明显不同，具有区分度 ✅
- 计算极快，可离线批量处理 ✅

---

## 四、资产合成方案（当前讨论焦点）

### 4.1 总体策略
- **训练阶段**：程序化生成大量**简化几何体组合**的机械手 URDF（Box + Cylinder + Capsule 近似）
- **测试/部署阶段**：直接使用真实机械手的精细 STL Mesh URDF
- **泛化路径**：简单几何体的多样化排列组合 → 学到对手部形态的泛化能力 → 零样本迁移真实手

### 4.2 合成 URDF 的硬性约束（由特征提取管线反推）

1. **每个 Link 的所有 Collision Geometry 的 `<origin>` 必须在同一局部坐标系下正确定义**
   - SDF 采样时需要把所有 Box 统一到 Link 局部坐标系下做并集查询
   
2. **Joint 坐标原点必须合理包围 Child Link 的 Collision Mesh**
   - SDF 采样球以 Joint 原点为球心，Child Link 的几何体应在球内

3. **关节旋转轴必须与 Child Link Collision Mesh 的实际旋转中心对齐**
   - 这是 Joint-centric 表征的基础前提

4. **Collision Geometry 的数量、类型不需要统一**
   - SDF 方案天然处理任意数量和类型的 Primitive 组合

### 4.3 合成变量空间（多样化来源）
- 手指数量（3~5 根）
- 每根手指的关节数量（2~4 个）
- 各连杆的长度、宽度、厚度（在合理解剖学范围内随机）
- 手掌尺寸和形状
- 手指在手掌上的排列方式（间距、角度）
- 拇指的对掌角度和位置

---

## 五、参考工作

| 工作 | 核心贡献 | 本项目借鉴点 |
|------|---------|------------|
| **GET-Zero** | Graph Transformer + 零样本手部迁移 | Graph Transformer 架构，跨手泛化思路 |
| **TRO-Grasp** | 图神经网络 + Link-centric 表征 | 图结构设计，Link 节点思路 |
| **BPS（Basis Point Set）** | 固定点集的点云编码 | 固定 Basis Points 的采样思路 |
| **Occupancy Networks** | 隐式几何表征 | SDF/Occupancy 统一表征的理论基础 |

---

## 六、待实施的下一步（VSCode Copilot 继续讨论）

1. **URDF 资产合成脚本**
   - 程序化生成参数化机械手 URDF
   - 保证 Joint/Link 坐标系对齐约束
   - 输出包含 Collision Geometry（Box/Cylinder 组合）的合法 URDF

2. **SDF 特征预处理管线**
   - 读取任意 URDF（合成的或真实的）
   - 对每个 Link 执行球形 SDF 采样
   - 输出每个 Link 的 64 维 SDF 特征向量，保存为查找表

3. **图结构构建模块**
   - 从 URDF 运动学链提取 SE(3)^rest 边特征
   - 拼接节点特征（关节状态 + SDF 查表）
   - 输出标准 PyTorch Geometric 格式的图数据

4. **图注意力网络实现**
   - 参考 GET-Zero 的 Graph Transformer
   - 同构边设计（SE(3) 边特征作为 Attention Bias）
   - 满足 >20Hz 推理频率约束
