# 电机控制指令发送频率分析

## 结论速览

| 参数 | 数值 |
|------|------|
| 物理仿真频率 (`sim.dt`) | **120 Hz**（步长 ≈ 8.33 ms） |
| 抽取因子 (`decimation`) | **4** |
| **电机控制指令发送频率** | **30 Hz**（周期 ≈ 33.33 ms） |

计算公式：

```
控制频率 = 仿真频率 / decimation = 120 Hz / 4 = 30 Hz
控制周期 Δt = sim.dt × decimation = (1/120) × 4 = 1/30 s ≈ 33.33 ms
```

---

## 代码路径与依据

### 1. ManagerBased 工作流 — LeapHand 关节空间环境

**文件**：`source/anymani/anymani/tasks/inhand/config/leaphand/leaphand_env_cfg.py`

```python
def __post_init__(self):
    super().__post_init__()
    self.decimation = 4            # 每 4 个物理步发出一次控制指令
    self.episode_length_s = 30.0
    self.sim.dt = 1.0 / 120.0     # 物理步长：120 Hz
    self.sim.render_interval = self.decimation
```

同文件中，所有派生环境（`LeapHandTactileEnvCfg`、`LeapHandSE3EnvCfg` 等）均继承上述参数，**控制频率统一为 30 Hz**。

---

### 2. ManagerBased 工作流 — LeapHand Round-Tip 环境

**文件**：`source/anymani/anymani/tasks/inhand/config/leaphand_round/inhand_round_base_env_cfg.py`

```python
def __post_init__(self):
    super().__post_init__()
    self.decimation = 4            # 同上，30 Hz 控制
    self.episode_length_s = 30.0
    self.sim.dt = 1.0 / 120.0
    self.sim.render_interval = self.decimation
```

---

### 3. Direct 工作流 — LeapHand 重定向环境

**文件**：`source/anymani/anymani/tasks/direct/leaphand/leap_hand_env_cfg.py`

```python
@configclass
class LeapHandEnvCfg(DirectRLEnvCfg):
    decimation = 4                  # 30 Hz 控制频率
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,                 # 120 Hz 物理仿真
        render_interval=decimation,
        ...
    )
```

---

### 4. 控制周期在 BC 数据录制中的显式计算

**文件**：`source/anymani/anymani/tasks/inhand/mdp/recorders/recorders.py`（约第 458–459 行）

```python
# 计算控制周期 Δt = dt * decimation
dt = self._env.sim.cfg.dt * self._env.cfg.decimation
```

该值作为元数据 `dt` 写入 HDF5 文件，供行为克隆训练时推导等效角速度使用：

```python
V_b = J_b @ (Δθ / Δt)   # 关节增量 / 控制周期 → 末端速度
```

---

### 5. 直接任务中 Δt 的其他用途

**文件**：`source/anymani/anymani/tasks/direct/leaphand/reorientation_env.py`

控制周期 `sim.dt × decimation` 还用于将剧集长度（秒）转换为步数：

```python
int(self.cfg.min_episode_length_s / (self.cfg.sim.dt * self.cfg.decimation))
```

---

## 机制说明：Decimation 与 Zero-Order Hold

Isaac Lab 的 `ManagerBasedRLEnv` / `DirectRLEnv` 遵循 **decimation（抽取）** 机制：

```
策略推理（30 Hz）
  │  输出动作 a_t
  ↓
  ┌─────────────────────────────────────┐
  │  物理步 1  → PD控制器 → 电机指令    │
  │  物理步 2  → PD控制器 → 电机指令    │  ← 同一动作 a_t 持续保持
  │  物理步 3  → PD控制器 → 电机指令    │
  │  物理步 4  → PD控制器 → 电机指令    │
  └─────────────────────────────────────┘
  │  环境返回新的观测
  ↓
策略推理（下一步，30 Hz）
```

- **目标关节位置**在每个 decimation 块开始时由策略设定，并在接下来的 4 个物理步内保持不变（Zero-Order Hold）。  
- **底层 PD 控制器**在每个物理步（120 Hz）都在运行，将关节向目标位置驱动。  
- **对外可观测的"动作频率"**，即策略每秒下发新指令的次数，为 **30 Hz**。

---

## 各环境频率汇总

| 环境 ID | 配置文件 | sim.dt | decimation | 控制频率 |
|---------|---------|--------|------------|----------|
| `AnyMani-LeapHand-Joint-v0` | `leaphand_env_cfg.py` | 1/120 s | 4 | **30 Hz** |
| `AnyMani-LeapHand-SE3-v0` | `leaphand_env_cfg.py` | 1/120 s | 4 | **30 Hz** |
| `AnyMani-LeapHand-Tactile-v0` | `leaphand_env_cfg.py` | 1/120 s | 4 | **30 Hz** |
| `AnyMani-LeapHand-RoundTip-v0` | `inhand_round_base_env_cfg.py` | 1/120 s | 4 | **30 Hz** |
| Direct reorientation | `leap_hand_env_cfg.py` | 1/120 s | 4 | **30 Hz** |

---

## 参考文件

- `source/anymani/anymani/tasks/inhand/config/leaphand/leaphand_env_cfg.py`
- `source/anymani/anymani/tasks/inhand/config/leaphand_round/inhand_round_base_env_cfg.py`
- `source/anymani/anymani/tasks/direct/leaphand/leap_hand_env_cfg.py`
- `source/anymani/anymani/tasks/direct/leaphand/reorientation_env.py`
- `source/anymani/anymani/tasks/inhand/mdp/recorders/recorders.py`
- `source/anymani/ideas/robotics.ipynb`（decimation 机制演示）
