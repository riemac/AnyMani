# Delegate Context

## 背景

AnyMani 项目的 inhand 任务模块经历了多轮迭代，积累了多种失败/废弃的环境变体。用户要求进行"代码出清"，彻底删除不再使用的环境及其关联的所有文件。

项目根目录：`/home/hac/isaac/AnyMani/source/anymani/anymani/tasks/inhand/`

## 本次委派目标

### 要删除的环境及关联文件

**1. 环境注册（`inhand/config/leaphand/__init__.py`）**
删除以下环境的 `gym.register` 代码和对应的 import：
- `AnyMani-LeapHand-SE3-v0` / `AnyMani-LeapHand-SE3-Play-v0`
- `AnyMani-LeapHand-SE3-Tactile-v0` / `AnyMani-LeapHand-SE3-Tactile-Play-v0`
- `AnyMani-LeapHand-Affine-v0` / `AnyMani-LeapHand-Affine-Play-v0`
- `Template-Leaphand-Rot-Manager-v0`（已废弃模板）

**2. 环境配置类（`inhand/config/leaphand/leaphand_env_cfg.py`）**
删除以下配置类（保留 Joint/Tactile 相关的）：
- `LeapHandSe3EnvCfg` / `LeapHandSe3EnvCfg_PLAY`
- `LeapHandSe3TactileEnvCfg` / `LeapHandSe3TactileEnvCfg_PLAY`
- `LeapHandAffineEnvCfg` / `LeapHandAffineEnvCfg_PLAY`

**3. Agent configs（`inhand/config/leaphand/agents/`）**
删除文件：
- `rl_games_ppo_cfg_se3.yaml`
- `rl_games_ppo_cfg_se3_tactile.yaml`
- `skrl_ppo_cfg.yaml`
- `skrl_amp_cfg.yaml`
- `rsl_rl_ppo_cfg.py`
更新 `agents/__init__.py` 移除对已删除文件的引用

**4. MDP 动作模块（`inhand/mdp/actions/`）**
删除文件：
- `se3_actions.py` + `se3_actions_cfg.py`
- `affine_formation.py` + `affine_formation_cfg.py`
- `floating_base_kinematic.py` + `floating_base_kinematic_cfg.py`
更新 `actions/__init__.py` 移除对已删除文件的 import

**5. MDP 组件库（`inhand/inhand_env_cfg.py`）**
删除 SE3/Affine 相关的配置组件（如 `Se3ActionsCfg`、`AffineActionsCfg`、`Se3ObservationsCfg`、`Se3RewardsCfg` 等），保留 Joint/Tactile 相关的

**6. 其他 MDP 文件**
检查 `observations.py`、`rewards_action.py`、`rewards_task.py` 等文件中是否有仅被 SE3/Affine 使用的函数/类，如有则删除

### 要保留的环境

- `AnyMani-LeapHand-Joint-v0` (+Play)
- `AnyMani-LeapHand-Tactile-v0` (+Play)
- `AnyMani-LeapHand-RoundTip-v0`（`leaphand_round/` 整个目录保留）
- `Template-Leaphand-Direct-v0`、`Template-Leaphand-ContinuousRot-Direct-v0`（`direct/` 整个目录保留）
- `leaphand_stable_env_cfg.py` 保留
- `functional/` 目录保留
- rl_games agent configs：`rl_games_ppo_cfg.yaml`、`rl_games_ppo_cfg_tactile.yaml` 保留

### 注意事项

- 删除文件前先确认没有被保留环境引用
- 更新所有 `__init__.py` 确保 import 正确
- 对于 `inhand_env_cfg.py` 中的共享组件（如被多个环境使用的），只删除仅被废弃环境使用的部分
- 删除后运行 `python -c "import anymani.tasks"` 验证不会报 ImportError

## 参考资料

- 主配置文件：`inhand/config/leaphand/leaphand_env_cfg.py`
- MDP 组件库：`inhand/inhand_env_cfg.py`
- 环境注册：`inhand/config/leaphand/__init__.py`
- Agent configs：`inhand/config/leaphand/agents/`

## 历史记录

| 轮次 | 说明 |
| --- | --- |
| 1 | 主代理盘点了所有环境，用户确认保留 Joint/Tactile/RoundTip/Direct，删除 SE3/Affine 及关联文件，彻底清理（含 MDP 组件、agent configs、floating_base_kinematic、skrl/rsl_rl configs）。 |
