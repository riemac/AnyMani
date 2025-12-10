## 工程问题记录

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
  - asset.root_physx_view.get_jacobians() 返回的是几何雅可比，参考点在{b}(末端)，参考系在{w}(World坐标系)

* **IsaacSim 模块导入:**
  某些IsaacSim模块（如 `isaacgym`, `omni.isaac` 等）只能在Applauncher启动IsaacSim环境后导入使用，否则会报找不到错误，这是正常的。