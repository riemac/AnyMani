# 任务：快速运动适应模块实现
参照 `In-Hand Object Rotation via Rapid Motor Adaptation`论文，其项目代码库为 `hora`(基于Isaac Gym)。
主要目的是实现物体的泛化，适配到AnyRotate项目中(主要环境配置文件为 `inhand_base_env_cfg.py`)，可以考虑新增一环境配置文件专门适用本任务idea

## 具体需求
具体需求：
1. 基于Isaac Lab框架，为LeapHand灵巧手实现RMA训练管道
2. 参考hora项目（基于Isaac Gym）的实现方式，但适配到Isaac Lab的架构
3. 实现物体泛化能力，使机械手能够适应不同形状、重量、摩擦系数的物体
4. 按照项目规范，需要先进行充分的技术调研，然后制定详细的实现规划
5. 实现应包括：
   - RMA网络架构（特权信息编码器、适应模块等）
   - 训练管道（teacher-student框架）
   - 环境配置和奖励函数设计
   - 评估和测试脚本

## 关键疑惑
1. IsaacLab中的MangaerBasedRLEnv架构，其观察是直接发送给Base Policy的。在训练时，前向传播和反向传导似乎是rl_games内部处理的。这意味着，如果想要实现前置的特权信息编码器，似乎不能靠observation mdp，因为它会被视作Base Policy的输入。这面临一个工程组织架构上的挑战。怎么才能把物体特权信息/本体感受与动作历史序列输入给物体属性编码器/RMA模块，而不经由Base Policy呢。可参考rl_games源码和dextrah项目里有没有类似实现
   > 目前可能方案：不是去改 observation mdp “绕开”policy输入，而是把特权信息做成独立 group，然后在 自定义 network forward 里选择是否使用它（stage1 用，stage2/部署禁用）。这样同一套 env/runner 兼容三阶段
2. DirectRLRnv似乎更灵活，ManagerBasedRLEnv可以做到吗？需要修改有关环境包装的源码吗？
   > 发现：ManagerBasedRLEnv 可以做 RMA：它天然输出“按 group 命名”的观测 dic；rl_games 侧能拿到 {"obs": {"policy":..., "priv_info":..., "proprio_hist":...}, "states": {...}};再配合 IsaacLab 的 rl_games.py:52（支持 obs_groups + concate_obs_group=False），rl_games 侧能拿到 {"obs": {"policy":..., "priv_info":..., "proprio_hist":...}, "states": {...}} 这种多路输入。自定义网络里就能把 μ/φ 放在 base policy 前面，而不是硬塞进同一个扁平 observation。
3. 训练时，为实现物体的泛化，是分别训练不同的物体旋转策略再蒸馏成单一策略，还是对物体类型进行域随机化？hora里是怎么做的？
   > 目前调研：hora 是单一策略 + 域随机化 + 多物体资产采样。那问题来了， AnyRotate-IsaacLab 如何实现？
4. 物体的数据集来源于什么？Hora的操作物体可以迁移到AnyRotate这里来吗
5. 训练是物体编码器和Base Policy联合优化吗（论文里是这样的）？RMA的监督训练是离线还是在线？
   > 构思方案：stage1 用 rl_games PPO 联合训练 μ+π（critic 继续用 states）；
   > stage2 用单独脚本做监督（冻结 π/μ，只训 φ），更接近 hora 的实现，也更省改 rl_games 算法栈(这里考虑是离线还是在线)

## Training Pipeline
/home/hac/isaac/AnyRotate/source/leaphand/leaphand/task_md/9a2e5de4-e747-4316-8c3a-7aaeef1b76a7.png
![alt text](9a2e5de4-e747-4316-8c3a-7aaeef1b76a7.png)