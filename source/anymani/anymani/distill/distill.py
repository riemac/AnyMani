r"""AnyMani distill package façade.

`distill` 是训练管线与网络架构的所有权边界：RL teacher、IL / distillation、
模型 adapter、训练入口和 smoke 测试都应从这里进入，而不是散落到项目根部脚本。

当前可用入口：

- `python -m anymani.distill.train`：GM teacher rl_games PPO 训练；
- `python -m anymani.distill.smoke`：GM teacher 环境随机动作 smoke。
"""
