r"""RL agent configuration package for `anymani.distill.rl`.

IsaacLab 的 registry 通过 `rl_games_cfg_entry_point` 从本 package 读取 YAML。
把 YAML 放在 `distill/rl/agents` 下，是为了让 AnyMani RL 训练管线在 distill 内自包含，
不再依赖 `tasks/<hand>/config/.../agents`、临时 MVP 脚本或项目根脚本目录。
"""
