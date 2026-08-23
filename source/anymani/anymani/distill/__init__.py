r"""AnyMani 跨手型训练管线、物理表征与共享模型 package。

``distill`` 是 stage-independent component 与 stage orchestration 的所有权边界：

```text
representations/  # physical source / field / query / target，不含神经网络
models/           # SSL、RL、IL 共用的 adapter / backbone / decoder / heads
objectives/       # representation 与未来 RL 的数学 loss
methods/          # task-free 科学方法聚合根
ssl/              # self-supervised pretraining stage
rl/               # rl_games PPO training / playback / adapters
il/               # imitation learning / teacher-student distillation stage
registries/       # 未来独立 component registry
```

当前 tracked RL 入口为：

- ``python -m anymani.distill.train --task <gym-id>``；
- ``python -m anymani.distill.play --task <gym-id> --checkpoint <ckpt>``。

geometry SSL 的可运行入口为 ``python -m anymani.distill.ssl.pretrain``，普通 CLI flags 描述一次运行，schema 5 Python preset 提供方法与默认配置。IL 尚未建立 trainer。geometry SSL 保持 task-free PyTorch/Warp 依赖边界，RL wrapper 与 Isaac Sim runtime 由对应阶段入口加载。

运行时 reset/step/PhysX 语义验证位于 ``source/anymani/anymani/smokes/``；distill 不复制
tasks-owned env cfg，也不 import、解析或要求 Research vault 存在。
"""
