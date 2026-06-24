r"""AnyMani distill package façade.

`distill` 是训练管线与网络架构的所有权边界：RL teacher、IL / distillation、
模型 adapter、训练入口和 smoke 测试都应从这里进入，而不是散落到项目根部脚本。

当前 RL 入口：

- `python -m anymani.distill.train --task AnyMani-GM-SingleAsset-MLP-v0`：
  single-asset MLP PPO 训练入口。

运行时 reset/step / PhysX 语义验证不再放在 distill 临时脚本中，而是放到
`source/anymani/anymani/smokes/` 下通过显式 pytest smoke 执行。
"""
