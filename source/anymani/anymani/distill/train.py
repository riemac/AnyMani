r"""Reserved entry for the future main AnyMani training pipeline.

当前 3-URDF 异构手 MLP 可行性验证属于 MVP，不占用主入口；请使用：

```bash
python -m anymani.distill.train_mvp \
  --task AnyMani-GM-Heterogeneous-MLP-Smoke-v0 \
  --num_envs 300 \
  --max_iterations 1 \
  --headless
```

后续正式 teacher specialist / student distillation pipeline 稳定后，再把主训练命令
收敛到本文件。
"""
