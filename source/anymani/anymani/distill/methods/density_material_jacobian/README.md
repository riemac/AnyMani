# Density + Relational Material Jacobian Method

该 method 在统一 PALM/JOINT/TIP encoder 上联合训练 Gaussian density 与 fixed-material anchor-relational Jacobian。它复用 immutable geometry source cache、Sobol state measure 和 resident window；拥有独立的 physical sample、batch/padding、model assembly、objectives、FairGrad backward、evaluation 与 retained artifact。

## 信息流

```text
GeometrySource + q
  -> one POE/current-screw pass
  -> spatial queries -> density-only Warp teacher
  -> fixed owner-local material identities -> Gamma teacher
  -> StaticGeometryEvidence + q -> unified Z
  -> density reader / material-Jacobian reader
```

Gamma 的四通道顺序固定为 `height, radius, dot, chirality`。Radius 在面内距离退化时使用独立 mask；其余通道仍保留。非祖先 edge 的 point Jacobian 与所有一阶 targets 精确为零。

## Retained 边界

训练完成后只发布 `encoder.` namespace。Density reader、Gamma reader、queries、Warp cache、material identities、teacher targets、ancestor masks 和 objectives 都不进入 PPO artifact。

## 当前实验入口

```bash
/home/hac/isaac/env_isaaclab/bin/python -m anymani.distill.ssl.pretrain \
  --config geometry_ssl_density_material_jacobian_v0_8_0 \
  --device cuda:0
```

正式 snapshot 使用 8 个 anchor realizations、4/16/64 mm canonical density bands、每 joint `2 active + 1 structural-zero` Gamma edges、entity permutation、20% joint-sign rewrite 和两任务 FairGrad。
