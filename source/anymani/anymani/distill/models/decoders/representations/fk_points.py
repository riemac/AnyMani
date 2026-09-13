r"""Frame-origin 与 physical-landmark FK baseline decoder contract。

decoder 从有效 JOINT/TIP owner latent 预测明确命名的 3D points；输出逻辑形状保持
``[B, G, P, 3]``，其中 $B$ 是 batch size，$G$ 是有效 semantic group/token 数，$P$ 是
每组 landmark 数，最后一维为 hand-frame 米制坐标。

必须独立支持：

- prior-art reproduction：预测 authored URDF joint/tip frame origin，保留其 local-frame
  gauge sensitivity；
- stronger physical baseline：预测 collision centroid、surface landmarks 或经 sidecar 审核
  的 distal physical points。

两条路线不能混用 target 名称或误报为同一 FK objective。decoder 与 field/Gaussian decoder
一样默认 pretraining-only；PPO 继承 shared adapter + backbone，而不是永久消费预测 points。
具体 shared/per-type MLP、landmark 数与 pose/orientation 扩展尚未裁定。
"""
