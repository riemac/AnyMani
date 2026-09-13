r"""FK/self-modeling 的两级论文基线 target contract。

必须同时保留：

1. URDF frame-origin FK：尽量复现 GET-Zero 类 per-joint/frame-origin target。它便宜、
   易监督，但对 arbitrary link/joint frame rewrite 敏感；
2. Physical-landmark FK：预测经语义确认的 collision centroid、surface landmark 或 distal
   physical points。它仍是低维 point target，但减少“只修复任意 frame origin 就能获益”
   这一替代解释。

两种基线都应与 field candidates 使用相同 asset/q split、shared-backbone budget 与 PPO
fine-tune protocol。若 field 只击败 frame-origin FK 而不能击败 physical-landmark FK，
论文结论应收缩为 target 定义改进，而不能宣称 dense physical field 必然更优。
"""
