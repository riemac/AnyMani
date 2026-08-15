r"""Training-time decoder package。

decoder 将 retained backbone latent 映射到 SSL/IL 所需的重建目标。representation decoder
默认在预训练结束后丢弃，不进入 PPO forward；是否保留某个 decoder 只能由下游任务证据
裁定，不能因为 checkpoint 中存在参数就自动部署。
"""
