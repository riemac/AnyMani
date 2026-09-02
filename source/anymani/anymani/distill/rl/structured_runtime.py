r"""冻结N040 provider与structured actor/critic的共享-Z runtime package。

每个policy batch只执行一次geometry provider；actor/critic接收同一``GeometryTokenBatch``。Provider参数冻结且不
进入optimizer；actor/critic trainable namespaces与parameter sets完全分离。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from anymani.distill.models.heterogeneous_policy import StructuredActorCriticPackage
from anymani.distill.models.structured_heterogeneous import (
    GeometryTokenBatch,
    StructuredActorObservation,
    StructuredActorOutput,
    StructuredCriticObservation,
    StructuredCriticOutput,
)
from anymani.distill.rl.runtime.retained_geometry import RetainedGeometryBatch, RetainedGeometryProvider


@dataclass(frozen=True)
class StructuredGeometryContext:
    r"""Actor/critic共享的精简tokens与完整retained graph batch。"""

    tokens: GeometryTokenBatch
    retained: RetainedGeometryBatch


class StructuredHeterogeneousRuntime(nn.Module):
    r"""一次N040 resolve、独立actor/critic forward与checkpoint identity owner。"""

    def __init__(
        self,
        geometry_provider: RetainedGeometryProvider,
        policy: StructuredActorCriticPackage,
    ) -> None:
        r"""绑定冻结provider与trainable package并验证参数边界。"""

        super().__init__()
        self.geometry_provider = geometry_provider
        self.policy = policy
        self.geometry_provider.requires_grad_(False)
        self.geometry_provider.eval()
        actor_ids, critic_ids = self.policy.trainable_parameter_sets()
        if not actor_ids.isdisjoint(critic_ids):
            raise ValueError("structured actor and critic parameters must be disjoint")

    @property
    def anymani_identity(self) -> dict:
        r"""返回checkpoint restore前必须exact匹配的provider identity。"""

        return self.geometry_provider.identity

    def train(self, mode: bool = True) -> StructuredHeterogeneousRuntime:
        r"""切换actor/critic training state，同时强制N040保持eval。"""

        super().train(mode)
        self.geometry_provider.eval()
        return self

    def resolve_geometry(
        self,
        prototype_index: torch.Tensor,
        actor_observation: StructuredActorObservation,
    ) -> StructuredGeometryContext:
        r"""从opaque local rows与current q计算一次共享$Z^e$。"""

        # Builder链会导入robots/Isaac；延迟到AppLauncher后的真实runtime调用，保持纯PPO tests可收集。
        from anymani.distill.rl.runtime.structured_geometry import resolve_structured_geometry

        tokens, retained = resolve_structured_geometry(
            self.geometry_provider, prototype_index, actor_observation
        )
        return StructuredGeometryContext(tokens=tokens, retained=retained)

    def actor_forward(
        self,
        actor_observation: StructuredActorObservation,
        geometry: StructuredGeometryContext,
    ) -> StructuredActorOutput:
        r"""运行deployable actor；不读取critic observation。"""

        return self.policy.actor(actor_observation, geometry.tokens)

    def critic_forward(
        self,
        critic_observation: StructuredCriticObservation,
        geometry: StructuredGeometryContext,
    ) -> StructuredCriticOutput:
        r"""运行independent privileged critic。"""

        return self.policy.critic(critic_observation, geometry.tokens)

    def actor_parameters(self):
        r"""返回actor optimizer parameter iterator。"""

        return self.policy.actor.parameters()

    def critic_parameters(self):
        r"""返回critic optimizer parameter iterator。"""

        return self.policy.critic.parameters()


__all__ = ["StructuredGeometryContext", "StructuredHeterogeneousRuntime"]
