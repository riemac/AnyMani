r"""Critic-first连续证据、干预顺序与开销停止门合同。"""

from __future__ import annotations

from anymani.distill.diagnostics.evaluation.rl.critic_health import (
    CriticCheckpointEvidence,
    recommend_critic_intervention,
)


def _evidence(
    update: int,
    *,
    span: float = 20.0,
    return_span: float = 20.0,
    advantage_span: float = 20.0,
    normalizer_pure: bool = True,
) -> CriticCheckpointEvidence:
    r"""构造可独立控制Actor/Critic尺度、normalizer purity与完整梯度可靠性的四资产证据。"""

    return CriticCheckpointEvidence(
        update=update,
        return_target_std=(1.0, 1.0, 1.0, return_span),
        advantage_std=(1.0, 1.0, 1.0, advantage_span),
        explained_variance=(-0.5, -0.1, 0.3, 0.5),
        critic_gradient_norm=(1.0, 2.0, 4.0, span),
        critic_top1_norm_fraction=0.55,
        actor_negative_pair_fraction=0.30,
        gradient_probe_seconds=1.0,
        epoch_total_seconds=10.0,
        value_normalizer_pure=normalizer_pure,
        actor_full_gradient_reliable=True,
        critic_full_gradient_reliable=True,
    )


def test_normalizer_purity_precedes_every_scale_or_gradient_intervention() -> None:
    r"""训练统计被诊断读取污染时必须先修pure estimator，不得越级组合梯度。"""

    decision = recommend_critic_intervention((_evidence(500, normalizer_pure=False),))
    assert decision.action == "fix_value_normalizer_purity"


def test_advantage_normalization_requires_two_consecutive_actor_scale_checkpoints() -> None:
    r"""逐资产Actor尺度持续失配才建议归一化，不再要求Critic explained variance为负。"""

    first = _evidence(500)
    assert recommend_critic_intervention((first,)).action == "none"
    decision = recommend_critic_intervention((first, _evidence(1000)))
    assert decision.action == "per_asset_advantage_normalization"


def test_actor_pcgrad_requires_per_asset_normalization_healthy_critic_and_reliable_full_gradients() -> None:
    r"""末层负余弦不能触发PCGrad；可靠完整Actor梯度与健康Critic是共同前提。"""

    checkpoints = (_evidence(500), _evidence(1000))
    pcgrad = recommend_critic_intervention(
        checkpoints,
        applied_interventions=("per_asset_advantage_normalization",),
        critic_healthy=True,
    )
    assert pcgrad.action == "pcgrad_actor"
    unreliable = CriticCheckpointEvidence(**{**_evidence(1000).__dict__, "actor_full_gradient_reliable": False})
    assert (
        recommend_critic_intervention(
            (_evidence(500), unreliable),
            applied_interventions=("per_asset_advantage_normalization",),
            critic_healthy=True,
        ).action
        == "none"
    )


def test_unhealthy_critic_uses_popart_then_reliable_fairgrad() -> None:
    r"""Actor尺度校准后，只有持续Critic失配才按PopArt→FairGrad-c推进。"""

    checkpoints = (_evidence(500), _evidence(1000))
    popart = recommend_critic_intervention(checkpoints, applied_interventions=("per_asset_advantage_normalization",))
    assert popart.action == "popart"
    fairgrad = recommend_critic_intervention(
        checkpoints,
        applied_interventions=("per_asset_advantage_normalization", "popart"),
    )
    assert fairgrad.action == "fairgrad_critic"


def test_gradient_solver_overhead_above_twenty_percent_stops_branch() -> None:
    r"""常驻solver超过steady epoch的20%时，开销门优先于继续叠加算法。"""

    expensive = CriticCheckpointEvidence(
        **{
            **_evidence(1500).__dict__,
            "gradient_probe_seconds": 2.01,
            "epoch_total_seconds": 10.0,
        }
    )
    decision = recommend_critic_intervention(
        (expensive,),
        applied_interventions=("fairgrad_critic",),
        critic_healthy=True,
    )
    assert decision.action == "stop_gradient_solver"
