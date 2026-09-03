r"""Strict v5 Sobol、joint comfort、包络、硬门与低秩CEM纯Torch合同。"""

from __future__ import annotations

import torch

from anymani.pregrasp.mvp80_strict_search import (
    CEM_PROPOSAL_CENTER_COUNTS,
    fixed_position_envelope,
    initial_envelope,
    initial_joint_candidates,
    low_rank_cem_candidates,
    sobol_bank,
    strict_pass_mask,
)
from anymani.pregrasp.strict_gate import MVP80_STRICT_GOOD_PREGRASP_GATE
from anymani.tasks.hetero.config.generated.good_pregrasp_identity import (
    GOOD_PREGRASP_CATALOG_ROOT,
    GOOD_PREGRASP_GENERATION_DIGEST,
    GOOD_PREGRASP_REQUIRE_STRICT,
)
from anymani.tasks.hetero.config.generated.good_pregrasp_identity_v4 import (
    GOOD_PREGRASP_GENERATION_DIGEST as V4_GENERATION_DIGEST,
)
from anymani.tasks.hetero.config.generated.strict_good_pregrasp_identity import (
    STRICT_GOOD_PREGRASP_GENERATION_IDENTITY,
)


def test_strict_v5_identity_locks_sobol_top32_cem_and_gate_digest() -> None:
    r"""Generation key必须覆盖完整搜索预算与唯一strict predicate。"""

    identity = STRICT_GOOD_PREGRASP_GENERATION_IDENTITY
    assert identity["initial_proposal"]["type"] == "scrambled_sobol"
    assert identity["initial_proposal"]["candidates_per_asset"] == 256
    assert identity["physics_screen"]["geometry_top_k_per_asset"] == 32
    assert identity["refinement"]["rounds_max"] == 3
    assert identity["refinement"]["proposals_per_round_per_failed_asset"] == 128
    assert identity["refinement"]["full_physics_candidates_per_round"] == 128
    assert identity["hard_gate"]["angular_velocity_kind"] == "total_l2"
    assert identity["hard_gate"]["gate_digest"] == MVP80_STRICT_GOOD_PREGRASP_GATE.digest
    assert GOOD_PREGRASP_REQUIRE_STRICT is True
    assert GOOD_PREGRASP_CATALOG_ROOT.endswith("_v5")
    assert GOOD_PREGRASP_GENERATION_DIGEST != V4_GENERATION_DIGEST


def test_sobol_stream_is_row_local_deterministic_and_order_invariant() -> None:
    r"""改变cohort顺序不能改变同一formal row的13D Sobol proposals。"""

    first = sobol_bank((16, 32), candidate_count=8, seed=7, device="cpu")
    repeated = sobol_bank((32, 16), candidate_count=8, seed=7, device="cpu")
    torch.testing.assert_close(first[0], repeated[1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(first[1], repeated[0], rtol=0.0, atol=0.0)
    assert first.shape == (2, 8, 13)


def test_initial_joint_candidates_keep_active_margin_and_zero_ghosts() -> None:
    r"""所有Sobol synergy q必须位于至少10% comfort domain，ghost始终为0。"""

    lower = torch.full((2, 16), -1.0)
    upper = torch.full((2, 16), 2.0)
    active = torch.ones(2, 16, dtype=torch.bool)
    active[1, 10:] = False
    sobol = sobol_bank((1, 2), candidate_count=32, seed=11, device="cpu")
    q, margin = initial_joint_candidates(lower, upper, active, sobol)
    assert q.shape == (2, 32, 16) and margin.shape == (2, 32)
    assert bool((margin >= 0.11 - 1.0e-6).all())
    assert torch.equal(q[1, :, 10:], torch.zeros_like(q[1, :, 10:]))


def test_envelope_selects_three_separated_fingers_with_direct_or_sobol_position() -> None:
    r"""合成四TIP环绕cube center时，自动pair满足10 cm与30°几何门。"""

    tips = torch.tensor(
        [
            [
                [-0.04, 0.08, 0.055],
                [0.00, 0.12, 0.055],
                [0.04, 0.08, 0.055],
                [0.00, 0.04, 0.055],
            ]
        ]
    )
    active = torch.ones(1, 4, dtype=torch.bool)
    position = torch.tensor([[0.0, 0.08, 0.054]])
    direct = fixed_position_envelope(tips, active, position)
    assert float(direct.tip_center_distances_m.max().item()) < 0.10
    assert float(direct.sector_min_deg.item()) >= 30.0

    sobol = torch.full((1, 13), 0.5)
    generated = initial_envelope(tips, active, sobol)
    assert generated.object_position_h_m.shape == (1, 3)
    assert float(generated.tip_center_distances_m.max().item()) < 0.10


def test_strict_pass_mask_applies_total_angular_and_every_other_gate() -> None:
    r"""恰在边界通过；总角速度2.01 rad/s单独使候选失败。"""

    common = {
        "joint_margin": torch.tensor([0.10]),
        "distances": torch.tensor([[0.08, 0.09, 0.10]]),
        "sector_deg": torch.tensor([30.0]),
        "penetration_m": torch.tensor([0.0005]),
        "displacement_m": torch.tensor([0.005]),
        "tilt_deg": torch.tensor([10.0]),
        "peak_linear_m_s": torch.tensor([0.25]),
        "palm_fraction": torch.tensor([0.50]),
    }
    def evaluate(angular: float) -> torch.Tensor:
        r"""只改变总角速度，其余八个门保持边界值。"""

        return strict_pass_mask(
            joint_margin=common["joint_margin"],
            distances=common["distances"],
            sector_deg=common["sector_deg"],
            penetration_m=common["penetration_m"],
            displacement_m=common["displacement_m"],
            tilt_deg=common["tilt_deg"],
            peak_linear_m_s=common["peak_linear_m_s"],
            peak_angular_rad_s=torch.tensor([angular]),
            palm_fraction=common["palm_fraction"],
        )

    assert bool(evaluate(2.0).item())
    assert not bool(evaluate(2.01).item())


def test_low_rank_cem_is_deterministic_bounded_and_preserves_ghosts() -> None:
    r"""4D joint PCA+3D position CEM输出固定shape、comfort margin和deterministic stream。"""

    torch.manual_seed(5)
    elite_q = torch.randn(2, 16, 16) * 0.05 + 0.5
    elite_position = torch.randn(2, 16, 3) * 0.002 + torch.tensor((0.0, 0.08, 0.054))
    lower = torch.zeros(2, 16)
    upper = torch.ones(2, 16)
    active = torch.ones(2, 16, dtype=torch.bool)
    active[1, 12:] = False
    first_q, first_position, first_centers = low_rank_cem_candidates(
        elite_q,
        elite_position,
        lower,
        upper,
        active,
        candidate_count=128,
        seed=17,
        round_index=1,
        asset_keys=(101, 202),
    )
    second_q, second_position, second_centers = low_rank_cem_candidates(
        elite_q,
        elite_position,
        lower,
        upper,
        active,
        candidate_count=128,
        seed=17,
        round_index=1,
        asset_keys=(101, 202),
    )
    torch.testing.assert_close(first_q, second_q, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first_position, second_position, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first_centers, second_centers, rtol=0.0, atol=0.0)
    assert torch.equal(
        torch.bincount(first_centers[0], minlength=16), torch.tensor(CEM_PROPOSAL_CENTER_COUNTS)
    )
    assert bool((first_q[0] >= 0.10).all() and (first_q[0] <= 0.90).all())
    assert torch.equal(first_q[1, :, 12:], torch.zeros_like(first_q[1, :, 12:]))
    assert bool((first_position[..., 2] >= 0.055).all() and (first_position[..., 2] <= 0.065).all())

    swapped_q, swapped_position, _ = low_rank_cem_candidates(
        elite_q.flip(0),
        elite_position.flip(0),
        lower.flip(0),
        upper.flip(0),
        active.flip(0),
        candidate_count=128,
        seed=17,
        round_index=1,
        asset_keys=(202, 101),
    )
    torch.testing.assert_close(swapped_q.flip(0), first_q, rtol=0.0, atol=0.0)
    torch.testing.assert_close(swapped_position.flip(0), first_position, rtol=0.0, atol=0.0)
