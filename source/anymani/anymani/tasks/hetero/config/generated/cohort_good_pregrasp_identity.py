'Strict Top-8 pregrasp generation identity for variable-size resolved cohorts. Preserve strict-v5 proposal, physics, and gate values; replace manifest-row random streams with source/physical identity streams. Publish only after all locked cohort members pass. Object, scale, and physics identity stay unchanged.'

from __future__ import annotations

import json
import os
from typing import Any

from anymani.pregrasp.schema import stable_digest

from .strict_good_pregrasp_identity import (
    STRICT_GOOD_PREGRASP_CEM_CANDIDATES,
    STRICT_GOOD_PREGRASP_CEM_ELITES,
    STRICT_GOOD_PREGRASP_CEM_ROUNDS,
    STRICT_GOOD_PREGRASP_GENERATION_IDENTITY,
    STRICT_GOOD_PREGRASP_OBJECT_SCALE,
    STRICT_GOOD_PREGRASP_PHYSICS_DIGEST,
    STRICT_GOOD_PREGRASP_PHYSICS_IDENTITY,
    STRICT_GOOD_PREGRASP_PHYSICS_TOP_K,
    STRICT_GOOD_PREGRASP_REQUIRE_STRICT,
    STRICT_GOOD_PREGRASP_SEED,
    STRICT_GOOD_PREGRASP_SOBOL_CANDIDATES,
)

# The catalog path is run/evaluation data location; it does not change exact-key physics or search identity.
# Separate development or acceptance catalogs keep unseen-asset additions from changing the training index digest.
# Without an override, keep the training default and enforce its original resume identity.
COHORT_GOOD_PREGRASP_CATALOG_ROOT = os.environ.get(
    "ANYMANI_HETERO_GOOD_PREGRASP_CATALOG_ROOT",
    "outputs/pregrasp/catalogs/heterogeneous_rotation/strict-v1/dexcube/scale-1p1",
)  # The same physical, object, scale, physics, and generation identity can reuse the same Top-8 content.
if not COHORT_GOOD_PREGRASP_CATALOG_ROOT.strip():
    raise ValueError("cohort good-pregrasp catalog root must not be blank")
COHORT_GOOD_PREGRASP_EVIDENCE_ROOT = "outputs/pregrasp/search/heterogeneous_rotation/strict-v1/dexcube/scale-1p1"
'Per-cohort candidate evidence is stored under an ID and lock-digest subdirectory.'
COHORT_GOOD_PREGRASP_OBJECT_SCALE = STRICT_GOOD_PREGRASP_OBJECT_SCALE  # DexCube scale is dimensionless and fixed at 1.1.
COHORT_GOOD_PREGRASP_SEED = STRICT_GOOD_PREGRASP_SEED  # Global proposal seed: 20260902.
COHORT_GOOD_PREGRASP_SOBOL_CANDIDATES = STRICT_GOOD_PREGRASP_SOBOL_CANDIDATES  # Initial proposal count per asset: 256.
COHORT_GOOD_PREGRASP_PHYSICS_TOP_K = STRICT_GOOD_PREGRASP_PHYSICS_TOP_K  # Initial geometry shortlist for physics: Top-32.
COHORT_GOOD_PREGRASP_CEM_ROUNDS = STRICT_GOOD_PREGRASP_CEM_ROUNDS  # At most three refinement rounds.
COHORT_GOOD_PREGRASP_CEM_CANDIDATES = STRICT_GOOD_PREGRASP_CEM_CANDIDATES  # Each failed asset gets 128 candidates per round.
COHORT_GOOD_PREGRASP_CEM_ELITES = STRICT_GOOD_PREGRASP_CEM_ELITES  # Fit the low-rank distribution from the 16 highest-quality physical elites.
COHORT_GOOD_PREGRASP_REQUIRE_STRICT = STRICT_GOOD_PREGRASP_REQUIRE_STRICT  # Runtime rechecks every Top-8 member against the hard gate.
COHORT_GOOD_PREGRASP_PHYSICS_IDENTITY = STRICT_GOOD_PREGRASP_PHYSICS_IDENTITY  # Physics identity includes 120 Hz, material, and solver settings.
COHORT_GOOD_PREGRASP_PHYSICS_DIGEST = STRICT_GOOD_PREGRASP_PHYSICS_DIGEST  # SHA-256 of the physics identity.


def cohort_good_pregrasp_generation_identity() -> dict[str, Any]:
    'Return the strict generation protocol with cohort cardinality and physical-stream semantics. Deep-copy through JSON so updates cannot mutate the historical MVP80 identity; include the stream key and all-selected publication rule in the digest.'

    identity = json.loads(json.dumps(STRICT_GOOD_PREGRASP_GENERATION_IDENTITY))  # Make an independent JSON-safe protocol copy.
    identity["algorithm"] = "cohort-strict-good-pregrasp-v1"  # Keep the cohort stream distinct from row-keyed strict-v5.
    identity["refinement"]["random_stream_key"] = "source_content_and_physical_geometry_sha256"  # Reuse streams across manifests.
    identity["publication"] = {
        "top_k_per_asset": 8,  # Each exact hand-object-scale key contains eight ranked candidates.
        "require_all_selected_assets": True,  # Do not publish an index if any selected member lacks a complete Top-8.
        "cardinality_source": "resolved_cohort_lock",  # The frozen cohort lock defines A; do not hard-code 80.
    }
    return identity


COHORT_GOOD_PREGRASP_GENERATION_IDENTITY = cohort_good_pregrasp_generation_identity()  # Authoritative JSON-safe protocol.
COHORT_GOOD_PREGRASP_GENERATION_DIGEST = stable_digest(COHORT_GOOD_PREGRASP_GENERATION_IDENTITY)  # Fields included in the exact key.

__all__ = [
    "COHORT_GOOD_PREGRASP_CATALOG_ROOT",
    "COHORT_GOOD_PREGRASP_CEM_CANDIDATES",
    "COHORT_GOOD_PREGRASP_CEM_ELITES",
    "COHORT_GOOD_PREGRASP_CEM_ROUNDS",
    "COHORT_GOOD_PREGRASP_EVIDENCE_ROOT",
    "COHORT_GOOD_PREGRASP_GENERATION_DIGEST",
    "COHORT_GOOD_PREGRASP_GENERATION_IDENTITY",
    "COHORT_GOOD_PREGRASP_OBJECT_SCALE",
    "COHORT_GOOD_PREGRASP_PHYSICS_DIGEST",
    "COHORT_GOOD_PREGRASP_PHYSICS_IDENTITY",
    "COHORT_GOOD_PREGRASP_PHYSICS_TOP_K",
    "COHORT_GOOD_PREGRASP_REQUIRE_STRICT",
    "COHORT_GOOD_PREGRASP_SEED",
    "COHORT_GOOD_PREGRASP_SOBOL_CANDIDATES",
    "cohort_good_pregrasp_generation_identity",
]
