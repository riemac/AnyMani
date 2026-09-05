r"""Build the A24 transfer pilot without dropping any original A16 member.

The new A8 contains two LEAP-right topologies with four fingertips and 8/12
active joints, each represented by one mother and three selected variants.
They are SSL-training assets but absent from the A16 policy-training support.
This isolates new control topologies from a simultaneous change in tip count;
it is not a claim of geometry-encoder-unseen generalization.

The A8 lock is for preparing only the missing pregrasps. The A24 lock is for
mixed old/new training and evaluation, with all old members kept first.
Existing locks are immutable inputs and output replacement is forbidden.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from anymani.assets.bank.cohort import load_hand_asset_cohort, write_hand_asset_cohort_lock
from anymani.assets.bank.path_utils import resolve_anymani_root


def main() -> None:
    r"""Publish source-resolved locks; physical lowering and pregrasps run separately."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume", action="store_true", help="Validate and adopt matching already-published pilot locks."
    )
    args = parser.parse_args()
    root = resolve_anymani_root() / "source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1"
    cohorts = root / "cohorts"
    old = load_hand_asset_cohort(cohorts / "pure-leap-right-a16.lock.yaml", require_geometry_semantics=True)
    retained = tuple((member.source_alias, member.source_row) for member in old.members)
    assert len(retained) == 16 and len(set(retained)) == 16

    # These are policy-unseen topology representatives, not extra old-topology geometries.
    added = tuple(("ssl", row) for row in (2096, 2100, 2104, 2098, 2800, 2801, 2809, 2814))
    assert not set(retained).intersection(added)
    new_mothers = ("right_t3_i1_m1_r3", "right_t4_i1_m3_r4")
    selection = {
        "algorithm": "explicit-a16-plus-two-topologies-v1",
        "base_cohort": old.cohort_id,
        "new_mothers": list(new_mothers),
        "new_active_dof": [8, 12],
        "new_tip_count": 4,
        "representatives_per_new_mother": 4,
        "policy_exposure": "new topologies relative to A16; geometry SSL train exposure retained",
        "member_order": "all old A16 members followed by the two new four-member blocks",
    }
    plans = (
        ("pure-leap-right-pilot-new-a8", added, {"ssl": root / "ssl.yaml"}, "new-pregrasp-preparation"),
        (
            "pure-leap-right-pilot-a24",
            retained + added,
            {"ppo": root / "ppo.yaml", "ssl": root / "ssl.yaml"},
            "old-new-transfer",
        ),
    )
    for name, _, _, _ in plans:
        if (cohorts / f"{name}.lock.yaml").exists() and not args.resume:
            raise FileExistsError(cohorts / f"{name}.lock.yaml")

    # The standard writer re-resolves asset semantics, rather than editing lock JSON by hand.
    for name, members, sources, purpose in plans:
        path = cohorts / f"{name}.lock.yaml"
        expected_selection = {**selection, "purpose": purpose}
        if path.exists():
            existing = load_hand_asset_cohort(path, require_geometry_semantics=True)
            assert existing.cohort_id == name and dict(existing.selection) == expected_selection
        else:
            path = write_hand_asset_cohort_lock(
                path,
                cohort_id=name,
                source_manifests=sources,
                member_coordinates=members,
                selection=expected_selection,
                require_geometry_semantics=True,
            )
        # The writer/adoption already validates parent assets. This final order check
        # reads the published declaration without resolving entire source banks again.
        document = json.loads(path.read_text())
        actual = tuple((member["source_alias"], member["source_row"]) for member in document["members"])
        assert actual == members
        mothers = Counter(member["provenance"]["mother_name"] for member in document["members"])
        assert set(mothers) == set(new_mothers) | ({"right_t4_i4_m4_r4"} if len(members) == 24 else set())
        print(
            json.dumps({"cohort": name, "assets": len(members), "members_per_mother": dict(mothers), "path": str(path)})
        )


if __name__ == "__main__":
    main()
