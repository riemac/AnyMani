r"""Build the LEAP-origin transfer ladder without dropping previous training members.

The new A8 contains two LEAP-right topologies with four fingertips and 8/12
active joints, each represented by one mother and three selected variants.
They are SSL-training assets but absent from the A16 policy-training support.
This isolates new control topologies from a simultaneous change in tip count;
it is not a claim of geometry-encoder-unseen generalization.

The A8 lock is for preparing only the missing pregrasps. The A24 lock is for
mixed old/new training and evaluation, with all old members kept first.
Existing locks are immutable inputs and output replacement is forbidden.
The Allegro stage preserves A24 and adds two four-tip lineages for A32.
The separate six-member holdout has unseen three-tip topologies and never
enters either training cohort. All selections have geometry-SSL exposure.
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
    parser.add_argument("--stage", choices=("leap", "allegro", "holdout"), default="leap")
    parser.add_argument(
        "--resume", action="store_true", help="Validate and adopt matching already-published pilot locks."
    )
    args = parser.parse_args()
    root = resolve_anymani_root() / "source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1"
    cohorts = root / "cohorts"
    base_names = {"leap": "pure-leap-right-a16", "allegro": "pure-leap-right-pilot-a24"}
    old = (
        load_hand_asset_cohort(cohorts / f"{base_names[args.stage]}.lock.yaml", require_geometry_semantics=True)
        if args.stage != "holdout"
        else None
    )
    retained = tuple((member.source_alias, member.source_row) for member in old.members) if old else ()
    old_mothers = {member.provenance.mother_name for member in old.members} if old else set()
    assert len(retained) == {"leap": 16, "allegro": 24, "holdout": 0}[args.stage]
    assert len(set(retained)) == len(retained)

    # Fixed representatives span recorded mutation descriptors, not measured policy performance.
    coordinates = {
        "leap": tuple(("ssl", row) for row in (2096, 2100, 2104, 2098, 2800, 2801, 2809, 2814)),
        "allegro": tuple(("ppo", row) for row in (16, 17, 24, 28, 80, 81, 85, 93)),
        "holdout": tuple(("ssl", row) for row in (3984, 3986, 3999)) + tuple(("ppo", row) for row in (368, 370, 372)),
    }
    added = coordinates[args.stage]
    assert not set(retained).intersection(added)
    new_mothers = {
        "leap": ("right_t3_i1_m1_r3", "right_t4_i1_m3_r4"),
        "allegro": ("right_t3_i3_m2_r2", "right_t4_i2_m3_r2"),
        "holdout": ("right_t3_i3_m4", "right_t4_m4_r3"),
    }[args.stage]
    selection = {
        "algorithm": {
            "leap": "explicit-a16-plus-two-topologies-v1",
            "allegro": "explicit-a24-plus-two-allegro-topologies-v1",
            "holdout": "explicit-policy-unseen-three-tip-topologies-v1",
        }[args.stage],
        "base_cohort": old.cohort_id if old else None,
        "new_mothers": list(new_mothers),
        "new_active_dof": [8, 12] if args.stage == "leap" else [10, 11],
        "new_tip_count": 3 if args.stage == "holdout" else 4,
        "representatives_per_new_mother": 3 if args.stage == "holdout" else 4,
        "policy_exposure": (
            "new topologies relative to A16; geometry SSL train exposure retained"
            if args.stage == "leap"
            else "policy-unseen relative to the inherited training chain; geometry SSL train exposure retained"
        ),
        "member_order": (
            "all old A16 members followed by the two new four-member blocks"
            if args.stage == "leap"
            else "all prior training members first; then fixed mother-wise representative blocks"
        ),
    }
    new_name = {
        "leap": "pure-leap-right-pilot-new-a8",
        "allegro": "pure-allegro-right-pilot-new-a8",
        "holdout": "right-transfer-unseen-a6",
    }[args.stage]
    new_sources = {alias: root / f"{alias}.yaml" for alias, _ in added}
    plans = [
        (
            new_name,
            added,
            new_sources,
            "policy-unseen-evaluation" if args.stage == "holdout" else "new-pregrasp-preparation",
        )
    ]
    if old:
        union_name = "pure-leap-right-pilot-a24" if args.stage == "leap" else "right-leap-allegro-pilot-a32"
        plans.append(
            (union_name, retained + added, {"ppo": root / "ppo.yaml", "ssl": root / "ssl.yaml"}, "old-new-transfer")
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
        assert set(mothers) == set(new_mothers) | (old_mothers if len(members) > len(added) else set())
        expected_groups = (
            ["single_palm_leap"] * 3 + ["single_palm_allegro"] * 3
            if args.stage == "holdout"
            else ["single_palm_allegro" if args.stage == "allegro" else "single_palm_leap"] * len(added)
        )
        assert [member["provenance"]["group_name"] for member in document["members"][-len(added) :]] == expected_groups
        print(
            json.dumps({"cohort": name, "assets": len(members), "members_per_mother": dict(mothers), "path": str(path)})
        )


if __name__ == "__main__":
    main()
