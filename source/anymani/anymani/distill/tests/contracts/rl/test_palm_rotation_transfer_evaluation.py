r"""Frozen cross-cohort evaluation keeps policy semantics and shared reset states fixed."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest
from anymani.distill.diagnostics.evaluation.rl.palm_rotation_transfer import validate_transfer_evaluation


def _seal(identity):
    r"""Recompute the fixture identity after an intentional test mutation."""
    payload = {key: value for key, value in identity.items() if key != "identity_digest"}
    identity["identity_digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    return identity


@pytest.fixture
def transfer_pair(tmp_path):
    r"""One old asset expands to two assets while the old reset payload stays identical."""
    catalog = tmp_path / "catalog"
    history = catalog / "index_history"
    history.mkdir(parents=True)
    old_bytes = json.dumps({"entries": [{"key_digest": "k0", "entry_digest": "e0"}]}).encode()
    new_bytes = json.dumps(
        {
            "entries": [
                {"key_digest": "k0", "entry_digest": "e0"},
                {"key_digest": "k1", "entry_digest": "e1"},
            ]
        }
    ).encode()
    old_digest = hashlib.sha256(old_bytes).hexdigest()
    (history / f"{old_digest}.json").write_bytes(old_bytes)
    (catalog / "index.json").write_bytes(new_bytes)
    source = _seal(
        {
            "identity_schema_version": "4.0.0",
            "task_id": "rotation",
            "task_contract": {"object_scale": 1.1},
            "policy": {"arm": "direct_token", "actor_contact": "tip-only-binary"},
            "transport_abi": {"joints": 16},
            "training": {"seed": 43, "cohort_id": "old"},
            "implementation": {"files": {"actor.py": "same-source"}},
            "manifest": {"support_asset_count": 1, "path": "old.lock"},
            "pregrasp": {"catalog_root": "catalog", "index_sha256": old_digest, "ordered_key_digests": ["k0"]},
            "geometry_provider": {
                "asset_ids": ["a0"],
                "physical_geometry_hashes": ["p0"],
                "identity_digest": "old-geometry",
                "base_provider_identity_digest": "old-base",
                "identity_schema_version": "2.0.0",
                "provider_type": "retained",
                "precision": {"output_dtype": "float32"},
                "retained_artifact": {"sha256": "fixed-encoder"},
            },
        }
    )
    runtime = copy.deepcopy(source)
    runtime["manifest"] = {"support_asset_count": 2, "path": "new.lock"}
    runtime["pregrasp"]["index_sha256"] = hashlib.sha256(new_bytes).hexdigest()
    runtime["pregrasp"]["ordered_key_digests"] = ["k0", "k1"]
    runtime["geometry_provider"].update(asset_ids=["a0", "a1"], physical_geometry_hashes=["p0", "p1"])
    return source, _seal(runtime), tmp_path


def test_transfer_accepts_new_assets_but_preserves_shared_resets(transfer_pair):
    source, runtime, root = transfer_pair
    result = validate_transfer_evaluation(runtime, source, root=root)
    assert result["shared_asset_count"] == 1 and result["new_asset_count"] == 1
    assert result["shared_pregrasp_records_equal"] is True


@pytest.mark.parametrize("field", ["task_contract", "policy", "transport_abi", "training", "implementation"])
def test_transfer_does_not_hide_non_population_changes(transfer_pair, field):
    source, runtime, root = transfer_pair
    if field == "implementation":
        runtime[field]["files"]["actor.py"] = "changed-source"
    else:
        runtime[field]["unexpected"] = True
    _seal(runtime)
    with pytest.raises(RuntimeError):
        validate_transfer_evaluation(runtime, source, root=root)


def test_transfer_rejects_changed_geometry_precision(transfer_pair):
    source, runtime, root = transfer_pair
    runtime["geometry_provider"]["precision"]["output_dtype"] = "float16"
    _seal(runtime)
    with pytest.raises(RuntimeError, match="geometry"):
        validate_transfer_evaluation(runtime, source, root=root)


def test_transfer_rejects_replaced_shared_grasp_payload(transfer_pair):
    source, runtime, root = transfer_pair
    index = root / "catalog/index.json"
    document = json.loads(index.read_text())
    document["entries"][0]["entry_digest"] = "replacement"
    index.write_text(json.dumps(document))
    runtime["pregrasp"]["index_sha256"] = hashlib.sha256(index.read_bytes()).hexdigest()
    _seal(runtime)
    with pytest.raises(RuntimeError, match="pregrasp"):
        validate_transfer_evaluation(runtime, source, root=root)


def test_catalog_growth_without_support_change_is_compatible(transfer_pair):
    source, runtime, root = transfer_pair
    runtime["manifest"] = copy.deepcopy(source["manifest"])
    runtime["geometry_provider"] = copy.deepcopy(source["geometry_provider"])
    runtime["pregrasp"]["ordered_key_digests"] = ["k0"]
    _seal(runtime)
    result = validate_transfer_evaluation(runtime, source, root=root)
    assert result["new_asset_count"] == 0 and result["support_changed"] is False
