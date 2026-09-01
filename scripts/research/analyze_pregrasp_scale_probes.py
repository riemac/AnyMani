r"""聚合三次prestartup probe并识别实际DexCube mass/inertia scale law。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _observed_exponent(value_ratio: float, scale_ratio: float) -> float:
    r"""由$y\propto s^k$的两个样本恢复指数$k=\log(y_1/y_0)/\log(s_1/s_0)$。"""

    if value_ratio <= 0.0 or scale_ratio <= 0.0 or math.isclose(scale_ratio, 1.0):
        raise ValueError("scale-law ratios must be positive and use distinct scales")
    return math.log(value_ratio) / math.log(scale_ratio)


def main() -> int:
    r"""验证三scale身份/frame并写出实际物理缩放证据。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-1p1", type=Path, required=True)
    parser.add_argument("--probe-1p2", type=Path, required=True)
    parser.add_argument("--probe-1p25", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    documents = {
        1.1: json.loads(args.probe_1p1.read_text()),
        1.2: json.loads(args.probe_1p2.read_text()),
        1.25: json.loads(args.probe_1p25.read_text()),
    }
    if len({document["object_sha256"] for document in documents.values()}) != 1:
        raise RuntimeError("scale probes resolved different DexCube bytes")
    if len({json.dumps(document["canonical_artifacts"], sort_keys=True) for document in documents.values()}) != 1:
        raise RuntimeError("scale probes changed the ordered hand/canonical identities")
    for scale, document in documents.items():
        if not math.isclose(float(document["scale"]), scale):
            raise RuntimeError("probe scale field disagrees with its input identity")
        if max(document["frame_roundtrip_position_error_m"]) > 1.0e-6:
            raise RuntimeError("hand-frame position roundtrip exceeds 1e-6 m")
        if max(document["frame_roundtrip_quaternion_one_minus_abs_dot"]) > 1.0e-6:
            raise RuntimeError("hand-frame quaternion roundtrip exceeds 1e-6")

    baseline_scale = 1.2
    baseline = documents[baseline_scale]
    baseline_mass = float(baseline["object_mass_kg"][0][0])
    baseline_inertia = float(baseline["object_inertia_kg_m2"][0][0])
    samples = []
    for scale, document in documents.items():
        sensors = list(document["contact_sensors"].values())
        valid_separations = [value["min_separation_m"] for value in sensors if value["min_separation_m"] is not None]
        mass = float(document["object_mass_kg"][0][0])
        inertia = float(document["object_inertia_kg_m2"][0][0])
        sample = {
            "scale": scale,
            "mass_kg": mass,
            "inertia_xx_kg_m2": inertia,
            "mass_ratio_vs_1p2": mass / baseline_mass,
            "inertia_ratio_vs_1p2": inertia / baseline_inertia,
            "contact_points": sum(int(value["contact_points"]) for value in sensors),
            "min_separation_m": min(valid_separations) if valid_separations else None,
            "max_penetration_depth_m": max(float(value["penetration_depth_m"]) for value in sensors),
        }
        if not math.isclose(scale, baseline_scale):
            sample["observed_mass_exponent"] = _observed_exponent(mass / baseline_mass, scale / baseline_scale)
            sample["observed_inertia_exponent"] = _observed_exponent(
                inertia / baseline_inertia, scale / baseline_scale
            )
        samples.append(sample)

    non_baseline = [sample for sample in samples if not math.isclose(sample["scale"], baseline_scale)]
    analysis = {
        "artifact_type": "anymani.pregrasp.scale_physics_analysis",
        "schema_version": "1.0.0",
        "object_sha256": baseline["object_sha256"],
        "dataset_rows": baseline["dataset_rows"],
        "samples": samples,
        "observed_mass_exponent_mean": sum(sample["observed_mass_exponent"] for sample in non_baseline)
        / len(non_baseline),
        "observed_inertia_exponent_mean": sum(sample["observed_inertia_exponent"] for sample in non_baseline)
        / len(non_baseline),
        "interpretation": "Current USD/PhysX scaling preserves object mass and scales principal inertia approximately as s^2.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    print(json.dumps(analysis, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
