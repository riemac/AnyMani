r"""验证CUDAGraph step修复后的三次独立B4096 structured actor性能复测。"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    r"""解析一个post-fix baseline、两个新repeats与输出路径。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    r"""读取单次artifact并执行compile、数值等价和strict latency门。"""

    payload = path.read_bytes()
    document = json.loads(payload)
    if document.get("artifact_type") != "anymani.hetero.structured_n040_actor_performance":
        raise ValueError(f"{path} is not a structured performance artifact")
    batch = document.get("batches", {}).get("4096", {})
    if not isinstance(batch, dict) or "compile_error" in batch:
        raise ValueError(f"{path} has no successful B4096 compiled route")
    equivalence = batch.get("compiled_numerical_equivalence")
    timing = batch.get("full_compiled_gated_actor")
    if not isinstance(equivalence, dict) or not isinstance(timing, dict):
        raise ValueError(f"{path} lacks numerical equivalence or full compiled timing")
    tolerance = float(equivalence["atol"])
    if float(equivalence["z_max_abs"]) > tolerance or float(equivalence["mean_max_abs"]) > tolerance:
        raise ValueError(f"{path} exceeds compiled numerical tolerance")
    p95 = float(timing["p95_ms"])
    if p95 >= 48.0 or document.get("strict_full_actor_gate_passed") is not True:
        raise ValueError(f"{path} fails strict p95<48 ms")
    if int(timing["warmup"]) != 20 or int(timing["samples"]) != 50:
        raise ValueError(f"{path} does not use 20 warmups and 50 CUDA events")
    repeat_id = str(document.get("repeat_id", "post-fix-v4" if index == 0 else ""))
    if not repeat_id:
        raise ValueError(f"{path} lacks an independent repeat identity")
    return document, {
        "repeat_id": repeat_id,
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "full_compiled_p95_ms": p95,
        "full_compiled_median_ms": float(timing["median_ms"]),
        "z_max_abs": float(equivalence["z_max_abs"]),
        "mean_max_abs": float(equivalence["mean_max_abs"]),
        "git_commit": document.get("git_commit"),
        "script_sha256": document.get("script_sha256"),
    }


def main() -> int:
    r"""写三次post-fix通过的aggregate performance证据。"""

    args = _parse_args()
    loaded = [_load(path, index) for index, path in enumerate(args.artifacts)]
    documents = [item[0] for item in loaded]
    repeats = [item[1] for item in loaded]
    if len({item["repeat_id"] for item in repeats}) != 3:
        raise ValueError("performance repeat IDs must be unique")
    provider_identities = [document["provider_identity"] for document in documents]
    if any(identity != provider_identities[0] for identity in provider_identities[1:]):
        raise ValueError("performance repeats differ in N040 provider identity")
    p95_values = [float(item["full_compiled_p95_ms"]) for item in repeats]
    artifact = {
        "artifact_type": "anymani.hetero.structured_n040_actor_performance_repeats",
        "schema_version": "1.0.0",
        "measurement_count": 3,
        "all_strict_gates_passed": True,
        "all_numerically_equivalent": True,
        "provider_identity": provider_identities[0],
        "full_compiled_p95_ms": {
            "values": p95_values,
            "median": statistics.median(p95_values),
            "max": max(p95_values),
        },
        "repeats": repeats,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **artifact["full_compiled_p95_ms"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
