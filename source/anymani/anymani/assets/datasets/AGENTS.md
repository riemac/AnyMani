# Dataset Build 快速查证

本目录保存 typed dataset template、selection lock、build state、build report 与最终 manifest。正式 build 结束后，优先使用下面的有界检查；不要递归读取全部 10624 个 sidecar 做 YAML 全量解析，这会比生成本身更慢并制造额外内存压力。

## 1. State 与进程拓扑

在仓库根目录执行：

```bash
cd /home/hac/isaac/AnyMani
source /home/hac/isaac/env_isaaclab/bin/activate

PYTHONPATH=/home/hac/isaac/AnyMani/source/anymani python -u -c '
from pathlib import Path
from collections import Counter
import yaml

root = Path("source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1")
state = yaml.safe_load((root / ".build_state.yaml").read_text())
report = yaml.safe_load((root / "build_report.yaml").read_text())
attempts = [a for task in state["tasks"].values() for a in task["attempts"]]
accepted = [a for a in attempts if a["status"] == "accepted"]
worker_pids = {a["worker_pid"] for a in attempts if a.get("worker_pid")}
service_pids = {a["sdf_service_pid"] for a in attempts if a.get("sdf_service_pid")}

print("schema", state["schema_version"])
print("published", report["published"])
print("status_counts", report["status_counts"])
print("attempt_statuses", Counter(a["status"] for a in attempts))
print("worker_pid_count", len(worker_pids))
print("gpu_service_pids", sorted(service_pids))
print("worker_cuda_initialized", sum(bool(a.get("worker_cuda_initialized")) for a in attempts))
print("markers", sum((Path(a["run_dir"]) / "DATASET_BUILD_ATTEMPT.yaml").is_file() for a in accepted))
print("variant_sidecars", sum(len(a["sidecar_paths"]) for a in accepted))
print("missing_sidecars", sum(not Path(p).is_file() for a in accepted for p in a["sidecar_paths"]))
print("missing_urdfs", sum(not Path(p).is_file() for a in accepted for p in a["urdf_paths"]))
print("run_dirs_unique", len({a["run_dir"] for a in accepted}))
'
```

对 768-task、10624-variant 的正式构建，预期至少满足：`published True`、`completed 768`、`accepted 768`、`worker_cuda_initialized 0`、`variant_sidecars 10624`、`missing_sidecars 0`、`missing_urdfs 0`、`run_dirs_unique 768`。central GPU service 按 train、validation、evaluation stage 顺序各启动一次，因此通常看到 3 个 service PID，而不是 768 个。

## 2. Manifest 顶层结构

只读取两个 manifest 的顶层 YAML，不展开全部 asset records：

```bash
python -c '
from pathlib import Path
import yaml

root = Path("source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1")
for name in ("ssl.yaml", "ppo.yaml"):
    document = yaml.safe_load((root / name).read_text())
    print(name, {
        "schema_version": document["schema_version"],
        "validation": sorted(document["validation"]),
        "evaluation": sorted(document["evaluation"]),
        "train_runs": len(document["train"]["runs"]),
    })
'
```

预期 `schema_version` 为 `2.0.0`，validation 包含 `unseen_variant_set` 与 `unseen_mother`，evaluation 另外包含 `official_zero_shot`。

## 3. 单个 sidecar 证书抽样

先从 build state 取第一份 sidecar，再用 `rg` 读取固定证书片段；不要对整个 sidecar 树运行递归 `yaml.safe_load`：

```bash
sample_sidecar=$(python -c 'from pathlib import Path; import yaml; root=Path("source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1"); state=yaml.safe_load((root/".build_state.yaml").read_text()); print(next(Path(p) for t in state["tasks"].values() for a in t["attempts"] if a["status"] == "accepted" for p in a["sidecar_paths"]))')
rg -n -C 4 "finger_spacing_certificate|mesh_sdf|pair_clearances" "$sample_sidecar"
```

正式 central route 的抽样证书应显示 `complete: true`、`device: cuda`、`requested_backend: warp`、`actual_backend: warp`，且 `fallback_events` 为空。

## 4. GPU 与内核日志

构建结束后检查是否还有残留 GPU compute process：

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

正常结束时应无输出。构建期间或结束后检查 Xid、host OOM kill 与驱动分配错误：

```bash
journalctl -k --since "YYYY-MM-DD HH:MM:SS" --no-pager | rg "NVRM: Xid|oom-kill|Out of memory: Killed process|Killed process|NV_ERR_NO_MEMORY"
```

`Xid`、`oom-kill`、`Killed process` 表示失败风险；单独出现 `NV_ERR_NO_MEMORY` 但没有进程退出、Xid 或 host OOM 时，记录发生时间并与 GPU service stage 启动时间对齐，不能仅凭 CLI 的 `published=True` 忽略。

## 5. 中断恢复

恢复命令默认只审计，不执行删除：

```bash
PYTHONPATH=/home/hac/isaac/AnyMani/source/anymani python -m anymani.assets.scripts.dataset recover \
  --template source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1/template.yaml \
  --strategy adopt
```

只有核对 `recovery_report.yaml` 后才使用 `--apply`。rollback 只能删除当前 invocation 的 `DATASET_BUILD_ATTEMPT.yaml` marker 精确拥有的 run roots；不要按时间戳或“最新目录”手工删除。
