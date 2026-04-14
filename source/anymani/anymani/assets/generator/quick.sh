# 记录指令，方便使用
#
# 用法说明：
# 1. 先 `cd /home/hac/isaac/AnyMani`
# 2. 执行 `python`
# 3. 把下面某段 ```python``` 代码块整体粘进去运行
#
# 这里刻意只记录最短反馈回路，不再额外包 runner。
# 所有正式入口仍然收口在 `HandGeneratorCfg / HandGenerator`。

# --- 单样本：single_palm_allegro + full connectivity ---
```python
import sys
from pathlib import Path

repo_root = Path.cwd()  # 约定从 `/home/hac/isaac/AnyMani` 仓库根目录启动 `python`
sys.path.insert(0, str(repo_root / "source" / "anymani"))  # 让 `import anymani...` 直接可用

from anymani.assets.generator.hand_generator import HandGenerator, HandGeneratorCfg

result = HandGenerator(
    HandGeneratorCfg(
        mode="full",
        artifact_level="bundle",
        output_dir=repo_root / "source" / "anymani" / "anymani" / "assets" / "generated",
        hand_preset="single_palm_allegro",
        connectivity_preset="allegro_full",
        output_layout="recursive",
    )
).generate()

print(result.urdf_path)     # full connectivity 的 hand.urdf
print(result.sidecar_path)  # full connectivity 的 hand.yaml
```

# --- 单样本：single_palm_allegro + reduced connectivity ---
```python
import sys
from pathlib import Path

repo_root = Path.cwd()  # 约定从 `/home/hac/isaac/AnyMani` 仓库根目录启动 `python`
sys.path.insert(0, str(repo_root / "source" / "anymani"))  # 让 `import anymani...` 直接可用

from anymani.assets.generator.hand_generator import HandGenerator, HandGeneratorCfg

result = HandGenerator(
    HandGeneratorCfg(
        mode="full",
        artifact_level="bundle",
        output_dir=repo_root / "source" / "anymani" / "anymani" / "assets" / "generated",
        hand_preset="single_palm_allegro",
        connectivity_preset="allegro_t3_i2_m2_r2",
        output_layout="recursive",
    )
).generate()

print(result.hand_cfg.dof_count)  # 这里应是 $3+2+2+2=9$ 个 revolute DOF
print(result.urdf_path)           # reduced connectivity 的 hand.urdf
```

# --- 小规模枚举：single_palm_leap / single_palm_allegro 各跑两条 connectivity ---
```python
import sys
from pathlib import Path

repo_root = Path.cwd()  # 约定从 `/home/hac/isaac/AnyMani` 仓库根目录启动 `python`
sys.path.insert(0, str(repo_root / "source" / "anymani"))  # 让 `import anymani...` 直接可用

from anymani.assets.generator.hand_generator import HandGenerator, HandGeneratorCfg

generator = HandGenerator(
    HandGeneratorCfg(
        mode="made",
        artifact_level="bundle",
        output_dir=repo_root / "source" / "anymani" / "anymani" / "assets" / "generated",
        sampling_strategy="enumerate",
        hand_preset_names=("single_palm_allegro", "single_palm_leap"),
        connectivity_preset_names=(
            "allegro_full",
            "allegro_t3_i2_m2_r2",
            "leap_full",
            "leap_t3_i2_m2_r2",
        ),
        max_enumerate=4,
        output_layout="recursive",
    )
)

for result in generator.generate_batch():
    print(result.metadata["base_hand_preset"], result.metadata["connectivity_preset"], result.urdf_path)
```
