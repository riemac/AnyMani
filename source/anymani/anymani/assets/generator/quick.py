"""pre-made connectivity quick entry.

这个文件的定位非常直接：给当前 `assets/generator/` 子项目提供一个
**可直接运行** 的最短脚本入口，用来：

1. 列出 Allegro / LEAP 当前全部合法 connectivity preset 名；
2. 同时枚举 `single_palm_allegro` 与 `single_palm_leap` 的全部合法 pre-made 变体；
3. 为每个 bundle 打印：
   - `base_hand_preset`
   - `connectivity_preset`
   - `hand.urdf` 路径

# NOTE:
这不是新的 façade，也不是新的 runner。真正的用户入口仍然只有：

- `HandGeneratorCfg`
- `HandGenerator`

这个脚本只是把“你当前反复会运行的那几段最短指令”收成一个单文件，方便：

- 直接 `python quick.py`
- 在 notebook 里整段复制
- 或者把其中某些常量手工改小，做局部巡检

# NOTE:
默认配置刻意采用：

- `hand_presets = ["single_palm_allegro", "single_palm_leap"]`
- `connectivity_presets = None`

这正对应当前 pre-made 打通阶段的主线语义：
**自动展开 Allegro + LEAP 的全部合法 connectivity 变体。**

若只想看单个 family，只需要删掉 `HAND_PRESETS` 里另一项即可；
若只想看某几个 connectivity，则把 `CONNECTIVITY_PRESETS` 改成显式映射即可。
"""

from __future__ import annotations

import sys
from pathlib import Path


# 仓库根目录固定从当前脚本位置反推：
# `.../AnyMani/source/anymani/anymani/assets/generator/quick.py`
#   parents[0] = generator
#   parents[1] = assets
#   parents[2] = anymani
#   parents[3] = anymani
#   parents[4] = source
#   parents[5] = AnyMani repo root
REPO_ROOT = Path(__file__).resolve().parents[5]

# 让 `import anymani...` 直接指向当前工作区源码，而不是其它 Python 环境中的旧缓存包。
SOURCE_ROOT = REPO_ROOT / "source" / "anymani"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from anymani.assets.generator.hand_generator import HandGenerator, HandGeneratorCfg
from anymani.assets.presets.connectivity_presets import list_hand_connectivity_preset_names


# ============================================================================
#  运行常量
# ============================================================================


# 默认把两条 canonical base hand 都纳入 pre-made 枚举空间。
# 若只想看单个 family，删掉其中一项即可。
HAND_PRESETS = ["single_palm_allegro", "single_palm_leap"]

# `None` 的科研语义是：
# “对每个 hand preset，自动展开其所属 family 的全部合法 connectivity preset”。
#
# 若只想看少量 connectivity，把它改成例如：
# {
#     "single_palm_allegro": ["allegro_full", "allegro_t3_i2_m2_r2"],
#     "single_palm_leap": ["leap_full", "leap_t3_i2_m2_r2"],
# }
CONNECTIVITY_PRESETS: dict[str, list[str]] | None = None

# 产物目录默认仍落到项目自己的 `assets/generated/`。
OUTPUT_DIR = REPO_ROOT / "source" / "anymani" / "anymani" / "assets" / "generated"

# 当前 quick 入口默认就是为了人工巡检 URDF，因此直接导出 `bundle`。
ARTIFACT_LEVEL = "bundle"

# 递归目录更适合后续接 post-mutate，也更适合按 family / connectivity 浏览。
OUTPUT_LAYOUT = "recursive"


# ============================================================================
#  运行函数
# ============================================================================


def print_connectivity_registry() -> None:
    r"""打印当前 Allegro / LEAP 的全部合法 connectivity preset 名。

    这一步的价值不只是“看看数量”，而是让你在运行枚举前先明确：

    - 当前 registry 里到底有哪些合法名字；
    - Allegro / LEAP 两个 family 的空间规模分别是多少；
    - 若后续你要人工挑选 subset，应从哪些稳定名字里选。
    """

    for family in ("allegro", "leap"):
        names = list_hand_connectivity_preset_names(family)  # 当前 family 的全部合法 hand-level connectivity 名
        print(f"=== {family} ({len(names)} variants) ===")
        for name in names:
            print(name)
        print()


def enumerate_premade_bundles():
    r"""枚举当前配置下的全部 pre-made bundle。

    Returns:
        list: `HandGenerationResult` 列表；每个元素都应携带已导出的 `hand.urdf` 路径。
    """

    generator = HandGenerator(
        HandGeneratorCfg(
            mode="made",  # 当前只走 pre-made，不接 post-mutate
            artifact_level=ARTIFACT_LEVEL,  # 直接导出 bundle，方便 VS Code URDF 插件巡检
            output_dir=OUTPUT_DIR,
            sampling_strategy="enumerate",  # 当前要的是显式遍历离散合法空间，不是随机 sample
            hand_presets=HAND_PRESETS,
            connectivity_presets=CONNECTIVITY_PRESETS,
            output_layout=OUTPUT_LAYOUT,
        )
    )
    return list(generator.generate_batch())


def main() -> int:
    r"""脚本主入口。

    执行顺序刻意保持朴素：

    1. 先打印当前合法 registry；
    2. 再执行全量 pre-made 枚举；
    3. 最后逐行打印每个 bundle 的 provenance 与 `hand.urdf` 路径。
    """

    print_connectivity_registry()  # 先把合法空间完整打出来，方便人工核对
    results = enumerate_premade_bundles()  # 再真正执行 pre-made bundle 导出

    print(f"generated {len(results)} bundles under {OUTPUT_DIR}")
    for result in results:
        print(
            result.metadata["base_hand_preset"],  # canonical base hand 锚点
            result.metadata["connectivity_preset"],  # 当前 connectivity 变体名
            result.urdf_path,  # 直接给出 hand.urdf 路径，便于人工打开
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
