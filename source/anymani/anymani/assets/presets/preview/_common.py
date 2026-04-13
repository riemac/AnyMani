"""preset quick-check 脚本共享辅助。

这些脚本的定位很明确：给 preset 调参提供最短反馈回路。
因此这里的公共逻辑也刻意只保留最薄的一层：

1. 把 `source/anymani` 加进 `sys.path`
2. 统一输出目录策略（默认临时目录，可显式覆写）
3. 统一打印导出结果

之所以把这批脚本放在 `assets/presets/preview/`，是因为它们服务的对象本来就是
preset，而不是更广义的仓库级脚本系统。
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile


# 当前文件位于：
#   source/anymani/anymani/assets/presets/preview/_common.py
# 因此：
# - parents[6] -> 仓库根目录 AnyMani/
# - parents[4] -> 可直接放进 `sys.path` 的 source/anymani/
REPO_ROOT = Path(__file__).resolve().parents[6]
SOURCE_ROOT = Path(__file__).resolve().parents[4]


def bootstrap_python_path() -> None:
    """确保脚本以文件路径直跑时，也能稳定导入 `anymani` 包。"""

    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))


def resolve_output_dir(output_dir: str | None, *, prefix: str) -> Path:
    """解析脚本输出目录。

    - 用户显式传 `--output-dir` 时，尊重用户路径
    - 否则默认创建一个持久到会话结束的临时目录，便于 VS Code 直接打开检查
    """

    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    return Path(tempfile.mkdtemp(prefix=f"{prefix}_"))


def print_export_result(*, label: str, output_dir: Path, written: list[Path] | None = None) -> None:
    """统一打印 quick-check 产物位置。"""

    written = written or []
    if written:
        print(f"[INFO] {label} exported:")
        for path in written:
            print(f"  - {path}")
    else:
        print(f"[INFO] {label} finished. Output directory: {output_dir}")


def infer_family_from_palm_preset(preset_name: str) -> str | None:
    """从当前 palm preset 名推断 family。

    当前 quick-check 脚本只支持项目内已注册的 `com_*` / `single_box_*` 命名。
    """

    if preset_name.startswith("com_"):
        return preset_name.removeprefix("com_")
    if preset_name.startswith("single_box_"):
        return preset_name.removeprefix("single_box_")
    return None


__all__ = [
    "REPO_ROOT",
    "SOURCE_ROOT",
    "bootstrap_python_path",
    "resolve_output_dir",
    "print_export_result",
    "infer_family_from_palm_preset",
]
