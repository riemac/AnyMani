r"""研究入口的不可覆盖与数据位置传播合同，不启动Isaac。

生成器的prestartup配置段以AST边界执行，只验证解析、成员轴和catalog位置；
AppLauncher及后续物理代码不进入本测试，不把这些检查当作真实reset验收。
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.research import build_pure_leap_right_cohorts as builder


@pytest.mark.parametrize("output_kind", ["default", "absolute", "home-relative"])
def test_base_cohort_builder_never_overwrites_existing_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output_kind: str
) -> None:
    r"""默认名称和显式output都保护已冻结文件，不只保护自定义cohort-id。"""
    output = tmp_path / "pure-leap-right-a64.lock.yaml"
    output.write_text("frozen evidence\n")
    monkeypatch.setattr(builder, "COHORT_ROOT", tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    requested = {"default": None, "absolute": output, "home-relative": Path("~/pure-leap-right-a64.lock.yaml")}[output_kind]
    monkeypatch.setattr(
        builder,
        "_parse_args",
        lambda: SimpleNamespace(scale=64, output=requested, cohort_id=None, exclude_mother=[]),
    )

    def forbidden_publish(*args: object, **kwargs: object) -> None:
        r"""存在性拒绝必须早于昂贵source解析和任何writer副作用。"""
        pytest.fail("existing cohort reached the writer")

    monkeypatch.setattr(builder, "write_lineage_cohort_lock", forbidden_publish)
    with pytest.raises(FileExistsError):
        builder.main()
    assert output.read_text() == "frozen evidence\n"


def _generator_prestartup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    declared_catalog: str | None,
    environment_catalog: str | None = None,
    explicit_catalog: str | None = None,
) -> dict:
    r"""执行真实CLI的纯配置前缀；一旦到AppLauncher import便截断。"""
    path = Path(__file__).resolve().parents[1] / "generate_heterogeneous_mvp80_pregrasp_strict.py"
    module = ast.parse(path.read_text())
    stop = next(
        index for index, node in enumerate(module.body) if isinstance(node, ast.ImportFrom) and node.module == "isaaclab.app"
    )
    prefix = ast.Module(body=module.body[:stop], type_ignores=[])
    lock = tmp_path / "shard.canonical.lock.yaml"
    selection = {} if declared_catalog is None else {"catalog_root": declared_catalog}
    lock.write_text(json.dumps({"cohort_id": "fixture", "members": [{"cohort_index": 0}], "selection": selection}))
    argv = [str(path), "--cohort-lock", str(lock)]
    if explicit_catalog is not None:
        argv += ["--catalog", explicit_catalog]
    with monkeypatch.context() as scoped:
        scoped.setattr("sys.argv", argv)
        for name in ("ANYMANI_HETERO_COHORT_LOCK", "ANYMANI_HETERO_ASSET_ROWS", "ANYMANI_HETERO_NUM_ENVS"):
            scoped.setenv(name, "")  # prefix里的直接env赋值也在测试后恢复
        if environment_catalog is None:
            scoped.delenv("ANYMANI_HETERO_GOOD_PREGRASP_CATALOG_ROOT", raising=False)
        else:
            scoped.setenv("ANYMANI_HETERO_GOOD_PREGRASP_CATALOG_ROOT", environment_catalog)
        namespace = {"__name__": "prestartup_contract", "__file__": str(path)}
        exec(compile(prefix, str(path), "exec"), namespace)  # noqa: S102 - 仅执行仓库内真实纯配置段
    return namespace


def test_shard_catalog_is_used_without_repeating_cli_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r"""只给cohort锁也应使用其中记录的私有目录，而不是回落训练共享目录。"""
    target = str(tmp_path / "private catalog")
    namespace = _generator_prestartup(tmp_path, monkeypatch, declared_catalog=target)
    assert namespace["ARGS"].catalog == Path(target)


def test_environment_catalog_agrees_with_generator_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r"""env覆盖同时进入默认位置，避免后续script/runtime默认值一致性检查误拒绝。"""
    target = str(tmp_path / "environment catalog")
    namespace = _generator_prestartup(tmp_path, monkeypatch, declared_catalog=None, environment_catalog=target)
    assert namespace["ARGS"].catalog == namespace["DEFAULT_COHORT_CATALOG"] == Path(target)


def test_conflicting_implicit_catalog_locations_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r"""env与分片声明冲突时要求显式选择，不猜测哪个目录可写。"""
    with pytest.raises(ValueError, match="catalog location conflict"):
        _generator_prestartup(
            tmp_path, monkeypatch, declared_catalog=str(tmp_path / "declared"), environment_catalog=str(tmp_path / "env")
        )


def test_explicit_catalog_is_a_deliberate_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r"""显式CLI仍可把相同exact-key生成送入另一个独占目录。"""
    target = str(tmp_path / "explicit")
    namespace = _generator_prestartup(
        tmp_path,
        monkeypatch,
        declared_catalog=str(tmp_path / "declared"),
        environment_catalog=str(tmp_path / "env"),
        explicit_catalog=target,
    )
    assert namespace["ARGS"].catalog == Path(target)
