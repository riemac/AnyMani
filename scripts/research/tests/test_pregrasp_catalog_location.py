r"""预抓取数据位置可隔离，物理/搜索协议与默认训练位置保持不变。

这些测试不启动Kit、不读取策略，也不把目录选择当作预抓取有效性证书。
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from anymani.tasks.hetero.config.generated import cohort_good_pregrasp_identity as identity_module


def test_unset_catalog_keeps_existing_training_default(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""未显式选择目录时，已训练检查点仍使用原来的数据位置。"""
    monkeypatch.delenv("ANYMANI_HETERO_GOOD_PREGRASP_CATALOG_ROOT", raising=False)
    module = importlib.reload(identity_module)
    assert module.COHORT_GOOD_PREGRASP_CATALOG_ROOT == (
        "outputs/pregrasp/catalogs/heterogeneous_rotation/strict-v1/dexcube/scale-1p1"
    )


def test_explicit_catalog_changes_only_location(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r"""评估目录独立于训练目录，exact-key物理与搜索身份不随位置变化。"""
    before = importlib.reload(identity_module)
    physics = before.COHORT_GOOD_PREGRASP_PHYSICS_DIGEST
    generation = before.COHORT_GOOD_PREGRASP_GENERATION_DIGEST
    target = tmp_path / "isolated evaluation catalog"  # 路径允许空格，不在此处创建文件
    with monkeypatch.context() as scoped:
        scoped.setenv("ANYMANI_HETERO_GOOD_PREGRASP_CATALOG_ROOT", str(target))
        module = importlib.reload(identity_module)
        assert str(target) == module.COHORT_GOOD_PREGRASP_CATALOG_ROOT
        assert physics == module.COHORT_GOOD_PREGRASP_PHYSICS_DIGEST
        assert generation == module.COHORT_GOOD_PREGRASP_GENERATION_DIGEST
        assert module.COHORT_GOOD_PREGRASP_OBJECT_SCALE == 1.1
        assert module.COHORT_GOOD_PREGRASP_REQUIRE_STRICT
        assert not target.exists()
    importlib.reload(identity_module)  # 还原同一pytest进程中随后测试看到的默认配置


def test_blank_catalog_is_not_an_implicit_working_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""空路径须失败，不能无意中从当前工作目录读取另一份index。"""
    with monkeypatch.context() as scoped:
        scoped.setenv("ANYMANI_HETERO_GOOD_PREGRASP_CATALOG_ROOT", " ")
        with pytest.raises(ValueError, match="catalog root must not be blank"):
            importlib.reload(identity_module)
    importlib.reload(identity_module)
