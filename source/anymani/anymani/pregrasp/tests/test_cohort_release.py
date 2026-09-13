r"""严格初态准入后的角色补位规则；仅测试来源键，不伪造物理证书。"""

from __future__ import annotations

import pytest


def _roles():
    r"""一条谱系的训练4、开发2、终验4与备用4，编号仅为合成来源坐标。"""

    training = {("ssl", "/mother"): ("ssl#0", "ssl#1", "ssl#2", "ssl#3")}
    strata = [
        {
            "source_alias": "ssl",
            "mother_path": "/mother",
            "mother_name": "right-fixture",
            "development": ["ssl#4", "ssl#5"],
            "acceptance": ["ssl#6", "ssl#7", "ssl#8", "ssl#9"],
            "reserve": ["ssl#10", "ssl#11", "ssl#12", "ssl#13"],
        }
    ]
    return training, strata


def test_admission_replacements_use_fixed_role_then_reserve_order() -> None:
    r"""三个角色各缺一项时，先跳过未准入备用，再依次补位且互不复用。"""

    from anymani.pregrasp.scripts.release_cohort import select_admitted_roles

    training, strata = _roles()
    available = {f"ssl#{index}" for index in range(14)} - {"ssl#3", "ssl#5", "ssl#8", "ssl#10"}
    result = select_admitted_roles(training, strata, available)
    assert result["ready"]
    assert result["role_keys"] == {
        "training": ["ssl#0", "ssl#1", "ssl#2", "ssl#11"],
        "development": ["ssl#4", "ssl#12"],
        "acceptance": ["ssl#6", "ssl#7", "ssl#13", "ssl#9"],
    }
    assert [(row["old"], row["new"]) for row in result["replacements"]] == [
        ("ssl#3", "ssl#11"),
        ("ssl#5", "ssl#12"),
        ("ssl#8", "ssl#13"),
    ]
    all_keys = [key for keys in result["role_keys"].values() for key in keys]
    assert len(all_keys) == len(set(all_keys)) == 10


def test_admission_reports_shortage_without_shrinking_declared_roles() -> None:
    r"""备用不足保留原声明位置和具体缺口，不把较小集合伪装成完整准入。"""

    from anymani.pregrasp.scripts.release_cohort import select_admitted_roles

    training, strata = _roles()
    available = {"ssl#0", "ssl#1", "ssl#2", "ssl#4", "ssl#6", "ssl#7", "ssl#9", "ssl#11"}
    result = select_admitted_roles(training, strata, available)
    assert not result["ready"]
    assert {role: len(keys) for role, keys in result["role_keys"].items()} == {
        "training": 4,
        "development": 2,
        "acceptance": 4,
    }
    assert [item["role"] for item in result["unavailable"]] == ["development", "acceptance"]


def test_admission_cannot_replace_missing_mother_with_variant() -> None:
    r"""训练首项是母体；母体缺少严格初态时不从变体池替代拓扑基准。"""

    from anymani.pregrasp.scripts.release_cohort import select_admitted_roles

    training, strata = _roles()
    result = select_admitted_roles(training, strata, {f"ssl#{i}" for i in range(1, 14)})
    assert not result["ready"]
    assert result["role_keys"]["training"][0] == "ssl#0"
    assert result["unavailable"][0]["reason"] == "mother-not-admitted"


def test_admission_rejects_duplicate_source_roles() -> None:
    r"""同一来源不能预先占据两个角色，再依靠就绪状态掩盖重复。"""

    from anymani.pregrasp.scripts.release_cohort import select_admitted_roles

    training, strata = _roles()
    strata[0]["development"][0] = "ssl#1"
    with pytest.raises(ValueError, match="duplicate"):
        select_admitted_roles(training, strata, {f"ssl#{i}" for i in range(14)})
