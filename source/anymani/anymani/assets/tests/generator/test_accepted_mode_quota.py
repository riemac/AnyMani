r"""post-mutate accepted/output self_mode quota 测试。

这组测试先锁住纯分配器语义。generator 集成层会在现有 post-mutate quick
测试基础上继续扩展；这里最重要的是避免概率 dict 再退回 proposal prior。
"""

from __future__ import annotations

from assets.generator._accepted_mode_quota import LIMIT_TWEAK_MODE_ORDER, allocate_accepted_mode_quota


def test_allocate_accepted_mode_quota_exact_case():
    r"""整数概率应直接得到对应 accepted quota。"""

    quota = allocate_accepted_mode_quota({"identity": 0.2, "general": 0.8}, 10)

    assert quota == {"identity": 2, "general": 8}


def test_allocate_accepted_mode_quota_largest_remainder():
    r"""小数部分更大的 mode 应先拿到剩余槽位。"""

    quota = allocate_accepted_mode_quota(
        {
            "identity": 0.34,
            "general": 0.33,
            "index_ring": 0.33,
        },
        5,
    )

    assert quota == {"identity": 2, "general": 2, "index_ring": 1}


def test_allocate_accepted_mode_quota_tie_break_ignores_dict_order():
    r"""小数部分相同的 tie-break 必须由固定 mode 顺序决定。"""

    first = allocate_accepted_mode_quota({"general": 0.5, "identity": 0.5}, 1)
    second = allocate_accepted_mode_quota({"identity": 0.5, "general": 0.5}, 1)

    assert first == {"identity": 1}
    assert second == {"identity": 1}


def test_allocate_accepted_mode_quota_omits_zero_probability_modes():
    r"""零概率 mode 不应被采样，也不应出现在正 quota 表里。"""

    quota = allocate_accepted_mode_quota({"identity": 0.0, "general": 1.0}, 3)

    assert quota == {"general": 3}


def test_allocate_accepted_mode_quota_supports_limit_tweak_mode_order():
    r"""`limit_tweak` 的 accepted quota tie-break 也应使用自己的固定 mode 顺序。"""

    quota = allocate_accepted_mode_quota(
        {"disturb": 0.5, "identity": 0.5},
        1,
        mode_order=LIMIT_TWEAK_MODE_ORDER,
        label="limit_tweak.self_mode",
    )

    assert quota == {"identity": 1}
