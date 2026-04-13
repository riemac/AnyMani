r"""单位语义回归测试。

这组测试专门锁住当前资产生成系统的长度单位契约：

1. builder 裸浮点数一律按 SI(m) 解释，不再做“像不像 cm”的隐式猜测；
2. finger preset 为了贴合人工测量习惯，可以显式写 `cm(...)`；
3. mount preset 默认保持 m 直写，但 single-box 这类人工测量锚点仍允许 `cm(...)`。

# NOTE:
这不是纯软件工程上的“风格统一”，而是科研调参可追溯性的要求。
如果 builder 继续偷偷猜单位，那么同一份数值在 review 时就无法唯一解释。
"""

from __future__ import annotations

import math

from assets.builder._utils import _to_si
from assets.presets import get_finger_builder_preset, get_mount_preset


def test_builder_scalar_lengths_are_now_interpreted_as_si_meters():
    r"""builder 裸 float 必须按 SI(m) 直读，不再隐式当成厘米。

    这里直接锁住最核心的契约变化：

    - `2.7` 现在代表 $2.7\text{m}$，而不是偷偷变成 $0.027\text{m}$；
    - `0.027` 仍代表 $0.027\text{m}$；
    - builder 若要吃厘米，应由上游显式写成 `cm(2.7)` 后再传入。
    """

    assert math.isclose(_to_si(2.7), 2.7, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(_to_si(0.027), 0.027, rel_tol=0.0, abs_tol=1e-12)


def test_finger_presets_store_measurement_anchors_via_explicit_cm_helpers():
    r"""finger preset 仍可按厘米心智录入，但落到 cfg 上后应已是米制。

    这里验证的是 preset 层“显式 `cm(...)`、内部结果为 m”的契约。
    """

    allegro = get_finger_builder_preset("allegro_non_thumb_v1")
    thumb = get_finger_builder_preset("allegro_thumb_v1")

    assert math.isclose(allegro.width, 0.027, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(allegro.height, 0.020, rel_tol=0.0, abs_tol=1e-12)
    assert all(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
        for actual, expected in zip(allegro.length, [0.018, 0.054, 0.038, 0.022], strict=True)
    )
    assert math.isclose(allegro.mesh_offsets[2], -0.006, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(allegro.tip["radius"], 0.012, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(allegro.tip["height"], 0.010, rel_tol=0.0, abs_tol=1e-12)

    assert all(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
        for actual, expected in zip(thumb.lengths, [0.045, 0.017, 0.043, 0.040], strict=True)
    )
    assert math.isclose(thumb.cmc1_width, 0.035, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(thumb.cmc1_height, 0.034, rel_tol=0.0, abs_tol=1e-12)
    assert all(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
        for actual, expected in zip(thumb.cmc1_offset, (0.009, 0.0145), strict=True)
    )
    assert all(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
        for actual, expected in zip(thumb.non_cmc1_offset, [-0.002, 0.0, -0.009], strict=True)
    )


def test_mount_presets_keep_meter_defaults_while_allowing_cm_encoded_single_box_data():
    r"""mount preset 应保持“真实 family 默认 m，人工 single-box 可显式 cm(...)”。

    这里同时锁住两类来源：

    - `allegro`：真实 family 锚点，文件里按 m 直写；
    - `single_box_allegro`：人工测量锚点，文件里允许 `cm(...)`。
    """

    allegro = get_mount_preset("allegro")
    single_box = get_mount_preset("single_box_allegro")

    assert math.isclose(allegro["thumb"].pos[0], -0.0182, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(allegro["index"].pos[1], 0.0435, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(single_box["thumb"].pos[0], 0.0245, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(single_box["thumb"].pos[1], 0.0305, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(single_box["index"].pos[0], 0.044, rel_tol=0.0, abs_tol=1e-12)
