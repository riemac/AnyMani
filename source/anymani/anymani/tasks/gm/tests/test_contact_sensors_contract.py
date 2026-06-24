r"""Pure sidecar contract tests for GM contact sensor layout.

这些测试不启动 Isaac Sim，也不导入 `inhand_env_cfg.py`。它们只验证
`tasks/gm/contact_sensors.py` 对 `hand.yaml -> hand_cfg` 的科研合同：contact topology
来自 sidecar 的 palm / joint child / `is_tip`，而不是 env cfg 中的固定四指名称。
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any

import pytest
import yaml
from anymani.assets.bank.path_utils import resolve_anymani_root
from anymani.tasks.gm.contact_sensors import (
    build_contact_sensor_layout_from_assets,
    build_contact_sensor_layout_from_sidecar,
    make_contact_sensor_cfg,
)


@dataclass(frozen=True)
class _FakeAsset:
    r"""测试用 resolved hand asset，只保留 contact layout 需要的两个字段。"""

    asset_id: str
    """fake asset id；用于 strict validation 报错定位。"""

    sidecar: dict[str, Any]
    """fake hand sidecar；结构对齐真实 `HandContainer.sidecar`。"""


def _real_default_sidecar() -> dict[str, Any]:
    r"""读取当前 GM 默认 post-mutate run 中的一个真实 hand sidecar。"""

    repo_root = resolve_anymani_root()
    sidecar_path = repo_root / (
        "source/anymani/anymani/assets/generated/2026-06-10_11-30-08/"
        "single_palm_leap/right_t4_i4_m4_r4/2026-06-11_14-20-22/0b6fbfce/hand.yaml"
    )
    return yaml.safe_load(sidecar_path.read_text(encoding="utf-8"))  # 真实 sidecar，含完整 `hand_cfg`


def _minimal_sidecar(*, palm: str = "core", tips: tuple[str, ...] = ("solo_tip",)) -> dict[str, Any]:
    r"""构造非四指 topology 的最小 sidecar，用于验证 layout 不依赖固定 finger 名称。"""

    joints: list[dict[str, Any]] = [
        {"name": "root", "child": "root_link", "is_tip": False},
    ]
    for tip_index, tip_link in enumerate(tips):
        joints.append({"name": f"tip_{tip_index}", "child": tip_link, "is_tip": True})

    return {
        "hand_cfg": {
            "palm": {"name": palm},
            "fingers": [
                {
                    "name": "noncanonical_finger",
                    "joints": joints,
                }
            ],
        }
    }


def test_contact_layout_parses_real_default_hand_cfg_tip_and_non_tip_links() -> None:
    r"""真实 default sidecar 应能推导 fingertip、palm 与 non-tip links。"""

    layout = build_contact_sensor_layout_from_sidecar(_real_default_sidecar(), asset_id="0b6fbfce")

    assert layout.palm_link_name == "palm"
    assert layout.finger_link_chains == (
        ("index_root_fixed_link", "index_mcp1", "index_mcp2", "index_pip", "index_dip", "index_tip"),
        ("middle_root_fixed_link", "middle_mcp1", "middle_mcp2", "middle_pip", "middle_dip", "middle_tip"),
        ("ring_root_fixed_link", "ring_mcp1", "ring_mcp2", "ring_pip", "ring_dip", "ring_tip"),
        ("thumb_cmc1", "thumb_cmc2", "thumb_mcp", "thumb_dip", "thumb_tip"),
    )
    assert layout.fingertip_link_names == ("index_tip", "middle_tip", "ring_tip", "thumb_tip")
    assert layout.fingertip_sensor_names == tuple(f"contact_{link_name}" for link_name in layout.fingertip_link_names)
    assert layout.non_tip_link_names[0] == "palm"  # palm bad-contact sensor 必须显式存在
    assert "index_mcp1" in layout.non_tip_link_names  # 非 tip revolute link 应进入 bad-contact 集合
    assert "index_tip" not in layout.non_tip_link_names  # fingertip 不应同时被算作 non-tip penalty


def test_contact_layout_accepts_noncanonical_finger_names_and_counts() -> None:
    r"""layout 应只依赖 `is_tip` 与 child link，不依赖 index/middle/ring/thumb 固定四指。"""

    layout = build_contact_sensor_layout_from_sidecar(
        _minimal_sidecar(palm="central_palm", tips=("alpha_tip", "beta_tip")),
        asset_id="two_tip_fixture",
    )

    assert layout.palm_link_name == "central_palm"
    assert layout.finger_link_chains == (("root_link", "alpha_tip", "beta_tip"),)
    assert layout.fingertip_link_names == ("alpha_tip", "beta_tip")
    assert layout.non_tip_link_names == ("central_palm", "root_link")
    assert layout.fingertip_sensor_names == ("contact_alpha_tip", "contact_beta_tip")
    assert layout.non_tip_sensor_names == ("contact_central_palm", "contact_root_link")


def test_contact_layout_strict_validation_fails_on_mismatched_assets() -> None:
    r"""`validate_all_assets=True` 应在 selected assets contact topology 不一致时 fail fast。"""

    assets = (
        _FakeAsset("one_tip", _minimal_sidecar(tips=("solo_tip",))),
        _FakeAsset("two_tip", _minimal_sidecar(tips=("solo_tip", "extra_tip"))),
    )

    with pytest.raises(ValueError, match="do not share the same GM contact layout"):
        build_contact_sensor_layout_from_assets(assets, validate_all_assets=True)


def test_contact_layout_default_validation_reads_first_asset_only() -> None:
    r"""默认 `validate_all_assets=False` 应返回首个 asset layout，不扫描整个 selection。"""

    assets = (
        _FakeAsset("first", _minimal_sidecar(tips=("first_tip",))),
        _FakeAsset("mismatch", _minimal_sidecar(tips=("other_tip", "extra_tip"))),
    )

    layout = build_contact_sensor_layout_from_assets(assets, validate_all_assets=False)

    assert layout.source_asset_id == "first"
    assert layout.fingertip_link_names == ("first_tip",)


def test_contact_sensor_cfg_uses_single_link_object_filter() -> None:
    r"""每个 generated sensor cfg 应绑定单个 robot link，并只过滤到 object prim。"""

    class _ContactSensorCfg:
        r"""IsaacLab `ContactSensorCfg` 的最小测试替身。"""

        def __init__(self, **kwargs):
            r"""保存所有 keyword 字段，模拟 configclass 的开放属性语义。"""

            self.__dict__.update(kwargs)

    isaaclab = types.ModuleType("isaaclab")
    sensors = types.ModuleType("isaaclab.sensors")
    sensors.ContactSensorCfg = _ContactSensorCfg
    previous_isaaclab = sys.modules.get("isaaclab")
    previous_sensors = sys.modules.get("isaaclab.sensors")
    sys.modules["isaaclab"] = isaaclab
    sys.modules["isaaclab.sensors"] = sensors
    try:
        cfg = make_contact_sensor_cfg("solo_tip", debug_vis=True)
    finally:
        if previous_isaaclab is None:
            sys.modules.pop("isaaclab", None)
        else:
            sys.modules["isaaclab"] = previous_isaaclab
        if previous_sensors is None:
            sys.modules.pop("isaaclab.sensors", None)
        else:
            sys.modules["isaaclab.sensors"] = previous_sensors

    assert cfg.prim_path == "{ENV_REGEX_NS}/Robot/solo_tip"
    assert cfg.filter_prim_paths_expr == ["{ENV_REGEX_NS}/object"]
    assert cfg.track_friction_forces is True
    assert cfg.debug_vis is True
