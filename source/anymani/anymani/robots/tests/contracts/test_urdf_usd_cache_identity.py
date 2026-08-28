r"""URDF→USD 稳定 cache identity 的纯文件合同；不启动 Isaac Sim。"""

from __future__ import annotations

from pathlib import Path

from anymani.robots.usd_cache import build_urdf_usd_cache_dir


def _write_urdf(root: Path, *, mesh_bytes: bytes = b"mesh-v1") -> Path:
    r"""写一份引用本地 mesh 的最小 URDF，供 dependency hash contract 使用。"""

    mesh_dir = root / "meshes"
    mesh_dir.mkdir(parents=True)
    (mesh_dir / "tip.obj").write_bytes(mesh_bytes)
    urdf_path = root / "hand.urdf"
    urdf_path.write_text(
        '<robot name="cache"><link name="palm"><visual><geometry>'
        '<mesh filename="meshes/tip.obj"/></geometry></visual></link></robot>',
        encoding="utf-8",
    )
    return urdf_path


def _cache_dir(
    urdf_path: Path,
    cache_root: Path,
    *,
    converter_config: dict | None = None,
    isaac_sim_version: str = "5.1.0",
    asset_row: int = 0,
) -> Path:
    r"""以固定 IsaacLab/converter identity 调用纯 cache-key builder。"""

    return build_urdf_usd_cache_dir(
        urdf_path=urdf_path,
        converter_config=converter_config or {"fix_base": True, "merge_fixed_joints": False},
        isaaclab_version="0.54.3",
        isaac_sim_version=isaac_sim_version,
        converter_implementation_sha256="converter-source-v1",
        canonical_identity={"schema_digest": "schema-v1", "asset_row": asset_row},
        cache_root=cache_root,
    )


def test_cache_key_tracks_urdf_mesh_converter_and_isaac_versions(tmp_path: Path) -> None:
    r"""所有会改变 USD 物理内容的输入都必须使 cache directory 改变。"""

    urdf = _write_urdf(tmp_path / "asset")
    baseline = _cache_dir(urdf, tmp_path / "cache")

    urdf.write_text(urdf.read_text(encoding="utf-8").replace('name="cache"', 'name="changed"'), encoding="utf-8")
    changed_urdf = _cache_dir(urdf, tmp_path / "cache")
    urdf.write_text(urdf.read_text(encoding="utf-8").replace('name="changed"', 'name="cache"'), encoding="utf-8")
    (urdf.parent / "meshes" / "tip.obj").write_bytes(b"mesh-v2")
    changed_mesh = _cache_dir(urdf, tmp_path / "cache")
    changed_cfg = _cache_dir(urdf, tmp_path / "cache", converter_config={"fix_base": False})
    changed_sim = _cache_dir(urdf, tmp_path / "cache", isaac_sim_version="5.2.0")

    assert len({baseline, changed_urdf, changed_mesh, changed_cfg, changed_sim}) == 5
    assert baseline.name != ""  # 末级目录是完整 input digest，不是 asset ID 或 selection row


def test_selection_asset_row_does_not_change_physical_usd_key(tmp_path: Path) -> None:
    r"""同一 canonical URDF 在不同 selection row 中必须命中同一个物理 USD cache。"""

    urdf = _write_urdf(tmp_path / "asset")

    assert _cache_dir(urdf, tmp_path / "cache", asset_row=0) == _cache_dir(
        urdf,
        tmp_path / "cache",
        asset_row=2047,
    )
