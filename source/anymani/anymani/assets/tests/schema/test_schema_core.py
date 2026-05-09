from assets.asset_schema_core import (
    BoxGeometryCfg,
    EllipticCylinderGeometryCfg,
    InertialCfg,
    JointPropertiesCfg,
    MeshGeometryCfg,
    PoseCfg,
    make_geometry_cfg,
)


def test_pose_cfg_from_value_supports_vector6():
    pose = PoseCfg.from_value((1, 2, 3, 4, 5, 6))
    assert pose.pos == (1.0, 2.0, 3.0)
    assert pose.rpy == (4.0, 5.0, 6.0)


def test_make_geometry_cfg_supports_box_mesh_and_elliptic_cylinder():
    box = make_geometry_cfg({"type": "box", "size": (0.01, 0.02, 0.03)})
    mesh = make_geometry_cfg("foo/bar.obj")
    elliptic = make_geometry_cfg({"type": "elliptic_cylinder", "radius_x": 0.01, "radius_z": 0.02, "length": 0.03})
    assert isinstance(box, BoxGeometryCfg)
    assert isinstance(mesh, MeshGeometryCfg)
    assert isinstance(elliptic, EllipticCylinderGeometryCfg)


def test_inertial_cfg_from_primitives():
    box = InertialCfg.from_box((0.01, 0.02, 0.03), density=1000.0)
    cylinder = InertialCfg.from_cylinder(0.01, 0.03, density=800.0, principal_axis="x")
    elliptic = InertialCfg.from_elliptic_cylinder(0.01, 0.02, 0.03, density=850.0, principal_axis="y")
    sphere = InertialCfg.from_sphere(0.01, density=900.0)
    assert box.mass > 0.0
    assert cylinder.inertia.ixx > 0.0
    assert elliptic.inertia.iyy > 0.0
    assert sphere.inertia.izz > 0.0


def test_elliptic_cylinder_inertia_degenerates_to_cylinder_when_radii_match():
    r"""当 $r_x=r_z$ 时，椭圆柱惯量应退化回标准圆柱惯量。"""

    cylinder = InertialCfg.from_cylinder(0.01, 0.03, density=800.0, principal_axis="y")
    elliptic = InertialCfg.from_elliptic_cylinder(0.01, 0.01, 0.03, density=800.0, principal_axis="y")
    assert elliptic.mass == cylinder.mass
    assert elliptic.inertia.ixx == cylinder.inertia.ixx
    assert elliptic.inertia.iyy == cylinder.inertia.iyy
    assert elliptic.inertia.izz == cylinder.inertia.izz


def test_joint_properties_cfg_normalizes_optional_friction():
    r"""joint properties 应能表达 LEAP 官方 `<joint_properties friction="..."/>` 语义。"""

    cfg = JointPropertiesCfg(friction="0.0")
    assert cfg.friction == 0.0
