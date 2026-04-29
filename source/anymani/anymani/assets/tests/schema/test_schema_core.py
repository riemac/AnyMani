from assets.asset_schema_core import (
    BoxGeometryCfg,
    InertialCfg,
    MeshGeometryCfg,
    PoseCfg,
    make_geometry_cfg,
)


def test_pose_cfg_from_value_supports_vector6():
    pose = PoseCfg.from_value((1, 2, 3, 4, 5, 6))
    assert pose.pos == (1.0, 2.0, 3.0)
    assert pose.rpy == (4.0, 5.0, 6.0)


def test_make_geometry_cfg_supports_box_and_mesh():
    box = make_geometry_cfg({"type": "box", "size": (0.01, 0.02, 0.03)})
    mesh = make_geometry_cfg("foo/bar.obj")
    assert isinstance(box, BoxGeometryCfg)
    assert isinstance(mesh, MeshGeometryCfg)


def test_inertial_cfg_from_primitives():
    box = InertialCfg.from_box((0.01, 0.02, 0.03), density=1000.0)
    cylinder = InertialCfg.from_cylinder(0.01, 0.03, density=800.0, principal_axis="x")
    sphere = InertialCfg.from_sphere(0.01, density=900.0)
    assert box.mass > 0.0
    assert cylinder.inertia.ixx > 0.0
    assert sphere.inertia.izz > 0.0
