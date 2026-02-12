import trimesh

from src.utils.mesh_io import get_units


def test_get_units_respects_millimeter_metadata():
    mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    unit = get_units("dummy.obj", {"Unit": "millimeter"}, mesh)
    assert unit == "mm"


def test_get_units_respects_centimeter_metadata():
    mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    unit = get_units("dummy.obj", {"Unit": "centimeter"}, mesh)
    assert unit == "cm"
