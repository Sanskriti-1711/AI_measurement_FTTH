import unittest
import numpy as np
import trimesh
from src.measurers import measure_rectangular, measure_circular

class TestMeasurers(unittest.TestCase):
    def test_measure_rectangular(self):
        # Create a simple box 1.0 x 0.5 x 0.2
        mesh = trimesh.creation.box(extents=[1.0, 0.5, 0.2])
        # Move it so Z is negative
        mesh.apply_translation([0, 0, -0.1])
        dims = measure_rectangular(mesh)
        self.assertAlmostEqual(dims['length'], 1.0, places=1)
        self.assertAlmostEqual(dims['width'], 0.5, places=1)

    def test_measure_circular(self):
        # Create an annulus (ring) which is more representative of a duct surface
        mesh = trimesh.creation.annulus(r_min=0.49, r_max=0.5, height=0.2)
        mesh.apply_translation([0, 0, -0.1])
        dims = measure_circular(mesh)
        # Diameter should be approx 1.0 (outer dia)
        self.assertAlmostEqual(dims['diameter'], 1.0, places=1)

if __name__ == '__main__':
    unittest.main()
