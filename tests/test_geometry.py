import unittest
import numpy as np
from theta_pipeline.geometry import rays, erp_coordinates, cube_rotations, project, homogeneous, ownership
from theta_pipeline.stages import component_center_path, rotations
from pycolmap import panorama


class GeometryTests(unittest.TestCase):
    def test_component_center_path_handles_empty_and_single_center(self):
        empty = component_center_path([], ['a'], 0)
        self.assertEqual(empty['panorama_order'], [])
        self.assertIsNone(empty['coordinate_y_span'])
        self.assertIsNone(empty['consecutive_distance_range'])

        records = [{'panorama_id': 'a', 'component': 0, 'status': 'registered',
                    'center_world': [1., 2., 3.]}]
        single = component_center_path(records, ['a'], 0)
        self.assertEqual(single['panorama_order'], ['a'])
        self.assertEqual(single['coordinate_y_span'], 0.)
        self.assertIsNone(single['consecutive_distance_median'])

    def test_component_center_path_never_connects_components(self):
        records = [
            {'panorama_id': 'a', 'component': 0, 'status': 'registered',
             'center_world': [0., 0., 0.]},
            {'panorama_id': 'b', 'component': 1, 'status': 'registered',
             'center_world': [1000., 0., 0.]},
        ]
        first = component_center_path(records, ['a', 'b'], 0)
        second = component_center_path(records, ['a', 'b'], 1)
        self.assertEqual(first['panorama_order'], ['a'])
        self.assertIsNone(first['consecutive_distance_range'])
        self.assertEqual(second['panorama_order'], ['b'])

    def test_full_resolution_ownership(self):
        # Regression: large batched matmul crashed Accelerate on macOS 15.3.
        rr = rotations()
        mask = ownership(rr[0], rr, 1024, 0)
        self.assertEqual(mask.shape, (1024, 1024))
        self.assertTrue((mask > 0).any())

    def test_cardinal_erp_coordinates(self):
        xyz = np.array([[0, 0, 1], [1, 0, 0], [-1, 0, 0], [0, -1, 0], [0, 1, 0]])
        u, v = erp_coordinates(xyz, 400, 200)
        np.testing.assert_allclose(u[:3], [199.5, 299.5, 99.5])
        np.testing.assert_allclose(v, [99.5, 99.5, 99.5, 0, 199])

    def test_matches_colmap_spherical_convention(self):
        dirs = rays(17).reshape(-1, 3) @ rotations()[8]
        u, v = erp_coordinates(dirs, 4096, 2048)
        upstream = panorama.spherical_img_from_cam((4096, 2048), dirs) - .5
        np.testing.assert_allclose(u, upstream[:, 0] % 4096, atol=.001)
        np.testing.assert_allclose(v, np.clip(upstream[:, 1], 0, 2047), atol=.001)

    def test_cube_covers_axes_with_common_center(self):
        rr = cube_rotations()
        axes = np.array([r.T @ np.array([0, 0, 1]) for r in rr.values()])
        self.assertEqual(len(set(map(tuple, axes))), 6)
        for r in rr.values():
            np.testing.assert_allclose(r @ r.T, np.eye(3))
            self.assertAlmostEqual(np.linalg.det(r), 1)
            np.testing.assert_allclose(homogeneous(r)[:3, 3], 0)
        # Every sphere ray lies inside at least one cube face.
        dirs = np.random.default_rng(42).normal(size=(2000, 3))
        covered = np.zeros(len(dirs), bool)
        for r in rr.values():
            face = dirs @ r.T
            covered |= (face[:, 2] > 0) & (np.abs(face[:, 0]) <= face[:, 2]) & (np.abs(face[:, 1]) <= face[:, 2])
        self.assertTrue(covered.all())

    def test_rig_reference_conversion(self):
        rr = rotations()
        world = homogeneous(cube_rotations()['right'], np.array([1., 2., 3.]))
        ref = homogeneous(rr[0]) @ world
        np.testing.assert_allclose(homogeneous(rr[0].T) @ ref, world, atol=1e-14)
        center = np.linalg.inv(world)[:3, 3]
        for r in rr:
            face = homogeneous(r @ rr[0].T) @ ref
            np.testing.assert_allclose(np.linalg.inv(face)[:3, 3], center, atol=1e-14)

    def test_projection_and_mask_seam(self):
        rgb = np.full((100, 200, 3), [22, 88, 199], dtype=np.uint8)
        result = project(rgb, cube_rotations()['back'], 32)
        np.testing.assert_array_equal(result, np.broadcast_to([22, 88, 199], result.shape))
        mask = np.full((100, 200), 255, np.uint8)
        mask[:, :10] = 0
        out = project(mask, cube_rotations()['back'], 32, mask=True)
        self.assertTrue((out == 0).any())
        self.assertTrue((out == 255).any())
        self.assertEqual(set(np.unique(out)), {0, 255})

    def test_ownership_assigns_to_nearest_axis(self):
        rr = rotations()
        for i, r in enumerate(rr):
            selected = ownership(r, rr, 32, i) > 0
            scores = (rays(32) @ r) @ np.array(rr)[:, 2, :].T
            self.assertTrue(np.all(scores[..., i][selected] >= np.max(scores, axis=-1)[selected] - 1e-12))


if __name__ == '__main__':
    unittest.main()
