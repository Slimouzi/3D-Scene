import unittest
import numpy as np
from theta_pipeline import gsplat_inspect as gi
from theta_pipeline.geometry import cube_rotations, erp_coordinates, rays


class InspectionTests(unittest.TestCase):
    def test_label_projection_matches_geometry_convention(self):
        labels = (np.arange(64 * 128).reshape(64, 128) % 251).astype(np.uint8)
        for rotation in cube_rotations().values():      # includes the seam (back) and both poles
            u, v = erp_coordinates(rays(16) @ rotation, 128, 64)
            expected = labels[np.clip(np.floor(v + .5).astype(int), 0, 63), np.floor(u + .5).astype(int) % 128]
            np.testing.assert_array_equal(gi.project_labels(labels, rotation, 16), expected)

    def test_region_metrics_and_empty_regions(self):
        reference = np.full((8, 8, 3), 100, np.uint8)
        rendered = reference.copy()
        rendered[:, :4] = 110
        weight = np.ones((8, 8))
        weight[0] = 0
        left = np.zeros((8, 8), bool)
        left[:, :4] = True
        rows = gi.region_metrics(reference, rendered, weight, {'left': left, 'right': ~left,
                                                               'none': np.zeros_like(left)})
        self.assertAlmostEqual(rows['left']['mean_abs_error'], 10 / 255)
        self.assertEqual(rows['left']['pixels'], 28)            # excluded row not counted
        self.assertEqual(rows['right']['mean_abs_error'], 0)
        self.assertEqual(rows['none'], {'pixels': 0, 'psnr': None, 'mean_abs_error': None})

    def test_error_map_marks_excluded_pixels(self):
        reference = np.zeros((4, 4, 3), np.uint8)
        rendered = np.full((4, 4, 3), 255, np.uint8)
        weight = np.ones((4, 4))
        weight[0, 0] = 0
        image = gi.error_image(reference, rendered, weight)
        self.assertEqual(tuple(image[0, 0]), (38, 51, 140))
        self.assertEqual(tuple(image[1, 1]), (255, 229, 0))

    def test_contours_are_relative_and_within_valid(self):
        reference = np.zeros((20, 20, 3), np.uint8)
        reference[:, 10:] = 200
        valid = np.ones((20, 20), bool)
        valid[:, 9] = False
        mask = gi.contour_mask(reference, valid)
        self.assertTrue(mask[:, 10].all() and not mask[:, 9].any() and not mask[:, :8].any())

    def test_gaussian_statistics(self):
        means = np.array([[0, 0, 0], [5, 5, 5], [.5, .5, .5]], float)
        scales = np.array([[.01] * 3, [.5] * 3, [.02] * 3])
        stats = gi.gaussian_statistics(means, scales, np.array([.9, .9, .1]),
                                       np.array([[0, 0, 0], [1, 1, 1]], float), 1., .1)
        self.assertEqual(stats['count'], 3)
        self.assertEqual(stats['larger_than_prune_scale3d'], 1)
        self.assertEqual(stats['outside_sfm_points_box_plus_25pct'], 1)
        self.assertEqual(stats['outside_and_opaque_over_0_5'], 1)


if __name__ == '__main__':
    unittest.main()
