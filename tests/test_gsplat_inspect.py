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


    def test_sizes_are_normalized_by_scene_scale(self):
        scene_scale = 4.
        scales = np.array([[.2] * 3, [.3, .1, .1], [1.] * 3])      # max axes .2, .3, 1.0
        stats = gi.gaussian_statistics(np.zeros((3, 3)), scales, np.full(3, .9),
                                       np.array([[-1, -1, -1], [1, 1, 1]], float), scene_scale, .1)
        quantiles = stats['max_scale_over_scene_scale_quantiles']
        self.assertAlmostEqual(quantiles['max'], 1. / scene_scale)
        self.assertAlmostEqual(quantiles['p50'], .3 / scene_scale)
        # Threshold .1 x 4 = .4: only the 1.0 Gaussian is large (0.3 would be large if unnormalized).
        self.assertEqual(stats['larger_than_prune_scale3d'], 1)
        np.testing.assert_array_equal(gi.large_gaussians(scales, scene_scale, .1), [False, False, True])
        self.assertEqual(stats['scene_scale'], scene_scale)

    def test_train_face_selection_by_glass_and_furniture(self):
        from theta_pipeline.segmentation import LABELS
        labels = np.zeros((64, 128), np.uint8)
        labels[:, 60:68] = LABELS['glass']                   # straight ahead (front faces)
        labels[:, :8] = labels[:, -8:] = LABELS['furniture']  # behind (seam)
        rotations = {f'f{k}': r for k, r in enumerate(cube_rotations().values())}
        cameras = [{'name': n, 'panorama_id': 'p'} for n in sorted(rotations)]
        chosen, fractions = gi.select_train_faces(cameras, lambda pano: labels, rotations, 16, per_group=1)
        front, back = 'f0', 'f2'                             # cube order: front, right, back, ...
        self.assertEqual(chosen, [front, back])
        self.assertGreater(fractions[front]['glass_windows'], 0)
        self.assertGreater(fractions[back]['furniture'], 0)

    def test_summary_excludes_empty_faces(self):
        empty = {'pixels': 0, 'psnr': None, 'mean_abs_error': None}
        regions = lambda v: {k: ({'pixels': 1, 'psnr': v, 'mean_abs_error': 0} if k == 'glass' else empty)
                             for k in [*gi.REGIONS, 'contours', 'other']}
        faces = [{'excluded': False, 'psnr': 20., 'ssim': .8, 'regions': regions(15.)},
                 {'excluded': True, 'psnr': None, 'ssim': None, 'regions': regions(None)}]
        summary = gi.summarize(faces)
        self.assertEqual((summary['usable_faces'], summary['mean_psnr']), (1, 20.))
        self.assertEqual(summary['mean_region_psnr']['glass'], 15.)
        self.assertIsNone(summary['mean_region_psnr']['mirror'])


class CameraCheckTests(unittest.TestCase):
    def camera(self):
        from theta_pipeline.geometry import homogeneous
        rotation = cube_rotations()['right']
        return {'K': [[8., 0, 8], [0, 8, 8], [0, 0, 1]],
                'world_to_camera': homogeneous(rotation, np.array([.1, -.2, .3])).tolist(),
                'width': 16, 'height': 16}

    def test_exact_observations_have_zero_residual(self):
        from theta_pipeline import gsplat_camera_check as cc
        camera = self.camera()
        rng = np.random.default_rng(0)
        T = np.asarray(camera['world_to_camera'])
        cam_points = np.column_stack((rng.uniform(-1, 1, (20, 2)), rng.uniform(2, 4, 20)))
        world = (cam_points - T[:3, 3]) @ T[:3, :3]           # inverse rigid transform
        points = {k: {'xyz': world[k].tolist()} for k in range(20)}
        uv, depth = cc.project(camera['K'], camera['world_to_camera'], world)
        self.assertTrue((depth > 0).all())
        observations = [(k, tuple(uv[k])) for k in range(20)]
        errors, behind = cc.residuals(camera, observations, points)
        self.assertLess(errors.max(), 1e-9)
        self.assertEqual(behind, 0)
        shifted = [(k, (x + .5, y + .5)) for k, (x, y) in observations]
        self.assertAlmostEqual(float(np.median(cc.residuals(camera, shifted, points)[0])), np.sqrt(.5), places=9)
        points[0] = {'xyz': (-world[0] + 2 * (-T[:3, :3].T @ T[:3, 3])).tolist()}   # mirrored behind
        self.assertEqual(cc.residuals(camera, observations, points)[1], 1)
        self.assertEqual(cc.rigid(camera['world_to_camera'])[1], 1.0)
        image = cc.overlay(np.zeros((16, 16, 3), np.uint8), observations[1:3], points, camera)
        self.assertEqual(image.size, (16, 16))

if __name__ == '__main__':
    unittest.main()
