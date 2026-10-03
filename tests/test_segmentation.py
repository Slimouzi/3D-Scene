import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from theta_pipeline import stages
from theta_pipeline.segmentation import LABELS, POLICY, PROMPTS
from theta_pipeline.segmentation import __main__ as gpu
from theta_pipeline.segmentation import fusion
from theta_pipeline.segmentation.environment import pins, qualify
from theta_pipeline.segmentation.faces import coverage, segmentation_faces
from theta_pipeline.segmentation.provenance import code_digests
from theta_pipeline.segmentation.sam3_segmenter import SAM3Segmenter
from theta_pipeline.storage import Run, read, write

WIDTH = 512
GLASS = ((20, 60), (-15, 15))     # longitude, latitude (down positive) in degrees
MIRROR = ((-60, -30), (-10, 20))


def erp_pixels(region):
    (lon0, lon1), (lat0, lat1) = region
    u = lambda lon: int((lon / 360 + .5) * WIDTH)
    v = lambda lat: int((lat / 180 + .5) * WIDTH // 2)
    return slice(v(lat0), v(lat1)), slice(u(lon0), u(lon1))


class FakeSegmenter:
    """Window on blue pixels, mirror on red ones, plus masks the policy must reject."""

    def __init__(self, mode='normal'):
        self.mode, self.reported = mode, False

    def segment(self, image, prompts):
        if self.mode == 'fail':
            raise RuntimeError('CUDA out of memory')
        a = np.asarray(image).astype(int)
        out = {p: [] for p in prompts}
        blue = (a[..., 2] > 200) & (a[..., 0] < 80)
        red = (a[..., 0] > 200) & (a[..., 2] < 80)
        if blue.any() and not (self.mode == 'single' and self.reported):
            out['window'].append((blue, .9))
            self.reported = True
        if red.any() and self.mode != 'single':
            out['mirror'].append((red, .8))
        tiny = np.zeros(a.shape[:2], bool)
        tiny[a.shape[0] // 2, a.shape[1] // 2] = True
        edge = np.zeros(a.shape[:2], bool)
        edge[:2] = True
        out['tripod'].append((tiny, .7))
        out['person'].append((edge, .7))
        return out


class SegmentationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / 'input').mkdir()
        for name in ('a', 'b'):
            erp = np.full((WIDTH // 2, WIDTH, 3), 128, np.uint8)
            erp[erp_pixels(GLASS)] = (0, 0, 255)
            erp[erp_pixels(MIRROR)] = (255, 0, 0)
            erp[-1, -1] = ord(name)  # distinct content: the audit rejects duplicates
            Image.fromarray(erp).save(self.root / f'input/{name}.png')
        self.config = self.root / 'config.json'
        write(self.config, {'schema_version': 1, 'kind': 'diagnostic', 'input': 'input',
            'output': 'runs', 'erp_width': WIDTH, 'num_threads': 1, 'seed': 0,
            'max_features': 128, 'mapping_max_seconds': 10, 'masks': {},
            'auto_mask_backend': 'sam3'})
        self.provenance = {'model': 'facebook/sam3', 'checkpoint_sha256': 'f' * 64,
                           'git_commit': 'test', 'code_sha256': code_digests(),
                           'environment': {'qualified': True}}

    def prepared(self, run):
        run.stage('audit', stages.audit)
        run.stage('prepare', stages.prepare, requires=('audit',))
        run.stage('seg_faces', stages.seg_faces, requires=('prepare',))
        return read(run.path / 'segmentation/faces.json')

    def segment(self, run, faces, pano, mode='normal'):
        return gpu.segment_panorama(run.path, faces, pano, FakeSegmenter(mode), self.provenance)

    def test_faces_overlap_and_cover_zenith_nadir(self):
        faces = segmentation_faces(stages.rotations())
        stats = coverage([f['rotation'] for f in faces], WIDTH // 4)
        self.assertGreaterEqual(stats['min_face_overlap_fraction'], .2)
        self.assertGreaterEqual(stats['min_projections_per_direction'], 2)
        axes = np.array([np.asarray(f['rotation'])[2] for f in faces])
        self.assertTrue((axes[:, 1] < -.9).any() and (axes[:, 1] > .9).any())
        sfm = [f for f in faces if f['role'] == 'sfm']
        for f, r in zip(sfm, stages.rotations()):
            np.testing.assert_allclose(f['rotation'], r)

    def test_sfm_faces_are_reused(self):
        with Run(self.config, 'reuse').locked() as run:
            faces = self.prepared(run)
            entry = faces['panoramas']['a']['faces']
            self.assertEqual(entry['sfm00']['image'], 'prepare/images/pano_camera0/a.png')
            self.assertTrue(entry['pole00']['image'].startswith('segmentation/faces/'))

    def test_trial_then_all_panoramas(self):
        with Run(self.config, 'full').locked() as run:
            faces = self.prepared(run)
            self.assertEqual(faces['trial_panorama'], 'a')
            with self.assertRaisesRegex(RuntimeError, 'seg_trial'):
                gpu.select(run.path, run.state, faces)
            with self.assertRaisesRegex(RuntimeError, 'not the run trial'):
                gpu.select(run.path, run.state, faces, trial='b')
            self.assertEqual(self.segment(run, faces, 'a'), 'OK')
            run.stage('seg_trial', stages.seg_trial, requires=('seg_faces',))
            gate = read(run.path / 'segmentation/trial/trial_gate.json')
            self.assertEqual(gate['decision'], 'PASS', gate['gates'])
            self.assertTrue((run.path / 'segmentation/trial/semantic_masks.json').is_file())
            self.assertEqual(gpu.select(run.path, run.state, faces), ['a', 'b'])
            self.segment(run, faces, 'b')
            run.stage('auto_mask', stages.auto_mask, requires=('seg_trial',))

            masks = read(run.path / 'semantic_masks.json')
            self.assertEqual(masks['status'], 'ACCEPTED')
            self.assertTrue(masks['accepted_masks'])
            self.assertEqual(masks['multiview_panoramas'], {'glass': ['a', 'b'], 'mirror': ['a', 'b']})
            self.assertEqual(set(masks['masks'][0]), {'image_id', 'face_id', 'prompt', 'mask_path',
                'score', 'area_fraction', 'model', 'checkpoint_sha256', 'git_commit'})
            for name in ('mask_consistency.json', 'mask_provenance.json',
                         'navigation_constraints.json', 'semantic_masks_preview/a.jpg'):
                self.assertTrue((run.path / name).is_file(), name)
            consistency = read(run.path / 'mask_consistency.json')['panoramas']['a']
            rejected = {r['reason'] for r in consistency['rejected_masks']}
            self.assertEqual(rejected, {'too_small', 'outside_image'})
            self.assertTrue(all(c['result'] == 'PASS' for c in consistency['checks']))
            self.assertGreater(consistency['metrics']['groups']['glass']['pairwise_iou'], .5)

            artifacts = masks['panoramas'][0]['artifacts']
            geometry = np.asarray(Image.open(run.path / artifacts['geometry_mask']))
            appearance = np.asarray(Image.open(run.path / artifacts['appearance_mask']))
            unknown = np.asarray(Image.open(run.path / artifacts['unknown_mask']))
            labels = np.asarray(Image.open(run.path / artifacts['labels']))
            self.assertEqual(geometry.shape, (WIDTH // 2, WIDTH))
            rows, cols = erp_pixels(GLASS)
            center = ((rows.start + rows.stop) // 2, (cols.start + cols.stop) // 2)
            self.assertEqual(labels[center], LABELS['glass'])
            self.assertEqual(geometry[center], 0)
            self.assertEqual(appearance[center], POLICY['appearance_weight_reflective'])
            self.assertEqual(unknown[center], 0)
            self.assertEqual(geometry[5, 5], 255)
            self.assertEqual(geometry[-5, 5], 255)

            gates = stages.semantic_gates(run, masks, {'input_panoramas': 2})
            self.assertEqual({g['result'] for g in gates}, {'PASS'})
            navigation = read(run.path / 'navigation_constraints.json')
            self.assertEqual(navigation['status'], 'advisory')
            self.assertFalse(navigation['navigation_validated'])

        # New SfM experiment: geometry masks come from the ACCEPTED semantic run.
        config = json.loads(self.config.read_text())
        config['semantic_run'] = 'full'
        masked_config = self.root / 'masked.json'
        write(masked_config, config)
        with Run(masked_config, 'masked').locked() as run:
            self.assertEqual(run.mask_paths['a']['geometry'],
                             (self.root / 'runs/full/segmentation/fused/a/geometry_mask.png').resolve())
            run.stage('audit', stages.audit)
            run.stage('auto_mask', stages.auto_mask, requires=('audit',))
            run.stage('prepare', stages.prepare, requires=('audit',))
            self.assertEqual(read(run.path / 'semantic_masks.json')['status'], 'ACCEPTED')
            excluded = [np.asarray(Image.open(p)).min() for p in
                        (run.path / 'prepare/masks/geometry').rglob('*.png')]
            self.assertIn(0, excluded)

    def test_single_projection_glass_is_unknown_never_accepted(self):
        with Run(self.config, 'single').locked() as run:
            faces = self.prepared(run)
            self.segment(run, faces, 'a', mode='single')
            run.stage('seg_trial', stages.seg_trial, requires=('seg_faces',))
            gate = read(run.path / 'segmentation/trial/trial_gate.json')
            self.assertNotEqual(gate['decision'], 'PASS')
            self.assertEqual(gate['metrics']['groups']['glass']['validated_fraction'], 0)
            self.assertGreater(gate['metrics']['groups']['unknown_glass']['fraction'], 0)
            labels = np.asarray(Image.open(run.path / 'segmentation/trial/a/labels.png'))
            unknown = np.asarray(Image.open(run.path / 'segmentation/trial/a/unknown_mask.png'))
            geometry = np.asarray(Image.open(run.path / 'segmentation/trial/a/geometry_mask.png'))
            glassy = labels == LABELS['unknown_glass']
            self.assertTrue(glassy.any())
            self.assertFalse((labels == LABELS['glass']).any())
            self.assertTrue((unknown[glassy] == 255).all())
            # Unvalidated isolated candidates keep their texture for SfM...
            self.assertTrue((geometry[glassy] == 255).all())
            # ...but are blocked for navigation.
            navigation = read(run.path / 'segmentation/trial/navigation_constraints.json')
            self.assertIn('unknown_glass', navigation['blocked_unknown_labels'])
            with self.assertRaisesRegex(RuntimeError, 'refused'):
                gpu.select(run.path, run.state, faces)

    def test_failed_inference_produces_no_default_mask(self):
        with Run(self.config, 'failed').locked() as run:
            faces = self.prepared(run)
            self.assertEqual(self.segment(run, faces, 'a', mode='fail'), 'FAILED')
            run.stage('seg_trial', stages.seg_trial, requires=('seg_faces',))
            gate = read(run.path / 'segmentation/trial/trial_gate.json')
            self.assertEqual(gate['decision'], 'FAIL')
            self.assertFalse((run.path / 'segmentation/trial/a').exists())
            masks = read(run.path / 'segmentation/trial/semantic_masks.json')
            self.assertEqual(masks['status'], 'REJECTED')
            self.assertIsNone(masks['panoramas'][0]['artifacts'])

    def test_unavailable_model_is_unknown(self):
        with Run(self.config, 'unavailable').locked() as run:
            self.prepared(run)
        # This environment is not the qualified GPU one: the runner must report UNKNOWN.
        args = ['--run', str(run.path), '--config', str(self.config)]
        self.assertEqual(gpu.main(args + ['--trial', 'a']), 2)
        raw = read(run.path / 'segmentation/raw/a/masks.json')
        self.assertEqual(raw['status'], 'UNKNOWN')
        self.assertFalse(raw['masks'])
        self.assertFalse(raw['provenance']['environment']['qualified'])
        # --all is refused after an UNKNOWN trial.
        self.assertEqual(gpu.main(args + ['--all']), 1)
        self.assertFalse((run.path / 'segmentation/raw/b').exists())
        with Run(self.config, 'unavailable').locked() as run:
            with self.assertRaisesRegex(RuntimeError, 'UNKNOWN'):
                run.stage('seg_trial', stages.seg_trial, requires=('seg_faces',))
            gate = read(run.path / 'segmentation/trial/trial_gate.json')
            self.assertEqual(gate['decision'], 'UNKNOWN')
            self.assertEqual(run.state['stages']['seg_trial']['status'], 'failed')
            masks = read(run.path / 'segmentation/trial/semantic_masks.json')
            self.assertEqual(masks['status'], 'UNKNOWN')
            self.assertFalse(masks['accepted_masks'])
            self.assertFalse((run.path / 'segmentation/trial/a').exists())

    def test_gpu_lock_pins_and_mac_is_not_evidence(self):
        expected = pins()
        self.assertEqual(expected['torch'], '2.14.1')
        self.assertEqual(expected['triton'], '3.8.0')
        self.assertEqual(expected['cuda'], '13.0')
        self.assertRegex(expected['sam3_commit'], '^[0-9a-f]{40}$')
        self.assertRegex(expected['python'], r'^3\.12\.\d+$')
        if not qualify()['found'].get('cuda_available'):
            self.assertFalse(qualify()['qualified'])

    def test_masked_experiment_does_not_reuse_provisional_partition(self):
        class Masked:
            config = {'semantic_run': 'seg'}
            path = self.root
            sources = []

            def require(self, name):
                pass
        write(self.root / 'quality.json', {})
        write(self.root / 'gate_results.json', {'permissions': {'research_training': 'PASS'}})
        stages.partition(Masked())
        split = read(self.root / 'split.json')
        self.assertEqual(split['status'], 'not_proposed')
        self.assertIsNone(split['train'])

    def test_adapter_refuses_full_erp(self):
        with self.assertRaisesRegex(ValueError, 'ERP'):
            SAM3Segmenter.segment(None, Image.new('RGB', (8, 4)), PROMPTS)

    def test_pairwise_iou_and_seam_dilation(self):
        n = np.array([2, 2, 3])
        self.assertEqual(fusion.pairwise_iou(n, np.array([2, 0, 3])), 1.)
        self.assertEqual(fusion.pairwise_iou(n, np.array([1, 1, 0])), 0.)
        mask = np.zeros((8, 16), bool)
        mask[4, 0] = True
        self.assertTrue(fusion.wrap_dilate(mask, 1)[4, -1])

    def test_erp_directions_invert_projection(self):
        from theta_pipeline.geometry import erp_coordinates
        d = fusion.erp_directions(64)
        u, v = erp_coordinates(d, 64, 32)
        np.testing.assert_allclose(u, np.broadcast_to(np.arange(64), u.shape), atol=1e-4)
        np.testing.assert_allclose(v, np.broadcast_to(np.arange(32)[:, None], v.shape), atol=1e-4)


if __name__ == '__main__':
    unittest.main()
