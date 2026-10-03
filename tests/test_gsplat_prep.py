import unittest
import numpy as np
from PIL import Image
from test_split import CENTERS, SplitFixture, pano_color, views
from theta_pipeline import gsplat_prep, gsplat_preflight, split, stages
from theta_pipeline.storage import Run, read, write


class ColorTests(unittest.TestCase):
    def test_colors_use_only_train_observations(self):
        points = {1: {'xyz': [0, 0, 0], 'track': [('train/a.png', (1, 1)), ('held/b.png', (2, 2)),
                                                    ('train/c.png', (3, 3))]},
                  2: {'xyz': [1, 1, 1], 'track': [('held/b.png', (1, 1))]}}
        sampled = []

        def sample(name, xy):
            sampled.append(name)
            return {'train/a.png': [255, 0, 0], 'train/c.png': [0, 0, 255], 'held/b.png': [0, 255, 0]}[name]
        result = gsplat_prep.train_colors(points, [1], {'train/a.png', 'train/c.png'}, sample)
        self.assertEqual(sorted(sampled), ['train/a.png', 'train/c.png'])
        np.testing.assert_allclose(result['rgb'], [[.5, 0, .5]])
        self.assertEqual((result['train_observations'], result['heldout_observations_ignored']), (2, 1))
        with self.assertRaisesRegex(RuntimeError, 'no train observation'):
            gsplat_prep.train_colors(points, [2], {'train/a.png'}, sample)

    def test_bilinear_sampling_uses_colmap_pixel_centers(self):
        image = np.zeros((4, 4, 3))
        image[1, 2] = [100, 50, 25]
        np.testing.assert_allclose(gsplat_prep.bilinear_rgb(image, (2.5, 1.5)), [100, 50, 25])
        np.testing.assert_allclose(gsplat_prep.bilinear_rgb(image, (3.0, 1.5)), [50, 25, 12.5])

    def test_face_weights_zero_on_excluded_support(self):
        erp = np.full((64, 128), 255, np.uint8)
        erp[:, 60:68] = 0                       # excluded strip facing forward
        erp[:, 30:40] = 128                     # reflective weight elsewhere
        weights = gsplat_prep.face_weights(erp, np.eye(3), 16)
        self.assertEqual(weights[8, 8], 0)
        self.assertEqual(weights[8, 0], 255)
        self.assertTrue((weights[:, 7:9] == 0).all())


class SourceTests(unittest.TestCase):
    sets = {'train': ['a', 'b', 'c'], 'validation': ['d'], 'test': ['e']}

    def record(self, **changes):
        return {**self.sets, 'status': 'frozen', 'partition_sha256': split.partition_sha256(self.sets),
                'sfm_run': 'sfm', 'semantic_run': 'sem', **changes}

    def config(self, pinned):
        return {'sfm_run': 'sfm', 'semantic_run': 'sem', 'split': {'expected_partition_sha256': pinned}}

    gates = {'gsplat_allowed': {'evaluated': True}, 'permissions': {'evaluated_training': 'PASS'}}

    def test_only_a_frozen_pinned_permitted_split_is_accepted(self):
        pinned = split.partition_sha256(self.sets)
        self.assertEqual(split.verify_split_source(self.record(), self.gates, self.config(pinned)), [])
        cases = [
            (self.record(status='rejected'), self.gates, self.config(pinned), 'not frozen'),
            (self.record(), self.gates, self.config(None), 'not pinned'),
            (self.record(), self.gates, self.config('0' * 64), 'pinned'),
            (self.record(test=['a']), self.gates, self.config(pinned), 'recorded partition hash'),
            (self.record(), {'gsplat_allowed': {'evaluated': False}}, self.config(pinned), 'evaluated'),
            (self.record(sfm_run='other'), self.gates, self.config(pinned), 'sfm_run'),
        ]
        for record, gates, config, message in cases:
            problems = split.verify_split_source(record, gates, config)
            self.assertTrue(any(message in p for p in problems), (message, problems))

    def test_lock_pins_and_mac_is_not_evidence(self):
        pins = gsplat_preflight.pins()
        self.assertEqual(pins['torch'], '2.4.1+cu124')
        self.assertEqual(pins['gsplat'], '1.5.3+pt24cu124')
        self.assertEqual(pins['cuda'], '12.4')
        self.assertRegex(pins['python'], r'^3\.10\.\d+$')
        self.assertFalse(gsplat_preflight.qualify()['qualified'])


class PrepareTests(SplitFixture):
    """Frozen split -> pinned gsplat experiment -> train-only inputs -> start-of-training checks."""

    def fake_model(self):
        names = [v['sfm_name'] for v in views(CENTERS)]
        images = {n: {'width': 8, 'height': 8, 'K': [[4, 0, 4], [0, 4, 4], [0, 0, 1]],
                      'world_to_camera': np.eye(4).tolist()} for n in names}
        record = read(self.root / 'runs/exp/split.json')
        train, held = record['train'][0], record['test'][0]
        points = {1: {'xyz': [0, 0, 1], 'track': [(f'pano_camera0/{train}.png', (4, 4)),
                                                   (f'pano_camera3/{held}.png', (4, 4))]},
                  2: {'xyz': [1, 0, 1], 'track': [(f'pano_camera1/{p}.png', (4, 4)) for p in self.tracks[2]]}}
        return {'images': images, 'points': points}

    def prepared(self, pinned=True, name='gs'):
        with Run(self.config, 'exp').locked() as run:
            self.run_split(run)
        record = read(self.root / 'runs/exp/split.json')
        self.assertEqual(record['status'], 'frozen', record['checks'])
        config = {**self.base, 'semantic_run': 'sem', 'sfm_run': 'sfm', 'split_run': 'exp',
                  'split': {'expected_partition_sha256': record['partition_sha256'] if pinned is True else pinned}}
        write(self.root / f'{name}.json', config)
        original = gsplat_prep.load_model
        gsplat_prep.load_model = lambda path: self.fake_model()
        self.addCleanup(lambda: setattr(gsplat_prep, 'load_model', original))
        run = Run(self.root / f'{name}.json', name)
        with run.locked():
            run.stage('audit', stages.audit)
            run.stage('import_split', gsplat_prep.import_split, requires=('audit',))
            run.stage('gsplat_prepare', gsplat_prep.gsplat_prepare, requires=('import_split',))
        return run, record

    def test_prepared_inputs_are_train_only_and_verified(self):
        run, record = self.prepared()
        target = run.path / 'gsplat_inputs'
        train = read(target / 'cameras_train.json')['cameras']
        self.assertEqual(len(train), 12 * len(record['train']))
        self.assertTrue(all(c['set'] == 'train' and c['panorama_id'] in record['train'] for c in train))
        for group in ('validation', 'test'):
            cameras = read(target / f'cameras_{group}.json')['cameras']
            self.assertTrue(cameras and all(c['panorama_id'] in record[group] for c in cameras))
        points = np.load(target / 'points.npz')
        eligible = read(run.path / 'split_import/train_inputs.json')['init_points']['eligible_point_ids']
        self.assertEqual(points['ids'].tolist(), eligible)
        first = record['train'][0]
        row = points['ids'].tolist().index(1)
        np.testing.assert_allclose(points['rgb'][row], np.array(pano_color(first)) / 255)
        manifest = read(target / 'gsplat_inputs.json')
        self.assertGreaterEqual(manifest['points']['heldout_observations_ignored'], 1)
        self.assertIn('13 panoramas', manifest['limitation'])
        self.assertTrue(all(c['weights']['path'].startswith('gs/gsplat_inputs/weights/train/') for c in train))
        self.assertEqual(gsplat_preflight.verify_prep(run.path), [])
        self.assertTrue(gsplat_preflight.code_matches_prep(run.path))

    def test_start_of_training_detects_modified_inputs(self):
        run, record = self.prepared()
        camera = read(run.path / 'gsplat_inputs/cameras_train.json')['cameras'][0]
        Image.fromarray(np.zeros((8, 8), np.uint8)).save(self.root / 'runs' / camera['weights']['path'])
        problems = gsplat_preflight.verify_prep(run.path)
        self.assertTrue(any('changed' in p for p in problems), problems)
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            gsplat_preflight.load_verified(self.root / 'runs', camera['weights'])

    def test_unpinned_or_mismatched_partition_refused(self):
        for pinned, name in ((None, 'unpinned'), ('0' * 64, 'mismatch')):
            with self.assertRaisesRegex(RuntimeError, 'pinned'):
                self.prepared(pinned=pinned, name=name)


if __name__ == '__main__':
    unittest.main()
