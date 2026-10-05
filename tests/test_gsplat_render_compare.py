"""Read-only historical / 3DGUT comparison: same checkpoint, same faces, no test set, no writes in training/.

Needs torch (CPU is enough): a fake gsplat module records the rasterization arguments. Real
3DGUT rendering is checked on the VM by tests/test_gsplat_gpu.py.
"""
import sys
import tempfile
import types
import unittest
import weakref
from pathlib import Path
import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch, 'torch is not installed in this environment')
class RenderCompareTests(unittest.TestCase):
    def setUp(self):
        from theta_pipeline import gsplat_inspect, gsplat_preflight, gsplat_render_compare, gsplat_train
        from theta_pipeline.storage import digest, write
        self.modules = (gsplat_inspect, gsplat_preflight, gsplat_render_compare, gsplat_train)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.prep = prep = Path(temp.name) / 'runs' / 'prep'
        self.training = prep / 'training' / 'absgs-abs-seed0'
        (self.training / 'checkpoints').mkdir(parents=True)
        (self.training / 'checkpoints' / 'step_003000.pt').write_bytes(b'checkpoint')
        write(self.training / 'training.json', {'git_commit': 'g'})
        write(prep / 'gsplat_inputs/gsplat_inputs.json', {'partition_sha256': 'p', 'files': {}})
        camera = lambda name, group: {'name': name, 'panorama_id': Path(name).stem, 'set': group, 'width': 8}
        write(prep / 'gsplat_inputs/cameras_validation.json',
              {'cameras': [camera(f'pano_camera{k}/{p}.png', 'validation') for p in ('R0010006', 'R0010011')
                           for k in (0, 1)]})
        write(prep / 'gsplat_inputs/cameras_train.json', {'cameras': [camera('pano_camera0/R0010004.png', 'train')]})
        write(prep / 'gsplat_inputs/cameras_test.json', {'cameras': [camera('pano_camera0/R0010008.png', 'test')]})
        self.checkpoint_digest = digest(self.training / 'checkpoints' / 'step_003000.pt')
        self.calls, self.loads, self.read_paths, self.groups = [], [], [], []
        originals = []

        def patch(module, name, value):
            originals.append((module, name, getattr(module, name)))
            setattr(module, name, value)
        self.addCleanup(lambda: [setattr(m, n, v) for m, n, v in reversed(originals)])
        fake = types.ModuleType('gsplat')

        self.fault = None
        self.rgb_value = {'historique': .5, '3dgut': .4}
        self.returned, self.alive_at_call = [], []

        def rasterization(**kwargs):
            self.calls.append(kwargs)
            # Earlier outputs still referenced anywhere would inflate a GPU memory measurement.
            self.alive_at_call.append(sum(1 for ref in self.returned if ref() is not None))
            mode = '3dgut' if kwargs['with_ut'] else 'historique'
            if self.fault and self.fault[0] == mode and self.fault[1] == 'raise':
                raise RuntimeError('simulated 3DGUT kernel error')
            size = kwargs['width']
            renders = torch.full((1, size, size, 4), self.rgb_value[mode])
            renders[..., 3] = 2.
            alphas = torch.ones(1, size, size, 1)
            if self.fault and self.fault[0] == mode:
                if self.fault[1] == 'depth_nan':
                    renders[..., 3] = float('nan')
                elif self.fault[1] == 'alpha_nan':
                    alphas[:] = float('nan')
                elif self.fault[1] == 'rgb_inf':
                    renders[0, 0, 0, 0] = float('inf')
            radii = torch.tensor([[[3, 4], [0, 0]]], dtype=torch.int32)
            self.returned += [weakref.ref(renders), weakref.ref(alphas), weakref.ref(radii)]
            return renders, alphas, {'radii': radii}
        fake.rasterization = rasterization
        previous = sys.modules.get('gsplat')
        sys.modules['gsplat'] = fake
        self.addCleanup(lambda: sys.modules.__setitem__('gsplat', previous) if previous else sys.modules.pop('gsplat'))

        def restore(path, cfg, meta, device):
            self.loads.append(str(path))
            params = {'means': torch.zeros(2, 3), 'quats': torch.tensor([[1., 0, 0, 0]] * 2),
                      'scales': torch.zeros(2, 3), 'opacities': torch.zeros(2), 'sh0': torch.zeros(2, 1, 3),
                      'shN': torch.zeros(2, 3, 3)}
            self.params = params
            return 3000, params, None, None, None, None, {'sh_degree': 1, 'meta': {}}

        def load_cameras(prep, group, device, names=None):
            self.groups.append(group)
            n = len(names)
            return {'names': names, 'set': group, 'width': 8, 'height': 8,
                    'images': torch.full((n, 8, 8, 3), 120, dtype=torch.uint8),
                    'weights': torch.full((n, 8, 8), 255, dtype=torch.uint8),
                    'viewmats': torch.eye(4)[None].repeat(n, 1, 1),
                    'Ks': torch.tensor([[[4., 0, 4], [0, 4, 4], [0, 0, 1]]]).repeat(n, 1, 1)}

        class Sources:
            def __init__(self, output, manifest):
                self.rotations = {name: np.eye(3) for name in (
                    'pano_camera0/R0010006.png', 'pano_camera1/R0010006.png', 'pano_camera0/R0010011.png',
                    'pano_camera1/R0010011.png', 'pano_camera0/R0010004.png')}

            def labels(self, pano):
                return np.full((16, 32), 4, np.uint8)
        inspect, preflight, compare, train = self.modules
        patch(preflight, 'verify_prep', lambda p: [])
        patch(preflight, 'qualify', lambda: {'problems': []})
        patch(train, 'restore', restore)
        patch(train, 'load_cameras', load_cameras)
        patch(train, 'config_sha256', lambda cfg: 'c')
        patch(inspect, 'Sources', Sources)
        patch(inspect, 'select_train_faces', lambda cams, labels, rot, size, n: ([c['name'] for c in cams][:n], {}))
        original_read = compare.read
        patch(compare, 'read', lambda path: (self.read_paths.append(str(path)), original_read(path))[1])
        self.cfg = {'name': 'absgs-abs-seed0'}

    def snapshot(self):
        from theta_pipeline.storage import digest
        return {str(p.relative_to(self.training)): digest(p) for p in sorted(self.training.rglob('*')) if p.is_file()}

    def run_compare(self, train_faces=1, repetitions=3):
        return self.modules[2].compare(self.prep, self.cfg, 'step_003000.pt', train_faces, repetitions, 1)

    def index(self, target):
        from theta_pipeline.storage import read
        return read(target.parent / 'index.json')

    def test_same_checkpoint_same_faces_and_recorded_parameters(self):
        from theta_pipeline.storage import read
        before = self.snapshot()
        target = self.run_compare()
        self.assertEqual(len(self.loads), 1)                                 # loaded once for both modes
        result = read(target / 'comparison.json')
        self.assertTrue(result['valid'])
        self.assertEqual(result['checkpoint_loads'], 1)
        self.assertEqual(set(result['checkpoint_sha256_by_mode'].values()), {self.checkpoint_digest})
        modes = {(c['with_ut'], c['with_eval3d']) for c in self.calls}
        self.assertEqual(modes, {(False, False), (True, True)})
        shared = ('width', 'height', 'sh_degree', 'packed', 'render_mode', 'rasterize_mode', 'camera_model', 'eps2d',
                  'near_plane')
        self.assertEqual(len({tuple(c[k] for k in shared) for c in self.calls}), 1)
        self.assertEqual(len({id(c['means']) for c in self.calls}), 1)     # the very same parameter tensors
        historic = [c['viewmats'] for c in self.calls if not c['with_ut']]
        gut = [c['viewmats'] for c in self.calls if c['with_ut']]
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(historic, gut)))
        self.assertEqual(result['sh_degree'], 1)
        self.assertEqual(result['renderers']['3dgut']['with_ut'], True)
        self.assertEqual(result['renderers']['3dgut']['with_eval3d'], True)
        for key in ('gsplat', 'torch', 'cuda', 'driver', 'device'):
            self.assertIn(key, result['environment'])
        self.assertIn('analysis_git_commit', result)
        self.assertEqual(len(result['faces']), 5)
        self.assertEqual(sorted(p.name for p in target.glob('*.jpg')),
                         sorted([f['sheet'] for f in result['faces']] + ['overview.jpg']))
        self.assertEqual(self.snapshot(), before)                            # nothing written in training/
        self.assertNotIn('training', str(target.relative_to(self.prep.resolve())))
        self.assertEqual(self.index(target)['attempts'][-1]['status'], 'completed')

    def test_measurements_per_mode(self):
        from theta_pipeline.storage import read
        result = read(self.run_compare(repetitions=3) / 'comparison.json')
        for face in result['faces']:
            for mode in ('historique', '3dgut'):
                entry = face[mode]
                self.assertEqual(len(entry['times_ms']), 3)
                self.assertEqual(entry['projection']['visible_gaussians'], 1)
                self.assertEqual(entry['projection']['radius_max'], 4.)
                self.assertEqual(entry['repetition_spread']['rgb_max_abs_diff'], 0.)
                for key in ('alpha_mean', 'alpha_below_half_fraction', 'depth_p05', 'depth_p50', 'depth_p95'):
                    self.assertIn(key, entry)
        perf = result['performance']
        self.assertEqual(perf['historique']['renders'], 5 * 3)
        self.assertIn('peak_render_memory_mb', perf['3dgut'])
        self.assertIn('baseline_memory_mb', perf['3dgut'])
        self.assertEqual(result['warmup'], 1)
        warmups = 2 * 2                                                      # one per mode and per set
        self.assertEqual(len(self.calls), warmups + 5 * 2 * 3)

    def test_invalid_raw_outputs_are_declared(self):
        from theta_pipeline.storage import read
        for fault in ('depth_nan', 'alpha_nan', 'rgb_inf'):
            self.fault = ('3dgut', fault)
            target = self.run_compare(train_faces=0, repetitions=1)
            result = read(target / 'comparison.json')
            self.assertFalse(result['valid'], fault)
            self.assertEqual({x['mode'] for x in result['invalid_outputs']}, {'3dgut'})
            self.assertEqual(len(result['invalid_outputs']), 4)
            face = result['faces'][0]
            self.assertIsNone(face['3dgut']['psnr'], fault)                 # never a metric from invalid output
            self.assertIsNotNone(face['historique']['psnr'])
            key = {'depth_nan': 'depth', 'alpha_nan': 'alpha', 'rgb_inf': 'rgb'}[fault]
            self.assertGreater(face['3dgut']['non_finite'][key], 0)
            self.assertIn('INSPECTION INVALIDE', (target / 'comparison.md').read_text())
            self.assertIn(face['camera'], (target / 'comparison.md').read_text())

    def test_invalid_inspection_returns_non_zero_and_is_not_completed(self):
        from theta_pipeline.storage import read, write
        config = self.prep.parent / 'cfg.json'
        write(config, self.cfg)
        self.fault = ('3dgut', 'depth_nan')
        code = self.modules[2].main(['--prep', str(self.prep), '--config', str(config),
                                     '--checkpoint', 'step_003000.pt', '--repetitions', '1'])
        self.assertNotEqual(code, 0)
        folder = next((self.prep / 'inspection' / 'render-compare').iterdir())
        attempt = read(folder / 'index.json')['attempts'][-1]
        self.assertEqual(attempt['status'], 'invalid')
        self.assertFalse(attempt['valid'])
        report = folder / attempt['folder'] / 'comparison.md'
        self.assertTrue(report.is_file())                                 # the report is kept
        self.assertIn('INSPECTION INVALIDE', report.read_text())
        self.fault = None
        self.assertEqual(self.modules[2].main(['--prep', str(self.prep), '--config', str(config),
                                               '--checkpoint', 'step_003000.pt', '--repetitions', '1']), 0)

    def test_no_earlier_render_output_is_alive_during_a_render(self):
        self.run_compare(train_faces=1, repetitions=3)
        self.assertGreater(len(self.alive_at_call), 10)
        self.assertEqual(max(self.alive_at_call), 0)

    def test_overview_error_uses_the_same_rgb_as_the_metrics(self):
        from PIL import Image
        from theta_pipeline.storage import read
        self.rgb_value = {'historique': .5, '3dgut': 1.5}            # raw above 1, displayed as white
        original = self.modules[3].load_cameras

        def white(prep, group, device, names=None):
            data = original(prep, group, device, names)
            data['images'][:] = 255
            return data
        self.modules[3].load_cameras = white
        target = self.run_compare(train_faces=0, repetitions=1)
        result = read(target / 'comparison.json')
        self.assertEqual(result['faces'][0]['3dgut']['psnr'], 100.)    # clamped 1.0 equals the white reference
        overview = np.asarray(Image.open(target / 'overview.jpg').convert('RGB')).astype(int)
        size, x0, y0 = 160, 260, 24
        error_3dgut = overview[y0 + size // 2, x0 + 4 * size + size // 2]
        error_historic = overview[y0 + size // 2, x0 + 2 * size + size // 2]
        self.assertLess(int(error_3dgut.max()), 40)                    # black: no error, as the metric says
        self.assertGreater(int(error_historic[0]), 200)                # the historical render really differs

    def test_failed_attempt_then_retry(self):
        from theta_pipeline.storage import read
        self.fault = ('3dgut', 'raise')
        with self.assertRaisesRegex(RuntimeError, 'simulated 3DGUT kernel error'):
            self.run_compare(train_faces=0)
        self.fault = None
        target = self.run_compare(train_faces=0)
        attempts = self.index(target)['attempts']
        self.assertEqual([(a['attempt'], a['status']) for a in attempts], [(1, 'failed'), (2, 'completed')])
        self.assertIn('simulated 3DGUT kernel error', attempts[0]['error'])
        self.assertTrue((target.parent / 'attempt-001' / 'status.json').is_file())   # earlier attempt kept
        self.assertEqual(read(target.parent / 'attempt-001' / 'status.json')['status'], 'failed')
        self.assertEqual(target.name, 'attempt-002')
        self.assertTrue(read(target / 'comparison.json')['valid'])

    def test_repeated_runs_share_one_identity(self):
        first = self.run_compare(train_faces=0)
        second = self.run_compare(train_faces=0)
        self.assertEqual(first.parent, second.parent)
        self.assertEqual([a['status'] for a in self.index(second)['attempts']], ['completed', 'completed'])
        self.assertTrue((first / 'comparison.json').is_file())

    def test_test_set_never_used(self):
        target = self.run_compare()
        self.assertNotIn('test', self.groups)
        self.assertFalse(any('cameras_test' in p for p in self.read_paths))
        from theta_pipeline.storage import read
        result = read(target / 'comparison.json')
        self.assertFalse(result['test_loaded'])
        self.assertNotIn('R0010008', ' '.join(f['camera'] for f in result['faces']))

    def test_checkpoint_hash_is_verified(self):
        train = self.modules[3]
        original = train.restore

        def tampering(path, cfg, meta, device):
            Path(path).write_bytes(b'changed')
            return original(path, cfg, meta, device)
        train.restore = tampering
        with self.assertRaisesRegex(RuntimeError, 'changed during the comparison'):
            self.run_compare(train_faces=0)

    def test_report_presents_regions_and_performance(self):
        text = (self.run_compare(train_faces=0) / 'comparison.md').read_text()
        for expected in ('## Régions', '| validation | R0010006 | glass |', 'absente', '## Performance et projection',
                         'ne quantifient pas', 'overview.jpg'):
            self.assertIn(expected, text)

    def test_unknown_renderer_refused(self):
        inspect = self.modules[0]
        with self.assertRaisesRegex(ValueError, 'unknown renderer'):
            inspect.render_face({}, {}, 0, 0, 'n2')
        self.assertEqual(sorted(inspect.RENDERERS), ['3dgut', 'historique'])
        self.assertEqual({k: v for k, v in inspect.RENDERERS['historique'].items() if k not in ('with_ut', 'with_eval3d')},
                         {k: v for k, v in inspect.RENDERERS['3dgut'].items() if k not in ('with_ut', 'with_eval3d')})

if __name__ == '__main__':
    unittest.main()
