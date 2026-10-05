"""Read-only historical / 3DGUT comparison: same checkpoint, same faces, no test set, no writes in training/.

Needs torch (CPU is enough): a fake gsplat module records the rasterization arguments. Real
3DGUT rendering is checked on the VM by tests/test_gsplat_gpu.py.
"""
import sys
import tempfile
import types
import unittest
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

        def rasterization(**kwargs):
            self.calls.append(kwargs)
            value = .4 if kwargs['with_ut'] else .5
            size = kwargs['width']
            renders = torch.full((1, size, size, 4), value)
            renders[..., 3] = 2.
            return renders, torch.ones(1, size, size, 1), {}
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

    def run_compare(self, train_faces=1):
        return self.modules[2].compare(self.prep, self.cfg, 'step_003000.pt', train_faces)

    def test_same_checkpoint_same_faces_and_recorded_parameters(self):
        from theta_pipeline.storage import read
        before = self.snapshot()
        target = self.run_compare()
        self.assertEqual(len(self.loads), 1)                                 # loaded once for both modes
        result = read(target / 'comparison.json')
        self.assertEqual(result['checkpoint_loads'], 1)
        self.assertEqual(set(result['checkpoint_sha256_by_mode'].values()), {self.checkpoint_digest})
        modes = [(c['with_ut'], c['with_eval3d']) for c in self.calls]
        self.assertEqual(modes, [(False, False), (True, True)] * 5)
        for historic, gut in zip(self.calls[::2], self.calls[1::2]):
            for key in ('viewmats', 'Ks', 'width', 'height', 'sh_degree', 'packed', 'render_mode', 'rasterize_mode',
                        'camera_model', 'eps2d', 'near_plane'):
                value_a, value_b = historic[key], gut[key]
                same = torch.equal(value_a, value_b) if torch.is_tensor(value_a) else value_a == value_b
                self.assertTrue(same, key)
            self.assertIs(historic['means'], gut['means'])                   # the very same parameter tensors
        self.assertEqual(result['sh_degree'], 1)
        self.assertEqual(result['renderers']['3dgut']['with_ut'], True)
        self.assertEqual(result['renderers']['3dgut']['with_eval3d'], True)
        self.assertIn('gsplat', result['environment'])
        self.assertIn('analysis_git_commit', result)
        names = [f['camera'] for f in result['faces']]
        self.assertEqual(len(names), 5)
        for face in result['faces']:
            self.assertEqual(set(face) >= {'historique', '3dgut', 'sheet'}, True)
            self.assertTrue((target / face['sheet']).is_file())
        self.assertEqual(sorted(p.name for p in target.glob('*.jpg')), sorted(f['sheet'] for f in result['faces']))
        self.assertEqual(self.snapshot(), before)                            # nothing written in training/
        self.assertNotIn('training', str(target.relative_to(self.prep.resolve())))

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

    def test_rerun_never_overwrites(self):
        self.run_compare(train_faces=0)
        with self.assertRaisesRegex(RuntimeError, 'never overwritten'):
            self.run_compare(train_faces=0)

    def test_unknown_renderer_refused(self):
        inspect = self.modules[0]
        with self.assertRaisesRegex(ValueError, 'unknown renderer'):
            inspect.render_face({}, {}, 0, 0, 'n2')
        self.assertEqual(sorted(inspect.RENDERERS), ['3dgut', 'historique'])
        self.assertEqual({k: v for k, v in inspect.RENDERERS['historique'].items() if k not in ('with_ut', 'with_eval3d')},
                         {k: v for k, v in inspect.RENDERERS['3dgut'].items() if k not in ('with_ut', 'with_eval3d')})


if __name__ == '__main__':
    unittest.main()
