import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from theta_pipeline import gsplat_divergence as gd
from theta_pipeline import gsplat_sheet as gs
from theta_pipeline.storage import write

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
load = lambda name: json.loads((ROOT / f'configs/gsplat-{name}.json').read_text())


class SettingsTests(unittest.TestCase):
    def test_paired_arms_identical_until_3000(self):
        for short, long in (('ctrl-3k-s0', 'ctrl-10k-s0'), ('ctrl-3k-s1', 'ctrl-10k-s1')):
            self.assertEqual(gd.settings_differences(gd.effective_settings(load(short), 3000),
                                                     gd.effective_settings(load(long), 3000)), [])

    def test_unpaired_arms_differ_from_iteration_0(self):
        differences = gd.settings_differences(gd.effective_settings(load('ctrl-3k-s0'), 3000),
                                              gd.effective_settings(load('ctrl-3k-s1'), 3000))
        self.assertIn('strategy', [d['what'] for d in differences])
        self.assertIn('large_pruning', [d['what'] for d in differences])
        self.assertEqual(gd.first_divergence(differences, {'first_difference': {}}, [])['at'],
                         'before iteration 0 (settings or schedule)')

    def test_lr_or_sh_change_is_detected(self):
        changed = {**load('ctrl-10k-s0'), 'means_lr_decay_steps': 10000}
        differences = gd.settings_differences(gd.effective_settings(load('ctrl-3k-s0'), 3000),
                                              gd.effective_settings(changed, 3000))
        self.assertIn('means_lr_factor', [d['what'] for d in differences])
        changed = {**load('ctrl-10k-s0'), 'sh_degree_interval': 500}
        differences = gd.settings_differences(gd.effective_settings(load('ctrl-3k-s0'), 3000),
                                              gd.effective_settings(changed, 3000))
        self.assertEqual(next(d for d in differences if d['what'] == 'sh_degree')['first_index'], 500)


class TrajectoryTests(unittest.TestCase):
    def test_first_logged_difference(self):
        rows = [{'step': s, 'camera': f'c{s}', 'sh_degree': 0, 'lr_means': 1., 'gaussians': 10, 'loss': .1}
                for s in (100, 200, 300)]
        other = [dict(r) for r in rows]
        other[1]['loss'] = .1000001
        other[2]['camera'] = 'x'
        result = gd.trajectory_differences(rows, other, 3000)
        self.assertEqual(result['first_difference']['loss']['step'], 200)
        self.assertEqual(result['first_difference']['camera']['step'], 300)
        self.assertNotIn('gaussians', result['first_difference'])
        divergence = gd.first_divergence([], result, [])
        self.assertEqual((divergence['at'], divergence['kind']), (200, 'logged loss'))


@unittest.skipUnless(torch, 'torch is not installed in this environment')
class CheckpointTests(unittest.TestCase):
    def save(self, folder, means, generator_seed=0):
        from theta_pipeline import gsplat_train
        params = torch.nn.ParameterDict({'means': torch.nn.Parameter(means)})
        optimizers = {'means': torch.optim.Adam([{'params': params['means'], 'lr': .1, 'name': 'means'}])}
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizers['means'], gsplat_train.MeansLrFactor(.01, 3000))
        return gsplat_train.save_checkpoint(folder, 1000, params, optimizers, scheduler, {'grad2d': None},
                                            torch.Generator().manual_seed(generator_seed), {}, 0)

    def test_checkpoint_comparison(self):
        with tempfile.TemporaryDirectory() as temp:
            a = self.save(Path(temp) / 'a', torch.zeros(4, 3))
            b = self.save(Path(temp) / 'b', torch.zeros(4, 3))
            c = self.save(Path(temp) / 'c', torch.full((4, 3), 1e-6), generator_seed=1)
            same = gd.checkpoint_differences(a, b)
            self.assertTrue(same['camera_generator_equal'])
            self.assertEqual(same['params']['means']['status'], 'identical')
            different = gd.checkpoint_differences(a, c)
            self.assertFalse(different['camera_generator_equal'])
            self.assertAlmostEqual(different['params']['means']['max_abs_diff'], 1e-6, places=12)
            divergence = gd.first_divergence([], {'first_difference': {}}, [same, different])
            self.assertEqual(divergence['detail'], {'params': ['means'], 'state': ['camera_generator_equal']})

    def test_shape_mismatch_is_non_comparable_never_infinite(self):
        with tempfile.TemporaryDirectory() as temp:
            a = self.save(Path(temp) / 'a', torch.zeros(4, 3))
            b = self.save(Path(temp) / 'b', torch.zeros(5, 3))
            result = gd.checkpoint_differences(a, b)
            means = result['params']['means']
            self.assertFalse(means['comparable'])
            self.assertEqual(means['status'], gd.NON_COMPARABLE)
            self.assertIsNone(means['max_abs_diff'])
            self.assertEqual((means['short']['shape'], means['long']['shape']), ([4, 3], [5, 3]))
            text = gd.report({'short': 's', 'long': 'l', 'until': 3000, 'settings_differences': [],
                              'trajectory': {'logged_steps_compared': 0, 'first_difference': {}},
                              'checkpoints': [result], 'first_divergence': gd.first_divergence(
                                  [], {'first_difference': {}}, [result])})
            self.assertNotIn('inf', text)
            self.assertIn(gd.NON_COMPARABLE, text)

    def test_nan_and_inf_are_recorded(self):
        a = torch.tensor([0., float('nan'), 1., float('inf')])
        b = torch.tensor([0., 0., 1.5, float('inf')])
        result = gd.compare_tensors(a, b)
        self.assertEqual((result['short']['nan'], result['short']['inf']), (1, 1))
        self.assertEqual(result['short']['dtype'], 'float32')
        self.assertEqual(result['max_abs_diff'], .5)            # finite entries only
        self.assertEqual(result['non_finite_entries'], 2)

    def test_refused_operation_is_named(self):
        def kernel():
            raise RuntimeError('index_add_cuda_ does not have a deterministic implementation, but you set ...')
        try:
            kernel()
        except RuntimeError as error:
            refused = gd.refusal(error)
        self.assertTrue(refused['message'].startswith('index_add_cuda_'))
        self.assertTrue(any('kernel' in frame for frame in refused['frames']))


class SheetTests(unittest.TestCase):
    def inspection(self, prep, name, checkpoint, faces, test_loaded=False):
        folder = prep / 'inspection' / f'{name}-{Path(checkpoint).stem}-v2'
        write(folder / 'inspection.json', {'test_loaded': test_loaded})
        rows = []
        for face in faces:
            stem = face.replace('/', '__').removesuffix('.png')
            files = {}
            for panel in ('reference', 'render', 'error', 'depth'):
                path = folder / 'validation-full' / stem / f'{panel}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.full((8, 8, 3), 100, np.uint8)).save(path)
                files[panel] = f'{stem}/{panel}.png'
            rows.append({'camera': face, 'psnr': 18., 'low_alpha_fraction_of_valid': .01, 'files': files,
                         'regions': {'glass': {'psnr': 14., 'pixels': 5}, 'contours': {'psnr': 15., 'pixels': 5},
                                     'furniture': {'psnr': None, 'pixels': 0}}})
        write(folder / 'validation-full' / 'inspection.json', {'faces': rows})

    def test_sheet_and_refusals(self):
        faces = ['pano_camera0/R.png', 'pano_camera1/R.png']
        with tempfile.TemporaryDirectory() as temp:
            prep = Path(temp)
            self.inspection(prep, 'ctrl-3k-s0', 'step_003000.pt', faces)
            self.inspection(prep, 'ctrl-10k-s0', 'step_010000.pt', faces)
            columns = ['ctrl-3k-s0:step_003000.pt', 'ctrl-10k-s0:step_010000.pt']
            self.assertEqual(gs.main(['--prep', temp, '--name', 'x', '--columns', *columns, '--size', '16']), 0)
            image = Image.open(prep / 'comparisons/x/validation_sheet.jpg')
            self.assertEqual(image.size, (3 * 16, 40 + 2 * (3 * 16 + 22)))
            text = (prep / 'comparisons/x/validation_sheet.md').read_text()
            self.assertIn('ctrl-10k-s0 @ 10000', text)
            self.assertIn('| absente |', text)                    # furniture has no pixels in the fixture
            self.assertIn('panoramas de validation R', text)
            self.inspection(prep, 'ctrl-3k-s1', 'step_003000.pt', faces[:1])
            with self.assertRaisesRegex(RuntimeError, 'same validation faces'):
                gs.load_columns(prep, [columns[0], 'ctrl-3k-s1:step_003000.pt'])
            self.inspection(prep, 'leak', 'step_003000.pt', faces, test_loaded=True)
            with self.assertRaisesRegex(RuntimeError, 'test set'):
                gs.load_columns(prep, ['leak:step_003000.pt'])


if __name__ == '__main__':
    unittest.main()


class CheckpointDiagnosisTests(unittest.TestCase):
    """Obstruction test geometry, non-finite accounting, report; no GPU needed."""

    def camera(self):
        return np.eye(4).tolist(), [[8., 0, 8], [0, 8, 8], [0, 0, 1]], 16

    def test_reference_depth_and_near_gaussians(self):
        from theta_pipeline import gsplat_checkpoint_diag as cd
        T, K, size = self.camera()
        points = np.array([[0, 0, 2.], [.1, 0, 4.], [0, 0, -3.], [50, 0, 1.]])   # behind and outside ignored
        reference = cd.reference_depth(T, K, size, points, quantile=0.)
        self.assertEqual(reference, 2.)
        means = np.array([[0, 0, .5], [0, 0, 1.5], [0, 0, -.2], [30, 0, .5]])
        near = cd.near_gaussians(T, K, size, means, reference)          # depth < 0.5 x 2.0, inside, in front
        np.testing.assert_array_equal(near, [True, False, False, False])
        self.assertFalse(cd.near_gaussians(T, K, size, means, None).any())
        self.assertIsNone(cd.reference_depth(T, K, size, np.array([[0, 0, -1.]])))

    def test_population_reports_non_finite_values(self):
        from theta_pipeline import gsplat_checkpoint_diag as cd
        params = {'means': np.array([[0, 0, 0], [np.nan, 0, 0], [1, 1, 1.]]),
                  'scales': np.log(np.full((3, 3), .1)), 'opacities': np.array([2., 0., np.inf]),
                  'quats': np.zeros((3, 4)), 'sh0': np.zeros((3, 1, 3)), 'shN': np.zeros((3, 3, 3))}
        result = cd.population(params, np.array([[0, 0, 0], [1, 1, 1.]]), 2., .1)
        self.assertEqual(result['count_total'], 3)
        self.assertEqual(result['non_finite']['means'], {'nan': 1, 'inf': 0})
        self.assertEqual(result['non_finite']['opacities'], {'nan': 0, 'inf': 1})
        self.assertEqual(result['count'], 1)                            # statistics over finite Gaussians only

    def test_report_and_test_set_refusal(self):
        from theta_pipeline import gsplat_checkpoint_diag as cd
        face = lambda psnr: {'psnr': psnr, 'ssim': .7, 'low_alpha_fraction_of_valid': .01, 'near_gaussians': 3,
                             'near_opaque_gaussians': 1, 'near_alpha_fraction_of_valid': .02}
        pop = {'count_total': 10, 'non_finite': {'means': {'nan': 0, 'inf': 0}}, 'opacity_quantiles': {},
               'max_scale_over_scene_scale_quantiles': {'p95': .01, 'max': .2},
               'outside_sfm_points_box_plus_25pct': 2, 'outside_and_opaque_over_0_5': 1}
        result = {'training': 't', 'set': 'validation', 'panorama': 'R0010011', 'hypothesis': 'h', 'near_rule': 'r',
                  'checkpoints': {'step_002000.pt': {'population': pop}, 'step_003000.pt': {'population': pop}},
                  'faces': {'pano_camera0/R0010011.png': {'step_002000.pt': face(18.), 'step_003000.pt': face(16.5)}}}
        text = cd.report(result)
        self.assertIn('-1.50', text)
        self.assertIn('aucune', text)
        with self.assertRaisesRegex(RuntimeError, 'test set'):
            cd.diagnose('unused', {}, ['a', 'b'], 'R0010008', group='test')
