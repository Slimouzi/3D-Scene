"""Experiment A (AbsGS): versioned design, preflight code rule, absgrad wiring, paired analysis."""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from theta_pipeline import gsplat_compare
from theta_pipeline.gsplat_preflight import TRAINING_MODULES, code_differences
from theta_pipeline.gsplat_train import densification_schedule
from theta_pipeline.storage import write

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = json.loads((ROOT / 'configs/experiments/absgs-3k.json').read_text())
load = lambda name: json.loads((ROOT / f'configs/gsplat-{name}.json').read_text())


class DesignTests(unittest.TestCase):
    def test_experiment_derives_from_gsplat_v2(self):
        prep = json.loads((ROOT / EXPERIMENT['prep_config']).read_text())
        self.assertEqual(EXPERIMENT['prep_config'], 'configs/salon-gsplat-v2.json')
        self.assertEqual(prep['split_run'], EXPERIMENT['split_run'])
        self.assertEqual(prep['split']['expected_partition_sha256'], EXPERIMENT['partition_sha256'])
        self.assertEqual(EXPERIMENT['prep_run'], 'salon-gsplat-009')
        self.assertGreaterEqual(len(EXPERIMENT['seeds']), 3)

    def test_pairs_differ_only_by_the_densification_criterion(self):
        reference = load('ctrl-3k-s0')
        for pair in EXPERIMENT['pairs']:
            base, abs_ = load(pair['baseline']), load(pair['variant'])
            self.assertEqual(base['seed'], pair['seed'])
            self.assertEqual(abs_['seed'], pair['seed'])
            strip = lambda c: {k: v for k, v in c.items() if k not in ('name', 'strategy', 'schedule')}
            self.assertEqual(strip(base), strip(abs_))
            self.assertEqual(base['strategy'], reference['strategy'])
            self.assertEqual(abs_['strategy'], {**reference['strategy'], 'absgrad': True, 'grow_grad2d': .0008})
            for cfg in (base, abs_):
                schedule = densification_schedule(cfg)
                self.assertEqual(schedule['refine'], list(range(600, 2500, 100)))
                self.assertEqual(schedule['large_pruning'], [])        # no global size pruning

    def test_six_seeds_with_unchanged_variants(self):
        self.assertEqual(EXPERIMENT['seeds'], [0, 1, 2, 3, 4, 5])
        self.assertEqual([p['seed'] for p in EXPERIMENT['pairs']], EXPERIMENT['seeds'])
        for seed in (3, 4, 5):
            for arm in ('base', 'abs'):
                new, old = load(f'absgs-{arm}-seed{seed}'), load(f'absgs-{arm}-seed0')
                strip = lambda c: {k: v for k, v in c.items() if k not in ('name', 'seed', 'schedule')}
                self.assertEqual(strip(new), strip(old))
                self.assertEqual(new['seed'], seed)
        self.assertIn('none excluded', EXPERIMENT['seed_status'])

    def test_test_set_stays_reserved(self):
        self.assertTrue(any('R0010008' in item and 'evaluate-test' in item for item in EXPERIMENT['not_used']))


class PreflightCodeRuleTests(unittest.TestCase):
    def test_training_code_may_change_preparation_code_may_not(self):
        prep = {'gsplat_train.py': 'a', 'gsplat_prep.py': 'b', 'split.py': 'c', 'segmentation/fusion.py': 'd'}
        trained = {**prep, 'gsplat_train.py': 'a2', 'gsplat_compare.py': 'new'}
        self.assertEqual(code_differences(prep, trained),
                         {'preparation': [], 'training': ['gsplat_compare.py', 'gsplat_train.py']})
        for module in ('gsplat_prep.py', 'split.py', 'segmentation/fusion.py'):
            changed = {**prep, module: 'x'}
            self.assertEqual(code_differences(prep, changed)['preparation'], [module])
        self.assertNotIn('gsplat_prep.py', TRAINING_MODULES)
        self.assertNotIn('split.py', TRAINING_MODULES)


@unittest.skipUnless(torch, 'torch is not installed in this environment')
class AbsgradWiringTests(unittest.TestCase):
    def test_render_forwards_absgrad_to_rasterization(self):
        from theta_pipeline import gsplat_train
        calls = []
        fake = types.ModuleType('gsplat')
        fake.rasterization = lambda **kwargs: (calls.append(kwargs) or (torch.zeros(1, 2, 2, 3), None, {}))
        original = sys.modules.get('gsplat')
        sys.modules['gsplat'] = fake
        try:
            params = {'means': torch.zeros(1, 3), 'quats': torch.zeros(1, 4), 'scales': torch.zeros(1, 3),
                      'opacities': torch.zeros(1), 'sh0': torch.zeros(1, 1, 3), 'shN': torch.zeros(1, 8, 3)}
            for flag in (False, True):
                gsplat_train.render(params, torch.eye(4)[None], torch.eye(3)[None], 2, 2, 0, absgrad=flag)
        finally:
            if original is None:
                sys.modules.pop('gsplat')
            else:
                sys.modules['gsplat'] = original
        self.assertEqual([c['absgrad'] for c in calls], [False, True])


class PairedAnalysisTests(unittest.TestCase):
    def make(self, prep, name, values):
        folder = prep / 'training' / name
        write(folder / 'training.json', {'status': 'completed', 'config': load(name), 'history': [{}]})
        cameras = [{'camera': f'pano_camera{k}/{pano}.png', 'psnr': value, 'excluded': False}
                   for pano, value in values.items() for k in range(2)]
        mean = sum(values.values()) / len(values)
        (folder / 'validation.jsonl').write_text(json.dumps(
            {'step': 3000, 'mean_psnr': mean, 'mean_ssim': .7, 'usable_cameras': len(cameras),
             'cameras': cameras}) + '\n')
        (folder / 'train.jsonl').write_text(json.dumps({'step': 3000, 'gaussians': 10, 'loss': .1}) + '\n')
        write(folder / 'selection.json', {'step': 3000, 'test_used': False})

    def test_variant_minus_baseline_overall_and_per_panorama(self):
        with tempfile.TemporaryDirectory() as temp:
            prep = Path(temp)
            gains = {0: (.3, -.1), 1: (.2, .1), 2: (.4, 0.)}
            for seed, (first, second) in gains.items():
                self.make(prep, f'absgs-base-seed{seed}', {'R0010006': 18., 'R0010011': 17.})
                self.make(prep, f'absgs-abs-seed{seed}', {'R0010006': 18. + first, 'R0010011': 17. + second})
            result = gsplat_compare.paired(EXPERIMENT, prep, {'psnr_db': .244, 'source': 'ctrl'})
            by_pano = result['summary']['by_panorama']
            self.assertAlmostEqual(by_pano['R0010006']['mean'], .3)
            self.assertEqual((by_pano['R0010011']['positive'], by_pano['R0010011']['negative']), (1, 1))
            self.assertAlmostEqual(result['summary']['psnr']['mean'], (.1 + .15 + .2) / 3)
            text = gsplat_compare.paired_report(result)
            self.assertIn('AbsGS − défaut', text)
            self.assertIn('panorama R0010011', text)



class AbsoluteAndCoverageTests(unittest.TestCase):
    """Seed 2-style check: is a gain driven by a weak baseline? Regions absent are never zero."""

    def make(self, prep, name, values, ssim=.7, curve=(16., 17.)):
        folder = prep / 'training' / name
        write(folder / 'training.json', {'status': 'completed', 'config': load(name), 'history': [{}]})
        cameras = [{'camera': f'pano_camera{k}/{pano}.png', 'psnr': value, 'ssim': ssim, 'excluded': False}
                   for pano, value in values.items() for k in range(2)]
        cameras.append({'camera': 'pano_camera9/R0010011.png', 'psnr': None, 'ssim': None, 'excluded': True})
        mean = sum(values.values()) / len(values)
        rows = [{'step': 1000, 'mean_psnr': curve[0], 'mean_ssim': .6, 'usable_cameras': 4},
                {'step': 2000, 'mean_psnr': curve[1], 'mean_ssim': .65, 'usable_cameras': 4},
                {'step': 3000, 'mean_psnr': mean, 'mean_ssim': ssim, 'usable_cameras': 4, 'cameras': cameras}]
        (folder / 'validation.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        (folder / 'train.jsonl').write_text(json.dumps({'step': 3000, 'gaussians': 10}) + '\n')
        write(folder / 'selection.json', {'step': 3000, 'checkpoint': 'step_003000.pt', 'test_used': False})
        faces = [{'panorama_id': pano, 'regions': {
                    'glass': {'pixels': 50, 'psnr': value - 3}, 'mirror': {'pixels': 0, 'psnr': None},
                    'contours': {'pixels': 30, 'psnr': value - 2}}}
                 for pano, value in values.items() for _ in range(2)]
        write(prep / 'inspection' / f'{name}-step_003000-v2' / 'validation-full' / 'inspection.json', {'faces': faces})

    def test_absolute_values_deviation_and_absent_regions(self):
        with tempfile.TemporaryDirectory() as temp:
            prep = Path(temp)
            baselines = {0: 18., 1: 18.1, 2: 17.0}            # seed 2: weak baseline
            for seed, value in baselines.items():
                self.make(prep, f'absgs-base-seed{seed}', {'R0010006': value, 'R0010011': value - 1})
                self.make(prep, f'absgs-abs-seed{seed}', {'R0010006': 18.2, 'R0010011': 17.2}, ssim=.72)
            result = gsplat_compare.paired(EXPERIMENT, prep, {'psnr_db': .244, 'source': 'salon-gsplat-007'})
            seed2 = next(r for r in result['pairs'] if r['seed'] == 2)
            self.assertAlmostEqual(seed2['absolute']['baseline']['minus_median_of_other_seeds'], 16.5 - 17.55)
            self.assertAlmostEqual(seed2['absolute']['variant']['minus_median_of_other_seeds'], 0.)
            pano = seed2['absolute']['baseline']['panoramas']['R0010011']
            self.assertEqual((pano['usable'], pano['faces']), (2, 3))
            self.assertEqual([c['step'] for c in seed2['absolute']['baseline']['curve']], [1000, 2000, 3000])
            coverage = seed2['coverage']['R0010006']
            self.assertTrue(coverage['mirror']['absent'])
            self.assertIsNone(coverage['mirror']['mean_psnr'])
            self.assertEqual(coverage['glass']['faces_present'], 2)
            self.assertIsNone(seed2['regions_by_panorama']['R0010006']['mirror'])
            self.assertTrue(seed2['coverage_identical_between_arms'])
            text = gsplat_compare.paired_report(result)
            self.assertIn('Résultats limités aux 2 panoramas de validation R0010006 et R0010011', text)
            self.assertIn('absente (0/2)', text)
            self.assertIn('Repère historique', text)
            self.assertIn('ne prouvent pas la robustesse', text)
            self.assertIn('n’isole pas à lui seul le non-déterminisme GPU', text)
            self.assertNotIn('| mirror | 0.000', text)

if __name__ == '__main__':
    unittest.main()
