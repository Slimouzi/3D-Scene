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


if __name__ == '__main__':
    unittest.main()
