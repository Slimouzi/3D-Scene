import json
import tempfile
import unittest
from pathlib import Path
from theta_pipeline import gsplat_compare
from theta_pipeline.gsplat_train import MeansLrFactor, densification_schedule, means_lr_factor
from theta_pipeline.storage import write

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = json.loads((ROOT / 'configs/experiments/ctrl-duration-pruning.json').read_text())
ARMS = {json.loads((ROOT / p).read_text())['name']: json.loads((ROOT / p).read_text()) for p in EXPERIMENT['arms']}
FACTOR_KEYS = {'name', 'steps', 'strategy', 'schedule'}


class DesignTests(unittest.TestCase):
    def test_arms_differ_only_by_the_two_factors(self):
        base = ARMS['ctrl-3k-s0']
        for name, cfg in ARMS.items():
            same = {k: v for k, v in cfg.items() if k not in FACTOR_KEYS}
            self.assertEqual(same, {k: v for k, v in base.items() if k not in FACTOR_KEYS}, name)
            only_reset = {k: v for k, v in cfg['strategy'].items() if k != 'reset_every'}
            self.assertEqual(only_reset, {k: v for k, v in base['strategy'].items() if k != 'reset_every'})
        self.assertEqual(sorted((c['steps'], c['strategy']['reset_every']) for c in ARMS.values()),
                         [(3000, 1000), (3000, 3000), (10000, 1000), (10000, 3000)])

    def test_schedules_are_the_ones_declared(self):
        for name, cfg in ARMS.items():
            schedule = densification_schedule(cfg)
            self.assertEqual(schedule['refine'], list(range(600, 2500, 100)), name)
            self.assertEqual(schedule['opacity_reset'], [], name)
            expected = list(range(1100, 2500, 100)) if name.endswith('s1') else []
            self.assertEqual(schedule['large_pruning'], expected, name)

    def test_position_lr_decays_over_3000_steps_then_holds(self):
        for cfg in ARMS.values():
            factor = means_lr_factor(cfg)
            self.assertAlmostEqual(factor(0), 1.)
            self.assertAlmostEqual(factor(3000), .01)
            self.assertAlmostEqual(factor(9999), .01)
        # Default (no decay steps): identical to the former ExponentialLR, gamma = ratio ** (1 / steps).
        default = MeansLrFactor(.01, 3000)
        for i in (0, 1, 1500, 2999):
            self.assertAlmostEqual(default(i), (.01 ** (1 / 3000)) ** i)

    def test_reference_and_test_set_untouched_by_design(self):
        self.assertIn('test set (reserved; evaluate-test --final not run)', EXPERIMENT['not_used'])
        self.assertIn('l4-short-001 kept intact', EXPERIMENT['reference'])


class CompareTests(unittest.TestCase):
    def make(self, prep, name, steps, rows, gaussians=100, test_used=False):
        folder = prep / 'training' / name
        cfg = ARMS.get(name, {'steps': steps, 'schedule': {}})
        write(folder / 'training.json', {'status': 'completed', 'config': cfg})
        (folder / 'validation.jsonl').write_text(''.join(
            json.dumps({'step': s, 'mean_psnr': p, 'mean_ssim': .7, 'usable_cameras': 12}) + '\n' for s, p in rows))
        (folder / 'train.jsonl').write_text(json.dumps({'step': steps, 'gaussians': gaussians}) + '\n')
        write(folder / 'selection.json', {'step': rows[-1][0], 'test_used': test_used})

    def test_effects_and_noise(self):
        with tempfile.TemporaryDirectory() as temp:
            prep, ref = Path(temp) / 'prep', Path(temp) / 'ref'
            self.make(prep, 'ctrl-3k-s0', 3000, [(3000, 18.0)])
            self.make(prep, 'ctrl-3k-s1', 3000, [(3000, 18.5)])
            self.make(prep, 'ctrl-10k-s0', 10000, [(3000, 18.1), (10000, 19.0)])
            self.make(ref, 'l4-short-001', 3000, [(3000, 18.2)])
            result = gsplat_compare.compare(EXPERIMENT, prep, (ref, 'l4-short-001'))
            self.assertAlmostEqual(result['effects']['duration (last step, longer - shorter)']['s0'], 1.0)
            self.assertIsNone(result['effects']['duration (last step, longer - shorter)']['s1'])   # arm not run
            self.assertAlmostEqual(result['effects']['duration inside the longer arm (last - step 3000)']['s0'], .9)
            self.assertAlmostEqual(result['effects']['pruning (s1 - s0, same duration)'][3000], .5)
            self.assertAlmostEqual(result['noise']['replicate_minus_reference_psnr_3000'], -.2)
            self.assertEqual(result['arms']['ctrl-10k-s1']['status'], 'not run')
            self.assertIn('Bruit', gsplat_compare.report(result))

    def test_refuses_an_arm_that_used_the_test_set(self):
        with tempfile.TemporaryDirectory() as temp:
            prep = Path(temp)
            self.make(prep, 'ctrl-3k-s0', 3000, [(3000, 18.0)], test_used=True)
            with self.assertRaisesRegex(RuntimeError, 'test set'):
                gsplat_compare.compare(EXPERIMENT, prep)


if __name__ == '__main__':
    unittest.main()
