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
    def make(self, prep, name, steps, rows, gaussians=100, test_used=False, loss_offset=0.):
        folder = prep / 'training' / name
        cfg = ARMS.get(name, {'steps': steps, 'schedule': {}})
        write(folder / 'training.json', {'status': 'completed', 'config': cfg, 'history': [{}]})
        (folder / 'validation.jsonl').write_text(''.join(
            json.dumps({'step': s, 'mean_psnr': p, 'mean_ssim': .7, 'usable_cameras': 12}) + '\n' for s, p in rows))
        log = [{'step': s, 'loss': .2 - s / 1e5 + loss_offset, 'gaussians': gaussians + s // 100,
                'max_memory_gb': 1. + s / 1e4, 'seconds': s / 10.} for s in range(100, steps + 1, 100)]
        (folder / 'train.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in log))
        write(folder / 'selection.json', {'step': rows[-1][0], 'test_used': test_used})

    def inspection(self, prep, name, step, glass):
        region = {'contours': 15., 'glass': glass, 'furniture': 17., 'mirror': None,
                  'unvalidated_reflective': None, 'other': 20.}
        block = {'mean_psnr': 18., 'mean_ssim': .7, 'usable_faces': 12, 'mean_region_psnr': region}
        write(prep / 'inspection' / f'{name}-step_{step:06d}-v2/inspection.json', {
            'checkpoint': f'step_{step:06d}.pt', 'results': {'validation/full': block, 'train/full': block},
            'gaussians': {'larger_than_prune_scale3d': 3, 'outside_sfm_points_box_plus_25pct': 4,
                          'outside_and_opaque_over_0_5': 1,
                          'max_scale_over_scene_scale_quantiles': {'p50': .01, 'p95': .05, 'p99': .08, 'max': .4}}})

    def test_effects_pairing_resources_and_noise(self):
        with tempfile.TemporaryDirectory() as temp:
            prep, ref = Path(temp) / 'prep', Path(temp) / 'ref'
            self.make(prep, 'ctrl-3k-s0', 3000, [(3000, 18.0)])
            self.make(prep, 'ctrl-3k-s1', 3000, [(3000, 18.5)])
            self.make(prep, 'ctrl-10k-s0', 10000, [(3000, 18.1), (10000, 19.0)], loss_offset=.001)
            self.make(ref, 'l4-short-001', 3000, [(3000, 18.2)])
            self.inspection(prep, 'ctrl-10k-s0', 3000, 12.)
            self.inspection(prep, 'ctrl-10k-s0', 10000, 13.5)
            result = gsplat_compare.compare(EXPERIMENT, prep, (ref, 'l4-short-001'))
            effects = result['effects']
            self.assertAlmostEqual(effects['duration (last step, longer - shorter)']['s0'], 1.0)
            self.assertIsNone(effects['duration (last step, longer - shorter)']['s1'])   # arm not run
            self.assertAlmostEqual(effects['duration inside the longer arm (last - step 3000)']['s0'], .9)
            self.assertAlmostEqual(effects['pruning (s1 - s0, same duration)'][3000], .5)
            self.assertAlmostEqual(result['noise']['replicate_minus_reference_psnr_3000'], -.2)
            pair = result['pairing_at_3000']['s0']
            self.assertAlmostEqual(pair['psnr_3000_long_minus_short'], .1)
            self.assertEqual(pair['gaussians_3000_long_minus_short'], 0)
            self.assertEqual(pair['logged_steps_compared'], 30)
            self.assertAlmostEqual(pair['max_abs_loss_gap'], .001)
            self.assertIsNone(result['pairing_at_3000']['s1'])
            arm = result['arms']['ctrl-10k-s0']
            self.assertEqual((arm['gaussians_3000'], arm['gaussians_last']), (130, 200))
            self.assertAlmostEqual(arm['max_memory_gb'], 2.)
            self.assertEqual(arm['inspection_last']['validation']['regions']['glass'], 13.5)
            self.assertNotIn('loss_curve', arm)
            text = gsplat_compare.report(result)
            for expected in ('Bruit', 'Appariement', 'step_010000.pt', '0.0500 / 0.4000', 'contours'):
                self.assertIn(expected, text)

    def test_refuses_an_arm_that_used_the_test_set(self):
        with tempfile.TemporaryDirectory() as temp:
            prep = Path(temp)
            self.make(prep, 'ctrl-3k-s0', 3000, [(3000, 18.0)], test_used=True)
            with self.assertRaisesRegex(RuntimeError, 'test set'):
                gsplat_compare.compare(EXPERIMENT, prep)


class PairedSeedTests(unittest.TestCase):
    paired = json.loads((ROOT / 'configs/experiments/paired-seeds-3k.json').read_text())

    def test_pairs_differ_only_by_pruning(self):
        for pair in self.paired['pairs']:
            a, b = (json.loads((ROOT / f"configs/gsplat-{pair[k]}.json").read_text()) for k in ('s0', 's1'))
            self.assertEqual(a['seed'], pair['seed'])
            self.assertEqual(b['seed'], pair['seed'])
            strip = lambda c: {k: v for k, v in c.items() if k not in ('name', 'strategy', 'schedule')}
            self.assertEqual(strip(a), strip(b))
            self.assertEqual(a['strategy'], ARMS['ctrl-3k-s0']['strategy'])
            self.assertEqual(b['strategy'], ARMS['ctrl-3k-s1']['strategy'])
            self.assertEqual(densification_schedule(b)['large_pruning'], list(range(1100, 2500, 100)))
            self.assertEqual(densification_schedule(a)['large_pruning'], [])

    def test_paired_statistics(self):
        with tempfile.TemporaryDirectory() as temp:
            prep = Path(temp)
            maker = CompareTests()
            for seed, (p0, p1) in {0: (18.0, 18.2), 1: (17.9, 18.0), 2: (18.1, 18.0)}.items():
                names = (f'pair-3k-s0-seed{seed}', f'pair-3k-s1-seed{seed}')
                for name, value in zip(names, (p0, p1)):
                    maker.make(prep, name, 3000, [(3000, value)])
            experiment = {**self.paired, 'pairs': [p for p in self.paired['pairs'] if p['seed'] in (0, 1, 2)]}
            self.assertEqual(len(self.paired['pairs']), 6)
            result = gsplat_compare.paired(experiment, prep, {'psnr_db': .244, 'source': 'ctrl'})
            self.assertEqual([r['seed'] for r in result['pairs']], [0, 1, 2])
            psnr = result['summary']['psnr']
            self.assertEqual((psnr['n'], psnr['positive'], psnr['negative']), (3, 2, 1))
            self.assertAlmostEqual(psnr['mean'], (0.2 + 0.1 - 0.1) / 3)
            self.assertAlmostEqual(psnr['median'], .1)
            noise = result['summary']['psnr_vs_noise']
            self.assertEqual((noise['seeds_beyond_noise'], noise['seeds']), (0, 3))
            self.assertAlmostEqual(noise['abs_mean_over_noise'], (0.2 / 3) / .244)
            text = gsplat_compare.paired_report(result)
            self.assertIn('médiane', text)
            self.assertIn('Repère historique : 0.244 dB', text)
            self.assertIn('pas un seuil statistique', text)

if __name__ == '__main__':
    unittest.main()
