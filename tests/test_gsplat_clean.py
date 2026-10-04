"""Correction experiment: train-only rule, one cleaned model, tracked degradations (no GPU)."""
import json
import unittest
from pathlib import Path
import numpy as np
from theta_pipeline import gsplat_clean as gc
from theta_pipeline.gsplat_inspect import detail_ratio

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = json.loads((ROOT / 'configs/experiments/correction-near-train.json').read_text())
RULE = EXPERIMENT['rule']


def camera(name, group, z_offset=0.):
    T = np.eye(4)
    T[2, 3] = z_offset
    return {'name': name, 'set': group, 'width': 16, 'K': [[8., 0, 8], [0, 8, 8], [0, 0, 1]],
            'world_to_camera': T.tolist()}


class RuleTests(unittest.TestCase):
    def host(self):
        # 0: far and small; 1: near the first camera only; 2: far but its oriented extent reaches near.
        q = np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0
        return {'means': np.array([[0, 0, 3.], [0, 0, .5], [0, 0, 3.]]),
                'quats': np.array([[1., 0, 0, 0], [1., 0, 0, 0], q]),
                'scales': np.log(np.array([[.05] * 3, [.01] * 3, [.9, .01, .01]])),
                'opacities': np.array([3., 3., 3.])}

    def test_union_over_train_faces_only(self):
        points = np.array([[0, 0, 2.], [.1, 0, 4.]])
        cameras = [camera('pano_camera0/a.png', 'train'), camera('pano_camera1/a.png', 'train', z_offset=-1.)]
        remove, counts = gc.removal_set(cameras, self.host(), points, RULE)
        np.testing.assert_array_equal(remove, [False, True, True])
        self.assertEqual(counts['pano_camera0/a.png'], 2)
        with self.assertRaisesRegex(RuntimeError, 'train cameras only'):
            gc.removal_set(cameras + [camera('pano_camera0/v.png', 'validation')], self.host(), points, RULE)

    def test_design_uses_train_views_and_all_absgs_trainings(self):
        absgs = json.loads((ROOT / 'configs/experiments/absgs-3k.json').read_text())
        self.assertEqual(RULE['views'], 'train')
        self.assertEqual(EXPERIMENT['trainings'], absgs['arms'])
        self.assertEqual(len(EXPERIMENT['trainings']), 12)
        self.assertEqual(EXPERIMENT['partition_sha256'], absgs['partition_sha256'])
        self.assertIn('validation cameras to define the rule', EXPERIMENT['not_used'])


class MetricTests(unittest.TestCase):
    def test_detail_ratio(self):
        reference = np.zeros((16, 16, 3))
        reference[:, 0::4] = reference[:, 1::4] = 255         # period-4 stripes: non-zero central gradient
        blurred = np.full((16, 16, 3), 127.5)
        mask = np.ones((16, 16), bool)
        self.assertAlmostEqual(detail_ratio(reference, reference, mask), 1.)
        self.assertAlmostEqual(detail_ratio(reference, blurred, mask), 0.)
        self.assertIsNone(detail_ratio(reference, blurred, np.zeros((16, 16), bool)))
        self.assertIsNone(detail_ratio(blurred, reference, mask))            # flat reference

    def test_detail_ratio_ignores_pixels_outside_the_region(self):
        rng = np.random.default_rng(0)
        reference = rng.uniform(0, 255, (32, 32, 3))
        mask = np.zeros((32, 32), bool)
        mask[8:24, 8:24] = True
        changed = reference.copy()
        changed[~mask] = rng.uniform(0, 255, (int((~mask).sum()), 3))     # outside only
        self.assertAlmostEqual(detail_ratio(reference, reference, mask), 1.)
        self.assertAlmostEqual(detail_ratio(reference, changed, mask), 1.)
        line = np.zeros((32, 32), bool)
        line[10, 5:25] = True                                            # no pixel with all four neighbours inside
        self.assertIsNone(detail_ratio(reference, changed, line))

    def face(self, name, pano, psnr, ssim, lum, furniture=None, mirror=None, excluded=False):
        region = lambda v: {'psnr': v, 'ssim': None if v is None else .7, 'detail_ratio': None if v is None else .9}
        return {'camera': name, 'panorama_id': pano, 'psnr': psnr, 'ssim': ssim, 'excluded': excluded,
                'render_luminance': lum, 'reference_luminance': .5,
                'regions': {'furniture': region(furniture), 'mirror': region(mirror), 'glass': region(None),
                            'contours': region(15.), 'other': region(20.)}}

    def test_summaries_deltas_degradations_and_report(self):
        full = [self.face('c0/A.png', 'A', 18., .70, .45, furniture=16., mirror=12.),
                self.face('c1/A.png', 'A', 19., .72, .50, furniture=17.),
                self.face('c0/B.png', 'B', 17., .60, .40),
                self.face('c1/B.png', 'B', None, None, None, excluded=True)]
        cleaned = [self.face('c0/A.png', 'A', 18.5, .71, .47, furniture=16.2, mirror=11.),
                   self.face('c1/A.png', 'A', 18.8, .70, .58, furniture=17.),
                   self.face('c0/B.png', 'B', 17.2, .61, .42),
                   self.face('c1/B.png', 'B', None, None, None, excluded=True)]
        f, c = gc.summarize(full), gc.summarize(cleaned)
        self.assertEqual((f['B']['faces'], f['B']['usable']), (2, 1))
        self.assertIsNone(f['B']['regions']['mirror']['psnr'])                 # absent, not zero
        d = gc.compare_summaries(f, c)
        self.assertAlmostEqual(d['A']['psnr'], (18.5 + 18.8 - 18. - 19.) / 2)
        self.assertAlmostEqual(d['A']['regions']['mirror']['psnr'], -1.)
        self.assertIsNone(d['B']['regions']['mirror']['psnr'])
        worse = gc.degradations(full, cleaned)
        self.assertEqual(worse['ssim_decreased'], ['c1/A.png'])
        self.assertEqual(worse['luminance_further_from_reference'], ['c1/A.png'])
        summary = {'experiment': 'x', 'prep_run': 'p', 'checkpoint': 'step_003000.pt', 'rule': RULE,
                   'trainings': [{'training': f'absgs-abs-seed{s}', 'removed': 5, 'removed_opaque': 2, 'delta': d,
                                  'degradations': worse} for s in (0, 1)]}
        text = gc.report(summary)
        for expected in ('seules vues d’entraînement', 'panoramas de validation A et B', '| absgs-abs | A | 2 |',
                         'Faces dont le SSIM baisse', 'jamais comme un score nul'):
            self.assertIn(expected, text)



class RunIdentityTests(unittest.TestCase):
    def test_identity_changes_with_rule_and_folders_are_never_overwritten(self):
        import tempfile
        trainings = {'absgs-abs-seed0': {'config_sha256': 'c', 'checkpoint_sha256': 'k'}}
        first = gc.run_identity(EXPERIMENT, 'm', trainings)
        self.assertEqual(first, gc.run_identity(json.loads(json.dumps(EXPERIMENT)), 'm', trainings))
        changed_rule = {**EXPERIMENT, 'rule': {**RULE, 'factor': .4}}
        self.assertNotEqual(first, gc.run_identity(changed_rule, 'm', trainings))
        self.assertNotEqual(first, gc.run_identity(EXPERIMENT, 'other-prep', trainings))
        self.assertNotEqual(first, gc.run_identity(EXPERIMENT, 'm', {'absgs-abs-seed0': {'config_sha256': 'c',
                                                                                      'checkpoint_sha256': 'k2'}}))
        with tempfile.TemporaryDirectory() as temp:
            folder = gc.new_folder(Path(temp) / first[:16])
            (folder / 'evaluation.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError, 'never overwritten'):
                gc.new_folder(Path(temp) / first[:16])
            self.assertEqual((folder / 'evaluation.json').read_text(), '{}')

if __name__ == '__main__':
    unittest.main()
