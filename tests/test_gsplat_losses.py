"""Masked losses, metrics, selection and checkpoint SH degree. Needs torch, not gsplat or a GPU.

Runs in .venv-gsplat on the VM (and in any environment with torch); skipped otherwise.
"""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch, 'torch is not installed in this environment')
class MaskedLossTests(unittest.TestCase):
    def setUp(self):
        from theta_pipeline import gsplat_train
        self.gt = gsplat_train
        g = torch.Generator().manual_seed(0)
        self.rendered = torch.rand(1, 24, 24, 3, generator=g, dtype=torch.float64)
        self.target = torch.rand(1, 24, 24, 3, generator=g, dtype=torch.float64)
        self.weight = torch.ones(1, 24, 24, dtype=torch.float64)
        self.weight[:, 8:16, 10:14] = 0          # excluded block
        self.weight[:, 0:4, :] = .5              # reflective weight

    def changed_outside(self, tensor):
        other = tensor.clone()
        other[:, 8:16, 10:14] = 1 - other[:, 8:16, 10:14]
        return other

    def test_excluded_pixels_change_neither_loss_nor_metrics(self):
        loss = self.gt.weighted_loss(self.rendered, self.target, self.weight, .2)
        changed = self.gt.weighted_loss(self.changed_outside(self.rendered), self.changed_outside(self.target),
                                        self.weight, .2)
        for a, b in zip(loss, changed):
            self.assertEqual(float(a), float(b))
        metrics = self.gt.view_metrics(self.rendered, self.target, self.weight)
        self.assertEqual(metrics, self.gt.view_metrics(self.changed_outside(self.rendered),
                                                       self.changed_outside(self.target), self.weight))

    def test_excluded_pixels_receive_no_gradient(self):
        rendered = self.rendered.clone().requires_grad_(True)
        loss, _, _ = self.gt.weighted_loss(rendered, self.target, self.weight, .2)
        loss.backward()
        self.assertTrue(bool((rendered.grad[:, 8:16, 10:14] == 0).all()))
        self.assertTrue(bool((rendered.grad[:, 8:16, 9] != 0).any()))     # valid neighbour still learns
        rendered = self.rendered.clone().requires_grad_(True)
        ssim = (self.weight * self.gt.ssim_map(rendered, self.target, self.weight > 0)).sum()
        ssim.backward()
        self.assertTrue(bool((rendered.grad[:, 8:16, 10:14] == 0).all()))

    def test_unmasked_ssim_matches_identity(self):
        valid = torch.ones(1, 24, 24, dtype=torch.bool)
        same = self.gt.ssim_map(self.target, self.target, valid)
        np.testing.assert_allclose(same.numpy(), 1, atol=1e-9)

    def test_fully_masked_view_is_excluded_not_100_db(self):
        zero = torch.zeros(1, 24, 24, dtype=torch.float64)
        metrics = self.gt.view_metrics(self.rendered, self.target, zero)
        self.assertIsNone(metrics['psnr'])
        self.assertTrue(metrics['excluded'])
        rows = [{'camera': 'a', **metrics},
                {'camera': 'b', **self.gt.view_metrics(self.rendered, self.target, self.weight)}]
        summary = self.gt.aggregate('validation', rows, 1)
        self.assertEqual(summary['usable_cameras'], 1)
        self.assertEqual(summary['excluded_cameras'], ['a'])
        self.assertEqual(summary['mean_psnr'], rows[1]['psnr'])
        self.assertIsNone(self.gt.aggregate('validation', rows[:1], 1)['mean_psnr'])
        with self.assertRaisesRegex(RuntimeError, 'no valid pixel'):
            self.gt.weighted_loss(self.rendered, self.target, zero, .2)

    def test_selection_refused_without_usable_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            rows = [{'step': 1000, 'checkpoint': 'step_001000.pt', 'mean_psnr': None, 'mean_ssim': None,
                     'sh_degree': 0, 'usable_cameras': 0}]
            (out / 'validation.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
            with self.assertRaisesRegex(RuntimeError, 'no usable validation view'):
                self.gt.select(out)
            rows.append({'step': 2000, 'checkpoint': 'step_002000.pt', 'mean_psnr': 20., 'mean_ssim': .7,
                         'sh_degree': 1, 'usable_cameras': 12})
            (out / 'validation.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
            self.gt.select(out)
            selection = json.loads((out / 'selection.json').read_text())
            self.assertEqual((selection['step'], selection['sh_degree']), (2000, 1))

    def test_train_refuses_fully_masked_train_view(self):
        data = {'names': ['a', 'b'], 'set': 'train',
                'weights': torch.stack([torch.ones(4, 4), torch.zeros(4, 4)]).to(torch.uint8)}
        with self.assertRaisesRegex(RuntimeError, r"without any valid pixel: \['b'\]"):
            self.gt.train(data, None, {}, {}, Path(tempfile.gettempdir()) / 'unused', {})

    def test_checkpoint_keeps_the_sh_degree_used(self):
        cfg = {'sh_degree': 3, 'init_scale': 1., 'init_opacity': .1,
               'lr': {'means': 1.6e-4, 'scales': 5e-3, 'quats': 1e-3, 'opacities': 5e-2, 'sh0': 2.5e-3,
                      'shN': 1.25e-4}}
        points = {'xyz': np.random.default_rng(0).normal(size=(10, 3)), 'rgb': np.full((10, 3), .5)}
        params, optimizers = self.gt.init_model(points, cfg, 1., 'cpu')
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizers['means'], gamma=.99)
        meta = {'config_sha256': 'c', 'manifest_sha256': 'm', 'partition_sha256': 'p', 'git_commit': 'g',
                'scene_scale': 1.}
        with tempfile.TemporaryDirectory() as temp:
            # Step 1000 trained its last iteration (index 999) at degree 999 // 1000 = 0.
            path = self.gt.save_checkpoint(Path(temp), 1000, params, optimizers, scheduler,
                                           {'grad2d': None, 'count': None, 'scene_scale': 1.},
                                           torch.Generator().manual_seed(0), meta, 0)
            *_, saved = self.gt.restore(path, cfg, meta, 'cpu')
            self.assertEqual(saved['sh_degree'], 0)


if __name__ == '__main__':
    unittest.main()
