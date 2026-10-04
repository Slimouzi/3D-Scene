"""gsplat GPU checks, run on the VM in .venv-gsplat before the first real training:

    THETA_GSPLAT_GPU=1 .venv-gsplat/bin/python -m unittest -v tests.test_gsplat_gpu

Skipped without THETA_GSPLAT_GPU. Uses a synthetic scene only: no project data is read.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
from theta_pipeline.gsplat_preflight import qualify

CONFIG = {'name': 'synthetic', 'steps': 20, 'seed': 0, 'sh_degree': 1, 'sh_degree_interval': 10,
          'ssim_lambda': .2, 'init_opacity': .1, 'init_scale': 1.,
          'lr': {'means': 1.6e-4, 'scales': 5e-3, 'quats': 1e-3, 'opacities': 5e-2, 'sh0': 2.5e-3, 'shN': 1.25e-4},
          'means_lr_final_ratio': .01,
          'strategy': {'refine_start_iter': 4, 'refine_stop_iter': 16, 'refine_every': 4, 'reset_every': 1000},
          'log_every': 5, 'validate_every': 10, 'checkpoint_every': 10}
META = {'config_sha256': 'c', 'manifest_sha256': 'm', 'partition_sha256': 'p', 'git_commit': 'g'}


def synthetic(group, count, device):
    import torch
    size = 32
    viewmats, names = [], []
    for k in range(count):
        angle = 2 * np.pi * k / count + (.3 if group != 'train' else 0)
        center = np.array([3 * np.sin(angle), 0, -3 * np.cos(angle)])
        forward = -center / np.linalg.norm(center)
        right = np.cross([0, 1, 0], forward)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rotation = np.stack((right, down, forward))
        view = np.eye(4)
        view[:3, :3], view[:3, 3] = rotation, -rotation @ center
        viewmats.append(view)
        names.append(f'{group}_{k}')
    images = torch.full((count, size, size, 3), 120, dtype=torch.uint8)
    images[:, 8:24, 8:24] = torch.tensor([200, 40, 40], dtype=torch.uint8)
    return {'names': names, 'set': group, 'images': images.to(device),
            'weights': torch.full((count, size, size), 255, dtype=torch.uint8, device=device),
            'Ks': torch.tensor([[[size, 0, size / 2], [0, size, size / 2], [0, 0, 1]]] * count,
                               dtype=torch.float32, device=device),
            'viewmats': torch.tensor(np.stack(viewmats), dtype=torch.float32, device=device),
            'width': size, 'height': size}


@unittest.skipUnless(os.environ.get('THETA_GSPLAT_GPU'), 'gsplat GPU checks run only on the VM')
class GsplatGpu(unittest.TestCase):
    def test_environment_matches_lock(self):
        result = qualify()
        self.assertTrue(result['qualified'], result['problems'])
        # Version checks and `import gsplat` miss backend dependencies: a CUDA render must succeed.
        self.assertIn('render_probe', result['found'])

    def test_render_mode_used_by_inspection(self):
        import torch
        from gsplat import rasterization
        d = 'cuda'
        renders, alphas, _ = rasterization(
            means=torch.tensor([[0., 0., 2.]], device=d), quats=torch.tensor([[1., 0., 0., 0.]], device=d),
            scales=torch.full((1, 3), .2, device=d), opacities=torch.tensor([.9], device=d),
            colors=torch.tensor([[1., 0., 0.]], device=d), viewmats=torch.eye(4, device=d)[None],
            Ks=torch.tensor([[[8., 0., 4.], [0., 8., 4.], [0., 0., 1.]]], device=d), width=8, height=8,
            render_mode='RGB+ED')
        self.assertEqual(tuple(renders.shape), (1, 8, 8, 4))
        self.assertAlmostEqual(float(renders[0, 4, 4, 3]), 2., delta=.3)     # expected depth
        self.assertGreater(float(alphas[0, 4, 4, 0]), .5)

    def test_absgs_densification_trains(self):
        from theta_pipeline import gsplat_train
        rng = np.random.default_rng(1)
        points = {'ids': np.arange(200), 'xyz': rng.uniform(-.5, .5, (200, 3)), 'rgb': rng.uniform(0, 1, (200, 3))}
        cfg = {**CONFIG, 'name': 'absgs', 'strategy': {**CONFIG['strategy'], 'absgrad': True, 'grow_grad2d': .0008}}
        with tempfile.TemporaryDirectory() as temp:
            gsplat_train.train(synthetic('train', 6, 'cuda'), synthetic('validation', 2, 'cuda'), points, cfg,
                               Path(temp), META)
            log = [json.loads(line) for line in (Path(temp) / 'train.jsonl').read_text().splitlines()]
        self.assertEqual(log[-1]['step'], 20)

    def test_numpy_projection_matches_gsplat(self):
        import torch
        from theta_pipeline import gsplat_checkpoint_diag as cd
        rng = np.random.default_rng(3)
        n, size = 2000, 64
        means = rng.uniform([-3, -3, -.5], [3, 3, 6], (n, 3))
        quats = rng.normal(size=(n, 4))
        scales = np.exp(rng.uniform(-4, 0, (n, 3)))
        opacity = rng.uniform(0, 1, n)
        view = np.eye(4)
        K = [[50., 0, 32], [0, 50., 32], [0, 0, 1]]
        ours = cd.project_like_gsplat(view, K, size, size, means, quats, scales, opacity)
        params = {'means': torch.tensor(means, dtype=torch.float32, device='cuda'),
                  'quats': torch.tensor(quats, dtype=torch.float32, device='cuda'),
                  'scales': torch.tensor(np.log(scales), dtype=torch.float32, device='cuda'),
                  'opacities': torch.logit(torch.tensor(opacity, dtype=torch.float32, device='cuda'))}
        theirs = cd.gsplat_radii(params, torch.eye(4, device='cuda'), torch.tensor(K, device='cuda'), size)
        mine = np.stack([ours['radius_x'], ours['radius_y']], -1)
        visible = (mine > 0).all(1) != (theirs > 0).all(1)
        self.assertLessEqual(int(visible.sum()), n // 200)            # float32 vs float64 at the culling edges
        both = (mine > 0).all(1) & (theirs > 0).all(1)
        self.assertLessEqual(int(np.abs(mine[both] - theirs[both]).max()), 1)

    def test_train_checkpoint_resume_and_refuse_foreign_config(self):
        from theta_pipeline import gsplat_train
        rng = np.random.default_rng(0)
        points = {'ids': np.arange(200), 'xyz': rng.uniform(-.5, .5, (200, 3)), 'rgb': rng.uniform(0, 1, (200, 3))}
        train, val = synthetic('train', 6, 'cuda'), synthetic('validation', 2, 'cuda')
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            gsplat_train.train(train, val, points, CONFIG, out, META)
            steps = [json.loads(line)['step'] for line in (out / 'train.jsonl').read_text().splitlines()]
            self.assertEqual(steps[-1], 20)
            self.assertEqual(sorted(p.name for p in (out / 'checkpoints').iterdir()),
                             ['step_000010.pt', 'step_000020.pt'])
            selection = json.loads((out / 'selection.json').read_text())
            self.assertFalse(selection['test_used'])
            # Checkpoint 10 trained its last iteration (index 9) at degree 9 // 10 = 0.
            import torch
            saved = torch.load(out / 'checkpoints/step_000010.pt', weights_only=False)
            validation = [json.loads(line) for line in (out / 'validation.jsonl').read_text().splitlines()]
            self.assertEqual(saved['sh_degree'], 0)
            self.assertEqual(validation[0]['sh_degree'], saved['sh_degree'])
            # Simulate an interruption after step 10 and resume.
            (out / 'checkpoints/step_000020.pt').unlink()
            gsplat_train.train(train, val, points, CONFIG, out, META, resume=True)
            steps = [json.loads(line)['step'] for line in (out / 'train.jsonl').read_text().splitlines()]
            self.assertEqual(steps[-1], 20)
            self.assertEqual(steps.count(15), 2)       # steps 11-20 were trained again from step 10
            self.assertTrue((out / 'checkpoints/step_000020.pt').exists())
            with self.assertRaisesRegex(RuntimeError, 'config_sha256'):
                gsplat_train.train(train, val, points, CONFIG, out, {**META, 'config_sha256': 'other'},
                                   resume=True)
            with self.assertRaisesRegex(RuntimeError, 'already has checkpoints'):
                gsplat_train.train(train, val, points, CONFIG, out, META)
            with self.assertRaisesRegex(RuntimeError, 'train set'):
                gsplat_train.train(val, val, points, CONFIG, Path(temp) / 'x', META)


if __name__ == '__main__':
    unittest.main()
