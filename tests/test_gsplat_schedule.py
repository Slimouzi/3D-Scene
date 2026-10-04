"""Densification/pruning schedule: computed from gsplat's conditions, checked against gsplat itself.

The comparison with DefaultStrategy needs torch and gsplat (CPU is enough): it runs in
.venv-gsplat on the VM and is skipped elsewhere.
"""
import json
import unittest
from pathlib import Path
from theta_pipeline.gsplat_train import densification_schedule

try:
    import torch
    from gsplat.strategy import DefaultStrategy
except ImportError:
    torch = None

CONFIGS = Path(__file__).resolve().parents[1] / 'configs'
SMALL = {'steps': 40, 'strategy': {'refine_start_iter': 5, 'refine_stop_iter': 30, 'refine_every': 5,
                                   'reset_every': 12}}


class ScheduleTests(unittest.TestCase):
    def test_documented_schedules(self):
        short = densification_schedule(json.loads((CONFIGS / 'gsplat-l4-short.json').read_text()))
        self.assertEqual((short['refine'][0], short['refine'][-1]), (600, 2400))
        self.assertEqual(short['large_pruning'], [])            # never active in l4-short-001
        long = densification_schedule(json.loads((CONFIGS / 'gsplat-l4-10k.json').read_text()))
        self.assertEqual((long['refine'][0], long['refine'][-1]), (600, 7400))
        self.assertEqual((long['large_pruning'][0], long['large_pruning'][-1]), (3100, 7400))
        for schedule in (short, long):
            self.assertEqual(schedule['opacity_reset'], [])

    def test_small_schedule(self):
        schedule = densification_schedule(SMALL)
        self.assertEqual(schedule['refine'], [10, 15, 20, 25])
        self.assertEqual(schedule['large_pruning'], [15, 20, 25])


@unittest.skipUnless(torch, 'torch and gsplat are required')
class AgainstGsplatTests(unittest.TestCase):
    def test_schedule_matches_default_strategy(self):
        strategy = DefaultStrategy(verbose=False, **SMALL['strategy'])
        n = 3
        params = torch.nn.ParameterDict({
            'means': torch.nn.Parameter(torch.zeros(n, 3)),
            'scales': torch.nn.Parameter(torch.log(torch.tensor([[.01] * 3, [.02] * 3, [5.] * 3]))),
            'quats': torch.nn.Parameter(torch.tensor([[1., 0, 0, 0]] * n)),
            'opacities': torch.nn.Parameter(torch.logit(torch.full((n,), .9))),
            'sh0': torch.nn.Parameter(torch.zeros(n, 1, 3)), 'shN': torch.nn.Parameter(torch.zeros(n, 3, 3))})
        optimizers = {k: torch.optim.Adam([{'params': params[k], 'lr': 1e-3, 'name': k}]) for k in params}
        state = strategy.initialize_state(scene_scale=1.)
        state.update(grad2d=torch.zeros(n), count=torch.zeros(n))     # normally filled by _update_state
        calls = {'refine': [], 'large_removed_at': None, 'opacity_reset': []}
        strategy._update_state = lambda *args, **kwargs: None
        strategy._grow_gs = lambda params, optimizers, state, step: calls['refine'].append(step) or (0, 0)
        original_prune = strategy._prune_gs

        def prune(params, optimizers, state, step):
            removed = original_prune(params, optimizers, state, step)
            if removed and calls['large_removed_at'] is None:
                calls['large_removed_at'] = step
            return removed
        strategy._prune_gs = prune
        import gsplat.strategy.default as default
        original_reset = default.reset_opa
        default.reset_opa = lambda **kwargs: calls['opacity_reset'].append(1)
        try:
            for step in range(SMALL['steps']):
                strategy.step_post_backward(params, optimizers, state, step, {}, packed=False)
        finally:
            default.reset_opa = original_reset
        expected = densification_schedule(SMALL)
        self.assertEqual(calls['refine'], expected['refine'])
        self.assertEqual(calls['large_removed_at'], expected['large_pruning'][0])
        self.assertEqual(len(params['means']), 2)
        self.assertEqual(calls['opacity_reset'], [])


if __name__ == '__main__':
    unittest.main()
