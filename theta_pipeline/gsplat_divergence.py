"""Where do two paired gsplat trainings first diverge? (read-only; checkpoints need torch)

    python -m theta_pipeline.gsplat_divergence --prep Output/runs/salon-gsplat-007 \
        --short configs/gsplat-ctrl-3k-s0.json --long configs/gsplat-ctrl-10k-s0.json --until 3000

Compares, up to `until`, what each run actually used: effective settings (learning rates,
position decay, SH schedule, densification/pruning schedule, cadence, seed), the logged
trajectory (camera drawn, position learning rate, SH degree, Gaussian count, loss) and
each common checkpoint (camera generator and RNG states, scheduler, optimizer steps,
strategy accumulators, parameters). The earliest difference is reported with its kind;
nothing is attributed to GPU non-determinism without a determinism probe
(`--probe-steps`), which replays the same start twice in a separate diagnostics folder.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import traceback
from pathlib import Path
from .gsplat_train import densification_schedule, means_lr_factor
from .storage import now, read, write

IGNORED = {'name', 'steps', 'schedule'}          # documentation and run length, not dynamics up to `until`


def effective_settings(cfg, until):
    """Settings that act on iterations i < until, as the trainer applies them."""
    schedule = densification_schedule({**cfg, 'steps': until})
    factor = means_lr_factor(cfg)
    return {'config': {k: v for k, v in cfg.items() if k not in IGNORED},
            'refine': schedule['refine'], 'large_pruning': schedule['large_pruning'],
            'opacity_reset': schedule['opacity_reset'],
            'means_lr_factor': [factor(i) for i in range(until + 1)],
            'sh_degree': [min(i // cfg['sh_degree_interval'], cfg['sh_degree']) for i in range(until)],
            'validated_steps': [s for s in range(1, until + 1)
                                if s % cfg['validate_every'] == 0 or s == cfg['steps']],
            'checkpoint_steps': [s for s in range(1, until + 1)
                                 if s % cfg['checkpoint_every'] == 0 or s == cfg['steps']]}


def settings_differences(a, b):
    out = []
    for key in sorted(set(a['config']) | set(b['config'])):
        if a['config'].get(key) != b['config'].get(key):
            out.append({'kind': 'setting', 'what': key, 'short': a['config'].get(key), 'long': b['config'].get(key)})
    for key in ('refine', 'large_pruning', 'opacity_reset', 'sh_degree', 'validated_steps', 'checkpoint_steps'):
        if a[key] != b[key]:
            first = next((i for i, (x, y) in enumerate(zip(a[key], b[key])) if x != y), min(len(a[key]), len(b[key])))
            out.append({'kind': 'schedule', 'what': key, 'first_index': first})
    lr = next((i for i, (x, y) in enumerate(zip(a['means_lr_factor'], b['means_lr_factor'])) if x != y), None)
    if lr is not None:
        out.append({'kind': 'schedule', 'what': 'means_lr_factor', 'iteration': lr})
    return out


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()] if Path(path).exists() else []


def trajectory_differences(short, long, until):
    """First logged step where each quantity differs (exact comparison)."""
    a = {r['step']: r for r in short if r['step'] <= until}
    b = {r['step']: r for r in long if r['step'] <= until}
    common = sorted(set(a) & set(b))
    first = {}
    for key in ('camera', 'sh_degree', 'lr_means', 'gaussians', 'loss', 'l1', 'ssim'):
        step = next((s for s in common if a[s].get(key) != b[s].get(key)), None)
        if step is not None:
            first[key] = {'step': step, 'short': a[step].get(key), 'long': b[step].get(key)}
    missing = sorted(set(a) ^ set(b))
    return {'logged_steps_compared': len(common), 'first_difference': first, 'steps_logged_by_one_only': missing}


NON_COMPARABLE = 'shape mismatch / non comparable'


def tensor_info(t):
    import torch
    floating = t.is_floating_point()
    return {'shape': list(t.shape), 'dtype': str(t.dtype).removeprefix('torch.'),
            'nan': int(torch.isnan(t).sum()) if floating else 0,
            'inf': int(torch.isinf(t).sum()) if floating else 0}


def compare_tensors(a, b):
    """Shapes first; only same-shape tensors get a numeric gap, over finite entries only."""
    import torch
    out = {'short': tensor_info(a), 'long': tensor_info(b)}
    if a.shape != b.shape or a.dtype != b.dtype:
        return {**out, 'comparable': False, 'status': NON_COMPARABLE, 'max_abs_diff': None}
    identical = bool(torch.equal(a, b))
    gap = None
    if not identical and a.numel() and a.is_floating_point():
        diff = (a.double() - b.double()).abs()
        finite = torch.isfinite(diff)
        gap = float(diff[finite].max()) if finite.any() else None
        out['non_finite_entries'] = int((~finite).sum())
    return {**out, 'comparable': True, 'status': 'identical' if identical else 'different', 'max_abs_diff': gap}


def checkpoint_differences(path_a, path_b):
    """Exact comparison of two checkpoints saved at the same step."""
    import torch
    a = torch.load(path_a, map_location='cpu', weights_only=False)
    b = torch.load(path_b, map_location='cpu', weights_only=False)
    out = {'step': a['step'], 'sh_degree_equal': a['sh_degree'] == b['sh_degree'],
           'camera_generator_equal': torch.equal(a['rng']['generator'], b['rng']['generator']),
           'torch_rng_equal': torch.equal(a['rng']['torch'], b['rng']['torch']),
           'cuda_rng_equal': len(a['rng']['cuda']) == len(b['rng']['cuda'])
           and all(torch.equal(x, y) for x, y in zip(a['rng']['cuda'], b['rng']['cuda'])),
           'scheduler_equal': {k: a['scheduler'][k] == b['scheduler'].get(k)
                               for k in ('last_epoch', 'base_lrs', '_last_lr') if k in a['scheduler']},
           'optimizer_steps': {}, 'params': {}, 'strategy_state': {}}
    for name, opt in a['optimizers'].items():
        steps = lambda o: sorted(float(s['step']) for s in o['state'].values() if 'step' in s)
        out['optimizer_steps'][name] = {'short': steps(opt)[:1], 'long': steps(b['optimizers'][name])[:1]}
    for name, tensor in a['params'].items():
        out['params'][name] = compare_tensors(tensor, b['params'][name])
    for key, value in a['strategy_state'].items():
        other = b['strategy_state'].get(key)
        if torch.is_tensor(value) and torch.is_tensor(other):
            out['strategy_state'][key] = compare_tensors(value, other)
        else:
            out['strategy_state'][key] = {'comparable': True, 'status': 'identical' if value == other else 'different'}
    return out


def environment():
    """Software and hardware identity of the diagnosis (driver via nvidia-smi when available)."""
    import torch
    from importlib.metadata import PackageNotFoundError, version
    info = {'torch': torch.__version__, 'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}
    try:
        info['gsplat'] = version('gsplat')
    except PackageNotFoundError:
        info['gsplat'] = None
    try:
        info['driver'] = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                        capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        info['driver'] = None
    return info


def first_divergence(settings, trajectory, checkpoints):
    """Earliest difference, ordered by the iteration at which it acts."""
    if settings:
        return {'at': 'before iteration 0 (settings or schedule)', 'differences': settings}
    events = [(v['step'], f'logged {k}', v) for k, v in trajectory['first_difference'].items()]
    for c in checkpoints:
        diffs = [k for k, v in c['params'].items() if v['status'] != 'identical']
        flags = [k for k in ('sh_degree_equal', 'camera_generator_equal', 'torch_rng_equal', 'cuda_rng_equal')
                 if not c[k]]
        if diffs or flags:
            events.append((c['step'], 'checkpoint', {'params': diffs, 'state': flags}))
    if not events:
        return {'at': None, 'note': 'no difference found in settings, logs or checkpoints'}
    step, kind, detail = min(events, key=lambda e: (e[0], e[1]))
    return {'at': step, 'kind': kind, 'detail': detail,
            'same_inputs_until_then': not settings,
            'interpretation': 'settings, schedules, camera order and RNG states are identical up to here: '
                              'compare with the determinism probe before attributing the gap to the GPU'}


def compare(prep, short_cfg, long_cfg, until):
    prep = Path(prep)
    a_dir, b_dir = (prep / 'training' / c['name'] for c in (short_cfg, long_cfg))
    settings = settings_differences(effective_settings(short_cfg, until), effective_settings(long_cfg, until))
    trajectory = trajectory_differences(jsonl(a_dir / 'train.jsonl'), jsonl(b_dir / 'train.jsonl'), until)
    checkpoints = []
    for step in range(short_cfg['checkpoint_every'], until + 1, short_cfg['checkpoint_every']):
        name = f'step_{step:06d}.pt'
        if (a_dir / 'checkpoints' / name).exists() and (b_dir / 'checkpoints' / name).exists():
            checkpoints.append(checkpoint_differences(a_dir / 'checkpoints' / name, b_dir / 'checkpoints' / name))
    return {'short': short_cfg['name'], 'long': long_cfg['name'], 'until': until,
            'settings_differences': settings, 'trajectory': trajectory, 'checkpoints': checkpoints,
            'first_divergence': first_divergence(settings, trajectory, checkpoints), 'created_at': now()}


def configure_determinism():
    """Explicit deterministic diagnosis mode; must run before the first CUDA/cuBLAS call."""
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def refusal(error):
    """The operation that refused deterministic mode, with the frames that called it."""
    frames = traceback.extract_tb(error.__traceback__)
    return {'message': str(error).splitlines()[0] if str(error) else type(error).__name__,
            'type': type(error).__name__,
            'frames': [f'{Path(f.filename).name}:{f.lineno} {f.name}' for f in frames[-6:]]}


def probe(prep, cfg, steps, deterministic=False):
    """Replay the first `steps` iterations twice with identical inputs; report the first gap."""
    import tempfile
    import torch
    from .gsplat_preflight import qualify, verify_prep
    from . import gsplat_train
    prep = Path(prep).resolve()
    problems = verify_prep(prep) + qualify()['problems']
    if problems:
        raise RuntimeError('; '.join(problems))
    data = gsplat_train.load_cameras(prep, 'train', 'cuda')
    points = gsplat_train.load_points(prep)
    probe_cfg = {**cfg, 'name': f"{cfg['name']}-probe", 'steps': steps, 'log_every': 1,
                 'validate_every': 10 ** 9, 'checkpoint_every': 10 ** 9}
    meta = {'config_sha256': 'probe', 'manifest_sha256': 'probe', 'partition_sha256': 'probe', 'git_commit': 'probe'}
    logs, refused = [], None
    (prep / 'diagnostics').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=prep / 'diagnostics') as temp:
        for run in ('a', 'b'):
            seed_everything(cfg['seed'])
            torch.cuda.synchronize()
            try:
                gsplat_train.train(data, None, points, probe_cfg, Path(temp) / run, meta)
            except RuntimeError as error:
                if not deterministic:
                    raise
                refused = refusal(error)
                break
            torch.cuda.synchronize()
            logs.append(jsonl(Path(temp) / run / 'train.jsonl'))
    mode = 'deterministic' if deterministic else 'default'
    result = {'config': cfg['name'], 'steps': steps, 'mode': mode, 'repetitions': len(logs),
              'same_inputs_and_seed': True, 'environment': environment(), 'created_at': now()}
    if refused:
        result.update(refused_operation=refused,
                      reading='deterministic mode refused an operation: it is named above with its call site')
    else:
        result['first_difference'] = trajectory_differences(*logs, steps)['first_difference']
        result['reading'] = ('torch deterministic algorithms constrain torch operations only; a difference that '
                             'remains points to custom CUDA kernels (gsplat rasterization/backward atomics), '
                             'not to the training settings' if deterministic else
                             'difference with identical inputs, seed and code: run-to-run non-determinism')
    write(prep / 'diagnostics' / f"determinism-{cfg['name']}-{steps}-{mode}.json", result)
    return result


def report(result):
    lines = [f"# Divergence {result['short']} / {result['long']} jusqu’à {result['until']}", '']
    env = result.get('environment')
    if env:
        lines += [f"Environnement : torch {env['torch']}, CUDA {env['cuda']}, cuDNN {env['cudnn']}, "
                  f"gsplat {env['gsplat']}, GPU {env['gpu']}, pilote {env['driver']}.", '']
    lines.append('Paramètres effectifs : ' + ('identiques' if not result['settings_differences'] else
                                              json.dumps(result['settings_differences'], ensure_ascii=False)))
    t = result['trajectory']
    lines += [f"Trajectoires journalisées : {t['logged_steps_compared']} étapes comparées.", '']
    for key, value in t['first_difference'].items():
        lines.append(f"- {key} : première différence à l’étape {value['step']} "
                     f"({value['short']} contre {value['long']})")
    if not t['first_difference']:
        lines.append('- aucune différence dans les journaux')
    lines += ['', '| Checkpoint | Générateur caméras | RNG torch | RNG CUDA | SH | Paramètres identiques | Écart max |',
              '|---|---|---|---|---|---|---:|']
    for c in result['checkpoints']:
        identical = [k for k, v in c['params'].items() if v['status'] == 'identical']
        incomparable = [k for k, v in c['params'].items() if not v['comparable']]
        gaps = [v['max_abs_diff'] for v in c['params'].values() if v['comparable'] and v['max_abs_diff'] is not None]
        gap = (f'{max(gaps):.3g}' if gaps else '0') + (f' ; {NON_COMPARABLE} : {", ".join(incomparable)}'
                                                       if incomparable else '')
        lines.append(f"| {c['step']} | {c['camera_generator_equal']} | {c['torch_rng_equal']} | {c['cuda_rng_equal']} | "
                     f"{c['sh_degree_equal']} | {len(identical)}/{len(c['params'])} | {gap} |")
    d = result['first_divergence']
    lines += ['', f"Première divergence : {d.get('at')} — {d.get('kind', '')} {json.dumps(d.get('detail', ''), ensure_ascii=False)}",
              d.get('interpretation', d.get('note', '')), '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description='First divergence between paired gsplat trainings')
    parser.add_argument('--prep', required=True)
    parser.add_argument('--short', required=True)
    parser.add_argument('--long', help='Paired longer arm (omit with --probe-only)')
    parser.add_argument('--until', type=int, default=3000)
    parser.add_argument('--probe-steps', type=int, help='Replay the short arm twice for this many steps (GPU)')
    parser.add_argument('--probe-only', action='store_true', help='Only run the determinism probe')
    parser.add_argument('--deterministic', action='store_true',
                        help='Probe with torch deterministic algorithms and CUBLAS_WORKSPACE_CONFIG=:4096:8')
    args = parser.parse_args(argv)
    try:
        if args.deterministic:
            if not args.probe_steps:
                raise RuntimeError('--deterministic applies to the probe: give --probe-steps')
            configure_determinism()
        short_cfg = read(args.short)
        if args.probe_only:
            if not args.probe_steps:
                raise RuntimeError('--probe-only needs --probe-steps')
            result = probe(args.prep, short_cfg, args.probe_steps, args.deterministic)
            print(json.dumps({k: result[k] for k in result if k in ('mode', 'repetitions', 'first_difference',
                                                                    'refused_operation')}, indent=2))
            return 0
        if not args.long:
            raise RuntimeError('--long is required unless --probe-only')
        long_cfg = read(args.long)
        result = compare(args.prep, short_cfg, long_cfg, args.until)
        try:
            result['environment'] = environment()
        except ImportError:
            result['environment'] = None
        if args.probe_steps:
            result['determinism_probe'] = probe(args.prep, short_cfg, args.probe_steps, args.deterministic)
        target = Path(args.prep) / 'diagnostics' / f"divergence-{short_cfg['name']}-{long_cfg['name']}"
        write(target / 'divergence.json', result)
        text = report(result)
        if args.probe_steps:
            p = result['determinism_probe']
            text += (f"\nSonde ({p['mode']}, même entrée, même graine, deux fois) : "
                     + json.dumps(p.get('first_difference') or p.get('refused_operation') or 'aucune différence',
                                  ensure_ascii=False) + '\n')
        (target / 'divergence.md').write_text(text)
    except Exception as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1
    print(target / 'divergence.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
