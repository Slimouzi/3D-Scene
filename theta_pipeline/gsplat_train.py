"""Evaluated gsplat training on the GPU VM (environment: requirements/gsplat.lock.txt).

    python -m theta_pipeline.gsplat_train train --prep Output/runs/<prep-run> --config configs/gsplat-l4-short.json
    python -m theta_pipeline.gsplat_train train ... --resume
    python -m theta_pipeline.gsplat_train evaluate-test --prep Output/runs/<prep-run> \
        --config configs/gsplat-l4-short.json --final

Protocol: losses and densification use cameras_train.json only; validation renders select
the checkpoint; the test set is rendered once, by evaluate-test --final, after training.
No quality threshold is applied: metrics are reported, never turned into PASS.
"""
import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
from .gsplat_preflight import load_verified, preflight
from .storage import digest, now, read, write

C0 = 0.28209479177387814


def config_sha256(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


# ---- data ---------------------------------------------------------------------

def load_cameras(prep, group, device, names=None):
    """Images and weights of one set (optionally a subset of faces), each hash-checked on read."""
    import torch
    output = Path(prep).parent
    entries = read(Path(prep) / f'gsplat_inputs/cameras_{group}.json')['cameras']
    if any(c['set'] != group for c in entries):
        raise RuntimeError(f'cameras_{group}.json contains another set')
    if names is not None:
        missing = set(names) - {c['name'] for c in entries}
        if missing:
            raise RuntimeError(f'{sorted(missing)} not in cameras_{group}.json')
        entries = [c for c in entries if c['name'] in set(names)]
    images, weights = [], []
    for c in entries:
        with Image.open(load_verified(output, c['image'])) as raw:
            images.append(np.asarray(raw.convert('RGB')))
        with Image.open(load_verified(output, c['weights'])) as raw:
            weights.append(np.asarray(raw.convert('L')))
    return {'names': [c['name'] for c in entries], 'set': group,
            'images': torch.from_numpy(np.stack(images)).to(device),
            'weights': torch.from_numpy(np.stack(weights)).to(device),
            'Ks': torch.tensor([c['K'] for c in entries], dtype=torch.float32, device=device),
            'viewmats': torch.tensor([c['world_to_camera'] for c in entries], dtype=torch.float32, device=device),
            'width': entries[0]['width'], 'height': entries[0]['height']}


def load_points(prep):
    output = Path(prep).parent
    manifest = read(Path(prep) / 'gsplat_inputs/gsplat_inputs.json')
    rel = next(r for r in manifest['files'] if r.endswith('gsplat_inputs/points.npz'))
    data = np.load(load_verified(output, {'path': rel, 'sha256': manifest['files'][rel]}))
    return {'ids': data['ids'], 'xyz': data['xyz'], 'rgb': data['rgb']}


# ---- model --------------------------------------------------------------------

def scene_scale(viewmats):
    import torch
    centers = torch.linalg.inv(viewmats)[:, :3, 3]
    return float((centers - centers.mean(0)).norm(dim=1).max() * 1.1)


def init_model(points, cfg, scale, device):
    import torch
    xyz = torch.tensor(points['xyz'], dtype=torch.float32)
    rgb = torch.tensor(points['rgb'], dtype=torch.float32)
    n = len(xyz)
    distances = torch.cdist(xyz, xyz)
    distances.fill_diagonal_(float('inf'))
    neighbours = distances.topk(min(3, n - 1), largest=False).values.mean(1).clamp_min(1e-7)
    degree = cfg['sh_degree']
    params = torch.nn.ParameterDict({
        'means': torch.nn.Parameter(xyz),
        'scales': torch.nn.Parameter(torch.log(neighbours * cfg['init_scale'])[:, None].repeat(1, 3)),
        'quats': torch.nn.Parameter(torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1)),
        'opacities': torch.nn.Parameter(torch.logit(torch.full((n,), cfg['init_opacity']))),
        'sh0': torch.nn.Parameter(((rgb - .5) / C0)[:, None, :]),
        'shN': torch.nn.Parameter(torch.zeros(n, (degree + 1) ** 2 - 1, 3)),
    }).to(device)
    return params, optimizers_for(params, cfg, scale)


def optimizers_for(params, cfg, scale):
    import torch
    lr = {**cfg['lr'], 'means': cfg['lr']['means'] * scale}
    return {name: torch.optim.Adam([{'params': params[name], 'lr': lr[name], 'name': name}], eps=1e-15)
            for name in params}


def render(params, viewmats, Ks, width, height, sh_degree):
    import torch
    from gsplat import rasterization
    colors = torch.cat([params['sh0'], params['shN']], 1)
    renders, _, info = rasterization(
        means=params['means'], quats=params['quats'], scales=torch.exp(params['scales']),
        opacities=torch.sigmoid(params['opacities']), colors=colors, viewmats=viewmats, Ks=Ks,
        width=width, height=height, sh_degree=sh_degree, packed=False)
    return renders, info


def densification_schedule(cfg):
    """0-based iterations at which gsplat 1.5.3 DefaultStrategy acts, from its own conditions.

    refine: grow + prune when start < i < stop, i % every == 0, i % reset_every >= pause.
    large_pruning: the subset with i > reset_every. opacity_reset: gsplat tests
    `i % reset_every == 0 & i > 0`, which operator precedence makes always false.
    """
    s = {'refine_start_iter': 500, 'refine_stop_iter': 15000, 'refine_every': 100, 'reset_every': 3000,
         'pause_refine_after_reset': 0, **cfg['strategy']}
    refine = [i for i in range(cfg['steps']) if s['refine_start_iter'] < i < s['refine_stop_iter']
              and i % s['refine_every'] == 0 and i % s['reset_every'] >= s['pause_refine_after_reset']]
    reset = [i for i in range(min(cfg['steps'], s['refine_stop_iter']))
             if (i % s['reset_every'] == 0 & i > 0)]
    return {'refine': refine, 'large_pruning': [i for i in refine if i > s['reset_every']],
            'opacity_reset': reset}


def schedule_summary(cfg):
    def span(values):
        return {'count': len(values), 'first': values[0] if values else None, 'last': values[-1] if values else None}
    return {k: span(v) for k, v in densification_schedule(cfg).items()}


# ---- losses and metrics ---------------------------------------------------------

def ssim_map(a, b, valid):
    """Masked SSIM (11x11 Gaussian, sigma 1.5), channel mean; inputs [N,H,W,3], valid [N,H,W].

    Window statistics are computed over valid pixels only, so an excluded pixel never
    enters any window: changing it changes neither the value nor the gradient.
    """
    import torch
    import torch.nn.functional as F
    v = valid.to(a.dtype)[:, None]
    x, y = a.permute(0, 3, 1, 2) * v, b.permute(0, 3, 1, 2) * v
    g = torch.exp(-(torch.arange(11, device=a.device, dtype=a.dtype) - 5) ** 2 / (2 * 1.5 ** 2))
    g = g / g.sum()
    window = (g[:, None] * g[None, :])[None, None]
    blur = lambda t: F.conv2d(t.reshape(-1, 1, *t.shape[-2:]), window, padding=5).reshape(t.shape)
    mass = blur(v).clamp_min(1e-6)
    mean = lambda t: blur(t) / mass
    mx, my = mean(x), mean(y)
    vx = (mean(x * x) - mx ** 2).clamp_min(0)
    vy = (mean(y * y) - my ** 2).clamp_min(0)
    cxy = mean(x * y) - mx * my
    c1, c2 = .01 ** 2, .03 ** 2
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))
    return s.mean(1) * valid.to(a.dtype)


def weighted_loss(rendered, target, weight, ssim_lambda):
    """Appearance weights (0 excluded, 0.5 reflective, 1 normal) scale every pixel term.

    A view without any valid pixel is refused: it would make an empty, meaningless loss.
    """
    total = weight.sum()
    if total <= 0:
        raise RuntimeError('training view has no valid pixel')
    l1 = (weight[..., None] * (rendered - target).abs()).sum() / (3 * total)
    ssim = (weight * ssim_map(rendered, target, weight > 0)).sum() / total
    return (1 - ssim_lambda) * l1 + ssim_lambda * (1 - ssim), l1, ssim


def view_metrics(rendered, target, weight):
    """Weighted PSNR/SSIM of one view, or None when it has no valid pixel (never 100 dB)."""
    import torch
    fraction = float((weight > 0).float().mean())
    total = weight.sum()
    if total <= 0:
        return {'psnr': None, 'ssim': None, 'evaluated_fraction': fraction, 'excluded': True}
    mse = (weight[..., None] * (rendered - target) ** 2).sum() / (3 * total)
    ssim = (weight * ssim_map(rendered, target, weight > 0)).sum() / total
    return {'psnr': float(-10 * torch.log10(mse.clamp_min(1e-10))), 'ssim': float(ssim),
            'evaluated_fraction': fraction, 'excluded': False}


def aggregate(group, rows, sh_degree):
    usable = [r for r in rows if not r['excluded']]
    return {'set': group, 'sh_degree': sh_degree, 'cameras': rows,
            'usable_cameras': len(usable), 'excluded_cameras': [r['camera'] for r in rows if r['excluded']],
            'mean_psnr': float(np.mean([r['psnr'] for r in usable])) if usable else None,
            'mean_ssim': float(np.mean([r['ssim'] for r in usable])) if usable else None}


def evaluate(params, data, sh_degree):
    """Per-view metrics; views without valid pixels are listed and left out of the means."""
    import torch
    rows = []
    with torch.no_grad():
        for i, name in enumerate(data['names']):
            rendered, _ = render(params, data['viewmats'][i:i + 1], data['Ks'][i:i + 1],
                                 data['width'], data['height'], sh_degree)
            rows.append({'camera': name, **view_metrics(rendered.clamp(0, 1),
                                                        data['images'][i:i + 1].float() / 255,
                                                        data['weights'][i:i + 1].float() / 255)})
    return aggregate(data['set'], rows, sh_degree)


# ---- checkpoints -------------------------------------------------------------

def save_checkpoint(folder, step, params, optimizers, scheduler, strategy_state, generator, meta, sh_degree):
    import torch
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'step_{step:06d}.pt'
    tmp = path.with_suffix('.tmp')
    torch.save({'step': step, 'meta': meta, 'sh_degree': sh_degree,
                'params': {k: v.detach().cpu() for k, v in params.items()},
                'optimizers': {k: o.state_dict() for k, o in optimizers.items()},
                'scheduler': scheduler.state_dict(),
                'strategy_state': {k: v.cpu() if hasattr(v, 'cpu') else v for k, v in strategy_state.items()},
                'rng': {'torch': torch.get_rng_state(),
                        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                        'generator': generator.get_state()}}, tmp)
    os.replace(tmp, path)
    return path


def latest_checkpoint(folder):
    found = sorted(Path(folder).glob('step_*.pt'))
    return found[-1] if found else None


def restore(path, cfg, meta, device):
    """Parameters, optimizer, scheduler, strategy and RNG state; refuses a foreign run."""
    import torch
    saved = torch.load(path, map_location='cpu', weights_only=False)
    for key in ('config_sha256', 'manifest_sha256', 'partition_sha256', 'git_commit'):
        if saved['meta'].get(key) != meta.get(key):
            raise RuntimeError(f'{path.name}: {key} differs from this run; cannot resume')
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(v.to(device)) for k, v in saved['params'].items()})
    optimizers = optimizers_for(params, cfg, saved['meta']['scene_scale'])
    for k, o in optimizers.items():
        o.load_state_dict(saved['optimizers'][k])
    state = {k: v.to(device) if hasattr(v, 'to') else v for k, v in saved['strategy_state'].items()}
    torch.set_rng_state(saved['rng']['torch'])
    if saved['rng']['cuda']:
        torch.cuda.set_rng_state_all(saved['rng']['cuda'])
    generator = torch.Generator()
    generator.set_state(saved['rng']['generator'])
    return saved['step'], params, optimizers, saved['scheduler'], state, generator, saved


# ---- training ----------------------------------------------------------------

def log_line(path, record):
    with open(path, 'a') as f:
        f.write(json.dumps(record) + '\n')


def train(train_data, val_data, points, cfg, out, meta, resume=False, device='cuda'):
    """Train on train_data only. val_data is rendered without gradients for selection."""
    if train_data['set'] != 'train' or (val_data is not None and val_data['set'] != 'validation'):
        raise RuntimeError('train() takes the train set for losses and the validation set for selection')
    empty = [name for name, w in zip(train_data['names'], train_data['weights']) if not bool((w > 0).any())]
    if empty:
        raise RuntimeError(f'train views without any valid pixel: {empty}')
    import torch
    from gsplat.strategy import DefaultStrategy
    out = Path(out)
    scale = scene_scale(train_data['viewmats'])
    meta = {**meta, 'scene_scale': scale}
    strategy = DefaultStrategy(verbose=False, **cfg['strategy'])
    checkpoint = latest_checkpoint(out / 'checkpoints') if resume else None
    if resume and checkpoint is None:
        raise RuntimeError('--resume given but no checkpoint exists')
    if checkpoint:
        start, params, optimizers, scheduler_state, state, generator, saved = restore(checkpoint, cfg, meta, device)
        meta['scene_scale'] = scale = saved['meta']['scene_scale']
    else:
        if (out / 'checkpoints').exists() and any((out / 'checkpoints').iterdir()):
            raise RuntimeError(f'{out} already has checkpoints: use --resume or a new config name')
        torch.manual_seed(cfg['seed'])
        start, (params, optimizers), scheduler_state = 0, init_model(points, cfg, scale, device), None
        state = strategy.initialize_state(scene_scale=scale)
        generator = torch.Generator().manual_seed(cfg['seed'])
    strategy.check_sanity(params, optimizers)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizers['means'], gamma=cfg['means_lr_final_ratio'] ** (1. / cfg['steps']))
    if scheduler_state:
        scheduler.load_state_dict(scheduler_state)
    n = len(train_data['names'])
    began = time.monotonic()
    for step in range(start, cfg['steps']):
        index = int(torch.randint(n, (1,), generator=generator))
        degree = min(step // cfg['sh_degree_interval'], cfg['sh_degree'])
        rendered, info = render(params, train_data['viewmats'][index:index + 1], train_data['Ks'][index:index + 1],
                                train_data['width'], train_data['height'], degree)
        strategy.step_pre_backward(params, optimizers, state, step, info)
        target = train_data['images'][index:index + 1].float() / 255
        weight = train_data['weights'][index:index + 1].float() / 255
        loss, l1, ssim = weighted_loss(rendered, target, weight, cfg['ssim_lambda'])
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
        done = step + 1
        if done % cfg['log_every'] == 0 or done == cfg['steps']:
            log_line(out / 'train.jsonl', {
                'step': done, 'camera': train_data['names'][index], 'loss': loss.item(), 'l1': l1.item(),
                'ssim': ssim.item(), 'gaussians': len(params['means']), 'sh_degree': degree,
                'lr_means': optimizers['means'].param_groups[0]['lr'],
                'seconds': round(time.monotonic() - began, 2),
                'max_memory_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
                if torch.cuda.is_available() else None})
        validate = val_data is not None and (done % cfg['validate_every'] == 0 or done == cfg['steps'])
        if done % cfg['checkpoint_every'] == 0 or done == cfg['steps'] or validate:
            # The degree actually used is stored, so any later evaluation renders identically.
            path = save_checkpoint(out / 'checkpoints', done, params, optimizers, scheduler, state, generator,
                                   meta, degree)
        if validate:
            result = evaluate(params, val_data, degree)
            log_line(out / 'validation.jsonl', {'step': done, 'checkpoint': path.name, **result})
            select(out)
    return params


def select(out):
    """Checkpoint choice from validation only: best mean PSNR over usable views, earliest step on ties."""
    rows = [json.loads(line) for line in (Path(out) / 'validation.jsonl').read_text().splitlines()]
    usable = [r for r in rows if r['mean_psnr'] is not None]
    if not usable:
        raise RuntimeError('no usable validation view: checkpoint selection refused')
    best = max(usable, key=lambda r: (r['mean_psnr'], -r['step']))
    write(Path(out) / 'selection.json', {'criterion': 'max mean validation PSNR over usable views, '
                                                      'earliest step on ties',
                                         'step': best['step'], 'checkpoint': best['checkpoint'],
                                         'sh_degree': best['sh_degree'],
                                         'validation_mean_psnr': best['mean_psnr'],
                                         'validation_mean_ssim': best['mean_ssim'],
                                         'validation_usable_cameras': best['usable_cameras'],
                                         'test_used': False})


# ---- command line -------------------------------------------------------------

def locked(prep):
    lock = (Path(prep) / '.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another process is using this run') from None
    return lock


def start(args):
    prep = Path(args.prep).resolve()
    cfg = read(args.config)
    out = prep / 'training' / cfg['name']
    report = preflight(prep)
    if not report['ok']:
        raise RuntimeError('Preflight refused training: ' + '; '.join(report['problems']))
    manifest = read(prep / 'gsplat_inputs/gsplat_inputs.json')
    meta = {'config_sha256': config_sha256(cfg),
            'manifest_sha256': digest(prep / 'gsplat_inputs/gsplat_inputs.json'),
            'partition_sha256': manifest['partition_sha256'], 'git_commit': report['git_commit']}
    return prep, cfg, out, report, manifest, meta


def command_train(args):
    prep, cfg, out, report, manifest, meta = start(args)
    lock = locked(prep)
    out.mkdir(parents=True, exist_ok=True)
    record = out / 'training.json'
    history = read(record)['history'] if record.exists() else []
    history.append({'started_at': now(), 'resume': args.resume, 'git_commit': report['git_commit']})
    status = {'schema_version': 1, 'name': cfg['name'], 'config': cfg, **meta,
              'prep_run': prep.name, 'split_run': manifest['split_run'],
              'environment': report['environment'], 'cameras': manifest['cameras'],
              'points': manifest['points'], 'protocol': manifest['protocol'],
              'limitation': manifest['limitation'],
              'quality_thresholds': 'none: metrics are reported, not judged',
              'densification_schedule': schedule_summary(cfg),
              'status': 'running', 'history': history}
    write(record, status)
    try:
        device = 'cuda'
        train_data = load_cameras(prep, 'train', device)
        val_data = load_cameras(prep, 'validation', device)
        train(train_data, val_data, load_points(prep), cfg, out, meta, args.resume, device)
        status.update(status='completed', finished_at=now(),
                      selection=read(out / 'selection.json'))
    except BaseException as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}', failed_at=now())
        raise
    finally:
        write(record, status)
        lock.close()
    print(json.dumps({'status': status['status'], 'selection': status.get('selection')}, indent=2))
    return 0


def command_evaluate_test(args):
    """The single final test evaluation, on the checkpoint chosen by validation."""
    if not args.final:
        raise RuntimeError('evaluate-test renders the reserved test set once: pass --final')
    prep, cfg, out, report, manifest, meta = start(args)
    lock = locked(prep)
    try:
        target = out / 'test_evaluation.json'
        if target.exists():
            raise RuntimeError(f'{target} exists: the test set has already been used')
        training = read(out / 'training.json')
        if training['status'] != 'completed':
            raise RuntimeError('training is not completed')
        selection = read(out / 'selection.json')
        _, params, *_, saved = restore(out / 'checkpoints' / selection['checkpoint'], cfg, meta, 'cuda')
        test_data = load_cameras(prep, 'test', 'cuda')
        result = evaluate(params, test_data, saved['sh_degree'])
        write(target, {'schema_version': 1, 'selection': selection, **meta, **result,
                       'limitation': manifest['limitation'],
                       'quality_thresholds': 'none: metrics are reported, not judged',
                       'evaluated_at': now()})
        write(out / 'selection.json', {**selection, 'test_used': True})
    finally:
        lock.close()
    print(json.dumps({'mean_psnr': result['mean_psnr'], 'mean_ssim': result['mean_ssim']}, indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description='Evaluated gsplat training (GPU)')
    parser.add_argument('command', choices=['train', 'evaluate-test'])
    parser.add_argument('--prep', required=True, help='Prepared run, e.g. Output/runs/salon-gsplat-006')
    parser.add_argument('--config', required=True, help='Training config, e.g. configs/gsplat-l4-short.json')
    parser.add_argument('--resume', action='store_true', help='Continue from the latest checkpoint')
    parser.add_argument('--final', action='store_true', help='Confirm the one-time test evaluation')
    args = parser.parse_args(argv)
    try:
        return command_train(args) if args.command == 'train' else command_evaluate_test(args)
    except Exception as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
