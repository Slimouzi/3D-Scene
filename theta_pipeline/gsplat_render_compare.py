"""Read-only comparison of one checkpoint rendered in two modes: 'historique' and '3dgut'.

    python -m theta_pipeline.gsplat_render_compare --prep Output/runs/salon-gsplat-009 \
        --config configs/gsplat-absgs-abs-seed0.json --checkpoint step_003000.pt \
        [--train-faces 3] [--repetitions 3] [--warmup 1]

The checkpoint is loaded once; both modes render exactly the same parameters with the same
poses, intrinsics, resolution, appearance weights and SH degree (the degree stored in the
checkpoint). Only the rasterization differs (gsplat_inspect.RENDERERS). Faces: every
validation face, plus optional train faces chosen by content; never the test set.

Raw outputs are checked before any clamp: a non-finite RGB, alpha or depth value makes that
face and mode invalid (no metric is computed from it) and the whole inspection invalid.
Each face is rendered `repetitions` times per mode after `warmup` untimed renders: GPU time
(synchronized), peak memory, visible Gaussians and projected radii are reported per mode, and
the spread between repetitions measures the variability of RENDERING only (not of training).

Runs: one stable scientific identity (checkpoint, config, preparation, renderers, faces,
repetitions) per folder; each execution is a new attempt-NNN with status running, failed or
completed. Earlier attempts are kept; a failed attempt never blocks a retry. Nothing is
written under training/; the checkpoint SHA-256 is checked before and after. The parameters
were optimized with the historical renderer: 3DGUT renders qualify the renderer on these
checkpoints, they are not a 3DGUT-trained model. No quality threshold.
"""
import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .storage import digest, now, read, write

MODES = ('historique', '3dgut')
REPORT_REGIONS = ('glass', 'mirror', 'unvalidated_reflective', 'furniture', 'contours', 'other')


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def run_identity(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def driver_version():
    try:
        return subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def non_finite(raw_rgb, alpha, depth):
    """Non-finite counts of RAW outputs (before any clamp: clamping would turn +inf into white)."""
    import torch
    count = lambda t: int((~torch.isfinite(t)).sum())
    return {'rgb': count(raw_rgb), 'alpha': count(alpha), 'depth': count(depth)}


def output_statistics(alpha, depth, weight):
    """Alpha and depth statistics on the valid pixels of the face."""
    a, d, valid = alpha.cpu().numpy(), depth.cpu().numpy(), weight.cpu().numpy() > 0
    covered = valid & (a > .5) & np.isfinite(d)
    q = lambda values, p: float(np.quantile(values, p)) if values.size else None
    return {'alpha_mean': float(a[valid].mean()) if valid.any() else None,
            'alpha_below_half_fraction': float((a[valid] < .5).mean()) if valid.any() else None,
            'depth_p05': q(d[covered], .05), 'depth_p50': q(d[covered], .5), 'depth_p95': q(d[covered], .95)}


def projection_statistics(meta):
    """Visible Gaussians and projected radii from gsplat's own projection (meta['radii'])."""
    radii = meta.get('radii') if isinstance(meta, dict) else None
    if radii is None:
        return {'visible_gaussians': None, 'radius_p50': None, 'radius_p95': None, 'radius_max': None}
    r = radii.reshape(-1, radii.shape[-1]).float().cpu().numpy()
    visible = (r > 0).all(1)
    largest = r[visible].max(1) if visible.any() else np.zeros(0)
    q = lambda p: float(np.quantile(largest, p)) if largest.size else None
    return {'visible_gaussians': int(visible.sum()), 'radius_p50': q(.5), 'radius_p95': q(.95),
            'radius_max': float(largest.max()) if largest.size else None}


def face_metrics(rgb, target, weight, labels, gsplat_train, inspect):
    """PSNR/SSIM, luminance and per-region metrics on the valid pixels (identical across modes)."""
    import torch
    from .gsplat_checkpoint_diag import luminance
    metrics = gsplat_train.view_metrics(rgb[None], target[None], weight[None])
    reference = target.cpu().numpy() * 255
    rendered = rgb.cpu().numpy() * 255
    w = weight.cpu().numpy()
    valid = w > 0
    regions = {name: np.isin(labels, codes) for name, codes in inspect.REGIONS.items()}
    regions['contours'] = inspect.contour_mask(reference.astype(np.uint8), valid)
    regions['other'] = valid & ~np.logical_or.reduce(list(regions.values()))
    rows = inspect.region_metrics(reference, rendered, w, regions)
    with torch.no_grad():
        ssim_map = gsplat_train.ssim_map(rgb[None], target[None], weight[None] > 0)[0].cpu().numpy()
    for name, mask in regions.items():
        inside = mask & valid
        rows[name]['ssim'] = float(ssim_map[inside].mean()) if inside.any() else None
    return {**metrics, 'render_luminance': luminance(rendered / 255, valid),
            'reference_luminance': luminance(reference / 255, valid), 'regions': rows}


class Timer:
    """GPU-synchronized wall time and peak memory of one render (CPU fallback without CUDA)."""

    def __init__(self, torch):
        self.torch, self.cuda = torch, torch.cuda.is_available()

    def __enter__(self):
        if self.cuda:
            self.torch.cuda.synchronize()
            self.torch.cuda.reset_peak_memory_stats()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.cuda:
            self.torch.cuda.synchronize()
        self.ms = (time.perf_counter() - self.start) * 1000
        self.peak = self.torch.cuda.max_memory_allocated() if self.cuda else None
        return False


def render_mode(inspect, params, data, i, degree, mode, repetitions, torch):
    """`repetitions` timed renders; returns the first one, timings, peak memory and repetition spread."""
    times, peaks, outputs, metas = [], [], [], []
    for _ in range(repetitions):
        with Timer(torch) as timer:
            raw, alpha, meta = inspect.render_raw(params, data, i, degree, mode)
        times.append(timer.ms)
        peaks.append(timer.peak)
        outputs.append((raw, alpha))
        metas.append(meta)
    first_raw, first_alpha = outputs[0]
    spread = {'rgb_max_abs_diff': 0., 'alpha_max_abs_diff': 0., 'depth_max_abs_diff': 0.}
    for raw, alpha in outputs[1:]:
        both = torch.isfinite(raw) & torch.isfinite(first_raw)
        diff = (raw - first_raw).abs()
        spread['rgb_max_abs_diff'] = max(spread['rgb_max_abs_diff'], float(diff[..., :3][both[..., :3]].max())
                                         if both[..., :3].any() else 0.)
        spread['depth_max_abs_diff'] = max(spread['depth_max_abs_diff'], float(diff[..., 3][both[..., 3]].max())
                                           if both[..., 3].any() else 0.)
        finite = torch.isfinite(alpha) & torch.isfinite(first_alpha)
        spread['alpha_max_abs_diff'] = max(spread['alpha_max_abs_diff'],
                                           float((alpha - first_alpha).abs()[finite].max()) if finite.any() else 0.)
    counts = {k: sum(non_finite(raw[..., :3], alpha, raw[..., 3])[k] for raw, alpha in outputs)
              for k in ('rgb', 'alpha', 'depth')}
    return {'raw': first_raw, 'alpha': first_alpha, 'meta': metas[0], 'times_ms': times,
            'peak_memory_bytes': max((p for p in peaks if p is not None), default=None),
            'repetition_spread': spread, 'non_finite': counts}


def summarize(faces):
    out = {}
    for mode in MODES:
        for group in sorted({f['set'] for f in faces}):
            for pano in sorted({f['panorama_id'] for f in faces if f['set'] == group}):
                rows = [f[mode] for f in faces if f['set'] == group and f['panorama_id'] == pano
                        and f[mode]['valid'] and not f[mode]['excluded']]
                mean = lambda values: statistics.fmean(values) if values else None
                region = lambda name, key: mean([r['regions'][name][key] for r in rows
                                                 if r['regions'].get(name, {}).get(key) is not None])
                out.setdefault(mode, {}).setdefault(group, {})[pano] = {
                    'usable': len(rows), 'psnr': mean([r['psnr'] for r in rows]), 'ssim': mean([r['ssim'] for r in rows]),
                    'luminance_minus_reference': mean([r['render_luminance'] - r['reference_luminance'] for r in rows]),
                    'regions': {name: {'psnr': region(name, 'psnr'), 'ssim': region(name, 'ssim'),
                                       'faces': sum(1 for r in rows if r['regions'].get(name, {}).get('psnr') is not None)}
                                for name in REPORT_REGIONS}}
    return out


def performance(faces):
    out = {}
    for mode in MODES:
        times = [t for f in faces for t in f[mode]['times_ms']]
        peaks = [f[mode]['peak_memory_bytes'] for f in faces if f[mode]['peak_memory_bytes'] is not None]
        visible = [f[mode]['projection']['visible_gaussians'] for f in faces
                   if f[mode]['projection']['visible_gaussians'] is not None]
        spread = [f[mode]['repetition_spread'] for f in faces]
        out[mode] = {'renders': len(times), 'time_ms_median': statistics.median(times) if times else None,
                     'time_ms_p95': float(np.quantile(times, .95)) if times else None,
                     'peak_memory_mb': max(peaks) / 2 ** 20 if peaks else None,
                     'visible_gaussians_median': statistics.median(visible) if visible else None,
                     'radius_p95_median': statistics.median([f[mode]['projection']['radius_p95'] for f in faces
                                                             if f[mode]['projection']['radius_p95'] is not None] or [0]),
                     'repetition_rgb_max_abs_diff': max((s['rgb_max_abs_diff'] for s in spread), default=0.),
                     'repetition_depth_max_abs_diff': max((s['depth_max_abs_diff'] for s in spread), default=0.)}
    return out


def overview(rows, size=160):
    """One sheet for all faces: reference, then render and error for each mode (shared scales)."""
    from .gsplat_inspect import error_image
    columns = ['reference'] + [f'{m} {k}' for m in MODES for k in ('rendu', 'erreur')]
    out = Image.new('RGB', (len(columns) * size + 260, len(rows) * size + 24), '#1d252c')
    draw = ImageDraw.Draw(out)
    for k, label in enumerate(columns):
        draw.text((260 + k * size + 4, 6), label, fill='white')
    for r, row in enumerate(rows):
        y = 24 + r * size
        draw.text((6, y + 6), row['label'], fill='white')
        panels = [row['reference']]
        for mode in MODES:
            rgb = row[mode]
            panels += [np.clip(np.nan_to_num(rgb, nan=1., posinf=1., neginf=0.) * 255, 0, 255).astype(np.uint8),
                       error_image(row['reference'], np.nan_to_num(rgb, nan=1., posinf=1., neginf=0.) * 255, row['weight'])]
        for k, panel in enumerate(panels):
            out.paste(Image.fromarray(panel).resize((size, size)), (260 + k * size, y))
    return out


def new_attempt(folder, identity_inputs):
    """Next attempt-NNN in the identity folder; earlier attempts are never touched."""
    folder.mkdir(parents=True, exist_ok=True)
    index_path = folder / 'index.json'
    index = read(index_path) if index_path.exists() else {'identity': identity_inputs, 'attempts': []}
    number = 1 + max((a['attempt'] for a in index['attempts']), default=0)
    attempt = folder / f'attempt-{number:03d}'
    attempt.mkdir()
    index['attempts'].append({'attempt': number, 'folder': attempt.name, 'status': 'running', 'started_at': now()})
    write(index_path, index)
    write(attempt / 'status.json', {'attempt': number, 'status': 'running', 'started_at': now()})
    return attempt, number


def set_status(folder, number, status, **extra):
    index = read(folder / 'index.json')
    for entry in index['attempts']:
        if entry['attempt'] == number:
            entry.update(status=status, finished_at=now(), **extra)
    write(folder / 'index.json', index)
    attempt = folder / f'attempt-{number:03d}'
    write(attempt / 'status.json', {**read(attempt / 'status.json'), 'status': status, 'finished_at': now(), **extra})


def compare(prep, cfg, checkpoint_name, train_faces=0, repetitions=3, warmup=1):
    import torch
    from . import gsplat_inspect as inspect, gsplat_train
    from .gsplat_checkpoint_diag import sheet
    from .gsplat_preflight import qualify, verify_prep
    from .segmentation.provenance import git_commit
    if repetitions < 1:
        raise ValueError('repetitions must be at least 1')
    prep = Path(prep).resolve()
    problems = verify_prep(prep) + qualify()['problems']
    if problems:
        raise RuntimeError('; '.join(problems))
    training = prep / 'training' / cfg['name']
    checkpoint = training / 'checkpoints' / checkpoint_name
    if not checkpoint.is_file():
        raise RuntimeError(f'{checkpoint_name} is not a checkpoint of {cfg["name"]}')
    status = read(training / 'training.json')
    manifest = read(prep / 'gsplat_inputs/gsplat_inputs.json')
    manifest_sha = digest(prep / 'gsplat_inputs/gsplat_inputs.json')
    meta = {'config_sha256': gsplat_train.config_sha256(cfg), 'manifest_sha256': manifest_sha,
            'partition_sha256': manifest['partition_sha256'], 'git_commit': status['git_commit']}
    sha = digest(checkpoint)
    sources = inspect.Sources(prep.parent, manifest)
    cameras = {'validation': read(prep / 'gsplat_inputs/cameras_validation.json')['cameras']}
    if train_faces:
        train = read(prep / 'gsplat_inputs/cameras_train.json')['cameras']
        chosen, _ = inspect.select_train_faces(train, sources.labels, sources.rotations, train[0]['width'], train_faces)
        cameras['train'] = [c for c in train if c['name'] in chosen]
    identity_inputs = {'checkpoint_sha256': sha, 'config_sha256': meta['config_sha256'],
                       'prep_manifest_sha256': manifest_sha, 'renderers': inspect.RENDERERS,
                       'faces': {g: [c['name'] for c in v] for g, v in cameras.items()},
                       'repetitions': repetitions, 'warmup': warmup}
    identity = run_identity(identity_inputs)
    folder = prep / 'inspection' / 'render-compare' / f"{cfg['name']}-{checkpoint.stem}-{identity[:12]}"
    target, number = new_attempt(folder, identity_inputs)
    try:
        _, params, *_, saved = gsplat_train.restore(checkpoint, cfg, meta, 'cuda')   # loaded once for both modes
        degree = saved['sh_degree']
        faces, invalid, overview_rows = [], [], []
        for group, entries in cameras.items():
            data = gsplat_train.load_cameras(prep, group, 'cuda', [c['name'] for c in entries])
            by_name = {c['name']: c for c in entries}
            for mode in MODES:                                         # untimed warm-up, not reported
                for _ in range(warmup):
                    inspect.render_raw(params, data, 0, degree, mode)
            for i, name in enumerate(data['names']):
                camera = by_name[name]
                target_rgb = data['images'][i].float() / 255
                weight = data['weights'][i].float() / 255
                labels = inspect.project_labels(sources.labels(camera['panorama_id']), sources.rotations[name],
                                                data['width'])
                row, variants = {'camera': name, 'panorama_id': camera['panorama_id'], 'set': group}, []
                thumbs = {'label': f'{group} {name}', 'reference': data['images'][i].cpu().numpy(),
                          'weight': weight.cpu().numpy()}
                for mode in MODES:
                    out = render_mode(inspect, params, data, i, degree, mode, repetitions, torch)
                    raw, alpha = out['raw'], out['alpha']
                    rgb, depth = raw[..., :3], raw[..., 3]
                    valid = not any(out['non_finite'].values())
                    entry = {'valid': valid, 'non_finite': out['non_finite'], 'times_ms': out['times_ms'],
                             'peak_memory_bytes': out['peak_memory_bytes'],
                             'repetition_spread': out['repetition_spread'],
                             'projection': projection_statistics(out['meta'])}
                    if valid:
                        entry.update(face_metrics(rgb.clamp(0, 1), target_rgb, weight, labels, gsplat_train, inspect))
                        entry.update(output_statistics(alpha, depth, weight))
                    else:
                        invalid.append({'set': group, 'camera': name, 'mode': mode, 'non_finite': out['non_finite']})
                        entry.update(psnr=None, ssim=None, excluded=True, regions={})
                    row[mode] = entry
                    shown = rgb.clamp(0, 1).cpu().numpy() if valid else np.nan_to_num(rgb.cpu().numpy(), nan=1.)
                    variants.append({'label': mode + ('' if valid else ' INVALIDE'), 'rgb': np.clip(shown, 0, 1),
                                     'alpha': np.nan_to_num(alpha.cpu().numpy()), 'depth': depth.cpu().numpy(),
                                     'weight': weight.cpu().numpy(), 'psnr': entry['psnr']})
                    thumbs[mode] = rgb.cpu().numpy()
                stem = f"{group}__{name.replace('/', '__').removesuffix('.png')}"
                sheet(name, data['images'][i].cpu().numpy(), variants, 'erreur absolue 0-0,25').save(
                    target / f'{stem}.jpg', quality=90)
                row['sheet'] = f'{stem}.jpg'
                faces.append(row)
                overview_rows.append(thumbs)
            del data
            torch.cuda.empty_cache()
        overview(overview_rows).save(target / 'overview.jpg', quality=88)
        if digest(checkpoint) != sha:
            raise RuntimeError(f'{checkpoint} changed during the comparison')
        result = {
            'schema_version': 2, 'tool': 'gsplat_render_compare', 'run_identity': identity, 'attempt': number,
            'valid': not invalid, 'invalid_outputs': invalid, 'training': cfg['name'], 'checkpoint': checkpoint.name,
            'checkpoint_sha256': sha, 'checkpoint_loads': 1, 'checkpoint_sha256_by_mode': {m: sha for m in MODES},
            'sh_degree': degree, 'config_sha256': meta['config_sha256'], 'prep_run': prep.name,
            'prep_manifest_sha256': manifest_sha, 'partition_sha256': manifest['partition_sha256'],
            'renderers': inspect.RENDERERS, 'repetitions': repetitions, 'warmup': warmup,
            'environment': {'gsplat': package_version('gsplat'), 'torch': package_version('torch'),
                            'cuda': torch.version.cuda, 'driver': driver_version(),
                            'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                            'python': platform.python_version()},
            'analysis_git_commit': git_commit(), 'sets': sorted(cameras), 'test_loaded': False,
            'shared_inputs': 'same checkpoint object, poses, intrinsics, resolution, weights and SH degree',
            'repetition_scope': 'variability of rendering the same parameters; says nothing about training '
                                'non-determinism',
            'caveat': 'parameters optimized with the historical renderer; 3DGUT renders do not represent a '
                      '3DGUT-trained model', 'faces': faces, 'summary': summarize(faces),
            'performance': performance(faces), 'created_at': now()}
        write(target / 'comparison.json', result)
        (target / 'comparison.md').write_text(report(result))
    except BaseException as error:
        set_status(folder, number, 'failed', error=f'{type(error).__name__}: {error}',
                   traceback=traceback.format_exc(limit=8))
        raise
    set_status(folder, number, 'completed', valid=result['valid'], invalid_outputs=len(invalid))
    return target


def report(result):
    fmt = lambda v, d=2: '—' if v is None else f'{v:.{d}f}'
    signed = lambda v, d=3: '—' if v is None else f'{v:+.{d}f}'
    env = result['environment']
    lines = [f"# Rendu historique / 3DGUT — {result['training']} / {result['checkpoint']} (tentative {result['attempt']})", '']
    if not result['valid']:
        lines += ['**INSPECTION INVALIDE** : sorties non finies (avant écrêtage) pour '
                  + ', '.join(f"{x['set']} {x['camera']} [{x['mode']}] {x['non_finite']}" for x in result['invalid_outputs'])
                  + '. Aucune métrique n’est calculée pour ces faces et modes.', '']
    lines += [f"Checkpoint `{result['checkpoint_sha256'][:16]}` chargé une fois et rendu dans les deux modes ; degré SH "
              f"{result['sh_degree']} ; gsplat {env['gsplat']}, torch {env['torch']}, CUDA {env['cuda']}, pilote "
              f"{env['driver']}, GPU {env['device']} ; commit d’analyse `{result['analysis_git_commit']}` ; identité "
              f"`{result['run_identity'][:12]}`. Jeu de test non utilisé.", '',
              'Paramètres : ' + '; '.join(f"{m} = {json.dumps(p, sort_keys=True)}" for m, p in result['renderers'].items()), '',
              f"Réserve : {result['caveat']}.", '', '## Synthèse par panorama', '',
              '| Ensemble | Panorama | PSNR historique | PSNR 3DGUT | ΔPSNR | SSIM historique | SSIM 3DGUT | '
              'Luminance − référence historique / 3DGUT |', '|---|---|---:|---:|---:|---:|---:|---|']
    summary = result['summary']
    for group, panos in summary['historique'].items():
        for pano, h in panos.items():
            g = summary['3dgut'][group][pano]
            delta = None if h['psnr'] is None or g['psnr'] is None else g['psnr'] - h['psnr']
            lines.append(f"| {group} | {pano} | {fmt(h['psnr'])} | {fmt(g['psnr'])} | {signed(delta)} | "
                         f"{fmt(h['ssim'], 4)} | {fmt(g['ssim'], 4)} | {signed(h['luminance_minus_reference'], 4)} / "
                         f"{signed(g['luminance_minus_reference'], 4)} |")
    lines += ['', '## Régions (PSNR / SSIM moyens ; faces où la région existe)', '',
              '| Ensemble | Panorama | Région | historique | 3DGUT | Faces |', '|---|---|---|---|---|---:|']
    for group, panos in summary['historique'].items():
        for pano, h in panos.items():
            for region in REPORT_REGIONS:
                a, b = h['regions'][region], summary['3dgut'][group][pano]['regions'][region]
                cell = lambda r: 'absente' if not r['faces'] else f"{fmt(r['psnr'])} / {fmt(r['ssim'], 4)}"
                lines.append(f"| {group} | {pano} | {region} | {cell(a)} | {cell(b)} | {a['faces']} |")
    perf = result['performance']
    lines += ['', f"## Performance et projection ({result['repetitions']} répétitions par face après {result['warmup']} "
                  'rendu(s) d’échauffement)', '',
              '| Mode | Rendus | Temps médian (ms) | p95 (ms) | Mémoire max (Mo) | Gaussiennes visibles (médiane) '
              '| Rayon p95 médian (px) | Écart max entre répétitions RGB / profondeur |', '|---|---:|---:|---:|---:|---:|---:|---|']
    for mode in MODES:
        p = perf[mode]
        lines.append(f"| {mode} | {p['renders']} | {fmt(p['time_ms_median'])} | {fmt(p['time_ms_p95'])} | "
                     f"{fmt(p['peak_memory_mb'], 1)} | {fmt(p['visible_gaussians_median'], 0)} | {fmt(p['radius_p95_median'], 1)} | "
                     f"{p['repetition_rgb_max_abs_diff']:.3g} / {p['repetition_depth_max_abs_diff']:.3g} |")
    lines += ['', 'Les répétitions mesurent la variabilité du rendu des mêmes paramètres ; elles ne quantifient pas '
                  'le non-déterminisme de l’entraînement.', '', '## Par face', '',
              '| Ensemble | Face | PSNR historique | PSNR 3DGUT | ΔPSNR | ΔSSIM | alpha < 0,5 hist. / 3DGUT | '
              'Profondeur p50 hist. / 3DGUT | Planche |', '|---|---|---:|---:|---:|---:|---|---|---|']
    for f in result['faces']:
        h, g = f['historique'], f['3dgut']
        dp = None if h['psnr'] is None or g['psnr'] is None else g['psnr'] - h['psnr']
        ds = None if h['ssim'] is None or g['ssim'] is None else g['ssim'] - h['ssim']
        lines.append(f"| {f['set']} | {f['camera']} | {fmt(h['psnr'])} | {fmt(g['psnr'])} | {signed(dp)} | "
                     f"{signed(ds, 4)} | {fmt(h.get('alpha_below_half_fraction'), 3)} / {fmt(g.get('alpha_below_half_fraction'), 3)} | "
                     f"{fmt(h.get('depth_p50'), 3)} / {fmt(g.get('depth_p50'), 3)} | [{f['sheet']}]({f['sheet']}) |")
    lines += ['', 'Planche globale : [overview.jpg](overview.jpg). Planches par face : lignes = modes, mêmes échelles '
                  '(erreur absolue 0-0,25 ; profondeur commune par face). Aucun seuil de qualité.', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Read-only comparison of one checkpoint rendered with the historical EWA rasterizer and with '
                    '3DGUT (with_ut=True, with_eval3d=True), on the same validation (and optional train) faces.')
    parser.add_argument('--prep', required=True, help='Prepared run, e.g. Output/runs/salon-gsplat-009')
    parser.add_argument('--config', required=True, help='Training config of the checkpoint, e.g. configs/gsplat-absgs-abs-seed0.json')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint file name, e.g. step_003000.pt')
    parser.add_argument('--train-faces', type=int, default=0,
                        help='Also compare N train faces per content group (glass windows, furniture); default 0')
    parser.add_argument('--repetitions', type=int, default=3, help='Timed renders per face and mode; default 3')
    parser.add_argument('--warmup', type=int, default=1, help='Untimed renders per mode before timing; default 1')
    args = parser.parse_args(argv)
    try:
        target = compare(args.prep, read(args.config), args.checkpoint, args.train_faces, args.repetitions, args.warmup)
    except Exception as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1
    print(target / 'comparison.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
