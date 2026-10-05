"""Read-only comparison of one checkpoint rendered in two modes: 'historique' and '3dgut'.

    python -m theta_pipeline.gsplat_render_compare --prep Output/runs/salon-gsplat-009 \
        --config configs/gsplat-absgs-abs-seed0.json --checkpoint step_003000.pt [--train-faces 3]

The checkpoint is loaded once; both modes render exactly the same parameters with the same
poses, intrinsics, resolution, appearance weights and SH degree (the degree stored in the
checkpoint). Only the rasterization differs (gsplat_inspect.RENDERERS). Faces: every
validation face, plus optional train faces chosen by content; never the test set. The
checkpoint SHA-256 is checked before and after; nothing is written under training/. The
parameters were optimized with the historical renderer: the 3DGUT renders qualify the
renderer on these checkpoints, they are not a 3DGUT-trained model. No quality threshold.
"""
import argparse
import hashlib
import json
import platform
import statistics
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import numpy as np
from .storage import digest, now, read, write

MODES = ('historique', '3dgut')
SETS = ('validation', 'train')


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def run_identity(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def face_metrics(rgb, target, weight, labels, gsplat_train, inspect):
    """PSNR/SSIM, luminance and per-region metrics on the valid pixels (identical across modes)."""
    import torch
    from .gsplat_checkpoint_diag import luminance
    metrics = gsplat_train.view_metrics(rgb[None], target[None], weight[None])
    reference = (target.cpu().numpy() * 255)
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


def summarize(faces):
    out = {}
    for mode in MODES:
        for group in sorted({f['set'] for f in faces}):
            for pano in sorted({f['panorama_id'] for f in faces if f['set'] == group}):
                rows = [f[mode] for f in faces if f['set'] == group and f['panorama_id'] == pano and not f[mode]['excluded']]
                mean = lambda key: statistics.fmean([r[key] for r in rows]) if rows else None
                out.setdefault(mode, {}).setdefault(group, {})[pano] = {
                    'usable': len(rows), 'psnr': mean('psnr'), 'ssim': mean('ssim'),
                    'luminance_minus_reference': statistics.fmean([r['render_luminance'] - r['reference_luminance']
                                                                   for r in rows]) if rows else None}
    return out


def compare(prep, cfg, checkpoint_name, train_faces=0):
    import torch
    from . import gsplat_inspect as inspect, gsplat_train
    from .gsplat_checkpoint_diag import sheet
    from .gsplat_clean import new_folder
    from .gsplat_preflight import qualify, verify_prep
    from .segmentation.provenance import git_commit
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
    _, params, *_, saved = gsplat_train.restore(checkpoint, cfg, meta, 'cuda')      # loaded once, shared by both modes
    degree = saved['sh_degree']
    sources = inspect.Sources(prep.parent, manifest)
    cameras = {'validation': read(prep / 'gsplat_inputs/cameras_validation.json')['cameras']}
    if train_faces:
        train = read(prep / 'gsplat_inputs/cameras_train.json')['cameras']
        chosen, _ = inspect.select_train_faces(train, sources.labels, sources.rotations, train[0]['width'], train_faces)
        cameras['train'] = [c for c in train if c['name'] in chosen]
    identity = run_identity({'checkpoint_sha256': sha, 'config_sha256': meta['config_sha256'],
                             'prep_manifest_sha256': manifest_sha, 'renderers': inspect.RENDERERS,
                             'faces': {g: [c['name'] for c in v] for g, v in cameras.items()}})
    target = new_folder(prep / 'inspection' / 'render-compare' / f"{cfg['name']}-{checkpoint.stem}-{identity[:12]}")
    faces = []
    for group, entries in cameras.items():
        data = gsplat_train.load_cameras(prep, group, 'cuda', [c['name'] for c in entries])
        by_name = {c['name']: c for c in entries}
        for i, name in enumerate(data['names']):
            camera = by_name[name]
            target_rgb = data['images'][i].float() / 255
            weight = data['weights'][i].float() / 255
            labels = inspect.project_labels(sources.labels(camera['panorama_id']), sources.rotations[name], data['width'])
            row, variants = {'camera': name, 'panorama_id': camera['panorama_id'], 'set': group}, []
            for mode in MODES:
                rgb, alpha, depth = inspect.render_face(params, data, i, degree, mode)
                row[mode] = face_metrics(rgb, target_rgb, weight, labels, gsplat_train, inspect)
                variants.append({'label': mode, 'rgb': rgb.cpu().numpy(), 'alpha': alpha.cpu().numpy(),
                                 'depth': depth.cpu().numpy(), 'weight': weight.cpu().numpy(), 'psnr': row[mode]['psnr']})
            stem = f"{group}__{name.replace('/', '__').removesuffix('.png')}"
            sheet(name, data['images'][i].cpu().numpy(), variants, 'erreur absolue 0-0,25').save(
                target / f'{stem}.jpg', quality=90)
            row['sheet'] = f'{stem}.jpg'
            faces.append(row)
        del data
        torch.cuda.empty_cache()
    if digest(checkpoint) != sha:
        raise RuntimeError(f'{checkpoint} changed during the comparison')
    result = {'schema_version': 1, 'tool': 'gsplat_render_compare', 'training': cfg['name'],
              'checkpoint': checkpoint.name, 'checkpoint_sha256': sha, 'checkpoint_loads': 1,
              'checkpoint_sha256_by_mode': {mode: sha for mode in MODES},
              'sh_degree': degree, 'config_sha256': meta['config_sha256'], 'prep_run': prep.name,
              'prep_manifest_sha256': manifest_sha, 'partition_sha256': manifest['partition_sha256'],
              'renderers': inspect.RENDERERS, 'run_identity': identity,
              'environment': {'gsplat': package_version('gsplat'), 'torch': package_version('torch'),
                              'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                              'python': platform.python_version()},
              'analysis_git_commit': git_commit(), 'sets': sorted(cameras), 'test_loaded': False,
              'shared_inputs': 'same checkpoint object, poses, intrinsics, resolution, weights and SH degree',
              'caveat': 'parameters optimized with the historical renderer; 3DGUT renders do not represent a '
                        '3DGUT-trained model', 'faces': faces, 'summary': summarize(faces), 'created_at': now()}
    write(target / 'comparison.json', result)
    (target / 'comparison.md').write_text(report(result))
    return target


def report(result):
    fmt = lambda v, d=2: '—' if v is None else f'{v:.{d}f}'
    signed = lambda v, d=3: '—' if v is None else f'{v:+.{d}f}'
    env = result['environment']
    lines = [f"# Rendu historique / 3DGUT — {result['training']} / {result['checkpoint']}", '',
             f"Checkpoint `{result['checkpoint_sha256'][:16]}` chargé une fois et rendu dans les deux modes ; degré SH "
             f"{result['sh_degree']} ; gsplat {env['gsplat']}, torch {env['torch']}, GPU {env['device']} ; commit d’analyse "
             f"`{result['analysis_git_commit']}`. Jeu de test non utilisé.", '',
             'Paramètres : ' + '; '.join(f"{m} = {json.dumps(p, sort_keys=True)}" for m, p in result['renderers'].items()), '',
             f"Réserve : {result['caveat']}.", '',
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
    lines += ['', '## Par face', '', '| Ensemble | Face | PSNR historique | PSNR 3DGUT | ΔPSNR | ΔSSIM | Planche |',
              '|---|---|---:|---:|---:|---:|---|']
    for f in result['faces']:
        h, g = f['historique'], f['3dgut']
        dp = None if h['psnr'] is None or g['psnr'] is None else g['psnr'] - h['psnr']
        ds = None if h['ssim'] is None or g['ssim'] is None else g['ssim'] - h['ssim']
        lines.append(f"| {f['set']} | {f['camera']} | {fmt(h['psnr'])} | {fmt(g['psnr'])} | {signed(dp)} | "
                     f"{signed(ds, 4)} | [{f['sheet']}]({f['sheet']}) |")
    lines += ['', 'Planches : lignes = modes, mêmes échelles (erreur absolue 0-0,25 ; profondeur commune par face). '
                  'Aucun seuil de qualité.', '']
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
    args = parser.parse_args(argv)
    try:
        target = compare(args.prep, read(args.config), args.checkpoint, args.train_faces)
    except Exception as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1
    print(target / 'comparison.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
