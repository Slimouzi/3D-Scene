"""Compare two checkpoints of one training on the same faces (GPU VM, read-only).

    python -m theta_pipeline.gsplat_checkpoint_diag --prep Output/runs/salon-gsplat-009 \
        --config configs/gsplat-absgs-base-seed2.json --checkpoints step_002000.pt step_003000.pt \
        --panorama R0010011

For every face of the panorama (validation set unless --set train): color, alpha and
expected depth at both checkpoints, the per-face metric change, and an obstruction test.
The obstruction hypothesis is checked, not assumed: Gaussians whose camera depth is below
`near_factor` x the 5th percentile depth of the SfM points seen by that face are rendered
alone, and the alpha they cover is measured. Population statistics (non-finite values,
sizes, opacities, positions) are reported per checkpoint. Checkpoints are only read; their
SHA-256 is checked before and after. The test set is never loaded.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .storage import digest, now, read, write

NEAR_FACTOR = .5


def camera_points(world_to_camera, K, size, xyz):
    """Camera depth and pixel position of world points; mask of points in front and inside the image."""
    T = np.asarray(world_to_camera, float)
    cam = np.asarray(xyz, float) @ T[:3, :3].T + T[:3, 3]
    z = cam[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = (cam @ np.asarray(K, float).T)[:, :2] / z[:, None]
    inside = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < size) & (uv[:, 1] >= 0) & (uv[:, 1] < size)
    return z, uv, inside


def reference_depth(world_to_camera, K, size, points_xyz, quantile=.05):
    """Low quantile of the depths of SfM points visible in the face (None if none is visible)."""
    z, _, inside = camera_points(world_to_camera, K, size, points_xyz)
    return float(np.quantile(z[inside], quantile)) if inside.any() else None


def near_gaussians(world_to_camera, K, size, means, reference, factor=NEAR_FACTOR):
    """Gaussians in front of the camera, projecting inside the face, nearer than factor x reference."""
    if reference is None:
        return np.zeros(len(means), bool)
    z, _, inside = camera_points(world_to_camera, K, size, means)
    return inside & (z < factor * reference)


def non_finite(arrays):
    return {name: {'nan': int(np.isnan(a).sum()), 'inf': int(np.isinf(a).sum())} for name, a in arrays.items()}


def population(params, points_xyz, scene_scale, prune_scale3d):
    from .gsplat_inspect import gaussian_statistics
    means, scales = params['means'], np.exp(params['scales'])
    opacities = 1 / (1 + np.exp(-params['opacities']))
    # Judged on raw parameters: a sigmoid turns an infinite opacity logit into a finite 1.
    finite = (np.isfinite(params['means']).all(1) & np.isfinite(params['scales']).all(1)
              & np.isfinite(params['opacities']))
    stats = gaussian_statistics(means[finite], scales[finite], opacities[finite], points_xyz, scene_scale,
                                prune_scale3d) if finite.any() else {'count': 0}
    return {'count_total': int(len(means)), 'non_finite': non_finite(params), **stats}


def render_layers(params, viewmat, K, size, degree, mask=None):
    """Color, alpha and expected depth of all Gaussians (or of a subset `mask`)."""
    import torch
    from gsplat import rasterization
    keep = slice(None) if mask is None else torch.from_numpy(mask).to(params['means'].device)
    with torch.no_grad():
        renders, alphas, _ = rasterization(
            means=params['means'][keep], quats=params['quats'][keep], scales=torch.exp(params['scales'][keep]),
            opacities=torch.sigmoid(params['opacities'][keep]),
            colors=torch.cat([params['sh0'][keep], params['shN'][keep]], 1), viewmats=viewmat[None], Ks=K[None],
            width=size, height=size, sh_degree=degree, packed=False, render_mode='RGB+ED')
    return (renders[0, ..., :3].clamp(0, 1).cpu().numpy(), alphas[0, ..., 0].cpu().numpy(),
            renders[0, ..., 3].cpu().numpy())


def diagnose(prep, cfg, checkpoints, panorama, group='validation'):
    if group not in ('validation', 'train'):
        raise RuntimeError('only validation or train faces may be diagnosed; the test set stays closed')
    import torch
    from . import gsplat_train
    from .gsplat_inspect import contact_sheet, depth_image, error_image
    from .gsplat_preflight import load_verified, qualify, verify_prep
    prep = Path(prep).resolve()
    problems = verify_prep(prep) + qualify()['problems']
    if problems:
        raise RuntimeError('; '.join(problems))
    training = prep / 'training' / cfg['name']
    status = read(training / 'training.json')
    manifest = read(prep / 'gsplat_inputs/gsplat_inputs.json')
    meta = {'config_sha256': gsplat_train.config_sha256(cfg),
            'manifest_sha256': digest(prep / 'gsplat_inputs/gsplat_inputs.json'),
            'partition_sha256': manifest['partition_sha256'], 'git_commit': status['git_commit']}
    files = {name: training / 'checkpoints' / name for name in checkpoints}
    before = {name: digest(path) for name, path in files.items()}
    cameras = [c for c in read(prep / f'gsplat_inputs/cameras_{group}.json')['cameras'] if c['panorama_id'] == panorama]
    if not cameras:
        raise RuntimeError(f'no {group} face of {panorama}')
    data = gsplat_train.load_cameras(prep, group, 'cuda', [c['name'] for c in cameras])
    order = {c['name']: c for c in cameras}
    rel = next(r for r in manifest['files'] if r.endswith('points.npz'))
    points = np.load(load_verified(prep.parent, {'path': rel, 'sha256': manifest['files'][rel]}))['xyz']
    target = prep / 'diagnostics' / f"checkpoints-{cfg['name']}-{panorama}-{'-vs-'.join(Path(c).stem for c in checkpoints)}"
    target.mkdir(parents=True, exist_ok=True)
    states, faces = {}, {}
    for name, path in files.items():
        _, params, *_, saved = gsplat_train.restore(path, cfg, meta, 'cuda')
        host = {k: v.detach().cpu().numpy() for k, v in params.items()}
        states[name] = {'step': saved['step'], 'sh_degree': saved['sh_degree'],
                        'population': population(host, points, saved['meta']['scene_scale'],
                                                 cfg['strategy'].get('prune_scale3d', .1))}
        for i, face in enumerate(data['names']):
            camera = order[face]
            size = data['width']
            reference = reference_depth(camera['world_to_camera'], camera['K'], size, points)
            near = near_gaussians(camera['world_to_camera'], camera['K'], size, host['means'], reference)
            rgb, alpha, depth = render_layers(params, data['viewmats'][i], data['Ks'][i], size, saved['sh_degree'])
            near_alpha = (render_layers(params, data['viewmats'][i], data['Ks'][i], size, saved['sh_degree'], near)[1]
                          if near.any() else np.zeros_like(alpha))
            target_rgb = data['images'][i].cpu().numpy()
            weight = data['weights'][i].float().cpu().numpy() / 255
            metrics = gsplat_train.view_metrics(torch.from_numpy(rgb)[None], data['images'][i:i + 1].float().cpu() / 255,
                                                data['weights'][i:i + 1].float().cpu() / 255)
            valid = weight > 0
            opacity = 1 / (1 + np.exp(-host['opacities']))
            faces.setdefault(face, {})[name] = {
                **metrics, 'low_alpha_fraction_of_valid': float(((alpha < .5) & valid).sum() / max(valid.sum(), 1)),
                'non_finite_render': int((~np.isfinite(rgb)).sum() + (~np.isfinite(depth)).sum()),
                'reference_depth_p05': reference, 'near_gaussians': int(near.sum()),
                'near_opaque_gaussians': int((near & (opacity > .5)).sum()),
                'near_alpha_fraction_of_valid': float(((near_alpha > .5) & valid).sum() / max(valid.sum(), 1))}
            panels = [('reference', target_rgb), ('render', (rgb * 255).round().astype(np.uint8)),
                      ('error', error_image(target_rgb, rgb * 255, weight)), ('alpha', (alpha * 255).astype(np.uint8)),
                      ('depth', depth_image(depth, alpha)), ('near alpha', (near_alpha * 255).astype(np.uint8))]
            stem = face.replace('/', '__').removesuffix('.png')
            contact_sheet(f'{face} @ {Path(name).stem}', panels, {k: None if metrics[k] is None else round(metrics[k], 3)
                                                                  for k in ('psnr', 'ssim')}).save(
                target / f'{stem}__{Path(name).stem}.jpg', quality=90)
        del params
        torch.cuda.empty_cache()
    after = {name: digest(path) for name, path in files.items()}
    if after != before:
        raise RuntimeError('a checkpoint changed during the diagnosis')
    result = {'schema_version': 1, 'training': cfg['name'], 'set': group, 'panorama': panorama,
              'checkpoints': {n: {**states[n], 'sha256': before[n]} for n in checkpoints},
              'faces': faces, 'near_rule': f'camera depth < {NEAR_FACTOR} x p05 depth of visible SfM points',
              'hypothesis': 'obstruction by Gaussians near the camera (to be checked, not established)',
              'test_loaded': False, 'created_at': now()}
    write(target / 'diagnosis.json', result)
    (target / 'diagnosis.md').write_text(report(result))
    return target


def report(result):
    a, b = list(result['checkpoints'])
    fmt = lambda v, d=2: '—' if v is None else f'{v:.{d}f}'
    lines = [f"# Diagnostic {result['training']} — {result['panorama']} ({result['set']}) : {a} → {b}", '',
             f"Hypothèse examinée : {result['hypothesis']}. Règle « proche » : {result['near_rule']}.", '',
             '| Face | PSNR ' + a + ' | PSNR ' + b + ' | Δ | alpha<0,5 ' + a + ' / ' + b
             + ' | proches (opaques) ' + a + ' / ' + b + ' | couverture proche ' + a + ' / ' + b + ' |',
             '|---|---:|---:|---:|---|---|---|']
    for face, values in result['faces'].items():
        x, y = values[a], values[b]
        delta = None if x['psnr'] is None or y['psnr'] is None else y['psnr'] - x['psnr']
        lines.append(f"| {face} | {fmt(x['psnr'])} | {fmt(y['psnr'])} | {fmt(delta)} | "
                     f"{fmt(x['low_alpha_fraction_of_valid'], 3)} / {fmt(y['low_alpha_fraction_of_valid'], 3)} | "
                     f"{x['near_gaussians']} ({x['near_opaque_gaussians']}) / {y['near_gaussians']} ({y['near_opaque_gaussians']}) | "
                     f"{fmt(x['near_alpha_fraction_of_valid'], 3)} / {fmt(y['near_alpha_fraction_of_valid'], 3)} |")
    lines += ['', '## Population de gaussiennes', '', '| Checkpoint | Nombre | Non finies | Opacité p05/p50/p95 '
              '| Taille ÷ scene_scale p95 / max | Hors boîte (opaques) |', '|---|---:|---|---|---|---|']
    for name, state in result['checkpoints'].items():
        pop = state['population']
        bad = {k: v for k, v in pop['non_finite'].items() if v['nan'] or v['inf']}
        size = pop.get('max_scale_over_scene_scale_quantiles') or {}
        lines.append(f"| {name} | {pop['count_total']} | {bad or 'aucune'} | {pop.get('opacity_quantiles', '—')} | "
                     f"{fmt(size.get('p95'), 4)} / {fmt(size.get('max'), 4)} | "
                     f"{pop.get('outside_sfm_points_box_plus_25pct', '—')} ({pop.get('outside_and_opaque_over_0_5', '—')}) |")
    lines += ['', 'Planches : une par face et par checkpoint (référence, rendu, erreur, alpha, profondeur, '
                  'couverture des seules gaussiennes proches). Aucune conclusion automatique ; jeu de test non chargé.', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Two-checkpoint diagnosis on the same faces (read-only)')
    parser.add_argument('--prep', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoints', nargs=2, required=True)
    parser.add_argument('--panorama', required=True)
    parser.add_argument('--set', default='validation', choices=['validation', 'train'])
    args = parser.parse_args(argv)
    try:
        target = diagnose(args.prep, read(args.config), args.checkpoints, args.panorama, args.set)
    except Exception as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1
    print(target / 'diagnosis.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
