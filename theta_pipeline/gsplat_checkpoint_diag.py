"""Compare two checkpoints of one training on the same faces (GPU VM, read-only).

    python -m theta_pipeline.gsplat_checkpoint_diag --prep Output/runs/salon-gsplat-009 \
        --config configs/gsplat-absgs-base-seed2.json --checkpoints step_002000.pt step_003000.pt \
        --panorama R0010011 --sheet-faces 1 5 6 9 10 --focus-faces 1 10

For every face of the panorama (validation unless --set train), at both checkpoints:
metrics, color, alpha, expected depth, and the mean luminance of render and reference on
the valid pixels (a dark veil is quantified, not explained). At the second checkpoint, an
in-memory ablation renders each face with all Gaussians, without those whose centre is
near the camera, and without those whose 3-sigma extent comes near the camera while their
projected footprint touches the face; metrics use the same valid pixels. Focus faces list
the Gaussians with the largest projected extent covering them. Sheets use identical scales
across variants (absolute error 0-0.25; one depth range per face). Diagnosis only: nothing
is pruned, checkpoints are only read (SHA-256 checked), the test set is never loaded.
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


SIGMA = 3.            # extent of a Gaussian, in standard deviations
NEAR_PLANE = .01      # gsplat culls centres nearer than this


def footprint(world_to_camera, K, size, means, scales):
    """Camera depth, projected 3-sigma radius (pixels) and whether the footprint touches the face."""
    z, uv, _ = camera_points(world_to_camera, K, size, means)
    extent = SIGMA * np.max(scales, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        radius = np.where(z > NEAR_PLANE, np.asarray(K)[0][0] * extent / z, 0.)
    touches = ((z > NEAR_PLANE) & (uv[:, 0] + radius >= 0) & (uv[:, 0] - radius < size)
               & (uv[:, 1] + radius >= 0) & (uv[:, 1] - radius < size))
    return z, radius, touches, extent


def near_extent_gaussians(world_to_camera, K, size, means, scales, reference, factor=NEAR_FACTOR):
    """Rendered Gaussians whose 3-sigma extent reaches nearer than factor x reference depth."""
    if reference is None:
        return np.zeros(len(means), bool)
    z, _, touches, extent = footprint(world_to_camera, K, size, means, scales)
    return touches & (z - extent < factor * reference)


def largest_covering(world_to_camera, K, size, means, scales, opacities, count=10):
    """Gaussians touching the face, ranked by opacity x covered area (radius capped at the face size)."""
    z, radius, touches, extent = footprint(world_to_camera, K, size, means, scales)
    index = np.flatnonzero(touches)
    score = opacities[index] * np.minimum(radius[index], size) ** 2
    top = index[np.argsort(-score, kind='stable')[:count]]
    return [{'index': int(i), 'depth': float(z[i]), 'extent_3sigma': float(extent[i]),
             'projected_radius_px': float(radius[i]), 'opacity': float(opacities[i]),
             'nearest_extent_depth': float(z[i] - extent[i])} for i in top]


def luminance(rgb, valid):
    """Mean Rec.709 luminance (0-1) over valid pixels."""
    y = np.asarray(rgb, float) @ np.array([.2126, .7152, .0722])
    return float(y[valid].mean()) if valid.any() else None


def depth_range(depths, alphas):
    values = np.concatenate([d[a > .5] for d, a in zip(depths, alphas)] or [np.zeros(0)])
    return (float(np.quantile(values, .02)), float(np.quantile(values, .98))) if values.size else (0., 1.)


def depth_with_range(depth, alpha, low, high):
    out = np.zeros(depth.shape, np.uint8)
    valid = alpha > .5
    out[valid] = (255 * (1 - np.clip((depth[valid] - low) / max(high - low, 1e-9), 0, 1))).astype(np.uint8)
    return out


def render_layers(params, viewmat, K, size, degree, keep=None):
    """Color, alpha and expected depth of all Gaussians, or of those where `keep` is true."""
    import torch
    from gsplat import rasterization
    sel = slice(None) if keep is None else torch.from_numpy(keep).to(params['means'].device)
    with torch.no_grad():
        renders, alphas, _ = rasterization(
            means=params['means'][sel], quats=params['quats'][sel], scales=torch.exp(params['scales'][sel]),
            opacities=torch.sigmoid(params['opacities'][sel]),
            colors=torch.cat([params['sh0'][sel], params['shN'][sel]], 1), viewmats=viewmat[None], Ks=K[None],
            width=size, height=size, sh_degree=degree, packed=False, render_mode='RGB+ED')
    return (renders[0, ..., :3].clamp(0, 1).cpu().numpy(), alphas[0, ..., 0].cpu().numpy(),
            renders[0, ..., 3].cpu().numpy())


def sheet(face, reference, variants, scale_note):
    """Rows: variants; columns: reference, render, error, alpha, depth. Shared depth range per face."""
    from .gsplat_inspect import error_image
    low, high = depth_range([v['depth'] for v in variants], [v['alpha'] for v in variants])
    size = min(reference.shape[0], 320)
    columns = ('reference', 'rendu', 'erreur', 'alpha', 'profondeur')
    out = Image.new('RGB', (len(columns) * size + 230, len(variants) * size + 44), '#1d252c')
    draw = ImageDraw.Draw(out)
    draw.text((6, 4), f'{face} — {scale_note}; profondeur {low:.3f}-{high:.3f}', fill='#f0c040')
    for k, label in enumerate(columns):
        draw.text((230 + k * size + 4, 24), label, fill='white')
    for r, v in enumerate(variants):
        y = 44 + r * size
        draw.text((6, y + 6), v['label'], fill='white')
        draw.text((6, y + 24), f"PSNR {v['psnr']:.2f}" if v['psnr'] is not None else 'PSNR —', fill='#9aa4ad')
        panels = [reference, (v['rgb'] * 255).round().astype(np.uint8), error_image(reference, v['rgb'] * 255, v['weight']),
                  (v['alpha'] * 255).astype(np.uint8), depth_with_range(v['depth'], v['alpha'], low, high)]
        for k, array in enumerate(panels):
            image = Image.fromarray(array if array.ndim == 3 else np.repeat(array[..., None], 3, -1))
            out.paste(image.resize((size, size)), (230 + k * size, y))
    return out


def face_number(name):
    """SfM face index of an image name such as pano_camera5/R0010011.png."""
    return int(Path(name).parent.name.removeprefix('pano_camera'))


def diagnose(prep, cfg, checkpoints, panorama, group='validation', sheet_faces=None, focus_faces=None):
    if group not in ('validation', 'train'):
        raise RuntimeError('only validation or train faces may be diagnosed; the test set stays closed')
    import torch
    from . import gsplat_train
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
    target = prep / 'diagnostics' / f"checkpoints-{cfg['name']}-{panorama}-{'-vs-'.join(Path(c).stem for c in checkpoints)}-v2"
    target.mkdir(parents=True, exist_ok=True)
    first, second = checkpoints
    states, faces, renders = {}, {}, {}
    for name, path in files.items():
        _, params, *_, saved = gsplat_train.restore(path, cfg, meta, 'cuda')
        host = {k: v.detach().cpu().numpy() for k, v in params.items()}
        scales, opacity = np.exp(host['scales']), 1 / (1 + np.exp(-host['opacities']))
        states[name] = {'step': saved['step'], 'sh_degree': saved['sh_degree'],
                        'population': population(host, points, saved['meta']['scene_scale'],
                                                 cfg['strategy'].get('prune_scale3d', .1))}
        for i, face in enumerate(data['names']):
            camera, size = order[face], data['width']
            target_rgb = data['images'][i].cpu().numpy()
            weight = data['weights'][i].float().cpu().numpy() / 255
            valid = weight > 0
            reference = reference_depth(camera['world_to_camera'], camera['K'], size, points)
            near_centre = near_gaussians(camera['world_to_camera'], camera['K'], size, host['means'], reference)
            near_extent = near_extent_gaussians(camera['world_to_camera'], camera['K'], size, host['means'], scales,
                                                reference)
            variants = {'complet': None}
            if name == second:
                variants.update({'sans proches (centre)': ~near_centre, 'sans proches (étendue)': ~near_extent})
            entry = faces.setdefault(face, {})
            for label, keep in variants.items():
                rgb, alpha, depth = render_layers(params, data['viewmats'][i], data['Ks'][i], size,
                                                  saved['sh_degree'], keep)
                metrics = gsplat_train.view_metrics(torch.from_numpy(rgb)[None],
                                                    data['images'][i:i + 1].float().cpu() / 255,
                                                    data['weights'][i:i + 1].float().cpu() / 255)
                entry[f'{name}/{label}'] = {
                    **metrics, 'render_luminance': luminance(rgb, valid),
                    'low_alpha_fraction_of_valid': float(((alpha < .5) & valid).sum() / max(valid.sum(), 1)),
                    'non_finite_render': int((~np.isfinite(rgb)).sum() + (~np.isfinite(depth)).sum()),
                    'removed_gaussians': 0 if keep is None else int((~keep).sum()),
                    'removed_opaque': 0 if keep is None else int((~keep & (opacity > .5)).sum())}
                renders.setdefault(face, []).append({'label': f'{Path(name).stem} {label}', 'rgb': rgb,
                                                     'alpha': alpha, 'depth': depth, 'weight': weight,
                                                     'psnr': metrics['psnr']})
            entry['reference_luminance'] = luminance(target_rgb / 255, valid)
            entry['reference_depth_p05'] = reference
            entry[f'{name}/near_centre'] = int(near_centre.sum())
            entry[f'{name}/near_extent'] = int(near_extent.sum())
            if focus_faces and face_number(face) in focus_faces:
                entry[f'{name}/largest_covering'] = largest_covering(camera['world_to_camera'], camera['K'], size,
                                                                     host['means'], scales, opacity)
        del params
        torch.cuda.empty_cache()
    if {name: digest(path) for name, path in files.items()} != before:
        raise RuntimeError('a checkpoint changed during the diagnosis')
    for i, face in enumerate(data['names']):
        if sheet_faces is None or face_number(face) in sheet_faces:
            sheet(face, data['images'][i].cpu().numpy(), renders[face],
                  'erreur absolue 0-0,25 (bleu : exclu)').save(
                target / f"face{face_number(face):02d}__{face.replace('/', '__').removesuffix('.png')}.jpg", quality=90)
    result = {'schema_version': 2, 'training': cfg['name'], 'set': group, 'panorama': panorama,
              'checkpoints': {n: {**states[n], 'sha256': before[n]} for n in checkpoints},
              'ablation_checkpoint': second, 'faces': faces,
              'near_rules': {'centre': f'centre depth < {NEAR_FACTOR} x p05 depth of visible SfM points',
                             'extent': f'(centre depth - {SIGMA:g} sigma max) < {NEAR_FACTOR} x p05 depth, '
                                       'footprint touching the face'},
              'hypothesis': 'obstruction by Gaussians near the camera (checked by in-memory ablation, not assumed)',
              'sheet_faces': sheet_faces, 'focus_faces': focus_faces, 'test_loaded': False, 'created_at': now()}
    write(target / 'diagnosis.json', result)
    (target / 'diagnosis.md').write_text(report(result))
    return target


def report(result):
    a, b = list(result['checkpoints'])
    fmt = lambda v, d=2: '—' if v is None else f'{v:.{d}f}'
    signed = lambda v, d=2: '—' if v is None else f'{v:+.{d}f}'
    get = lambda face, key, metric: (result['faces'][face].get(key) or {}).get(metric)
    lines = [f"# Diagnostic {result['training']} — {result['panorama']} ({result['set']}) : {a} → {b}", '',
             f"Hypothèse examinée : {result['hypothesis']}.",
             f"Règles « proche » : centre — {result['near_rules']['centre']} ; étendue — {result['near_rules']['extent']}.", '',
             '## Métriques et luminance (pixels valides)', '',
             f'| Face | PSNR {a} | PSNR {b} | Δ | Luminance référence | Rendu {a} | Rendu {b} | Écart rendu − référence {a} / {b} |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    for face, values in result['faces'].items():
        pa, pb = get(face, f'{a}/complet', 'psnr'), get(face, f'{b}/complet', 'psnr')
        ref = values.get('reference_luminance')
        la, lb = get(face, f'{a}/complet', 'render_luminance'), get(face, f'{b}/complet', 'render_luminance')
        diff = lambda x: None if x is None or ref is None else x - ref
        lines.append(f"| {face} | {fmt(pa)} | {fmt(pb)} | {signed(None if pa is None or pb is None else pb - pa)} | "
                     f"{fmt(ref, 3)} | {fmt(la, 3)} | {fmt(lb, 3)} | {signed(diff(la), 3)} / {signed(diff(lb), 3)} |")
    lines += ['', f'## Ablation en mémoire à {b} (mêmes pixels valides)', '',
              '| Face | PSNR complet | sans proches (centre) | sans proches (étendue) | SSIM complet / centre / étendue '
              '| Retirées centre (opaques) | Retirées étendue (opaques) |', '|---|---:|---:|---:|---|---|---|']
    for face in result['faces']:
        full, centre, extent = (f'{b}/complet', f'{b}/sans proches (centre)', f'{b}/sans proches (étendue)')
        p0 = get(face, full, 'psnr')
        cell = lambda key: (f"{fmt(get(face, key, 'psnr'))} ({signed(None if p0 is None or get(face, key, 'psnr') is None else get(face, key, 'psnr') - p0)})")
        lines.append(f"| {face} | {fmt(p0)} | {cell(centre)} | {cell(extent)} | "
                     f"{fmt(get(face, full, 'ssim'), 4)} / {fmt(get(face, centre, 'ssim'), 4)} / {fmt(get(face, extent, 'ssim'), 4)} | "
                     f"{get(face, centre, 'removed_gaussians')} ({get(face, centre, 'removed_opaque')}) | "
                     f"{get(face, extent, 'removed_gaussians')} ({get(face, extent, 'removed_opaque')}) |")
    focus = [(face, v) for face, v in result['faces'].items() if v.get(f'{b}/largest_covering')]
    if focus:
        lines += ['', f'## Gaussiennes de plus grande extension projetée couvrant les faces examinées ({b})', '']
        for face, values in focus:
            lines += [f'### {face}', '', '| Indice | Profondeur centre | Étendue 3σ | Rayon projeté (px) | Opacité '
                      '| Profondeur la plus proche de l’étendue |', '|---:|---:|---:|---:|---:|---:|']
            for g in values[f'{b}/largest_covering']:
                lines.append(f"| {g['index']} | {g['depth']:.3f} | {g['extent_3sigma']:.3f} | {g['projected_radius_px']:.0f} | "
                             f"{g['opacity']:.3f} | {g['nearest_extent_depth']:.3f} |")
            lines.append('')
    lines += ['', '## Population de gaussiennes', '', '| Checkpoint | Nombre | Non finies | Opacité p05/p50/p95 '
              '| Taille ÷ scene_scale p95 / max | Hors boîte (opaques) |', '|---|---:|---|---|---|---|']
    for name, state in result['checkpoints'].items():
        pop = state['population']
        bad = {k: v for k, v in pop['non_finite'].items() if v['nan'] or v['inf']}
        size = pop.get('max_scale_over_scene_scale_quantiles') or {}
        lines.append(f"| {name} | {pop['count_total']} | {bad or 'aucune'} | {pop.get('opacity_quantiles', '—')} | "
                     f"{fmt(size.get('p95'), 4)} / {fmt(size.get('max'), 4)} | "
                     f"{pop.get('outside_sfm_points_box_plus_25pct', '—')} ({pop.get('outside_and_opaque_over_0_5', '—')}) |")
    lines += ['', 'Planches : une par face retenue, lignes = variantes, échelles identiques entre variantes '
                  '(erreur absolue 0-0,25 ; une plage de profondeur commune par face). Diagnostic seulement : '
                  'aucun élagage n’est appliqué ni proposé ; jeu de test non chargé.', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Two-checkpoint diagnosis on the same faces (read-only)')
    parser.add_argument('--prep', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoints', nargs=2, required=True, help='first, second (ablation at the second)')
    parser.add_argument('--panorama', required=True)
    parser.add_argument('--set', default='validation', choices=['validation', 'train'])
    parser.add_argument('--sheet-faces', nargs='*', type=int, help='SfM face numbers to draw (default: all)')
    parser.add_argument('--focus-faces', nargs='*', type=int, help='Faces whose largest covering Gaussians are listed')
    args = parser.parse_args(argv)
    try:
        target = diagnose(args.prep, read(args.config), args.checkpoints, args.panorama, args.set,
                          args.sheet_faces, args.focus_faces)
    except Exception as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1
    print(target / 'diagnosis.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
