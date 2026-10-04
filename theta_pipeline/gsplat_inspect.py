"""Visual inspection of a selected gsplat checkpoint on validation views only (GPU VM).

    python -m theta_pipeline.gsplat_inspect --prep Output/runs/<prep-run> --config configs/<training>.json

Read-only: writes under <prep-run>/inspection/<training>-validation-<checkpoint>/ and never
touches the training directory. The test set is never loaded. For each validation face:
reference, render, weight mask, error map, alpha and expected depth, plus metrics per face
and per region (glass, mirror, unvalidated reflective, contours, other). Gaussian
population statistics help spot floaters. Nothing is judged: no quality threshold.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from .gsplat_preflight import load_verified, qualify, verify_prep
from .segmentation import LABELS
from .segmentation.provenance import git_commit
from .storage import digest, now, read, write

REGIONS = {'glass': (LABELS['glass'],), 'mirror': (LABELS['mirror'],),
           'unvalidated_reflective': (LABELS['unknown_glass'], LABELS['unknown_reflective'])}


def face_directions(size, rotation):
    """Panorama-frame unit ray of each face pixel center (as geometry.project, without cv2)."""
    y, x = np.mgrid[:size, :size]
    rays = np.stack(((x + .5 - size / 2) / (size / 2), (y + .5 - size / 2) / (size / 2), np.ones_like(x, float)), -1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays @ np.asarray(rotation)


def project_labels(labels, rotation, size):
    """Nearest-neighbour ERP label lookup for one face; labels are categorical."""
    height, width = labels.shape
    d = face_directions(size, rotation)
    lon = np.arctan2(d[..., 0], d[..., 2])
    lat = np.arctan2(d[..., 1], np.linalg.norm(d[..., [0, 2]], axis=-1))
    u = np.floor((lon / (2 * np.pi) + .5) * width).astype(np.int64) % width
    v = np.clip(np.floor((lat / np.pi + .5) * height).astype(np.int64), 0, height - 1)
    return labels[v, u]


def contour_mask(reference, valid, fraction=.10):
    """Strongest image gradients (top `fraction` of valid pixels): a relative contour set."""
    gray = reference.astype(float).mean(-1)
    gx, gy = np.zeros_like(gray), np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1] = gray[2:] - gray[:-2]
    magnitude = np.hypot(gx, gy)
    if not valid.any():
        return np.zeros_like(valid)
    # Flat faces have few non-zero gradients: never call a zero-gradient pixel a contour.
    return valid & (magnitude > 0) & (magnitude >= np.quantile(magnitude[valid], 1 - fraction))


def region_metrics(reference, rendered, weight, regions):
    """Weighted MSE-PSNR and mean absolute error per region; empty regions are None."""
    error = (rendered.astype(float) - reference.astype(float)) / 255
    rows = {}
    for name, mask in regions.items():
        w = weight * mask
        total = w.sum()
        if total <= 0:
            rows[name] = {'pixels': 0, 'psnr': None, 'mean_abs_error': None}
            continue
        mse = float((w[..., None] * error ** 2).sum() / (3 * total))
        rows[name] = {'pixels': int((w > 0).sum()), 'psnr': -10 * np.log10(max(mse, 1e-10)),
                      'mean_abs_error': float((w[..., None] * np.abs(error)).sum() / (3 * total))}
    return rows


def error_image(reference, rendered, weight):
    """Absolute error, mean over channels, black-to-yellow; excluded pixels in blue."""
    error = np.abs(rendered.astype(float) - reference.astype(float)).mean(-1) / 255
    scaled = np.clip(error / .25, 0, 1)
    out = np.stack((scaled, scaled ** 2 * .9, np.zeros_like(scaled)), -1)
    out[weight <= 0] = (.15, .2, .55)
    return (out * 255).astype(np.uint8)


def depth_image(depth, alpha):
    valid = alpha > .5
    out = np.zeros(depth.shape, np.uint8)
    if valid.any():
        low, high = np.quantile(depth[valid], [.02, .98])
        out[valid] = (255 * (1 - np.clip((depth[valid] - low) / max(high - low, 1e-9), 0, 1))).astype(np.uint8)
    return out


def gaussian_statistics(means, scales, opacities, points_xyz, scene_scale, prune_scale3d):
    """Population descriptors for floater review (no threshold is turned into a verdict)."""
    size = scales.max(1)
    low, high = points_xyz.min(0), points_xyz.max(0)
    margin = .25 * (high - low)
    outside = ((means < low - margin) | (means > high + margin)).any(1)
    return {'count': int(len(means)),
            'opacity_quantiles': dict(zip(('p05', 'p50', 'p95'), np.quantile(opacities, [.05, .5, .95]).round(4).tolist())),
            'max_scale_over_scene_scale_quantiles': dict(zip(
                ('p50', 'p95', 'p99', 'max'), (np.quantile(size, [.5, .95, .99]).tolist() + [float(size.max())]))),
            'larger_than_prune_scale3d': int((size > prune_scale3d * scene_scale).sum()),
            'outside_sfm_points_box_plus_25pct': int(outside.sum()),
            'outside_and_opaque_over_0_5': int((outside & (opacities > .5)).sum())}


def contact_sheet(name, panels, metrics):
    from PIL import ImageDraw
    size = panels[0][1].shape[0]
    thumb = min(size, 384)
    sheet = Image.new('RGB', (len(panels) * thumb, thumb + 40), '#1d252c')
    draw = ImageDraw.Draw(sheet)
    for k, (label, array) in enumerate(panels):
        image = Image.fromarray(array if array.ndim == 3 else np.repeat(array[..., None], 3, -1))
        sheet.paste(image.resize((thumb, thumb)), (k * thumb, 0))
        draw.text((k * thumb + 4, thumb + 4), label, fill='white')
    draw.text((4, thumb + 22), f"{name}  PSNR {metrics['psnr']}  SSIM {metrics['ssim']}", fill='#f0c040')
    return sheet


def inspect(prep, cfg):
    import torch
    from gsplat import rasterization
    from . import gsplat_train
    prep = Path(prep).resolve()
    output = prep.parent
    problems = verify_prep(prep)
    environment = qualify()
    if problems or not environment['qualified']:
        raise RuntimeError('; '.join(problems + environment['problems']))
    training = prep / 'training' / cfg['name']
    status = read(training / 'training.json')
    if status['status'] != 'completed':
        raise RuntimeError(f"training {cfg['name']} is {status['status']}")
    selection = read(training / 'selection.json')
    checkpoint = training / 'checkpoints' / selection['checkpoint']
    manifest = read(prep / 'gsplat_inputs/gsplat_inputs.json')
    meta = {'config_sha256': gsplat_train.config_sha256(cfg),
            'manifest_sha256': digest(prep / 'gsplat_inputs/gsplat_inputs.json'),
            'partition_sha256': manifest['partition_sha256'], 'git_commit': status['git_commit']}
    _, params, *_, saved = gsplat_train.restore(checkpoint, cfg, meta, 'cuda')
    degree = saved['sh_degree']
    data = gsplat_train.load_cameras(prep, 'validation', 'cuda')
    # Face orientation and semantic labels, hash-verified against their source runs.
    split_run = output / manifest['split_run']
    views_rel = 'sfm_import/views.json'
    if digest(split_run / views_rel) != read(split_run / 'run.json')['stages']['import_sfm']['artifacts'][views_rel]:
        raise RuntimeError(f'{split_run.name}/{views_rel} changed')
    rotations = {v['sfm_name']: np.asarray(v['T_face_from_panorama'])[:3, :3]
                 for v in read(split_run / views_rel)['views']}
    semantic = manifest['semantic_run']
    recorded = read(output / semantic / 'run.json')['stages']['auto_mask']['artifacts']
    cameras = read(prep / 'gsplat_inputs/cameras_validation.json')['cameras']
    target = prep / 'inspection' / f"{cfg['name']}-validation-{checkpoint.stem}"
    target.mkdir(parents=True, exist_ok=True)
    labels_cache, faces = {}, []
    for i, camera in enumerate(cameras):
        pano = camera['panorama_id']
        if pano not in labels_cache:
            rel = f'segmentation/fused/{pano}/labels.png'
            path = load_verified(output, {'path': f'{semantic}/{rel}', 'sha256': recorded.get(rel)})
            labels_cache[pano] = np.asarray(Image.open(path))
        with torch.no_grad():
            colors = torch.cat([params['sh0'], params['shN']], 1)
            renders, alphas, _ = rasterization(
                means=params['means'], quats=params['quats'], scales=torch.exp(params['scales']),
                opacities=torch.sigmoid(params['opacities']), colors=colors,
                viewmats=data['viewmats'][i:i + 1], Ks=data['Ks'][i:i + 1], width=data['width'],
                height=data['height'], sh_degree=degree, packed=False, render_mode='RGB+ED')
        rgb = renders[0, ..., :3].clamp(0, 1)
        metrics = gsplat_train.view_metrics(rgb[None], data['images'][i:i + 1].float() / 255,
                                            data['weights'][i:i + 1].float() / 255)
        reference = data['images'][i].cpu().numpy()
        rendered = (rgb.cpu().numpy() * 255).round().astype(np.uint8)
        weight = data['weights'][i].float().cpu().numpy() / 255
        alpha = alphas[0, ..., 0].cpu().numpy()
        depth = renders[0, ..., 3].cpu().numpy()
        labels = project_labels(labels_cache[pano], rotations[camera['name']], data['width'])
        valid = weight > 0
        regions = {name: np.isin(labels, codes) for name, codes in REGIONS.items()}
        regions['contours'] = contour_mask(reference, valid)
        regions['other'] = valid & ~np.logical_or.reduce(list(regions.values()))
        stem = camera['name'].replace('/', '__').removesuffix('.png')
        panels = [('reference', reference), ('render', rendered), ('weights', (weight * 255).astype(np.uint8)),
                  ('error', error_image(reference, rendered, weight)), ('alpha', (alpha * 255).astype(np.uint8)),
                  ('depth', depth_image(depth, alpha))]
        files = {}
        for label, array in panels:
            path = target / stem / f'{label}.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(array).save(path)
            files[label] = str(path.relative_to(target))
        sheet = target / f'{stem}.jpg'
        contact_sheet(camera['name'], panels, {k: None if metrics[k] is None else round(metrics[k], 3)
                                               for k in ('psnr', 'ssim')}).save(sheet, quality=90)
        faces.append({'camera': camera['name'], 'panorama_id': pano, **metrics,
                      'low_alpha_fraction_of_valid': float(((alpha < .5) & valid).sum() / max(valid.sum(), 1)),
                      'regions': region_metrics(reference, rendered, weight, regions),
                      'files': files, 'contact_sheet': sheet.name})
    points = np.load(load_verified(output, {'path': next(r for r in manifest['files'] if r.endswith('points.npz')),
                                            'sha256': next(v for r, v in manifest['files'].items()
                                                           if r.endswith('points.npz'))}))
    statistics = gaussian_statistics(params['means'].detach().cpu().numpy(),
                                     torch.exp(params['scales']).detach().cpu().numpy(),
                                     torch.sigmoid(params['opacities']).detach().cpu().numpy(),
                                     points['xyz'], saved['meta']['scene_scale'],
                                     cfg['strategy'].get('prune_scale3d', .1))
    usable = [f for f in faces if not f['excluded']]
    summary = {'schema_version': 1, 'set': 'validation', 'test_loaded': False,
               'training': cfg['name'], 'checkpoint': checkpoint.name, 'step': saved['step'],
               'sh_degree': degree, 'prep_run': prep.name, 'partition_sha256': manifest['partition_sha256'],
               'checkpoint_git_commit': saved['meta']['git_commit'], 'inspection_git_commit': git_commit(),
               'mean_psnr': float(np.mean([f['psnr'] for f in usable])) if usable else None,
               'mean_ssim': float(np.mean([f['ssim'] for f in usable])) if usable else None,
               'faces': faces, 'gaussians': statistics,
               'regions_definition': {'glass/mirror/unvalidated_reflective': 'semantic labels of the '
                                      'validation panorama projected per face (nearest)',
                                      'contours': 'top 10% image-gradient magnitude among valid pixels',
                                      'other': 'remaining valid pixels'},
               'limitation': manifest['limitation'], 'quality_thresholds': 'none', 'created_at': now()}
    write(target / 'inspection.json', summary)
    write_report(target, summary)
    return target


def write_report(target, summary):
    lines = [f"# Inspection validation — {summary['training']} / {summary['checkpoint']}", '',
             f"Étape {summary['step']}, degré SH {summary['sh_degree']}, partition `{summary['partition_sha256']}`. "
             'Jeu de test non chargé.', '',
             f"PSNR moyen {summary['mean_psnr']}, SSIM moyen {summary['mean_ssim']} (faces exploitables).", '',
             '| Face | PSNR | SSIM | Vitrage | Miroir | Reflets non validés | Contours | Autres | Alpha < 0,5 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt = lambda v: '—' if v is None else f'{v:.2f}'
    for face in summary['faces']:
        r = face['regions']
        lines.append(f"| [{face['camera']}]({face['contact_sheet']}) | {fmt(face['psnr'])} | {fmt(face['ssim'])} | "
                     + ' | '.join(fmt(r[k]['psnr']) for k in ('glass', 'mirror', 'unvalidated_reflective',
                                                             'contours', 'other'))
                     + f" | {face['low_alpha_fraction_of_valid']:.3f} |")
    g = summary['gaussians']
    lines += ['', 'Colonnes de régions : PSNR pondéré sur la région ; « — » si la région est absente de la face.', '',
              '## Gaussiennes', '', f"- Nombre : {g['count']}",
              f"- Opacité (p05/p50/p95) : {g['opacity_quantiles']}",
              f"- Plus grand axe / échelle de scène : {g['max_scale_over_scene_scale_quantiles']}",
              f"- Plus grandes que prune_scale3d : {g['larger_than_prune_scale3d']}",
              f"- Hors boîte des points SfM (+25 %) : {g['outside_sfm_points_box_plus_25pct']}, "
              f"dont opacité > 0,5 : {g['outside_and_opaque_over_0_5']}", '',
              'Aucun seuil de qualité : ces valeurs orientent l’inspection visuelle des planches.',
              f"Limite : {summary['limitation']}", '']
    (target / 'inspection.md').write_text('\n'.join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description='Validation-only visual inspection of a gsplat checkpoint')
    parser.add_argument('--prep', required=True)
    parser.add_argument('--config', required=True, help='Training config of the inspected run')
    args = parser.parse_args(argv)
    try:
        target = inspect(args.prep, read(args.config))
    except Exception as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1
    print(target / 'inspection.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
