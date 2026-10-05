"""Visual inspection of a selected gsplat checkpoint (GPU VM), never the test set.

    python -m theta_pipeline.gsplat_inspect --prep Output/runs/<prep-run> --config configs/<training>.json

Writes under <prep-run>/inspection/<training>-<checkpoint>-v2/ and never touches the
training directory. The selected checkpoint is copied; renders read the copy only, and
the original's SHA-256 is checked before and after. Exported for validation faces and for
train faces chosen by glass-window and furniture content, in two variants (full model,
and without Gaussians larger than prune_scale3d x scene_scale): reference, render, weight
mask, error map, alpha and expected depth, metrics per face and per region. Gaussians
outside the SfM point box are reported, never removed. No quality threshold.
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
           'unvalidated_reflective': (LABELS['unknown_glass'], LABELS['unknown_reflective']),
           'furniture': (LABELS['furniture'],)}
SELECTION = {'glass_windows': (LABELS['glass'], LABELS['unknown_glass']), 'furniture': (LABELS['furniture'],)}


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


def detail_ratio(reference, rendered, mask):
    """Mean luminance-gradient magnitude of the render over that of the reference, inside `mask`.

    Only pixels whose four central-difference neighbours also belong to `mask` are used, so
    pixels outside the region never enter the measure. Below 1: smoother than the reference
    (lost detail). Above 1 can be noise or artefacts as well as detail: it is not a proof of
    faithful detail. None if no such pixel exists or the reference is flat there.
    """
    mask = np.asarray(mask, bool)
    inner = np.zeros_like(mask)
    inner[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:] & mask[:-2, 1:-1] & mask[2:, 1:-1])
    if not inner.any():
        return None

    def gradient(image):
        y = np.asarray(image, float) @ np.array([.2126, .7152, .0722])
        gx, gy = np.zeros_like(y), np.zeros_like(y)
        gx[:, 1:-1] = y[:, 2:] - y[:, :-2]
        gy[1:-1] = y[2:] - y[:-2]
        return np.hypot(gx, gy)[inner]
    ref = gradient(reference).mean()
    return float(gradient(rendered).mean() / ref) if ref > 0 else None


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


def large_gaussians(scales, scene_scale, prune_scale3d):
    """Gaussians whose largest axis exceeds prune_scale3d x scene_scale (gsplat's pruning rule)."""
    return scales.max(1) > prune_scale3d * scene_scale


def gaussian_statistics(means, scales, opacities, points_xyz, scene_scale, prune_scale3d):
    """Population descriptors for floater review (no threshold is turned into a verdict).

    Sizes are reported normalized by scene_scale, the unit of gsplat's prune_scale3d.
    """
    size = scales.max(1) / scene_scale
    low, high = points_xyz.min(0), points_xyz.max(0)
    margin = .25 * (high - low)
    outside = ((means < low - margin) | (means > high + margin)).any(1)
    return {'count': int(len(means)),
            'opacity_quantiles': dict(zip(('p05', 'p50', 'p95'), np.quantile(opacities, [.05, .5, .95]).round(4).tolist())),
            'max_scale_over_scene_scale_quantiles': dict(zip(
                ('p50', 'p95', 'p99', 'max'), (np.quantile(size, [.5, .95, .99]).tolist() + [float(size.max())]))),
            'scene_scale': float(scene_scale), 'prune_scale3d': float(prune_scale3d),
            'larger_than_prune_scale3d': int(large_gaussians(scales, scene_scale, prune_scale3d).sum()),
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


def select_train_faces(cameras, labels_for, rotations, size, per_group=3):
    """Deterministic train faces showing most glass windows and most furniture (ties by name)."""
    fractions = {}
    for camera in cameras:
        labels = project_labels(labels_for(camera['panorama_id']), rotations[camera['name']], size)
        fractions[camera['name']] = {g: float(np.isin(labels, codes).mean()) for g, codes in SELECTION.items()}
    chosen = []
    for group in SELECTION:
        ranked = sorted(fractions, key=lambda n: (-fractions[n][group], n))
        chosen += [n for n in ranked if n not in chosen and fractions[n][group] > 0][:per_group]
    return chosen, {n: fractions[n] for n in chosen}


class Sources:
    """Hash-verified face rotations (split run) and semantic labels (semantic run)."""

    def __init__(self, output, manifest):
        self.output = output
        split_run = output / manifest['split_run']
        rel = 'sfm_import/views.json'
        if digest(split_run / rel) != read(split_run / 'run.json')['stages']['import_sfm']['artifacts'][rel]:
            raise RuntimeError(f'{split_run.name}/{rel} changed')
        self.rotations = {v['sfm_name']: np.asarray(v['T_face_from_panorama'])[:3, :3]
                          for v in read(split_run / rel)['views']}
        self.semantic = manifest['semantic_run']
        self.recorded = read(output / self.semantic / 'run.json')['stages']['auto_mask']['artifacts']
        self.cache = {}

    def labels(self, pano):
        if pano not in self.cache:
            rel = f'segmentation/fused/{pano}/labels.png'
            path = load_verified(self.output, {'path': f'{self.semantic}/{rel}', 'sha256': self.recorded.get(rel)})
            self.cache[pano] = np.asarray(Image.open(path))
        return self.cache[pano]


# Read-only rendering modes. 'historique' is the EWA rasterization used for training and every
# earlier inspection; '3dgut' renders the same parameters with gsplat's unscented-transform
# projection and 3-D (world-space) evaluation. Everything else is shared and explicit.
COMMON_RENDER = {'packed': False, 'render_mode': 'RGB+ED', 'rasterize_mode': 'classic', 'camera_model': 'pinhole',
                 'eps2d': .3, 'near_plane': .01}
RENDERERS = {'historique': {**COMMON_RENDER, 'with_ut': False, 'with_eval3d': False},
             '3dgut': {**COMMON_RENDER, 'with_ut': True, 'with_eval3d': True}}


def render_raw(params, data, i, degree, renderer='historique'):
    """Unclamped RGB+ED [H, W, 4], alpha [H, W] and gsplat's projection meta (radii, depths, ...)."""
    import torch
    from gsplat import rasterization
    if renderer not in RENDERERS:
        raise ValueError(f'unknown renderer {renderer}; choose among {sorted(RENDERERS)}')
    with torch.no_grad():
        renders, alphas, meta = rasterization(
            means=params['means'], quats=params['quats'], scales=torch.exp(params['scales']),
            opacities=torch.sigmoid(params['opacities']), colors=torch.cat([params['sh0'], params['shN']], 1),
            viewmats=data['viewmats'][i:i + 1], Ks=data['Ks'][i:i + 1], width=data['width'],
            height=data['height'], sh_degree=degree, **RENDERERS[renderer])
    return renders[0], alphas[0, ..., 0], meta


def render_face(params, data, i, degree, renderer='historique'):
    renders, alphas, _ = render_raw(params, data, i, degree, renderer)
    return renders[..., :3].clamp(0, 1), alphas, renders[..., 3]


def export_faces(target, cameras, data, params, degree, sources, write_panels=True, renderer='historique'):
    """Per-face metrics (PSNR, SSIM, luminance, per-region PSNR/SSIM/detail) and, optionally, panels."""
    import torch
    from . import gsplat_train
    from .gsplat_checkpoint_diag import luminance
    faces = []
    for i, camera in enumerate(cameras):
        rgb, alpha, depth = render_face(params, data, i, degree, renderer)
        metrics = gsplat_train.view_metrics(rgb[None], data['images'][i:i + 1].float() / 255,
                                            data['weights'][i:i + 1].float() / 255)
        reference = data['images'][i].cpu().numpy()
        rendered_float = rgb.cpu().numpy() * 255          # metrics use the unquantized render
        rendered = rendered_float.round().astype(np.uint8)
        weight = data['weights'][i].float().cpu().numpy() / 255
        alpha, depth = alpha.cpu().numpy(), depth.cpu().numpy()
        labels = project_labels(sources.labels(camera['panorama_id']), sources.rotations[camera['name']],
                                data['width'])
        valid = weight > 0
        regions = {name: np.isin(labels, codes) for name, codes in REGIONS.items()}
        regions['contours'] = contour_mask(reference, valid)
        regions['other'] = valid & ~np.logical_or.reduce(list(regions.values()))
        target_tensor = data['images'][i:i + 1].float() / 255
        with torch.no_grad():
            ssim_map = gsplat_train.ssim_map(rgb[None], target_tensor, data['weights'][i:i + 1] > 0)[0].cpu().numpy()
        region_rows = region_metrics(reference, rendered_float, weight, regions)
        for name, mask in regions.items():
            inside = mask & valid
            region_rows[name]['ssim'] = float(ssim_map[inside].mean()) if inside.any() else None
            region_rows[name]['detail_ratio'] = detail_ratio(reference, rendered_float, inside)
        lum = {'reference_luminance': luminance(reference / 255, valid),
               'render_luminance': luminance(rendered_float / 255, valid)}
        if not write_panels:
            faces.append({'camera': camera['name'], 'panorama_id': camera['panorama_id'], **metrics, **lum,
                          'low_alpha_fraction_of_valid': float(((alpha < .5) & valid).sum() / max(valid.sum(), 1)),
                          'regions': region_rows, 'files': {}, 'contact_sheet': None})
            continue
        stem = camera['name'].replace('/', '__').removesuffix('.png')
        panels = [('reference', reference), ('render', rendered), ('weights', (weight * 255).astype(np.uint8)),
                  ('error', error_image(reference, rendered_float, weight)), ('alpha', (alpha * 255).astype(np.uint8)),
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
        faces.append({'camera': camera['name'], 'panorama_id': camera['panorama_id'], **metrics, **lum,
                      'low_alpha_fraction_of_valid': float(((alpha < .5) & valid).sum() / max(valid.sum(), 1)),
                      'regions': region_rows, 'files': files, 'contact_sheet': sheet.name})
    return faces


def summarize(faces):
    usable = [f for f in faces if not f['excluded']]
    regions = {}
    for name in [*REGIONS, 'contours', 'other']:
        values = [f['regions'][name]['psnr'] for f in usable if f['regions'][name]['psnr'] is not None]
        regions[name] = float(np.mean(values)) if values else None
    return {'faces': len(faces), 'usable_faces': len(usable),
            'mean_psnr': float(np.mean([f['psnr'] for f in usable])) if usable else None,
            'mean_ssim': float(np.mean([f['ssim'] for f in usable])) if usable else None,
            'mean_region_psnr': regions}


def inspect(prep, cfg, train_faces=3, checkpoint=None, ablation=True):
    """Validation faces and selected train faces, with (optionally) and without large Gaussians.

    `checkpoint` names a file in the training's checkpoints (default: the one selected by
    validation), e.g. step_003000.pt to compare arms at the same step. It is copied; only the copy is read and filtered. The original
    file hash is checked before and after. Out-of-box Gaussians are reported, never removed.
    """
    import shutil
    import torch
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
    original = training / 'checkpoints' / (checkpoint or selection['checkpoint'])
    if not original.is_file():
        raise RuntimeError(f'{original.name} is not a checkpoint of {cfg["name"]}')
    original_sha = digest(original)
    root = prep / 'inspection' / f"{cfg['name']}-{original.stem}-v2"
    copy = root / 'checkpoint_copy' / original.name
    copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(original, copy)
    if digest(copy) != original_sha:
        raise RuntimeError('checkpoint copy differs from the original')
    manifest = read(prep / 'gsplat_inputs/gsplat_inputs.json')
    meta = {'config_sha256': gsplat_train.config_sha256(cfg),
            'manifest_sha256': digest(prep / 'gsplat_inputs/gsplat_inputs.json'),
            'partition_sha256': manifest['partition_sha256'], 'git_commit': status['git_commit']}
    _, params, *_, saved = gsplat_train.restore(copy, cfg, meta, 'cuda')
    degree, scale = saved['sh_degree'], saved['meta']['scene_scale']
    prune = cfg['strategy'].get('prune_scale3d', .1)
    large = large_gaussians(torch.exp(params['scales']).detach().cpu().numpy(), scale, prune)
    keep = torch.from_numpy(~large).to(params['means'].device)
    variants = {'full': params}
    if ablation:
        variants['without_large'] = {k: v.detach()[keep] for k, v in params.items()}
        torch.save({'source': original.name, 'source_sha256': original_sha, 'removed_large': int(large.sum()),
                    'rule': f'max axis > {prune} x scene_scale ({scale})',
                    'params': {k: v.cpu() for k, v in variants['without_large'].items()}},
                   root / 'checkpoint_copy' / 'without_large.pt')
    sources = Sources(output, manifest)
    train_cameras = read(prep / 'gsplat_inputs/cameras_train.json')['cameras']
    chosen, chosen_fractions = select_train_faces(train_cameras, sources.labels, sources.rotations,
                                                  train_cameras[0]['width'], train_faces)
    groups = {'validation': read(prep / 'gsplat_inputs/cameras_validation.json')['cameras'],
              'train': [c for c in train_cameras if c['name'] in chosen]}
    points = np.load(load_verified(output, {'path': next(r for r in manifest['files'] if r.endswith('points.npz')),
                                            'sha256': next(v for r, v in manifest['files'].items()
                                                           if r.endswith('points.npz'))}))
    results = {}
    for group, cameras in groups.items():
        data = gsplat_train.load_cameras(prep, group, 'cuda', [c['name'] for c in cameras])
        order = {name: k for k, name in enumerate(data['names'])}
        cameras = sorted(cameras, key=lambda c: order[c['name']])
        for variant, model in variants.items():
            target = root / f'{group}-{variant}'
            faces = export_faces(target, cameras, data, model, degree, sources)
            results[(group, variant)] = {'summary': summarize(faces), 'faces': faces, 'target': target}
            write(target / 'inspection.json', {'set': group, 'variant': variant, **summarize(faces), 'faces': faces})
        del data
        torch.cuda.empty_cache()
    if digest(original) != original_sha:
        raise RuntimeError('the original checkpoint changed during inspection')
    summary = {
        'schema_version': 2, 'training': cfg['name'], 'checkpoint': original.name,
        'checkpoint_choice': 'requested' if checkpoint else 'selected by validation',
        'checkpoint_sha256': original_sha, 'original_unchanged': True, 'step': saved['step'],
        'sh_degree': degree, 'prep_run': prep.name, 'partition_sha256': manifest['partition_sha256'],
        'test_loaded': False, 'checkpoint_git_commit': saved['meta']['git_commit'],
        'inspection_git_commit': git_commit(),
        'train_faces': {'rule': f'top {train_faces} train faces by projected glass-window (validated + '
                                f'unvalidated) fraction, then top {train_faces} by furniture fraction',
                        'faces': chosen_fractions},
        'gaussians': gaussian_statistics(params['means'].detach().cpu().numpy(),
                                         torch.exp(params['scales']).detach().cpu().numpy(),
                                         torch.sigmoid(params['opacities']).detach().cpu().numpy(),
                                         points['xyz'], scale, prune),
        'ablation': ({'removed_large_gaussians': int(large.sum()), 'rule': f'max axis > {prune} x scene_scale',
                      'out_of_box_gaussians': 'reported only, never removed'} if ablation else None),
        'results': {f'{g}/{v}': r['summary'] for (g, v), r in results.items()},
        'limitation': manifest['limitation'], 'quality_thresholds': 'none', 'created_at': now()}
    write(root / 'inspection.json', summary)
    write_report(root, summary, results)
    return root


def write_report(root, summary, results):
    fmt = lambda v: '—' if v is None else f'{v:.2f}'
    regions = [*REGIONS, 'contours', 'other']
    g = summary['gaussians']
    lines = [f"# Inspection v2 — {summary['training']} / {summary['checkpoint']}", '',
             f"Étape {summary['step']}, degré SH {summary['sh_degree']}. Original inchangé "
             f"(`{summary['checkpoint_sha256'][:12]}`) ; rendus sur une copie. Jeu de test non chargé.", '',
             '## Synthèse (PSNR pondéré moyen)', '',
             '| Ensemble / variante | Faces | PSNR | SSIM | ' + ' | '.join(regions) + ' |',
             '|---|---:|---:|---:|' + '---:|' * len(regions)]
    for key, r in summary['results'].items():
        lines.append(f"| {key} | {r['usable_faces']}/{r['faces']} | {fmt(r['mean_psnr'])} | {fmt(r['mean_ssim'])} | "
                     + ' | '.join(fmt(r['mean_region_psnr'][k]) for k in regions) + ' |')
    lines += ['', 'Train = faces apprises ; si le défaut y apparaît aussi, il ne vient pas de la généralisation.', '',
              '## Effet du retrait des grandes gaussiennes, par face', '',
              '| Ensemble | Face | PSNR complet | PSNR sans grandes | Écart |', '|---|---|---:|---:|---:|']
    for group in ('validation', 'train'):
        if (group, 'without_large') not in results:
            lines.append(f'| {group} | — | — | — | non calculé (--no-ablation) |')
            continue
        full = {f['camera']: f for f in results[(group, 'full')]['faces']}
        for face in results[(group, 'without_large')]['faces']:
            a, b = full[face['camera']]['psnr'], face['psnr']
            delta = None if a is None or b is None else b - a
            lines.append(f"| {group} | [{face['camera']}]({group}-without_large/{face['contact_sheet']}) | "
                         f"{fmt(a)} | {fmt(b)} | {fmt(delta)} |")
    lines += ['', '## Faces d’entraînement retenues', '']
    lines += [f"- {n} : vitrage {v['glass_windows']:.3f}, mobilier {v['furniture']:.3f}"
              for n, v in summary['train_faces']['faces'].items()]
    lines += ['', '## Gaussiennes (tailles normalisées par scene_scale)', '', f"- Nombre : {g['count']}",
              f"- scene_scale : {g['scene_scale']:.4f} ; seuil prune_scale3d : {g['prune_scale3d']}",
              f"- Plus grand axe / scene_scale : {g['max_scale_over_scene_scale_quantiles']}",
              f"- Au-delà du seuil (retirées dans la variante) : {g['larger_than_prune_scale3d']}",
              f"- Hors boîte des points SfM (+25 %) : {g['outside_sfm_points_box_plus_25pct']}, "
              f"dont opacité > 0,5 : {g['outside_and_opaque_over_0_5']} (signalées, non retirées)", '',
              f"Limite : {summary['limitation']}", '']
    (root / 'inspection.md').write_text('\n'.join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description='Inspection of a gsplat checkpoint (validation + selected train faces)')
    parser.add_argument('--prep', required=True)
    parser.add_argument('--config', required=True, help='Training config of the inspected run')
    parser.add_argument('--train-faces', type=int, default=3, help='Train faces per selection group')
    parser.add_argument('--checkpoint', help='Checkpoint file name (default: selected by validation)')
    parser.add_argument('--no-ablation', action='store_true', help='Skip the without-large-Gaussians variant')
    args = parser.parse_args(argv)
    try:
        root = inspect(args.prep, read(args.config), args.train_faces, args.checkpoint, not args.no_ablation)
    except Exception as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1
    print(root / 'inspection.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
