"""Correction experiment: one post-training cleaning rule defined from training views only.

    python -m theta_pipeline.gsplat_clean --experiment configs/experiments/correction-near-train.json \
        --prep Output/runs/salon-gsplat-009

Rule `near-extent-train-views-v1` (parameters fixed in the experiment file, before any
evaluation): a Gaussian is removed when, in at least one TRAIN face where gsplat renders
it, its oriented depth extent (sigma x standard deviation along the view axis) reaches
nearer than `factor` x the `reference_quantile` depth of the SfM points visible in that
face. Validation cameras never define the rule. The removal set is the union over train
faces, so each training yields ONE cleaned model, evaluated identically on every
validation face of every validation panorama, next to the unmodified model. Tracked:
PSNR, SSIM, luminance, region PSNR/SSIM/detail (furniture, mirror, glass, contours, other).
The original checkpoint is only read (SHA-256 checked); a cleaned copy with the removed
indices is written for traceability. The test set is never loaded. No threshold.
"""
import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
import numpy as np
from .gsplat_checkpoint_diag import near_extent_gaussians, reference_depth
from .storage import digest, now, read, write

RULE = 'near-extent-train-views-v1'


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def run_identity(experiment, prep_manifest_sha256, trainings):
    """Identity of one correction run: experiment (rule included), preparation, trainings and checkpoints."""
    return canonical_sha256({'experiment': experiment, 'prep_manifest_sha256': prep_manifest_sha256,
                             'trainings': trainings})


def new_folder(path):
    """Create a run folder; an existing one is never overwritten."""
    path = Path(path)
    if path.exists():
        raise RuntimeError(f'{path} already exists: a correction run is never overwritten')
    path.mkdir(parents=True)
    return path
TRACKED = ('furniture', 'mirror', 'glass', 'contours', 'other')


def removal_set(cameras, host, points, rule):
    """Union over TRAIN cameras of the Gaussians the rule classifies near; per-camera counts."""
    if any(c.get('set') != 'train' for c in cameras):
        raise RuntimeError('the cleaning rule is defined from train cameras only')
    means, quats = host['means'], host['quats']
    scales, opacity = np.exp(host['scales']), 1 / (1 + np.exp(-host['opacities']))
    remove = np.zeros(len(means), bool)
    counts = {}
    for camera in cameras:
        size = camera['width']
        reference = reference_depth(camera['world_to_camera'], camera['K'], size, points, rule['reference_quantile'])
        near = near_extent_gaussians(camera['world_to_camera'], camera['K'], size, means, quats, scales, opacity,
                                     reference, rule['factor'], rule['sigma'])
        counts[camera['name']] = int(near.sum())
        remove |= near
    return remove, counts


def summarize(faces):
    """Per validation panorama: means over usable faces; regions absent are None, never zero."""
    out = {}
    for pano in sorted({f['panorama_id'] for f in faces}):
        rows = [f for f in faces if f['panorama_id'] == pano and not f['excluded']]
        mean = lambda values: statistics.fmean(values) if values else None
        region = lambda name, key: mean([f['regions'][name][key] for f in rows
                                         if f['regions'].get(name, {}).get(key) is not None])
        out[pano] = {'faces': len([f for f in faces if f['panorama_id'] == pano]), 'usable': len(rows),
                     'psnr': mean([f['psnr'] for f in rows]), 'ssim': mean([f['ssim'] for f in rows]),
                     'luminance_minus_reference': mean([f['render_luminance'] - f['reference_luminance']
                                                        for f in rows if f.get('render_luminance') is not None]),
                     'regions': {name: {k: region(name, k) for k in ('psnr', 'ssim', 'detail_ratio')}
                                 for name in TRACKED}}
    return out


def degradations(full, cleaned):
    """Faces that lose SSIM, or whose luminance moves away from the reference, once cleaned."""
    before = {f['camera']: f for f in full}
    ssim_down, darker = [], []
    for face in cleaned:
        ref = before[face['camera']]
        if face['excluded'] or ref['excluded'] or None in (face.get('render_luminance'), ref.get('render_luminance')):
            continue
        if face['ssim'] < ref['ssim']:
            ssim_down.append(face['camera'])
        if abs(face['render_luminance'] - face['reference_luminance']) > abs(
                ref['render_luminance'] - ref['reference_luminance']):
            darker.append(face['camera'])
    return {'ssim_decreased': ssim_down, 'luminance_further_from_reference': darker}


def delta(a, b):
    return None if a is None or b is None else b - a


def compare_summaries(full, cleaned):
    out = {}
    for pano in full:
        f, c = full[pano], cleaned[pano]
        out[pano] = {key: delta(f[key], c[key]) for key in ('psnr', 'ssim', 'luminance_minus_reference')}
        out[pano]['regions'] = {name: {k: delta(f['regions'][name][k], c['regions'][name][k])
                                       for k in ('psnr', 'ssim', 'detail_ratio')} for name in TRACKED}
    return out


def clean_and_evaluate(prep, cfg, experiment, sources, train_cameras, validation, points, run_dir, provenance):
    """One training: rule from train views, one cleaned model, both models on all validation faces."""
    import torch
    from . import gsplat_train
    from .gsplat_inspect import export_faces
    rule = experiment['rule']
    training = prep / 'training' / cfg['name']
    status = read(training / 'training.json')
    manifest = read(prep / 'gsplat_inputs/gsplat_inputs.json')
    meta = {'config_sha256': gsplat_train.config_sha256(cfg),
            'manifest_sha256': digest(prep / 'gsplat_inputs/gsplat_inputs.json'),
            'partition_sha256': manifest['partition_sha256'], 'git_commit': status['git_commit']}
    checkpoint = training / 'checkpoints' / experiment['checkpoint']
    original = digest(checkpoint)
    _, params, *_, saved = gsplat_train.restore(checkpoint, cfg, meta, 'cuda')
    host = {k: v.detach().cpu().numpy() for k, v in params.items()}
    remove, counts = removal_set(train_cameras, host, points, rule)
    keep = torch.from_numpy(~remove).to(params['means'].device)
    cleaned = {k: v.detach()[keep] for k, v in params.items()}
    target = new_folder(run_dir / cfg['name'])
    stamp = {**provenance, 'training': cfg['name'], 'training_config_sha256': meta['config_sha256'],
             'checkpoint': checkpoint.name, 'checkpoint_sha256': original, 'sh_degree': saved['sh_degree']}
    torch.save({**stamp, 'source': checkpoint.name, 'source_sha256': original, 'rule': rule,
                'removed_indices': np.flatnonzero(remove), 'params': {k: v.cpu() for k, v in cleaned.items()}},
               target / f'cleaned_{checkpoint.name}')
    faces = {}
    for variant, model in (('full', params), ('cleaned', cleaned)):
        faces[variant] = export_faces(target / variant, validation['cameras'], validation['data'], model,
                                      saved['sh_degree'], sources, write_panels=experiment.get('write_panels', False))
    if digest(checkpoint) != original:
        raise RuntimeError(f'{checkpoint} changed during the correction experiment')
    full_summary, cleaned_summary = summarize(faces['full']), summarize(faces['cleaned'])
    result = {**stamp, 'seed': cfg['seed'],
              'gaussians': int(len(remove)), 'removed': int(remove.sum()),
              'removed_opaque': int((remove & (1 / (1 + np.exp(-host['opacities'])) > .5)).sum()),
              'train_faces_with_removals': sum(1 for v in counts.values() if v),
              'full': full_summary, 'cleaned': cleaned_summary,
              'delta': compare_summaries(full_summary, cleaned_summary),
              'degradations': degradations(faces['full'], faces['cleaned']), 'faces': faces}
    write(target / 'evaluation.json', result)
    del params, cleaned
    torch.cuda.empty_cache()
    return result


def run(experiment, prep):
    from . import gsplat_train
    from .gsplat_inspect import Sources
    from .gsplat_preflight import load_verified, qualify, verify_prep
    prep = Path(prep).resolve()
    problems = verify_prep(prep) + qualify()['problems']
    if problems:
        raise RuntimeError('; '.join(problems))
    manifest = read(prep / 'gsplat_inputs/gsplat_inputs.json')
    if manifest['partition_sha256'] != experiment['partition_sha256']:
        raise RuntimeError('prepared inputs do not use the experiment partition')
    train_cameras = read(prep / 'gsplat_inputs/cameras_train.json')['cameras']
    validation_cameras = read(prep / 'gsplat_inputs/cameras_validation.json')['cameras']
    validation = {'cameras': validation_cameras,
                  'data': gsplat_train.load_cameras(prep, 'validation', 'cuda', [c['name'] for c in validation_cameras])}
    rel = next(r for r in manifest['files'] if r.endswith('points.npz'))
    points = np.load(load_verified(prep.parent, {'path': rel, 'sha256': manifest['files'][rel]}))['xyz']
    sources = Sources(prep.parent, manifest)
    from .segmentation.provenance import git_commit
    configs = [read(path) for path in experiment['trainings']]
    trainings = {cfg['name']: {'config_sha256': gsplat_train.config_sha256(cfg),
                               'checkpoint_sha256': digest(prep / 'training' / cfg['name'] / 'checkpoints'
                                                           / experiment['checkpoint'])} for cfg in configs}
    manifest_sha = digest(prep / 'gsplat_inputs/gsplat_inputs.json')
    identity = run_identity(experiment, manifest_sha, trainings)
    provenance = {'run_identity': identity, 'experiment_sha256': canonical_sha256(experiment),
                  'prep_run': prep.name, 'prep_manifest_sha256': manifest_sha,
                  'analysis_git_commit': git_commit()}
    target = new_folder(prep / 'corrections' / experiment['name'] / identity[:16])
    write(target / 'experiment.json', {**provenance, 'experiment': experiment, 'trainings': trainings})
    results = []
    for cfg in configs:
        print(f"clean {cfg['name']}", flush=True)
        results.append(clean_and_evaluate(prep, cfg, experiment, sources, train_cameras, validation, points,
                                          target, provenance))
    summary = {**provenance, 'experiment': experiment['name'], 'rule': experiment['rule'],
               'checkpoint': experiment['checkpoint'], 'test_loaded': False, 'created_at': now(),
               'trainings': [{k: v for k, v in r.items() if k != 'faces'} for r in results]}
    write(target / 'summary.json', summary)
    (target / 'summary.md').write_text(report(summary))
    return target


def report(summary):
    fmt = lambda v, d=3: '—' if v is None else f'{v:+.{d}f}'
    rows = summary['trainings']
    panoramas = sorted({p for r in rows for p in r['delta']})
    rule = summary['rule']
    lines = [f"# Correction {summary['experiment']} — {summary['prep_run']}, {summary['checkpoint']}", '',
             f"Exécution `{summary.get('run_identity', '—')[:16]}` ; expérience `{summary.get('experiment_sha256', '—')[:16]}`, "
             f"préparation `{summary.get('prep_manifest_sha256', '—')[:16]}`, commit d’analyse "
             f"`{summary.get('analysis_git_commit')}`.", '',
             f"Règle `{rule['name']}`, définie sur les seules vues d’entraînement : retrait d’une gaussienne si, dans "
             f"au moins une face d’entraînement où gsplat la rend, son étendue ({rule['sigma']:g} σ selon l’axe de "
             f"visée) atteint moins de {rule['factor']} × le quantile {rule['reference_quantile']} des profondeurs "
             'SfM visibles. Un seul modèle nettoyé par entraînement, évalué sur toutes les faces de validation.',
             'Résultats limités aux panoramas de validation ' + ' et '.join(panoramas) + ' ; jeu de test non chargé.', '',
             '## Écarts nettoyé − complet par entraînement et par panorama', '',
             '| Entraînement | Retirées (opaques) | Panorama | ΔPSNR | ΔSSIM | Δ(luminance − référence) | '
             'ΔPSNR mobilier | Δdétail mobilier | ΔPSNR miroir | ΔSSIM miroir | ΔPSNR vitrage | ΔPSNR contours |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        for pano in panoramas:
            d = r['delta'][pano]
            g = d['regions']
            lines.append(f"| {r['training']} | {r['removed']} ({r['removed_opaque']}) | {pano} | {fmt(d['psnr'])} | "
                         f"{fmt(d['ssim'], 4)} | {fmt(d['luminance_minus_reference'], 4)} | {fmt(g['furniture']['psnr'])} | "
                         f"{fmt(g['furniture']['detail_ratio'])} | {fmt(g['mirror']['psnr'])} | {fmt(g['mirror']['ssim'], 4)} | "
                         f"{fmt(g['glass']['psnr'])} | {fmt(g['contours']['psnr'])} |")
    lines += ['', '## Dégradations suivies', '', '| Entraînement | Faces dont le SSIM baisse | '
              'Faces dont la luminance s’éloigne de la référence |', '|---|---|---|']
    for r in rows:
        lines.append(f"| {r['training']} | {len(r['degradations']['ssim_decreased'])} | "
                     f"{len(r['degradations']['luminance_further_from_reference'])} |")
    lines += ['', '## Synthèse par bras et par panorama (sur les graines)', '',
              '| Bras | Panorama | n | ΔPSNR moyen / médian | ΔSSIM moyen | > 0 / < 0 (PSNR) |', '|---|---|---:|---|---:|---|']
    for arm in sorted({r['training'].rsplit('-seed', 1)[0] for r in rows}):
        for pano in panoramas:
            values = [r['delta'][pano]['psnr'] for r in rows
                      if r['training'].startswith(arm + '-seed') and r['delta'][pano]['psnr'] is not None]
            ssim = [r['delta'][pano]['ssim'] for r in rows
                    if r['training'].startswith(arm + '-seed') and r['delta'][pano]['ssim'] is not None]
            if values:
                lines.append(f"| {arm} | {pano} | {len(values)} | {statistics.fmean(values):+.3f} / "
                             f"{statistics.median(values):+.3f} | {statistics.fmean(ssim):+.4f} | "
                             f"{sum(v > 0 for v in values)} / {sum(v < 0 for v in values)} |")
    lines += ['', 'Une région absente d’un panorama apparaît « — », jamais comme un score nul. Le détail '
                  '(rapport des gradients rendu / référence, pixels intérieurs à la région seulement) inférieur à 1 '
                  'signale un lissage ; supérieur à 1, il peut traduire du bruit autant que du détail. Aucun seuil.', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Correction experiment: one train-view cleaning rule')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--prep', required=True)
    args = parser.parse_args(argv)
    try:
        target = run(read(args.experiment), args.prep)
    except Exception as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1
    print(target / 'summary.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
