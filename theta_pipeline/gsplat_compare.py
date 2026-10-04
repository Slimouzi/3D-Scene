"""Compare the arms of a controlled gsplat experiment on validation only (CPU, files only).

    python -m theta_pipeline.gsplat_compare --experiment configs/experiments/ctrl-duration-pruning.json \
        --prep Output/runs/salon-gsplat-007 --reference Output/runs/salon-gsplat-006:l4-short-001

Reads training.json, validation.jsonl, selection.json, train.jsonl and, when present, the
inspection v2 summary. Never reads the test set. Reports differences, never verdicts: the
replicate-versus-reference gap is printed as the noise scale.
"""
import argparse
import json
import sys
from pathlib import Path
from .storage import now, read, write


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()] if Path(path).exists() else []


def inspection_at(prep, name, checkpoint):
    """Inspection v2 summary of a training at a given checkpoint file, or None."""
    path = Path(prep) / 'inspection' / f'{name}-{Path(checkpoint).stem}-v2/inspection.json'
    if not path.exists():
        return None
    data = read(path)
    keep = lambda r: {'mean_psnr': r['mean_psnr'], 'mean_ssim': r['mean_ssim'],
                      'usable_faces': r['usable_faces'], 'regions': r['mean_region_psnr']}
    return {'checkpoint': data['checkpoint'], 'validation': keep(data['results']['validation/full']),
            'train': keep(data['results']['train/full']), 'gaussians': data['gaussians']}


def panoramas_at(row):
    """Per validation panorama: faces, usable faces, mean PSNR and SSIM of one validation row."""
    out = {}
    for camera in (row or {}).get('cameras', []):
        entry = out.setdefault(Path(camera['camera']).stem, {'faces': 0, 'usable': 0, 'psnr': [], 'ssim': []})
        entry['faces'] += 1
        if not camera.get('excluded') and camera.get('psnr') is not None:
            entry['usable'] += 1
            entry['psnr'].append(camera['psnr'])
            if camera.get('ssim') is not None:
                entry['ssim'].append(camera['ssim'])
    mean = lambda v: sum(v) / len(v) if v else None
    return {k: {'faces': v['faces'], 'usable': v['usable'], 'psnr': mean(v['psnr']), 'ssim': mean(v['ssim'])}
            for k, v in sorted(out.items())}


def arm_summary(prep, name):
    folder = Path(prep) / 'training' / name
    if not (folder / 'training.json').exists():
        return {'name': name, 'status': 'not run'}
    training = read(folder / 'training.json')
    validation = {row['step']: row for row in jsonl(folder / 'validation.jsonl')}
    last = max(validation) if validation else None
    log = jsonl(folder / 'train.jsonl')
    logged = {row['step']: row for row in log}
    pick = lambda step, key: validation[step][key] if step in validation else None
    selection = read(folder / 'selection.json') if (folder / 'selection.json').exists() else {}
    by_panorama, panoramas = {}, {}
    for camera in (validation.get(3000) or {}).get('cameras', []):
        pano = Path(camera['camera']).stem
        entry = panoramas.setdefault(pano, {'faces': 0, 'usable': 0, 'psnr': [], 'ssim': []})
        entry['faces'] += 1
        if not camera.get('excluded') and camera.get('psnr') is not None:
            by_panorama.setdefault(pano, []).append(camera['psnr'])
            entry['usable'] += 1
            entry['psnr'].append(camera['psnr'])
            if camera.get('ssim') is not None:
                entry['ssim'].append(camera['ssim'])
    mean = lambda v: sum(v) / len(v) if v else None
    memory = [row['max_memory_gb'] for row in log if row.get('max_memory_gb') is not None]
    return {'name': name, 'status': training['status'], 'steps': training['config']['steps'],
            'factors': training['config'].get('schedule', {}).get('factors'),
            'val_psnr_3000': pick(3000, 'mean_psnr'), 'val_ssim_3000': pick(3000, 'mean_ssim'),
            'val_psnr_3000_by_panorama': {k: sum(v) / len(v) for k, v in sorted(by_panorama.items())},
            'panoramas_3000': {k: {'faces': v['faces'], 'usable': v['usable'], 'psnr': mean(v['psnr']),
                                   'ssim': mean(v['ssim'])} for k, v in sorted(panoramas.items())},
            'validation_curve': [{'step': step, 'mean_psnr': row['mean_psnr'], 'mean_ssim': row['mean_ssim'],
                                  'usable': row.get('usable_cameras')} for step, row in sorted(validation.items())],
            'selected_checkpoint': selection.get('checkpoint'),
            'val_psnr_selected': pick(selection.get('step'), 'mean_psnr'),
            'val_ssim_selected': pick(selection.get('step'), 'mean_ssim'),
            'panoramas_selected': panoramas_at(validation.get(selection.get('step'))),
            'last_step': last, 'val_psnr_last': pick(last, 'mean_psnr'), 'val_ssim_last': pick(last, 'mean_ssim'),
            'usable_validation_faces': pick(last, 'usable_cameras'),
            'selected_step': selection.get('step'), 'test_used': selection.get('test_used'),
            'gaussians_3000': logged.get(3000, {}).get('gaussians'),
            'gaussians_last': log[-1].get('gaussians') if log else None,
            'max_memory_gb': max(memory) if memory else None,
            'seconds': log[-1].get('seconds') if log else None,
            'runs': len(training.get('history', [])),
            'loss_curve': {row['step']: row['loss'] for row in log if 'loss' in row},
            'inspection_3000': inspection_at(prep, name, 'step_003000.pt'),
            'inspection_last': inspection_at(prep, name, f'step_{last:06d}.pt') if last else None}


def region_coverage(prep, name, checkpoint='step_003000.pt'):
    """Per validation panorama and region: faces where the region exists, pixels, mean PSNR.

    A region absent from every face is reported as absent (mean None), never as a zero score.
    """
    path = Path(prep) / 'inspection' / f'{name}-{Path(checkpoint).stem}-v2' / 'validation-full' / 'inspection.json'
    if not path.exists():
        return None
    out = {}
    for face in read(path)['faces']:
        pano = out.setdefault(face['panorama_id'], {})
        for region, values in face['regions'].items():
            entry = pano.setdefault(region, {'faces': 0, 'faces_present': 0, 'pixels': 0, 'psnr': []})
            entry['faces'] += 1
            if values['pixels'] > 0 and values['psnr'] is not None:
                entry['faces_present'] += 1
                entry['pixels'] += values['pixels']
                entry['psnr'].append(values['psnr'])
    for pano in out.values():
        for entry in pano.values():
            values = entry.pop('psnr')
            entry['mean_psnr'] = sum(values) / len(values) if values else None
            entry['absent'] = not values
    return out


def pairing(short, long):
    """Short and long arms share every setting up to step 3000: how close are they there?"""
    if not short or not long or 'loss_curve' not in short or 'loss_curve' not in long:
        return None
    common = sorted(set(short['loss_curve']) & set(long['loss_curve']))
    common = [s for s in common if s <= 3000]
    gaps = [abs(short['loss_curve'][s] - long['loss_curve'][s]) for s in common]
    return {'psnr_3000_long_minus_short': difference(long, short, 'val_psnr_3000'),
            'ssim_3000_long_minus_short': difference(long, short, 'val_ssim_3000'),
            'gaussians_3000_long_minus_short': difference(long, short, 'gaussians_3000'),
            'logged_steps_compared': len(common), 'max_abs_loss_gap': max(gaps) if gaps else None,
            'note': 'identical settings until 3000; residual gaps come from non-deterministic GPU '
                    'reductions and are read against the replicate noise'}


def difference(a, b, key):
    return None if a is None or b is None or a.get(key) is None or b.get(key) is None else a[key] - b[key]


def compare(experiment, prep, reference=None):
    arms = {Path(path).stem.removeprefix('gsplat-'): read(path) for path in experiment['arms']}
    rows = {name: arm_summary(prep, cfg['name']) for name, cfg in arms.items()}
    if any(r.get('test_used') for r in rows.values()):
        raise RuntimeError('an arm used the test set: the comparison is not valid')
    by_factor = {(cfg['schedule']['factors']['duration'], cfg['schedule']['factors']['pruning']): rows[name]
                 for name, cfg in arms.items()}
    durations = sorted({d for d, _ in by_factor})
    prunings = sorted({p for _, p in by_factor})
    effects = {'duration (last step, longer - shorter)': {
                   p: difference(by_factor.get((durations[-1], p)), by_factor.get((durations[0], p)), 'val_psnr_last')
                   for p in prunings},
               'duration inside the longer arm (last - step 3000)': {
                   p: difference({'v': (by_factor.get((durations[-1], p)) or {}).get('val_psnr_last')},
                                 {'v': (by_factor.get((durations[-1], p)) or {}).get('val_psnr_3000')}, 'v')
                   for p in prunings},
               'pruning (s1 - s0, same duration)': {
                   d: difference(by_factor.get((d, prunings[-1])), by_factor.get((d, prunings[0])), 'val_psnr_last')
                   for d in durations}}
    pairs = {p: pairing(by_factor.get((durations[0], p)), by_factor.get((durations[-1], p))) for p in prunings}
    noise = None
    if reference:
        ref_prep, ref_name = reference
        ref = arm_summary(ref_prep, ref_name)
        replicate = by_factor.get((durations[0], prunings[0]))
        noise = {'reference': f'{Path(ref_prep).name}/{ref_name}', 'reference_val_psnr_3000': ref.get('val_psnr_3000'),
                 'replicate_minus_reference_psnr_3000': difference(replicate, ref, 'val_psnr_3000'),
                 'replicate_minus_reference_ssim_3000': difference(replicate, ref, 'val_ssim_3000')}
    for row in rows.values():
        row.pop('loss_curve', None)
    return {'experiment': experiment['name'], 'prep_run': Path(prep).name, 'arms': rows, 'effects': effects,
            'pairing_at_3000': pairs,
            'noise': noise, 'set': 'validation only', 'quality_thresholds': 'none', 'created_at': now()}


REGIONS = ('contours', 'glass', 'furniture', 'mirror', 'unvalidated_reflective', 'other')


def report(result):
    plain = lambda v: '—' if v is None else (f'{v:.3f}' if isinstance(v, float) else str(v))
    signed = lambda v: '—' if v is None else (f'{v:+.3f}' if isinstance(v, float) else f'{v:+d}')
    arms = result['arms']
    lines = [f"# {result['experiment']} — {result['prep_run']} (validation uniquement)", '',
             'Comparaison expérimentale : elle ne valide ni la navigation ni la qualité produit.', '',
             '## Métriques de validation', '',
             '| Bras | Statut | PSNR@3000 | SSIM@3000 | PSNR fin | SSIM fin | Étape choisie |',
             '|---|---|---:|---:|---:|---:|---:|']
    for name, r in arms.items():
        lines.append(f"| {name} | {r['status']} | {plain(r.get('val_psnr_3000'))} | {plain(r.get('val_ssim_3000'))} | "
                     f"{plain(r.get('val_psnr_last'))} | {plain(r.get('val_ssim_last'))} | {plain(r.get('selected_step'))} |")
    lines += ['', '## Régions (PSNR moyen par région, faces de validation, inspection)', '',
              '| Bras | Checkpoint | ' + ' | '.join(REGIONS) + ' |', '|---|---|' + '---:|' * len(REGIONS)]
    for name, r in arms.items():
        for key in ('inspection_3000', 'inspection_last'):
            ins = r.get(key)
            if ins and (key == 'inspection_3000' or ins['checkpoint'] != (r.get('inspection_3000') or {}).get('checkpoint')):
                lines.append(f"| {name} | {ins['checkpoint']} | "
                             + ' | '.join(plain(ins['validation']['regions'].get(k)) for k in REGIONS) + ' |')
    lines += ['', 'Faces d’entraînement inspectées (mêmes faces pour tous les bras) : PSNR moyen', '']
    for name, r in arms.items():
        ins = [r.get('inspection_3000'), r.get('inspection_last')]
        lines.append(f"- {name} : " + ', '.join(f"{i['checkpoint']} {plain(i['train']['mean_psnr'])}" for i in ins if i)
                     if any(ins) else f'- {name} : non inspecté')
    lines += ['', '## Ressources', '',
              '| Bras | Gaussiennes@3000 | Gaussiennes fin | Grandes (> seuil) | Taille p95 / max (÷ scene_scale) '
              '| Hors boîte (opaques) | Mémoire max (Go) | Durée (s) | Exécutions |',
              '|---|---:|---:|---:|---|---:|---:|---:|---:|']
    for name, r in arms.items():
        g = (r.get('inspection_last') or r.get('inspection_3000') or {}).get('gaussians')
        if g:
            size = g['max_scale_over_scene_scale_quantiles']
            large = plain(g['larger_than_prune_scale3d'])
            sizes = f"{size['p95']:.4f} / {size['max']:.4f}"
            outside = f"{g['outside_sfm_points_box_plus_25pct']} ({g['outside_and_opaque_over_0_5']})"
        else:
            large = sizes = outside = '—'
        lines.append(f"| {name} | {plain(r.get('gaussians_3000'))} | {plain(r.get('gaussians_last'))} | {large} | "
                     f"{sizes} | {outside} | {plain(r.get('max_memory_gb'))} | {plain(r.get('seconds'))} | "
                     f"{plain(r.get('runs'))} |")
    lines += ['', '## Appariement court / long à 3 000 itérations', '']
    for pruning, pair in result['pairing_at_3000'].items():
        if pair is None:
            lines.append(f'- {pruning} : non disponible')
            continue
        lines.append(f"- {pruning} : ΔPSNR {signed(pair['psnr_3000_long_minus_short'])} dB, "
                     f"ΔSSIM {signed(pair['ssim_3000_long_minus_short'])}, "
                     f"Δgaussiennes {signed(pair['gaussians_3000_long_minus_short'])}, "
                     f"écart de perte max {plain(pair['max_abs_loss_gap'])} sur {pair['logged_steps_compared']} étapes")
    lines += ['', '## Effets (écarts de PSNR moyen en validation, dB)', '']
    for effect, values in result['effects'].items():
        lines.append(f'- {effect} : ' + ', '.join(f'{k} {signed(v)}' for k, v in values.items()))
    if result['noise']:
        n = result['noise']
        lines += ['', f"Bruit : réplique − référence ({n['reference']}) à l’étape 3000 = "
                      f"{signed(n['replicate_minus_reference_psnr_3000'])} dB (SSIM "
                      f"{signed(n['replicate_minus_reference_ssim_3000'])}). C’est un repère de variation entre "
                      'deux exécutions identiques, pas un seuil statistique.']
    lines += ['', 'Aucun gagnant n’est retenu sur le PSNR moyen seul : lire les régions (contours, vitrages, '
                  'mobilier) et les planches à 3 000 et en fin. Aucun seuil de qualité ; jeu de test non utilisé.', '']
    return '\n'.join(lines)


def paired(experiment, prep, noise=None):
    """Per-seed differences variant - baseline (s1 - s0) at 3000 on validation; mean, median, spread, signs.

    `noise` is the replicate gap in PSNR (dB), e.g. {'psnr_db': 0.244, 'source': ...}.
    """
    import statistics
    pairs = list(experiment['pairs'])
    if experiment.get('existing_seed0'):
        pairs.append({'seed': 0, 's0': 'ctrl-3k-s0', 's1': 'ctrl-3k-s1'})
    rows = []
    for pair in sorted(pairs, key=lambda p: p['seed']):
        a = arm_summary(prep, pair.get('baseline', pair.get('s0')))
        b = arm_summary(prep, pair.get('variant', pair.get('s1')))
        if a.get('test_used') or b.get('test_used'):
            raise RuntimeError('an arm used the test set: the comparison is not valid')
        row = {'seed': pair['seed'], 'status': (a['status'], b['status']),
               'psnr': difference(b, a, 'val_psnr_3000'), 'ssim': difference(b, a, 'val_ssim_3000'),
               'by_panorama': {k: difference(b.get('val_psnr_3000_by_panorama') or {},
                                             a.get('val_psnr_3000_by_panorama') or {}, k)
                               for k in sorted(set(a.get('val_psnr_3000_by_panorama') or {})
                                               | set(b.get('val_psnr_3000_by_panorama') or {}))}}
        row['selected'] = {'baseline_step': a.get('selected_step'), 'variant_step': b.get('selected_step'),
                           'psnr': difference(b, a, 'val_psnr_selected'), 'ssim': difference(b, a, 'val_ssim_selected'),
                           'by_panorama': {k: difference((b.get('panoramas_selected') or {}).get(k) or {},
                                                         (a.get('panoramas_selected') or {}).get(k) or {}, 'psnr')
                                           for k in sorted(set(a.get('panoramas_selected') or {})
                                                           | set(b.get('panoramas_selected') or {}))}}
        row['absolute'] = {role: {'psnr': arm.get('val_psnr_3000'), 'ssim': arm.get('val_ssim_3000'),
                                  'panoramas': arm.get('panoramas_3000'), 'selected': arm.get('selected_checkpoint'),
                                  'curve': arm.get('validation_curve')}
                           for role, arm in (('baseline', a), ('variant', b))}
        ca = region_coverage(prep, a['name'])
        cb = region_coverage(prep, b['name'])
        row['coverage'] = ca
        row['coverage_identical_between_arms'] = (
            None if ca is None or cb is None else
            {p: {k: v['faces_present'] for k, v in r.items()} for p, r in ca.items()}
            == {p: {k: v['faces_present'] for k, v in r.items()} for p, r in cb.items()})
        if ca and cb:
            row['regions_by_panorama'] = {
                pano: {region: (None if ca[pano][region]['absent'] or cb[pano][region]['absent']
                                else cb[pano][region]['mean_psnr'] - ca[pano][region]['mean_psnr'])
                       for region in ca[pano]} for pano in ca}
        ia, ib = a.get('inspection_3000'), b.get('inspection_3000')
        if ia and ib:
            row['regions'] = {k: difference(ib['validation']['regions'], ia['validation']['regions'], k)
                              for k in REGIONS}
            row['large_gaussians'] = ib['gaussians']['larger_than_prune_scale3d'] - ia['gaussians']['larger_than_prune_scale3d']
            row['outside'] = (ib['gaussians']['outside_sfm_points_box_plus_25pct']
                              - ia['gaussians']['outside_sfm_points_box_plus_25pct'])
            row['outside_opaque'] = (ib['gaussians']['outside_and_opaque_over_0_5']
                                     - ia['gaussians']['outside_and_opaque_over_0_5'])
        rows.append(row)

    def describe(values):
        values = [v for v in values if v is not None]
        if not values:
            return None
        return {'n': len(values), 'mean': statistics.fmean(values), 'median': statistics.median(values),
                'sd': statistics.stdev(values) if len(values) > 1 else None,
                'positive': sum(v > 0 for v in values), 'negative': sum(v < 0 for v in values)}
    summary = {'psnr': describe(r['psnr'] for r in rows), 'ssim': describe(r['ssim'] for r in rows),
               'large_gaussians': describe(r.get('large_gaussians') for r in rows),
               'outside': describe(r.get('outside') for r in rows),
               'outside_opaque': describe(r.get('outside_opaque') for r in rows),
               'regions': {k: describe((r.get('regions') or {}).get(k) for r in rows) for k in REGIONS},
               'by_panorama': {k: describe(r['by_panorama'].get(k) for r in rows)
                               for k in sorted({k for r in rows for k in r['by_panorama']})}}
    # Is a seed's gain driven by a weak baseline or by a strong variant? Distance to the other seeds.
    for row in rows:
        for role in ('baseline', 'variant'):
            others = [r['absolute'][role]['psnr'] for r in rows
                      if r is not row and r['absolute'][role]['psnr'] is not None]
            own = row['absolute'][role]['psnr']
            row['absolute'][role]['minus_median_of_other_seeds'] = (
                own - statistics.median(others) if own is not None and others else None)
    summary['validation_panoramas'] = sorted({p for r in rows for p in r['by_panorama']})
    summary['selected'] = {'psnr': describe(r['selected']['psnr'] for r in rows),
                           'ssim': describe(r['selected']['ssim'] for r in rows),
                           'by_panorama': {k: describe(r['selected']['by_panorama'].get(k) for r in rows)
                                           for k in summary['validation_panoramas']}}
    # Sign consistency of region differences, per panorama: a positive aggregate can hide reversals.
    summary['region_signs'] = {}
    for pano in summary['validation_panoramas']:
        for region in REGIONS:
            values = [(r['seed'], (r.get('regions_by_panorama') or {}).get(pano, {}).get(region)) for r in rows]
            known = [(seed, v) for seed, v in values if v is not None]
            if known:
                summary['region_signs'][f'{pano}/{region}'] = {
                    'positive_seeds': [s for s, v in known if v > 0], 'negative_seeds': [s for s, v in known if v < 0],
                    'values': {s: v for s, v in known}}
    if noise and summary['psnr']:
        gaps = [abs(r['psnr']) for r in rows if r['psnr'] is not None]
        summary['psnr_vs_noise'] = {'noise_db': noise['psnr_db'], 'source': noise.get('source'),
                                    'abs_mean_over_noise': abs(summary['psnr']['mean']) / noise['psnr_db'],
                                    'seeds_beyond_noise': sum(g > noise['psnr_db'] for g in gaps), 'seeds': len(gaps)}
    labels = experiment.get('labels', {'baseline': 's0', 'variant': 's1'})
    return {'experiment': experiment['name'], 'prep_run': Path(prep).name, 'pairs': rows, 'summary': summary,
            'labels': labels, 'difference': f"{labels['variant']} - {labels['baseline']} at step 3000, validation only", 'quality_thresholds': 'none', 'created_at': now()}


def paired_report(result):
    fmt = lambda v: '—' if v is None else (f'{v:+.3f}' if isinstance(v, float) else f'{v:+d}')
    plain = lambda v, d=3: '—' if v is None else f'{v:.{d}f}'
    labels = result.get('labels', {'baseline': 's0', 'variant': 's1'})
    base, variant = labels['baseline'], labels['variant']
    panoramas = result['summary'].get('validation_panoramas') or []
    scope = (f"Résultats limités aux {len(panoramas)} panoramas de validation " + ' et '.join(panoramas)
             if len(panoramas) > 1 else f"Résultats limités au panorama de validation {panoramas[0]}"
             if panoramas else 'Résultats limités aux panoramas de validation (non détaillés dans les journaux)')
    lines = [f"# {result['experiment']} — {result['prep_run']} : écarts appariés {variant} − {base} "
             f"à 3 000 (validation)", '', scope + '. Jeu de test non utilisé.', '',
             '| Graine | ΔPSNR | ΔSSIM | ' + ' | '.join(f'Δ{k}' for k in REGIONS)
             + ' | Δgrandes | Δhors boîte | Δhors boîte opaques |',
             '|---:|---:|---:|' + '---:|' * len(REGIONS) + '---:|---:|---:|']
    for r in result['pairs']:
        regions = r.get('regions') or {}
        lines.append(f"| {r['seed']} | {fmt(r['psnr'])} | {fmt(r['ssim'])} | "
                     + ' | '.join(fmt(regions.get(k)) for k in REGIONS)
                     + f" | {fmt(r.get('large_gaussians'))} | {fmt(r.get('outside'))} | {fmt(r.get('outside_opaque'))} |")
    lines += ['', '## Valeurs absolues par graine et par panorama (étape 3 000)', '',
              f'| Graine | Panorama | PSNR {base} | PSNR {variant} | SSIM {base} | SSIM {variant} | '
              f'Vues exploitables {base} / {variant} |', '|---:|---|---:|---:|---:|---:|---|']
    for r in result['pairs']:
        a, b = r['absolute']['baseline'], r['absolute']['variant']
        lines.append(f"| {r['seed']} | ensemble | {plain(a['psnr'])} | {plain(b['psnr'])} | {plain(a['ssim'], 4)} | "
                     f"{plain(b['ssim'], 4)} | — |")
        for pano in panoramas:
            pa, pb = (a['panoramas'] or {}).get(pano, {}), (b['panoramas'] or {}).get(pano, {})
            lines.append(f"| {r['seed']} | {pano} | {plain(pa.get('psnr'))} | {plain(pb.get('psnr'))} | "
                         f"{plain(pa.get('ssim'), 4)} | {plain(pb.get('ssim'), 4)} | "
                         f"{pa.get('usable', '—')}/{pa.get('faces', '—')} · {pb.get('usable', '—')}/{pb.get('faces', '—')} |")
    lines += ['', '## Référence faible ou variante forte ? (PSNR moins la médiane des autres graines)', '',
              f'| Graine | {base} | {variant} | Checkpoint choisi {base} / {variant} |', '|---:|---:|---:|---|']
    for r in result['pairs']:
        a, b = r['absolute']['baseline'], r['absolute']['variant']
        lines.append(f"| {r['seed']} | {fmt(a.get('minus_median_of_other_seeds'))} | "
                     f"{fmt(b.get('minus_median_of_other_seeds'))} | {a['selected']} / {b['selected']} |")
    lines += ['', 'Un gain dû à une référence faible apparaît comme une valeur négative dans la colonne de '
                  'référence ; un gain dû à la variante, comme une valeur positive dans sa colonne.', '',
              '## Courbes de validation (PSNR moyen)', '', '| Graine | Bras | ' + ' | '.join(
                  str(c['step']) for c in (result['pairs'][0]['absolute']['baseline']['curve'] or [])) + ' |',
              '|---:|---|' + '---:|' * len(result['pairs'][0]['absolute']['baseline']['curve'] or [])]
    for r in result['pairs']:
        for role, label in (('baseline', base), ('variant', variant)):
            lines.append(f"| {r['seed']} | {label} | "
                         + ' | '.join(plain(c['mean_psnr']) for c in (r['absolute'][role]['curve'] or [])) + ' |')
    coverage = next((r['coverage'] for r in result['pairs'] if r.get('coverage')), None)
    if coverage:
        lines += ['', '## Couverture des annotations régionales (faces où la région existe)', '',
                  '| Panorama | ' + ' | '.join(REGIONS) + ' |', '|---|' + '---|' * len(REGIONS)]
        for pano, regions in sorted(coverage.items()):
            cells = [('absente (0/{})'.format(regions[k]['faces']) if regions[k]['absent'] else
                      f"{regions[k]['faces_present']}/{regions[k]['faces']}") if k in regions else 'non calculée'
                     for k in REGIONS]
            lines.append(f'| {pano} | ' + ' | '.join(cells) + ' |')
        consistent = {r['seed']: r.get('coverage_identical_between_arms') for r in result['pairs']}
        lines += ['', f'Couverture identique entre bras (mêmes annotations) : {consistent}.',
                  'Une région absente n’a pas de score : elle est exclue, jamais comptée comme nulle.', '',
                  f'## Écarts {variant} − {base} par panorama et par région', '',
                  '| Graine | Panorama | ' + ' | '.join(REGIONS) + ' |', '|---:|---|' + '---:|' * len(REGIONS)]
        for r in result['pairs']:
            for pano, values in sorted((r.get('regions_by_panorama') or {}).items()):
                lines.append(f"| {r['seed']} | {pano} | " + ' | '.join(
                    'absente' if (r['coverage'] or {}).get(pano, {}).get(k, {}).get('absent') else fmt(values.get(k))
                    for k in REGIONS) + ' |')
    lines += ['', f'## Checkpoints sélectionnés par la validation ({variant} − {base})', '',
              'Comparaison complémentaire de celle à 3 000. Le checkpoint est choisi sur ces mêmes vues de '
              'validation : ce n’est pas une mesure indépendante de généralisation.', '',
              '| Graine | Étape choisie ' + base + ' / ' + variant + ' | ΔPSNR | ΔSSIM | '
              + ' | '.join(f'ΔPSNR {p}' for p in panoramas) + ' |',
              '|---:|---|---:|---:|' + '---:|' * len(panoramas)]
    for r in result['pairs']:
        sel = r['selected']
        lines.append(f"| {r['seed']} | {sel['baseline_step']} / {sel['variant_step']} | {fmt(sel['psnr'])} | "
                     f"{fmt(sel['ssim'])} | " + ' | '.join(fmt(sel['by_panorama'].get(p)) for p in panoramas) + ' |')
    chosen = result['summary']['selected']
    if chosen['psnr']:
        lines.append(f"\nMoyenne ΔPSNR aux checkpoints choisis : {chosen['psnr']['mean']:+.3f} dB "
                     f"(médiane {chosen['psnr']['median']:+.3f}, > 0 : {chosen['psnr']['positive']}/{chosen['psnr']['n']}).")
        for pano, d in chosen.get('by_panorama', {}).items():
            if d:
                lines.append(f"- {pano} : moyenne {d['mean']:+.3f}, médiane {d['median']:+.3f}, "
                             f"> 0 : {d['positive']}/{d['n']}, < 0 : {d['negative']}/{d['n']}")
    signs = result['summary'].get('region_signs') or {}
    if signs:
        lines += ['', '## Cohérence des signes par panorama et par région (étape 3 000)', '',
                  'Un écart agrégé positif peut masquer des reculs sur un panorama ou une région.', '',
                  '| Panorama / région | Graines en hausse | Graines en baisse | Écarts par graine |', '|---|---|---|---|']
        for key, value in signs.items():
            detail = ', '.join(f'{seed}: {v:+.2f}' for seed, v in sorted(value['values'].items()))
            lines.append(f"| {key} | {value['positive_seeds'] or '—'} | {value['negative_seeds'] or '—'} | {detail} |")
    lines += ['', '## Synthèse sur les graines', '',
              '| Quantité | n | moyenne | médiane | écart-type | > 0 | < 0 |', '|---|---:|---:|---:|---:|---:|---:|']
    items = [('PSNR', result['summary']['psnr']), ('SSIM', result['summary']['ssim']),
             ('grandes gaussiennes', result['summary']['large_gaussians']),
             ('hors boîte', result['summary']['outside']),
             ('hors boîte opaques', result['summary']['outside_opaque'])]
    items += [(f'région {k}', v) for k, v in result['summary']['regions'].items()]
    items += [(f'panorama {k}', v) for k, v in result['summary'].get('by_panorama', {}).items()]
    for label, d in items:
        if d:
            sd = '—' if d['sd'] is None else f"{d['sd']:.3f}"
            lines.append(f"| {label} | {d['n']} | {d['mean']:+.3f} | {d['median']:+.3f} | {sd} | "
                         f"{d['positive']} | {d['negative']} |")
    lines += ['', 'L’écart-type entre graines mesure la variabilité observée (graine, ordre des caméras et '
                  'non-déterminisme GPU confondus) ; il n’isole pas à lui seul le non-déterminisme GPU.']
    noise = result['summary'].get('psnr_vs_noise')
    if noise:
        lines += ['', f"Repère historique : {noise['noise_db']:.3f} dB, écart entre deux exécutions identiques mesuré "
                      f"sur une autre partition ({noise['source']}). Ce n’est pas un seuil statistique ; le rapport "
                      f"|moyenne ΔPSNR| / repère ({noise['abs_mean_over_noise']:.2f}) et le nombre de graines au-delà "
                      f"({noise['seeds_beyond_noise']}/{noise['seeds']}) ne prouvent pas la robustesse du gain."]
    lines += ['', 'Lecture : un écart moyen petit devant son écart-type, ou de signe instable, ne départage pas '
                  'les bras. Aucun gagnant sur le PSNR moyen seul ; lire les régions et les planches. ' + scope + '.', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Validation-only comparison of controlled gsplat arms')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--prep', required=True)
    parser.add_argument('--reference', help='<prep-run>:<training name> used as noise reference')
    parser.add_argument('--noise-from', help='comparison.json of the controlled experiment (replicate gap)')
    args = parser.parse_args(argv)
    try:
        experiment = read(args.experiment)
        reference = tuple(args.reference.rsplit(':', 1)) if args.reference else None
        target = Path(args.prep) / 'comparisons' / experiment['name']
        if 'pairs' in experiment:
            noise = None
            if args.noise_from:
                gap = read(args.noise_from)['noise']['replicate_minus_reference_psnr_3000']
                noise = {'psnr_db': abs(gap), 'source': args.noise_from}
            result = paired(experiment, Path(args.prep), noise)
            text = paired_report(result)
        else:
            result = compare(experiment, Path(args.prep), reference)
            text = report(result)
        write(target / 'comparison.json', result)
        (target / 'comparison.md').write_text(text)
    except Exception as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1
    print(target / 'comparison.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
