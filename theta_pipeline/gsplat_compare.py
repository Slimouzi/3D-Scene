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
    memory = [row['max_memory_gb'] for row in log if row.get('max_memory_gb') is not None]
    return {'name': name, 'status': training['status'], 'steps': training['config']['steps'],
            'factors': training['config'].get('schedule', {}).get('factors'),
            'val_psnr_3000': pick(3000, 'mean_psnr'), 'val_ssim_3000': pick(3000, 'mean_ssim'),
            'last_step': last, 'val_psnr_last': pick(last, 'mean_psnr'), 'val_ssim_last': pick(last, 'mean_ssim'),
            'usable_validation_faces': pick(last, 'usable_cameras'),
            'selected_step': selection.get('step'), 'test_used': selection.get('test_used'),
            'gaussians_3000': logged.get(3000, {}).get('gaussians'),
            'gaussians_last': log[-1]['gaussians'] if log else None,
            'max_memory_gb': max(memory) if memory else None,
            'seconds': log[-1]['seconds'] if log else None,
            'runs': len(training.get('history', [])),
            'loss_curve': {row['step']: row['loss'] for row in log},
            'inspection_3000': inspection_at(prep, name, 'step_003000.pt'),
            'inspection_last': inspection_at(prep, name, f'step_{last:06d}.pt') if last else None}


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
                      f"{signed(n['replicate_minus_reference_ssim_3000'])}). Un écart de PSNR de cet ordre "
                      'n’est pas distinguable d’une variation entre exécutions.']
    lines += ['', 'Aucun gagnant n’est retenu sur le PSNR moyen seul : lire les régions (contours, vitrages, '
                  'mobilier) et les planches à 3 000 et en fin. Aucun seuil de qualité ; jeu de test non utilisé.', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Validation-only comparison of controlled gsplat arms')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--prep', required=True)
    parser.add_argument('--reference', help='<prep-run>:<training name> used as noise reference')
    args = parser.parse_args(argv)
    try:
        experiment = read(args.experiment)
        reference = tuple(args.reference.rsplit(':', 1)) if args.reference else None
        result = compare(experiment, Path(args.prep), reference)
        target = Path(args.prep) / 'comparisons' / experiment['name']
        write(target / 'comparison.json', result)
        (target / 'comparison.md').write_text(report(result))
    except Exception as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1
    print(target / 'comparison.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
