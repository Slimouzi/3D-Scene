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


def arm_summary(prep, name):
    folder = Path(prep) / 'training' / name
    if not (folder / 'training.json').exists():
        return {'name': name, 'status': 'not run'}
    training = read(folder / 'training.json')
    validation = {row['step']: row for row in jsonl(folder / 'validation.jsonl')}
    last = max(validation) if validation else None
    log = jsonl(folder / 'train.jsonl')
    inspections = sorted((Path(prep) / 'inspection').glob(f'{name}-*-v2/inspection.json'))
    inspection = read(inspections[-1])['results'] if inspections else {}
    pick = lambda step, key: validation[step][key] if step in validation else None
    selection = read(folder / 'selection.json') if (folder / 'selection.json').exists() else {}
    return {'name': name, 'status': training['status'], 'steps': training['config']['steps'],
            'factors': training['config'].get('schedule', {}).get('factors'),
            'val_psnr_3000': pick(3000, 'mean_psnr'), 'val_ssim_3000': pick(3000, 'mean_ssim'),
            'last_step': last, 'val_psnr_last': pick(last, 'mean_psnr'), 'val_ssim_last': pick(last, 'mean_ssim'),
            'usable_validation_faces': pick(last, 'usable_cameras'),
            'selected_step': selection.get('step'), 'test_used': selection.get('test_used'),
            'gaussians_last': log[-1]['gaussians'] if log else None,
            'train_faces_full_psnr': inspection.get('train/full', {}).get('mean_psnr'),
            'validation_full_psnr_inspection': inspection.get('validation/full', {}).get('mean_psnr')}


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
    noise = None
    if reference:
        ref_prep, ref_name = reference
        ref = arm_summary(ref_prep, ref_name)
        replicate = by_factor.get((durations[0], prunings[0]))
        noise = {'reference': f'{Path(ref_prep).name}/{ref_name}', 'reference_val_psnr_3000': ref.get('val_psnr_3000'),
                 'replicate_minus_reference_psnr_3000': difference(replicate, ref, 'val_psnr_3000')}
    return {'experiment': experiment['name'], 'prep_run': Path(prep).name, 'arms': rows, 'effects': effects,
            'noise': noise, 'set': 'validation only', 'quality_thresholds': 'none', 'created_at': now()}


def report(result):
    fmt = lambda v: '—' if v is None else (f'{v:+.3f}' if isinstance(v, float) and abs(v) < 5 else
                                           f'{v:.3f}' if isinstance(v, float) else str(v))
    lines = [f"# {result['experiment']} — {result['prep_run']} (validation uniquement)", '',
             '| Bras | Statut | PSNR@3000 | SSIM@3000 | PSNR fin | SSIM fin | Étape choisie | Gaussiennes | PSNR train inspecté |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    plain = lambda v: '—' if v is None else (f'{v:.3f}' if isinstance(v, float) else str(v))
    for name, r in result['arms'].items():
        lines.append(f"| {name} | {r['status']} | {plain(r.get('val_psnr_3000'))} | {plain(r.get('val_ssim_3000'))} | "
                     f"{plain(r.get('val_psnr_last'))} | {plain(r.get('val_ssim_last'))} | {plain(r.get('selected_step'))} | "
                     f"{plain(r.get('gaussians_last'))} | {plain(r.get('train_faces_full_psnr'))} |")
    lines += ['', '## Effets (écarts de PSNR moyen en validation, dB)', '']
    for effect, values in result['effects'].items():
        lines.append(f'- {effect} : ' + ', '.join(f'{k} {fmt(v)}' for k, v in values.items()))
    if result['noise']:
        n = result['noise']
        lines += ['', f"Bruit : réplique − référence ({n['reference']}) à l’étape 3000 = "
                      f"{fmt(n['replicate_minus_reference_psnr_3000'])} dB. Un effet plus petit que cet écart "
                      'n’est pas distinguable d’une variation entre exécutions.']
    lines += ['', 'Aucun seuil de qualité ; jeu de test non utilisé.', '']
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
