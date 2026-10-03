"""GPU runner: segment the registered perspective faces of a run, then fuse them on CPU.

    python -m theta_pipeline.segmentation --run Output/runs/<run-id> --trial R0010004
    python -m theta_pipeline.segmentation --run Output/runs/<run-id> --all   # only after trial PASS

Exit codes: 0 PASS/ACCEPTED, 2 UNKNOWN, 3 FAIL/REJECTED, 1 refused or integrity error.
Segmentation writes only `segmentation/raw/<panorama>/`, under the run lock. Fusion runs
the CPU stage (seg-trial or auto-mask) with the CPU interpreter, which owns the decision.
"""
import argparse
import fcntl
import json
import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import numpy as np
from PIL import Image
from ..storage import digest, now, read, write
from . import MODEL_ID, POLICY, PROMPTS, slug
from .environment import qualify
from .provenance import PACKAGE, code_digests, git_commit
from .sam3_segmenter import ModelUnavailable, SAM3Segmenter

EXIT = {'PASS': 0, 'ACCEPTED': 0, 'UNKNOWN': 2, 'FAIL': 3, 'REJECTED': 3}


def verified(run_dir, state, stage, rel):
    """Read an artifact only if the CPU stage completed and its hash is unchanged."""
    entry = state['stages'].get(stage, {})
    if entry.get('status') != 'completed':
        raise RuntimeError(f'Run CPU stage {stage} first')
    expected = entry['artifacts'].get(rel)
    if expected is None or not (run_dir / rel).is_file() or digest(run_dir / rel) != expected:
        raise RuntimeError(f'{rel} changed or missing; use a new run-id')
    return read(run_dir / rel)


def select(run_dir, state, faces, trial=None):
    """--trial must name the registered trial panorama; --all requires a trial PASS."""
    if trial is not None:
        if trial != faces['trial_panorama']:
            raise RuntimeError(f"--trial {trial} is not the run trial panorama {faces['trial_panorama']}")
        return [trial]
    gate = verified(run_dir, state, 'seg_trial', 'segmentation/trial/trial_gate.json')
    if gate['decision'] != 'PASS':
        raise RuntimeError(f"Trial decision is {gate['decision']}: --all refused")
    return list(faces['panoramas'])


def segment_panorama(run_dir, faces, pano, segmenter, provenance):
    out = run_dir / 'segmentation/raw' / pano
    manifest = out / 'masks.json'
    if manifest.exists() and read(manifest)['status'] == 'OK':
        print(f'{pano}: already segmented', flush=True)
        return 'OK'
    entry = faces['panoramas'][pano]
    face_records, masks = [], []
    for face in faces['faces']:
        face_id = face['face_id']
        image_rel = entry['faces'][face_id]['image']
        try:
            if digest(run_dir / image_rel) != entry['faces'][face_id]['sha256']:
                raise RuntimeError(f'{image_rel} changed since seg_faces')
            with Image.open(run_dir / image_rel) as raw:
                image = raw.convert('RGB')
            result = segmenter.segment(image, PROMPTS)
        except Exception as error:
            # A failed face contributes no pixels: fusion never substitutes a default mask.
            face_records.append({'face_id': face_id, 'status': 'failed',
                                 'error': f'{type(error).__name__}: {error}'})
            continue
        count = 0
        for prompt, instances in result.items():
            for k, (mask, score) in enumerate(instances):
                rel = f'segmentation/raw/{pano}/{face_id}/{slug(prompt)}_{k:02d}.png'
                (run_dir / rel).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.asarray(mask, bool)).save(run_dir / rel, optimize=True)
                masks.append({'image_id': pano, 'face_id': face_id, 'prompt': prompt,
                              'mask_path': rel, 'score': score,
                              'area_fraction': float(np.mean(mask)), 'model': MODEL_ID,
                              'checkpoint_sha256': provenance['checkpoint_sha256'],
                              'git_commit': provenance['git_commit'],
                              'mask_sha256': digest(run_dir / rel)})
                count += 1
        face_records.append({'face_id': face_id, 'status': 'ok', 'masks': count})
        print(f'{pano} {face_id}: {count} masks', flush=True)
    failed = sum(f['status'] != 'ok' for f in face_records)
    status = 'OK' if not failed else 'FAILED' if failed == len(face_records) else 'PARTIAL'
    write(manifest, {'schema_version': 1, 'image_id': pano, 'status': status,
                     'source_sha256': entry['source_sha256'], 'faces': face_records,
                     'masks': masks, 'provenance': {**provenance, 'finished_at': now()}})
    return status


def write_unknown(run_dir, panoramas, reason, provenance):
    for pano in panoramas:
        manifest = run_dir / 'segmentation/raw' / pano / 'masks.json'
        if manifest.exists() and read(manifest)['status'] == 'OK':
            continue
        write(manifest, {'schema_version': 1, 'image_id': pano, 'status': 'UNKNOWN',
                         'reason': reason, 'faces': [], 'masks': [], 'provenance': provenance})


def segment(run_dir, args):
    """Returns the exit code when segmentation alone decides (UNKNOWN), else None."""
    with (run_dir / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another process is using this run') from None
        state = read(run_dir / 'run.json')
        if state['provenance']['code'] != code_digests():
            raise RuntimeError('This run was created by different code: clone the commit that '
                               'created it, or create the run again at this commit')
        faces = verified(run_dir, state, 'seg_faces', 'segmentation/faces.json')
        panoramas = select(run_dir, state, faces, args.trial)
        commit = git_commit()
        environment = qualify()
        problems = list(environment['problems'])
        if commit is None:
            problems.insert(0, 'not a git checkout')
        elif commit.endswith('-dirty'):
            problems.insert(0, f'uncommitted changes ({commit}): GPU runs need a fixed commit')
        try:
            sam3_version = version('sam3')
        except PackageNotFoundError:
            sam3_version = None
        provenance = {'model': MODEL_ID, 'sam3_version': sam3_version, 'git_commit': commit,
                      'code_sha256': code_digests(), 'environment': environment,
                      'faces_sha256': digest(run_dir / 'segmentation/faces.json'),
                      'prompts': list(PROMPTS), 'policy': POLICY, 'started_at': now(),
                      'checkpoint_sha256': None}
        if problems:
            write_unknown(run_dir, panoramas, '; '.join(problems), provenance)
            print('UNKNOWN: ' + '; '.join(problems), file=sys.stderr)
            return EXIT['UNKNOWN']
        try:
            segmenter = SAM3Segmenter(args.checkpoint)
        except ModelUnavailable as error:
            write_unknown(run_dir, panoramas, str(error), provenance)
            print(f'UNKNOWN: {error}', file=sys.stderr)
            return EXIT['UNKNOWN']
        provenance.update(checkpoint_sha256=segmenter.checkpoint_sha256,
                          checkpoint_file=segmenter.checkpoint_path.name,
                          environment={**environment, 'runtime': segmenter.environment})
        statuses = {p: segment_panorama(run_dir, faces, p, segmenter, provenance) for p in panoramas}
        print(json.dumps(statuses), flush=True)
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description='SAM 3 segmentation of perspective faces (GPU)')
    parser.add_argument('--run', required=True, help='Run directory, e.g. Output/runs/<run-id>')
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument('--trial', metavar='PANORAMA', help='Segment and gate the trial panorama only')
    scope.add_argument('--all', action='store_true', help='All panoramas; refused unless the trial is PASS')
    parser.add_argument('--config', default='configs/salon.json', help='Config that created the run')
    parser.add_argument('--checkpoint', help='Local sam3.pt (default: Hugging Face cache)')
    parser.add_argument('--cpu-python', default=os.environ.get('THETA_CPU_PYTHON', '.venv-sfm/bin/python'),
                        help='Interpreter of the CPU environment (sfm.lock.txt) used for fusion')
    args = parser.parse_args(argv)
    run_dir = Path(args.run).resolve()
    config = Path(args.config).resolve()
    try:
        if (config.parent / read(config)['output'] / run_dir.name).resolve() != run_dir:
            raise RuntimeError(f'{run_dir} is not a run of {config}')
        code = segment(run_dir, args)
        if code is not None:
            return code
        stage = 'seg-trial' if args.trial else 'auto-mask'
        cpu = subprocess.run([args.cpu_python, '-m', 'theta_pipeline', stage, '--config', str(config),
                              '--run-id', run_dir.name], cwd=PACKAGE.parent)
        # UNKNOWN stages fail on purpose (retryable) but still write their decision.
        result = run_dir / ('segmentation/trial/trial_gate.json' if args.trial else 'semantic_masks.json')
        decision = read(result)['decision' if args.trial else 'status'] if result.is_file() else None
        if decision is None:
            raise RuntimeError(f'CPU stage {stage} failed (exit {cpu.returncode})')
        print(f'{stage}: {decision}', flush=True)
        return EXIT[decision]
    except Exception as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
