"""Check that the cameras given to gsplat reproject SfM points onto their observations (CPU).

    python -m theta_pipeline.gsplat_camera_check --prep Output/runs/<prep-run>

Read-only. For train and validation cameras (the test set is not opened): image files
match their width/height and hash, intrinsics equal the SfM camera, world_to_camera is a
rigid transform equal to the SfM pose, init points equal the SfM positions, and every
2-D observation is compared with the projection K (R X + t). COLMAP and gsplat both
place pixel centers at +0.5, so no offset is applied. Overlays draw observations (red)
and projections (green) on a few faces. Errors are reported, not judged.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .gsplat_preflight import load_verified, verify_prep
from .storage import now, read, write

SETS = ('train', 'validation')


def project(K, world_to_camera, xyz):
    """Pixel coordinates and camera depth of world points."""
    T = np.asarray(world_to_camera, float)
    cam = np.asarray(xyz, float) @ T[:3, :3].T + T[:3, 3]
    uvw = cam @ np.asarray(K, float).T
    return uvw[:, :2] / uvw[:, 2:3], cam[:, 2]


def rigid(world_to_camera):
    R = np.asarray(world_to_camera, float)[:3, :3]
    return float(np.abs(R @ R.T - np.eye(3)).max()), float(np.linalg.det(R))


def residuals(camera, observations, points):
    """Reprojection errors (pixels) of one camera's observations, and points behind it."""
    if not observations:
        return np.zeros(0), 0
    ids = [pid for pid, _ in observations]
    xy = np.asarray([xy for _, xy in observations], float)
    uv, depth = project(camera['K'], camera['world_to_camera'], [points[pid]['xyz'] for pid in ids])
    front = depth > 0
    return np.linalg.norm(uv[front] - xy[front], axis=1), int((~front).sum())


def stats(values):
    if not len(values):
        return {'count': 0, 'median': None, 'p95': None, 'max': None}
    return {'count': int(len(values)), 'median': float(np.median(values)),
            'p95': float(np.quantile(values, .95)), 'max': float(values.max())}


def overlay(image, observations, points, camera):
    canvas = Image.fromarray(image).convert('RGB')
    draw = ImageDraw.Draw(canvas)
    uv, depth = project(camera['K'], camera['world_to_camera'], [points[p]['xyz'] for p, _ in observations])
    for (_, (x, y)), (u, v), z in zip(observations, uv, depth):
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=(255, 40, 40), width=2)
        if z > 0:
            draw.line((u - 5, v, u + 5, v), fill=(40, 255, 40), width=2)
            draw.line((u, v - 5, u, v + 5), fill=(40, 255, 40), width=2)
    return canvas


def check(prep, overlays=3):
    from .gsplat_prep import load_model, verified_source
    prep = Path(prep).resolve()
    output = prep.parent
    problems = verify_prep(prep)
    if problems:
        raise RuntimeError('; '.join(problems))
    manifest = read(prep / 'gsplat_inputs/gsplat_inputs.json')
    split_run = output / manifest['split_run']
    verified_source(split_run, ('import_sfm',))
    component = read(split_run / 'sfm_import/sfm_import.json')['component']
    model = load_model(split_run / f'sfm_import/sfm/sparse/{component}')
    points = model['points']
    by_image = {}
    for pid, point in points.items():
        for name, xy in point['track']:
            by_image.setdefault(name, []).append((pid, xy))
    rel = next(r for r in manifest['files'] if r.endswith('points.npz'))
    init = np.load(load_verified(output, {'path': rel, 'sha256': manifest['files'][rel]}))
    init_error = float(max((np.abs(np.asarray(points[int(i)]['xyz']) - x).max()
                            for i, x in zip(init['ids'], init['xyz'])), default=0.))
    target = prep / 'inspection' / 'camera_check'
    target.mkdir(parents=True, exist_ok=True)
    report = {'schema_version': 1, 'prep_run': prep.name, 'test_loaded': False,
              'convention': 'pixel centers at +0.5 in COLMAP and gsplat; no offset applied',
              'init_points_max_position_difference': init_error, 'sets': {}, 'created_at': now()}
    for group in SETS:
        cameras = read(prep / f'gsplat_inputs/cameras_{group}.json')['cameras']
        errors, issues, per_camera, behind = [], [], [], 0
        ranked = sorted(cameras, key=lambda c: (-len(by_image.get(c['name'], [])), c['name']))
        drawn = {c['name'] for c in ranked[:overlays]}
        for camera in cameras:
            name = camera['name']
            sfm = model['images'].get(name)
            with Image.open(load_verified(output, camera['image'])) as raw:
                image = np.asarray(raw.convert('RGB'))
            if image.shape[:2] != (camera['height'], camera['width']):
                issues.append(f"{name}: image {image.shape[1]}x{image.shape[0]} != camera "
                              f"{camera['width']}x{camera['height']}")
            if sfm is None:
                issues.append(f'{name}: absent from the SfM model')
                continue
            if np.abs(np.asarray(camera['K']) - sfm['K']).max() > 1e-9:
                issues.append(f'{name}: intrinsics differ from SfM')
            if np.abs(np.asarray(camera['world_to_camera']) - sfm['world_to_camera']).max() > 1e-9:
                issues.append(f'{name}: pose differs from SfM')
            orthogonality, determinant = rigid(camera['world_to_camera'])
            if orthogonality > 1e-6 or abs(determinant - 1) > 1e-6:
                issues.append(f'{name}: non-rigid world_to_camera')
            values, back = residuals(camera, by_image.get(name, []), points)
            errors.append(values)
            behind += back
            per_camera.append({'camera': name, **stats(values), 'behind_camera': back})
            if name in drawn:
                overlay(image, by_image.get(name, []), points, camera).save(
                    target / f"{group}__{name.replace('/', '__')}")
        report['sets'][group] = {'cameras': len(cameras), 'reprojection_px': stats(np.concatenate(errors)),
                                 'observations_behind_camera': behind, 'issues': issues,
                                 'per_camera': per_camera, 'overlays': sorted(drawn)}
    write(target / 'camera_check.json', report)
    lines = ['# Contrôle des caméras fournies à gsplat', '',
             f"Positions des points d’initialisation : écart maximal {init_error:.3g} avec le SfM.",
             'Convention : centres de pixels à +0,5 (COLMAP et gsplat), aucun décalage appliqué.', '',
             '| Ensemble | Caméras | Observations | Médiane (px) | p95 (px) | Max (px) | Derrière | Anomalies |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for group, r in report['sets'].items():
        e = r['reprojection_px']
        fmt = lambda v: '—' if v is None else f'{v:.3f}'
        lines.append(f"| {group} | {r['cameras']} | {e['count']} | {fmt(e['median'])} | {fmt(e['p95'])} | "
                     f"{fmt(e['max'])} | {r['observations_behind_camera']} | {len(r['issues'])} |")
    lines += ['', 'Superpositions : observations en rouge, projections en vert.', '']
    for group, r in report['sets'].items():
        lines += [f'- {group} : ' + ', '.join(f"`{group}__{n.replace('/', '__')}`" for n in r['overlays'])]
        lines += [f'  - {issue}' for issue in r['issues'][:20]]
    (target / 'camera_check.md').write_text('\n'.join(lines) + '\n')
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description='Reprojection check of the cameras given to gsplat')
    parser.add_argument('--prep', required=True)
    parser.add_argument('--overlays', type=int, default=3)
    args = parser.parse_args(argv)
    try:
        target = check(args.prep, args.overlays)
    except Exception as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1
    print(target / 'camera_check.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
