"""Segmentation faces: the 12 SfM faces reused, plus 45° offsets and polar faces."""
import numpy as np
from PIL import Image
from ..geometry import project, rays
from ..storage import digest, read, write
from . import POLICY, PROMPTS, TRIAL_PANORAMA


def yaw(degrees):
    t = np.deg2rad(degrees)
    return np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])


def pitch(degrees):
    t = np.deg2rad(degrees)
    return np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])


def segmentation_faces(sfm_rotations):
    """R_face_from_pano for each face; every sphere direction is seen by >=2 faces."""
    faces = [{'face_id': f'sfm{i:02d}', 'role': 'sfm', 'sfm_camera': i, 'rotation': r}
             for i, r in enumerate(sfm_rotations)]
    faces += [{'face_id': f'mid{i:02d}', 'role': 'overlap', 'rotation': r @ yaw(45).T}
              for i, r in enumerate(sfm_rotations)]
    faces += [{'face_id': f'pole{i:02d}', 'role': 'zenith_nadir', 'rotation': pitch(p) @ yaw(y)}
              for i, (p, y) in enumerate((p, y) for p in (-65, 65) for y in (0, 90, 180, 270))]
    return faces


def inside(directions, rotation, border_fraction=0.):
    f = [directions @ rotation[k] for k in range(3)]
    limit = (1 - border_fraction) * f[2]
    return (f[2] > 0) & (np.abs(f[0]) <= limit) & (np.abs(f[1]) <= limit)


def coverage(rotations, size):
    """Overlap and multiplicity measured on the face core (border band excluded)."""
    border = 2 * POLICY['face_border_px'] / size
    sphere = np.random.default_rng(0).normal(size=(200_000, 3))
    sphere /= np.linalg.norm(sphere, axis=1, keepdims=True)
    counts = sum(inside(sphere, r, border).astype(int) for r in rotations)
    overlaps = []
    for i, r in enumerate(rotations):
        d = rays(64).reshape(-1, 3) @ r
        seen = np.zeros(len(d), bool)
        for j, other in enumerate(rotations):
            if j != i:
                seen |= inside(d, other, border)
        overlaps.append(float(seen.mean()))
    return {'min_face_overlap_fraction': min(overlaps),
            'min_projections_per_direction': int(counts.min()),
            'two_view_sphere_fraction': float(np.mean(counts >= 2)),
            'includes_zenith_nadir': True}


def prepare_faces(run, sfm_rotations):
    run.require('prepare')
    size = run.config['erp_width'] // 4
    faces = segmentation_faces(sfm_rotations)
    stats = coverage([f['rotation'] for f in faces], size)
    if stats['min_face_overlap_fraction'] < POLICY['min_face_overlap_fraction']:
        raise RuntimeError(f'Face overlap below policy: {stats}')
    if stats['min_projections_per_direction'] < POLICY['min_consistent_projections']:
        raise RuntimeError(f'Some directions are seen by fewer than two faces: {stats}')
    stems = [p.stem for p in run.sources]
    capture = {r['panorama_id']: r for r in read(run.path / 'capture.json')['images']}
    outputs, panoramas = [], {}
    for pano in stems:
        with Image.open(run.path / 'prepare/erp' / f'{pano}.png') as raw:
            erp = raw.convert('RGB')
        entry = {}
        for face in faces:
            if face['role'] == 'sfm':
                # Reuse the exact SfM face, so masks and features share pixels.
                path = run.path / 'prepare/images' / f"pano_camera{face['sfm_camera']}/{pano}.png"
                if not path.is_file():
                    raise RuntimeError(f'Missing SfM face {path}')
            else:
                path = run.path / 'segmentation/faces' / face['face_id'] / f'{pano}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(project(erp, face['rotation'], size)).save(path)
                outputs.append(path)
            entry[face['face_id']] = {'image': str(path.relative_to(run.path)), 'sha256': digest(path)}
        panoramas[pano] = {'source_sha256': capture[pano]['sha256'], 'faces': entry}
    manifest = run.path / 'segmentation/faces.json'
    write(manifest, {'schema_version': 1,
                     'trial_panorama': TRIAL_PANORAMA if TRIAL_PANORAMA in stems else stems[0],
                     'face_size': size, 'hfov_deg': 90., 'prompts': list(PROMPTS),
                     'convention': 'rotation is R_face_from_panorama; x-right y-down z-forward',
                     'coverage': stats,
                     'faces': [{**f, 'rotation': np.asarray(f['rotation']).tolist()} for f in faces],
                     'panoramas': panoramas})
    return outputs + [manifest]
