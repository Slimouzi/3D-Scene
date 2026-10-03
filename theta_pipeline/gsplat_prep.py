"""CPU preparation of an evaluated gsplat experiment from a frozen AUTO-05 split.

The GPU trainer needs only torch and gsplat: everything requiring pycolmap or OpenCV is
done here, verified and hashed. Nothing from validation/test enters training inputs:

- cameras are written in three files (train, validation, test); the trainer's loss
  reads only cameras_train.json;
- point colors are recomputed by sampling train images at train observations only;
  SfM colors (which may mix held-out observations) are discarded;
- face weight maps come from the semantic run's appearance masks, projected per face
  with the conservative four-pixel rule (an excluded source pixel zeroes the weight).

Limitation, declared in every manifest: camera poses and point positions come from the
joint masked SfM of all 13 panoramas, so held-out stations took part in their estimation.
"""
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from . import split as auto05
from .geometry import project
from .segmentation.provenance import git_commit
from .storage import digest, now, read, write

STAGES = ('import_split', 'gsplat_prepare')
LIMITATION = ('Camera poses and initial point positions come from the joint masked SfM of all '
              '13 panoramas: validation and test stations took part in their estimation.')


def verified_source(source, stages):
    """Every artifact of the given completed stages, hash-checked against run.json."""
    state = read(source / 'run.json')
    for stage in stages:
        entry = state['stages'].get(stage, {})
        if entry.get('status') != 'completed':
            raise RuntimeError(f'{source.name}: stage {stage} not completed')
        for rel, sha in entry['artifacts'].items():
            if not (source / rel).is_file() or digest(source / rel) != sha:
                raise RuntimeError(f'{source.name}/{rel} changed or missing')
    return state


def import_split(run):
    """Import the frozen split after full re-validation (structure, separation, files)."""
    run.require('audit')
    output = run.path.parent
    source = output / run.config['split_run']
    state = verified_source(source, ('import_sfm', 'auto_split', 'split_gates'))
    record, gates = read(source / 'split.json'), read(source / 'gate_results.json')
    problems = auto05.verify_split_source(record, gates, run.config)
    if problems:
        raise RuntimeError('; '.join(problems))
    views = read(source / 'sfm_import/views.json')['views']
    component = read(source / 'sfm_import/sfm_import.json')['component']
    tracks = auto05.tracks_from_model(source / f'sfm_import/sfm/sparse/{component}')
    inputs = read(source / 'train_inputs.json')
    sets = {name: record[name] for name in auto05.SETS}
    checks = (auto05.validate(record, [p.stem for p in run.sources],
                              digest(source / 'sfm_import/poses.json'), views)
              + auto05.separation_checks(sets, inputs, views, tracks,
                                         run.config['sfm_run'], run.config['semantic_run'])
              + [auto05.verify_train_files(inputs, output)])
    failed = [c for c in checks if c['result'] != 'PASS']
    if failed:
        raise RuntimeError(f'Split re-validation failed: {failed}')
    target = run.path / 'split_import'
    target.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name in ('split.json', 'train_inputs.json', 'gate_results.json'):
        shutil.copyfile(source / name, target / name)
        outputs.append(target / name)
    manifest = target / 'split_import.json'
    write(manifest, {'schema_version': 1, 'split_run': run.config['split_run'],
                     'split_run_fingerprint': state['fingerprint'],
                     'partition_sha256': record['partition_sha256'], 'checks': checks,
                     'imported_at': now()})
    return outputs + [manifest]


def load_model(path):
    """Plain-data view of a COLMAP model: posed images and point tracks by image name."""
    import pycolmap as pc
    rec = pc.Reconstruction(path)
    images = {}
    for image in rec.images.values():
        if not image.has_pose:
            continue
        camera = rec.cameras[image.camera_id]
        pose = np.vstack((image.cam_from_world().matrix(), [0, 0, 0, 1]))
        images[image.name] = {'width': camera.width, 'height': camera.height,
                              'K': camera.calibration_matrix().tolist(),
                              'world_to_camera': pose.tolist()}
    points = {}
    for pid, point in rec.points3D.items():
        track = [(rec.images[e.image_id].name,
                  tuple(float(v) for v in rec.images[e.image_id].points2D[e.point2D_idx].xy))
                 for e in point.track.elements]
        points[int(pid)] = {'xyz': [float(v) for v in point.xyz], 'track': track}
    return {'images': images, 'points': points}


def bilinear_rgb(image, xy):
    """RGB at a COLMAP 2-D observation (pixel centers at +0.5), clamped to the image."""
    h, w = image.shape[:2]
    x = min(max(xy[0] - .5, 0.), w - 1.)
    y = min(max(xy[1] - .5, 0.), h - 1.)
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    top = (1 - fx) * image[y0, x0] + fx * image[y0, x1]
    bottom = (1 - fx) * image[y1, x0] + fx * image[y1, x1]
    return (1 - fy) * top + fy * bottom


def train_colors(points, eligible, train_images, sample):
    """Mean color of each eligible point over its train observations only.

    `sample(image_name, xy)` is only ever called with a train image name.
    """
    ids, xyz, rgb, used, ignored = [], [], [], 0, 0
    for pid in eligible:
        track = points[pid]['track']
        observations = [(name, xy) for name, xy in track if name in train_images]
        ignored += len(track) - len(observations)
        if not observations:
            raise RuntimeError(f'Point {pid} is eligible but has no train observation')
        ids.append(pid)
        xyz.append(points[pid]['xyz'])
        rgb.append(np.mean([sample(name, xy) for name, xy in observations], axis=0))
        used += len(observations)
    return {'ids': np.asarray(ids, np.int64), 'xyz': np.asarray(xyz, np.float64),
            'rgb': np.asarray(rgb, np.float64).reshape(-1, 3) / 255.,
            'train_observations': used, 'heldout_observations_ignored': ignored}


class VerifiedImages:
    """Loads each image once, after checking its SHA-256; records what was read."""

    def __init__(self, output, records):
        self.output, self.records, self.cache, self.read = Path(output), records, {}, []

    def __call__(self, name, xy):
        if name not in self.cache:
            record = self.records[name]
            path = self.output / record['path']
            if digest(path) != record['sha256']:
                raise RuntimeError(f"{record['path']} changed since the split")
            with Image.open(path) as raw:
                self.cache[name] = np.asarray(raw.convert('RGB'), np.float64)
            self.read.append(name)
        return bilinear_rgb(self.cache[name], xy)


def face_weights(erp_mask, rotation, size):
    """Appearance weight per face pixel; zero wherever any bilinear source pixel is zero."""
    weights = project(erp_mask, rotation, size)
    weights[project(erp_mask, rotation, size, mask=True) == 0] = 0
    return weights


def gsplat_prepare(run):
    run.require('import_split')
    output = run.path.parent
    config = run.config
    source = output / config['split_run']
    record = read(run.path / 'split_import/split.json')
    inputs = read(run.path / 'split_import/train_inputs.json')
    views = read(source / 'sfm_import/views.json')['views']
    component = read(source / 'sfm_import/sfm_import.json')['component']
    model = load_model(source / f'sfm_import/sfm/sparse/{component}')
    owner = record['face_assignment']
    sets = {name: set(record[name]) for name in auto05.SETS}
    recorded = {**auto05.recorded_digests(output, config['sfm_run'], 'prepare'),
                **auto05.recorded_digests(output, config['semantic_run'], 'auto_mask')}
    train_image_sha = {e['path']: e['sha256'] for e in inputs['images']}
    target = run.path / 'gsplat_inputs'
    cameras = {name: [] for name in auto05.SETS}
    outputs, masks = [], {}
    for view in sorted(views, key=lambda v: v['sfm_name']):
        name, pano, group = view['sfm_name'], view['panorama_id'], owner[view['sfm_name']]
        if group not in auto05.SETS or pano not in sets[group]:
            raise RuntimeError(f'{name}: face set {group} disagrees with its panorama')
        if name not in model['images']:
            raise RuntimeError(f'{name}: no pose in the SfM model')
        image_rel = f"{config['sfm_run']}/{view['image']}"
        image_sha = train_image_sha.get(image_rel) if group == 'train' else recorded.get(image_rel)
        if group == 'train' and image_sha is None:
            raise RuntimeError(f'{image_rel}: train image absent from train_inputs.json')
        mask_rel = (inputs['masks'][pano]['path'] if group == 'train' else
                    f"{config['semantic_run']}/segmentation/fused/{pano}/appearance_mask.png")
        mask_sha = inputs['masks'][pano]['sha256'] if group == 'train' else recorded.get(mask_rel)
        if pano not in masks:
            if not mask_sha or digest(output / mask_rel) != mask_sha:
                raise RuntimeError(f'{mask_rel} changed, missing or unrecorded')
            with Image.open(output / mask_rel) as raw:
                masks[pano] = np.asarray(raw.convert('L'))
        rotation = np.asarray(view['T_face_from_panorama'])[:3, :3]
        weights_path = target / 'weights' / group / name
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(face_weights(masks[pano], rotation, view['width'])).save(weights_path)
        outputs.append(weights_path)
        camera = model['images'][name]
        cameras[group].append({'name': name, 'panorama_id': pano, 'set': group,
                               'image': {'path': image_rel, 'sha256': image_sha},
                               'weights': {'path': str(weights_path.relative_to(output)),
                                           'sha256': digest(weights_path)},
                               'width': camera['width'], 'height': camera['height'],
                               'K': camera['K'], 'world_to_camera': camera['world_to_camera']})
    if len(cameras['train']) != len(inputs['images']):
        raise RuntimeError(f"{len(cameras['train'])} train cameras != {len(inputs['images'])} train images")
    for group, entries in cameras.items():
        path = target / f'cameras_{group}.json'
        write(path, {'schema_version': 1, 'set': group, 'cameras': entries})
        outputs.append(path)
    images = VerifiedImages(output, {c['name']: c['image'] for c in cameras['train']})
    eligible = inputs['init_points']['eligible_point_ids']
    colors = train_colors(model['points'], eligible, set(images.records), images)
    if set(images.read) - {c['name'] for c in cameras['train']}:
        raise RuntimeError('A non-train image was read for colors')
    points = target / 'points.npz'
    np.savez(points, ids=colors['ids'], xyz=colors['xyz'], rgb=colors['rgb'])
    outputs.append(points)
    manifest = target / 'gsplat_inputs.json'
    write(manifest, {
        'schema_version': 1, 'split_run': config['split_run'], 'sfm_run': config['sfm_run'],
        'semantic_run': config['semantic_run'], 'partition_sha256': record['partition_sha256'],
        'sets': {name: record[name] for name in auto05.SETS},
        'cameras': {name: len(entries) for name, entries in cameras.items()},
        'points': {'count': len(colors['ids']),
                   'excluded_heldout_only': len(inputs['init_points']['excluded_heldout_only_point_ids']),
                   'colors': 'recomputed from train observations only (bilinear, mean)',
                   'train_observations_used': colors['train_observations'],
                   'heldout_observations_ignored': colors['heldout_observations_ignored'],
                   'train_images_read': len(images.read)},
        'files': {str(p.relative_to(output)): digest(p) for p in outputs},
        'protocol': {**auto05.PROTOCOL,
                     'initialization': 'eligible SfM points; colors recomputed here from train views only'},
        'limitation': LIMITATION, 'git_commit': git_commit(), 'created_at': now()})
    return outputs + [manifest]
