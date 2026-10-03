"""Reproject face masks to the ERP, fuse projections, run automatic checks, write artifacts.

Rules (POLICY, uncalibrated):
- a pixel counts as observed by a face only inside its core (border band excluded);
- a group is validated on a pixel when >=2 faces agree and they are a majority of observers;
- a glass candidate without that consensus is unknown_glass: blocked for navigation,
  never free space; geometry exclusion follows POLICY['geometry_policy'];
- UNKNOWN (model, checkpoint, CUDA or environment unavailable) and FAILED (every face
  failed) panoramas get no mask file at all: no default mask exists.
Only small per-panorama summaries are kept in memory; masks are written as PNG at once.
"""
import shutil
from functools import lru_cache
import numpy as np
import cv2
from PIL import Image
from ..storage import digest, now, read, write
from . import GROUPS, LABELS, MODEL_ID, POLICY, PROMPTS
from .provenance import code_digests, git_commit

COLORS = {'no_inference': (40, 40, 40), 'dynamic': (230, 50, 200), 'mirror': (250, 210, 0),
          'glass': (0, 170, 255), 'unknown_glass': (255, 90, 0),
          'unknown_reflective': (255, 160, 120), 'window_frame': (120, 220, 120),
          'floor': (110, 80, 50), 'wall': (150, 150, 170), 'furniture': (170, 120, 200)}
GROUP_OF = {prompt: group for group, prompts in GROUPS.items() for prompt in prompts}
MANIFESTS = ('semantic_masks.json', 'mask_consistency.json', 'mask_provenance.json',
             'navigation_constraints.json')


@lru_cache(maxsize=2)
def erp_directions(width):
    """Unit directions of ERP pixel centers, inverse of geometry.erp_coordinates."""
    height = width // 2
    lon = ((np.arange(width) + .5) / width - .5) * 2 * np.pi
    lat = ((np.arange(height) + .5) / height - .5) * np.pi
    lon, lat = np.meshgrid(lon, lat)
    d = np.stack((np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)), -1)
    d.flags.writeable = False
    return d


class FaceLookup:
    """For each face: core ERP pixels it sees and the face pixel sampled for each."""

    def __init__(self, faces, size, width):
        d = erp_directions(width).reshape(-1, 3)
        self.width, self.size, self.entries = width, size, {}
        border = POLICY['face_border_px']
        for face in faces:
            r = np.asarray(face['rotation'])
            # Elementwise products: large BLAS calls crashed Accelerate on macOS 15.3.
            f = [d[:, 0] * r[k, 0] + d[:, 1] * r[k, 1] + d[:, 2] * r[k, 2] for k in range(3)]
            front = f[2] > 1e-9
            z = np.where(front, f[2], 1)
            col = np.floor(f[0] / z * size / 2 + size / 2).astype(np.int64)
            row = np.floor(f[1] / z * size / 2 + size / 2).astype(np.int64)
            core = (front & (col >= border) & (col < size - border)
                    & (row >= border) & (row < size - border))
            index = np.flatnonzero(core)
            self.entries[face['face_id']] = (index.astype(np.int32),
                                             (row[index] * size + col[index]).astype(np.int32))


def mask_rejection(mask, record):
    border = POLICY['face_border_px']
    if record['score'] < POLICY['confidence_threshold']:
        return 'low_score'
    if not mask.any():
        return 'empty'
    if not mask[border:-border, border:-border].any():
        return 'outside_image'
    if mask.mean() < POLICY['min_area_fraction']:
        return 'too_small'
    return None


def wrap_dilate(mask, radius):
    if radius <= 0 or not mask.any():
        return mask.copy()
    padded = np.pad(mask.astype(np.uint8), ((radius, radius), (radius, radius)), mode='edge')
    padded[:, :radius] = padded[:, -2 * radius:-radius]
    padded[:, -radius:] = padded[:, radius:2 * radius]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(padded, kernel)[radius:-radius, radius:-radius].astype(bool)


def wrap_labels(mask, connectivity):
    """Connected components on the ERP cylinder: columns 0 and W-1 are neighbours."""
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=connectivity)
    parent = np.arange(count)

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    left, right = labels[:, 0], labels[:, -1]
    pairs = [(left, right)]
    if connectivity == 8:
        pairs += [(left[1:], right[:-1]), (left[:-1], right[1:])]
    for a, b in pairs:
        both = (a > 0) & (b > 0)
        for x, y in set(zip(a[both].tolist(), b[both].tolist())):
            parent[root(x)] = root(y)
    return np.array([root(i) for i in range(count)])[labels]


def enclosed_holes(mask):
    """Non-mask regions enclosed by the mask. Only the poles (first/last rows) are outside."""
    if not mask.any():
        return np.zeros_like(mask)
    labels = wrap_labels(~mask, 4)
    outside = np.unique(np.concatenate((labels[0], labels[-1])))
    return ~mask & ~np.isin(labels, outside)


def connected_to(candidates, seeds):
    """Candidate components touching a seed pixel."""
    if not (candidates & seeds).any():
        return np.zeros_like(candidates)
    labels = wrap_labels(candidates, 8)
    touched = np.unique(labels[candidates & seeds])
    return candidates & np.isin(labels, touched)


def pairwise_iou(n, k):
    """Micro-averaged IoU over all face pairs observing each pixel (exact, O(pixels))."""
    n, k = n.astype(np.int64), k.astype(np.int64)
    intersection = (k * (k - 1) // 2).sum()
    union = (n * (n - 1) // 2 - (n - k) * (n - k - 1) // 2).sum()
    return float(intersection / union) if union else None


def fuse_panorama(run, faces, lookup, raw):
    """Vote counts per ERP pixel. Each face mask is read, used and released at once."""
    shape = (lookup.width // 2, lookup.width)
    observers = np.zeros(shape[0] * shape[1], np.uint8)
    votes = {g: np.zeros_like(observers) for g in GROUPS}
    ok = {f['face_id'] for f in raw['faces'] if f['status'] == 'ok'}
    by_face = {}
    for record in raw['masks']:
        by_face.setdefault(record['face_id'], []).append(record)
    accepted, rejected = [], []
    for face in faces['faces']:
        face_id = face['face_id']
        if face_id not in ok:
            continue
        index, sample = lookup.entries[face_id]
        observers[index] += 1
        hits = {}
        for record in by_face.get(face_id, []):
            path = run.path / record['mask_path']
            if digest(path) != record['mask_sha256']:
                raise RuntimeError(f"{record['mask_path']} changed after segmentation")
            with Image.open(path) as m:
                mask = np.asarray(m) > 0
            if mask.shape != (lookup.size, lookup.size):
                raise RuntimeError(f"{record['mask_path']}: unexpected size {mask.shape}")
            reason = mask_rejection(mask, record)
            if reason:
                rejected.append({**record, 'reason': reason})
                continue
            accepted.append(record)
            group = GROUP_OF[record['prompt']]
            hit = mask.reshape(-1)[sample]
            hits[group] = hits[group] | hit if group in hits else hit
        for group, hit in hits.items():
            votes[group][index[hit]] += 1
    return (observers.reshape(shape), {g: v.reshape(shape) for g, v in votes.items()},
            accepted, rejected)


def decide(observers, votes):
    n = observers.astype(np.int16)

    def validated(group):
        k = votes[group].astype(np.int16)
        return (k >= POLICY['min_consistent_projections']) & (k >= POLICY['consensus_fraction'] * n) & (n > 0)

    candidate = {g: votes[g] > 0 for g in GROUPS}
    ok = {g: validated(g) for g in GROUPS}
    risky = ('glass', 'mirror', 'reflection', 'dynamic')
    out = {'no_inference': n == 0, 'single_projection': n == 1,
           'dynamic': ok['dynamic'], 'glass': ok['glass'], 'mirror': ok['mirror'],
           'unknown_glass': candidate['glass'] & ~ok['glass'],
           'unknown_reflective': ((candidate['mirror'] & ~ok['mirror'])
                                  | (candidate['reflection'] & ~ok['reflection'])),
           'unknown_dynamic': candidate['dynamic'] & ~ok['dynamic'],
           'reflective_candidate': candidate['glass'] | candidate['mirror'] | candidate['reflection']}
    out['holes'] = enclosed_holes(ok['glass'] | ok['mirror'])
    seeds = np.logical_or.reduce([ok[g] for g in risky]) | out['holes']
    out['validated_exclusion'] = seeds | connected_to(
        np.logical_or.reduce([candidate[g] for g in risky]), seeds)
    labels = np.zeros(n.shape, np.uint8)
    for name in ('furniture', 'wall', 'floor', 'window_frame'):
        labels[ok[name]] = LABELS[name]
    for name in ('unknown_reflective', 'unknown_glass', 'glass', 'mirror', 'dynamic', 'no_inference'):
        labels[out[name]] = LABELS[name]
    labels[out['holes'] & ~out['no_inference']] = LABELS['unknown_glass']
    out['labels'] = labels
    out['unknown'] = (out['no_inference'] | out['single_projection'] | out['unknown_glass']
                      | out['unknown_reflective'] | out['unknown_dynamic'] | out['holes'])
    return out


def output_masks(decision, width):
    """geometry: 255 usable for SfM. appearance: render weight. unknown: 255 not validated."""
    radius = int(np.ceil(POLICY['geometry_margin_deg'] * width / 360))
    excluded = decision['validated_exclusion'] | decision['no_inference']
    geometry = ~wrap_dilate(excluded, radius)
    appearance = np.full(excluded.shape, 255, np.uint8)
    appearance[decision['reflective_candidate'] | decision['holes']] = POLICY['appearance_weight_reflective']
    appearance[wrap_dilate(decision['dynamic'], radius) | decision['no_inference']] = 0
    return {'geometry_mask': geometry.astype(np.uint8) * 255, 'appearance_mask': appearance,
            'unknown_mask': decision['unknown'].astype(np.uint8) * 255}


def seam_continuity(mask):
    """Mismatch between columns W-1 and 0, judged against adjacent interior column pairs.

    The ERP is circular: columns W-1 and 0 are as close as any adjacent pair, and a
    pointwise mask crossing the seam mismatches there only where a boundary crosses.
    A boundary near the seam legitimately gives a large raw rate (its slope), so the
    seam fails only if it exceeds both the threshold and the worst nearby interior pair
    by more than the threshold: a mask cut along the seam stays FAIL.
    """
    def rate(x, y):
        either = x | y
        return float((x != y).sum() / either.sum()) if either.any() else None
    width, k = mask.shape[1], POLICY['seam_reference_columns']
    seam = rate(mask[:, -1], mask[:, 0])
    reference = [r for j in (*range(width - 1 - k, width - 1), *range(k))
                 if (r := rate(mask[:, j], mask[:, j + 1])) is not None]
    worst = max(reference) if reference else 0.
    seam = seam or 0.
    threshold = POLICY['max_seam_mismatch']
    return {'seam_mismatch': seam, 'seam_rows': int((mask[:, -1] | mask[:, 0]).sum()),
            'reference_pairs': len(reference), 'reference_max': worst,
            'reference_median': float(np.median(reference)) if reference else None,
            'excess_over_reference': seam - worst, 'threshold': threshold,
            'passed': seam <= threshold or seam - worst <= threshold}


def roll_invariant(observers, votes, decision, masks, width):
    """Post-processing must commute with a horizontal rotation of the panorama."""
    shift = width // 2
    rolled = decide(np.roll(observers, shift, 1), {g: np.roll(v, shift, 1) for g, v in votes.items()})
    rolled_masks = output_masks(rolled, width)
    same = all(np.array_equal(np.roll(rolled_masks[k], -shift, 1), masks[k]) for k in masks)
    return same and np.array_equal(np.roll(rolled['labels'], -shift, 1), decision['labels'])


def solid_angle_fraction(mask):
    weights = np.cos(((np.arange(mask.shape[0]) + .5) / mask.shape[0] - .5) * np.pi)
    return float((mask * weights[:, None]).sum() / (weights.sum() * mask.shape[1]))


def check(name, passed, evidence, unknown=False):
    return {'name': name, 'result': 'UNKNOWN' if unknown else 'PASS' if passed else 'FAIL',
            'evidence': evidence}


def seam_check(observers, votes, decision, masks):
    width = decision['labels'].shape[1]
    seams = {'reflective_candidate': seam_continuity(decision['reflective_candidate']),
             'geometry_excluded': seam_continuity(masks['geometry_mask'] == 0),
             'unknown': seam_continuity(masks['unknown_mask'] > 0)}
    circular = roll_invariant(observers, votes, decision, masks, width)
    return check('seam_discontinuity', circular and all(v['passed'] for v in seams.values()),
                 {'roll_invariant': circular, 'masks': seams,
                  'rule': 'FAIL if post-processing is not roll invariant, or if a seam mismatch '
                          'exceeds max_seam_mismatch and the worst of the '
                          f"{POLICY['seam_reference_columns']} adjacent interior pairs on each side "
                          'by more than max_seam_mismatch'})


def panorama_checks(raw, observers, votes, decision, masks, rejected):
    failed = [f['face_id'] for f in raw['faces'] if f['status'] != 'ok']
    glass_votes = votes['glass'][decision['glass']]
    reflective = decision['glass'] | decision['mirror']
    holes = int(decision['holes'].sum())
    hole_fraction = holes / max(int((reflective | decision['holes']).sum()), 1)
    reasons = {}
    for r in rejected:
        reasons[r['reason']] = reasons.get(r['reason'], 0) + 1
    defaults = decision['no_inference'] & ((masks['geometry_mask'] > 0) | (masks['appearance_mask'] > 0))
    return [
        check('model_status', raw['status'] == 'OK', f"status {raw['status']}; failed faces: {failed}"),
        check('mask_rejection', True, {'rejected_by_reason': reasons}),
        check('glass_multi_projection',
              bool((glass_votes >= POLICY['min_consistent_projections']).all()),
              f'{int(decision["glass"].sum())} validated glass px, min votes '
              f'{int(glass_votes.min()) if glass_votes.size else None}; '
              f'{int(decision["unknown_glass"].sum())} px unknown_glass'),
        check('holes', hole_fraction <= POLICY['max_hole_fraction'],
              f'{holes} enclosed px ({hole_fraction:.4f} of glass/mirror) classified unknown'),
        seam_check(observers, votes, decision, masks),
        check('no_default_mask', not defaults.any(),
              f'{int(decision["no_inference"].sum())} px without inference, all excluded'),
    ]


def panorama_metrics(observers, votes, decision):
    groups = {g: {'candidate_fraction': solid_angle_fraction(votes[g] > 0),
                  'max_projections': int(votes[g].max()),
                  'pairwise_iou': pairwise_iou(observers, votes[g])} for g in GROUPS}
    for g in ('glass', 'mirror', 'dynamic'):
        groups[g]['validated_fraction'] = solid_angle_fraction(decision[g])
    for g in ('unknown_glass', 'unknown_reflective', 'unknown_dynamic', 'no_inference'):
        groups[g] = {'fraction': solid_angle_fraction(decision[g])}
    return {'two_view_coverage': solid_angle_fraction(observers >= 2),
            'unknown_fraction': solid_angle_fraction(decision['unknown']),
            'geometry_excluded_fraction': solid_angle_fraction(decision['validated_exclusion']),
            'groups': groups}


def save_outputs(run, mask_dir, preview_dir, pano, decision, masks, size):
    """Masks as PNG at the original resolution, labels at fusion resolution, preview JPEG."""
    paths = {}
    for name, value in masks.items():
        path = mask_dir / pano / f'{name}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(value).resize(size, Image.Resampling.NEAREST).save(path, optimize=True)
        paths[name] = path
    paths['labels'] = mask_dir / pano / 'labels.png'
    Image.fromarray(decision['labels']).save(paths['labels'], optimize=True)
    with Image.open(run.path / 'prepare/erp' / f'{pano}.png') as erp:
        shape = decision['labels'].shape
        preview = np.asarray(erp.convert('RGB').resize((shape[1], shape[0])), np.float32)
    for name, color in COLORS.items():
        hit = decision['labels'] == LABELS[name]
        preview[hit] = .45 * preview[hit] + .55 * np.asarray(color, np.float32)
    paths['preview'] = preview_dir / f'{pano}.jpg'
    paths['preview'].parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(preview.astype(np.uint8)).save(paths['preview'], quality=88)
    return paths


def neighbor_checks(results, order):
    """Glass/mirror presence agreement with adjacent captures (acquisition order).

    SfM poses are not used: masks must exist before the masked SfM run. Presence only.
    """
    fused = [p for p in order if results[p]['status'] in ('OK', 'PARTIAL')]
    checks = {}
    for i, pano in enumerate(fused):
        peers = [fused[j] for j in (i - 1, i + 1) if 0 <= j < len(fused)]
        entries = []
        for group in ('glass', 'mirror'):
            if not peers:
                entries.append(check(f'neighbor_{group}', False, 'no segmented neighbour', unknown=True))
                continue
            present = results[pano]['presence'][group]
            agreeing = [p for p in peers if results[p]['presence'][f'{group}_candidate']]
            entries.append(check(f'neighbor_{group}', not present or bool(agreeing),
                                 {'neighbors': peers, 'present': present,
                                  'neighbors_with_candidate': agreeing,
                                  'note': 'adjacent captures by file order; presence only'}))
        checks[pano] = entries
    return checks


def load_raw(run, pano):
    path = run.path / 'segmentation/raw' / pano / 'masks.json'
    if not path.is_file():
        raise RuntimeError(f'No GPU segmentation for {pano}: run '
                           f'`python -m theta_pipeline.segmentation --run {run.path}` on the GPU VM')
    return path, read(path)


def process(run, panoramas, mask_dir, preview_dir):
    faces = read(run.path / 'segmentation/faces.json')
    width = min(POLICY['fusion_max_width'], run.config['erp_width'])
    lookup = FaceLookup(faces['faces'], faces['face_size'], width)
    capture = {r['panorama_id']: r for r in read(run.path / 'capture.json')['images']}
    results, outputs = {}, []
    for pano in panoramas:
        raw_path, raw = load_raw(run, pano)
        outputs += [raw_path] + [run.path / r['mask_path'] for r in raw['masks']]
        if raw['status'] in ('UNKNOWN', 'FAILED'):
            results[pano] = {'raw': raw, 'status': raw['status'], 'paths': None}
            continue
        print(f'fuse {pano}', flush=True)
        observers, votes, accepted, rejected = fuse_panorama(run, faces, lookup, raw)
        decision = decide(observers, votes)
        masks = output_masks(decision, width)
        size = (capture[pano]['width'], capture[pano]['height'])
        paths = save_outputs(run, mask_dir, preview_dir, pano, decision, masks, size)
        outputs += list(paths.values())
        # Keep only summaries: full-resolution arrays are released with this iteration.
        results[pano] = {'raw': raw, 'status': raw['status'], 'accepted': accepted,
                         'rejected': rejected, 'paths': paths,
                         'presence': {'glass': bool(decision['glass'].any()),
                                      'mirror': bool(decision['mirror'].any()),
                                      'glass_candidate': bool((votes['glass'] > 0).any()),
                                      'mirror_candidate': bool((votes['mirror'] > 0).any())},
                         'metrics': panorama_metrics(observers, votes, decision),
                         'checks': panorama_checks(raw, observers, votes, decision, masks, rejected)}
    return faces, results, outputs


def provenance(run, faces, results):
    entries = {}
    for pano, result in results.items():
        raw = result['raw']
        entries[pano] = {'status': raw['status'], 'source_sha256': faces['panoramas'][pano]['source_sha256'],
                         'raw_manifest_sha256': digest(run.path / 'segmentation/raw' / pano / 'masks.json'),
                         'segmentation': raw['provenance']}
    segmentations = [e['segmentation'] for e in entries.values()]
    commits = {s.get('git_commit') for s in segmentations}
    checkpoints = {s.get('checkpoint_sha256') for s in segmentations}
    fusion_code = code_digests()
    commit = commits.pop() if len(commits) == 1 else None
    checkpoint = checkpoints.pop() if len(checkpoints) == 1 else None
    hashed = bool(commit and not commit.endswith('-dirty') and checkpoint
                  and all(s.get('code_sha256') == fusion_code for s in segmentations)
                  and all(s.get('environment', {}).get('qualified') for s in segmentations))
    return {'schema_version': 2, 'model': MODEL_ID, 'prompts': list(PROMPTS), 'policy': POLICY,
            'git_commit': commit, 'checkpoint_sha256': checkpoint,
            'faces_sha256': digest(run.path / 'segmentation/faces.json'),
            'fusion': {'git_commit': git_commit(), 'code_sha256': fusion_code, 'at': now()},
            'model_and_code_hashed': hashed,
            'requirement': 'single clean git commit, one checkpoint SHA-256, qualified GPU '
                           'environment, GPU code identical to fusion code',
            'panoramas': entries}


def write_manifests(run, folder, stems, results, prov, neighbors):
    records, panoramas, consistency = [], [], {}
    for pano in stems:
        r = results[pano]
        if r['paths'] is None:
            panoramas.append({'panorama_id': pano, 'status': r['status'],
                              'reason': r['raw'].get('reason') or 'every face failed',
                              'artifacts': None, 'checks_passed': False})
            continue
        records += [{k: m[k] for k in ('image_id', 'face_id', 'prompt', 'mask_path', 'score',
                                       'area_fraction', 'model', 'checkpoint_sha256', 'git_commit')}
                    for m in r['accepted']]
        checks = r['checks'] + neighbors.get(pano, [])
        consistency[pano] = {'metrics': r['metrics'], 'checks': checks,
                             'rejected_masks': [{k: m[k] for k in ('face_id', 'prompt', 'mask_path',
                                                                    'score', 'reason')}
                                                for m in r['rejected']]}
        groups = r['metrics']['groups']
        panoramas.append({'panorama_id': pano, 'status': r['status'],
                          'checks_passed': all(c['result'] == 'PASS' for c in checks),
                          'artifacts': {k: str(v.relative_to(run.path)) for k, v in r['paths'].items()},
                          'fractions': {g: groups[g].get('validated_fraction', groups[g].get('fraction'))
                                        for g in ('glass', 'mirror', 'dynamic', 'unknown_glass',
                                                  'unknown_reflective', 'no_inference')}})
    statuses = {r['status'] for r in results.values()}
    multiview = {g: [p for p in stems if results[p].get('presence', {}).get(g)] for g in ('glass', 'mirror')}
    accepted = (statuses == {'OK'} and prov['model_and_code_hashed']
                and all(p['checks_passed'] for p in panoramas)
                and all(len(v) >= 2 for v in multiview.values()))
    # UNKNOWN (nothing could be decided) stays distinct from REJECTED (a control failed).
    status = 'UNKNOWN' if 'UNKNOWN' in statuses else 'ACCEPTED' if accepted else 'REJECTED'
    navigation = {'schema_version': 2, 'status': 'advisory', 'navigation_validated': False,
                  'consumed_by': None,
                  'note': 'Advisory until a navigation builder consumes it; its presence does not '
                          'validate any navigation segment.',
                  'policy': 'unknown_glass, unknown_reflective and no_inference are blocked_unknown; '
                            'glass and mirror are obstacles; none of them is free_space',
                  'labels_legend': LABELS,
                  'blocked_unknown_labels': ['unknown_glass', 'unknown_reflective', 'no_inference'],
                  'obstacle_labels': ['glass', 'mirror'],
                  'panoramas': [{'panorama_id': p['panorama_id'],
                                 'labels': p['artifacts']['labels'] if p['artifacts'] else None,
                                 'unknown_glass_fraction': p['fractions']['unknown_glass']
                                 if p['artifacts'] else None,
                                 'all_pixels_unknown': p['artifacts'] is None} for p in panoramas]}
    manifest = {'schema_version': 3, 'status': status, 'backend': 'sam3', 'model': MODEL_ID,
                'scope': 'trial' if folder != run.path else 'all',
                'prompts': list(PROMPTS), 'groups': {g: list(v) for g, v in GROUPS.items()},
                'labels_legend': LABELS, 'policy': POLICY, 'error_policy': 'fail_closed_unknown',
                'accepted_masks': accepted, 'panoramas': panoramas,
                'multiview_panoramas': multiview, 'masks': records,
                'next_action': ('Run a new SfM experiment with "semantic_run" in its config'
                                if accepted else 'Inspect mask_consistency.json; '
                                'UNKNOWN or failed controls block training')}
    values = dict(zip(MANIFESTS, (manifest, {'schema_version': 2, 'policy': POLICY,
                                             'panoramas': consistency, 'multiview_panoramas': multiview},
                                  prov, navigation)))
    paths = []
    for name, value in values.items():
        write(folder / name, value)
        paths.append(folder / name)
    return manifest, paths


def trial(run):
    """Fuse the trial panorama and decide whether segmentation may extend to all panoramas."""
    run.require('seg_faces')
    pano = read(run.path / 'segmentation/faces.json')['trial_panorama']
    folder = run.path / 'segmentation/trial'
    faces, results, outputs = process(run, [pano], folder, folder / 'semantic_masks_preview')
    result = results[pano]
    prov = provenance(run, faces, results)
    manifest, paths = write_manifests(run, folder, [pano], results, prov, {})
    outputs += paths
    gate_path = folder / 'trial_gate.json'
    if result['status'] == 'UNKNOWN':
        write(gate_path, {'schema_version': 2, 'panorama_id': pano, 'decision': 'UNKNOWN',
                          'reason': result['raw'].get('reason'), 'decided_at': now()})
        # Stage fails so the trial can be retried in the same run once the model is available.
        raise RuntimeError(f"SAM 3 status UNKNOWN for {pano}: {result['raw'].get('reason')}")
    t = POLICY['trial']
    if result['paths'] is None:
        gates = [check('faces_processed', False, 'every face failed: no mask written')]
        m = None
    else:
        m = result['metrics']
        failed = sum(f['status'] != 'ok' for f in result['raw']['faces'])
        glass = m['groups']['glass']['validated_fraction']
        gates = [
            check('faces_processed', failed <= t['max_failed_faces'], f'{failed} failed faces'),
            check('two_view_coverage', m['two_view_coverage'] >= t['min_two_view_coverage'],
                  m['two_view_coverage']),
            check('unknown_fraction', m['unknown_fraction'] <= t['max_unknown_fraction'],
                  m['unknown_fraction']),
            # No glass detected is not a failure of the model, but it proves nothing either.
            check('glass_detected', glass > 0, f'validated glass fraction {glass}', unknown=glass == 0),
            *[check(f'{g}_projection_consistency',
                    (m['groups'][g]['pairwise_iou'] or 0) >= t['min_pairwise_iou'],
                    m['groups'][g]['pairwise_iou'], unknown=m['groups'][g]['pairwise_iou'] is None)
              for g in ('glass', 'mirror')],
            *[c for c in result['checks'] if c['name'] != 'mask_rejection'],
        ]
    gates.append(check('model_and_code_hashed', prov['model_and_code_hashed'], prov['requirement']))
    # A mirror outside the trial view is not an error; it stays UNKNOWN, not FAIL.
    blocking = [g for g in gates if g['name'] != 'mirror_projection_consistency']
    decision = ('FAIL' if any(g['result'] == 'FAIL' for g in blocking) else
                'PASS' if all(g['result'] == 'PASS' for g in blocking) else 'UNKNOWN')
    write(gate_path, {'schema_version': 2, 'panorama_id': pano, 'decision': decision,
                      'policy_version': POLICY['version'], 'calibrated': POLICY['calibrated'],
                      'semantic_status': manifest['status'], 'gates': gates, 'metrics': m,
                      'git_commit': prov['git_commit'], 'checkpoint_sha256': prov['checkpoint_sha256'],
                      'decided_at': now()})
    return outputs + [gate_path]


def all_panoramas(run):
    run.require('seg_trial')
    gate = read(run.path / 'segmentation/trial/trial_gate.json')
    if gate['decision'] != 'PASS':
        raise RuntimeError(f"Trial decision is {gate['decision']}: all-panorama fusion refused")
    stems = [p.stem for p in run.sources]
    faces, results, outputs = process(run, stems, run.path / 'segmentation/fused',
                                      run.path / 'semantic_masks_preview')
    prov = provenance(run, faces, results)
    neighbors = neighbor_checks(results, stems)
    manifest, paths = write_manifests(run, run.path, stems, results, prov, neighbors)
    if manifest['status'] == 'UNKNOWN':
        unknown = [p['panorama_id'] for p in manifest['panoramas'] if p['status'] == 'UNKNOWN']
        raise RuntimeError(f'SAM 3 status UNKNOWN for {unknown}: semantic_masks.json records UNKNOWN; '
                           'rerun the GPU step once the model is available')
    return outputs + paths


def source_semantics(config_dir, config):
    """Verified ACCEPTED semantic run used by a masked SfM experiment."""
    source = (config_dir / config['output'] / config['semantic_run']).resolve()
    state = read(source / 'run.json')
    entry = state['stages'].get('auto_mask', {})
    if entry.get('status') != 'completed':
        raise ValueError(f"semantic_run {config['semantic_run']}: auto_mask not completed")
    for name in MANIFESTS:
        if digest(source / name) != entry['artifacts'].get(name):
            raise ValueError(f"semantic_run {config['semantic_run']}: {name} changed or missing")
    manifest = read(source / 'semantic_masks.json')
    if manifest['status'] != 'ACCEPTED':
        raise ValueError(f"semantic_run {config['semantic_run']} is {manifest['status']}, not ACCEPTED")
    geometry = {p['panorama_id']: source / p['artifacts']['geometry_mask'] for p in manifest['panoramas']}
    for pano, path in geometry.items():
        if digest(path) != entry['artifacts'].get(str(path.relative_to(source))):
            raise ValueError(f'semantic_run geometry mask changed: {path}')
    return source, geometry


def import_semantics(run):
    """Copy the verified manifests of the semantic run into a masked SfM run."""
    run.require('audit')
    source, _ = source_semantics(run.config_dir, run.config)
    outputs = []
    for name in MANIFESTS:
        shutil.copyfile(source / name, run.path / name)
        outputs.append(run.path / name)
    write(run.path / 'semantic_import.json', {'schema_version': 1, 'semantic_run': run.config['semantic_run'],
                                              'files': {n: digest(source / n) for n in MANIFESTS},
                                              'geometry_masks': 'config masks.geometry from semantic_run'})
    return outputs + [run.path / 'semantic_import.json']
