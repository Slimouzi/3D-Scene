"""AUTO-05: deterministic spatial partition of panoramas and evaluated-training gates.

A split experiment imports, with hash verification, an ACCEPTED semantic run and a
masked SfM run; it never re-runs them. Method `auto05-hull-maxmin-v1`:

- unit: the panorama. Every face of a panorama belongs to its panorama's set.
- reconstructed centers are projected on their two principal axes (signs fixed
  deterministically), so the result depends neither on input order nor on scale;
- convex-hull vertices always stay in train, so held-out stations are interpolated,
  never extrapolated, from train;
- held-out sets are searched exhaustively among interior panoramas; a candidate is
  admissible only if every held-out panorama has >= min_train_neighbors train stations
  among its k nearest, lies inside the train hull and shares a covisibility edge with
  train, and if the train covisibility graph stays connected;
- among admissible sets, the one maximizing the minimum distance between held-out
  stations wins, ties by sorted panorama ids. Held-out stations ordered along the first
  axis are assigned test, validation, test, ... ;
- no admissible set, or too few panoramas: status UNKNOWN. There is no randomness.

Choices the ticket left open (fractions, k, minimum train neighbours) are parameters
recorded in split.json. They are structural, not quality thresholds.
"""
import hashlib
import json
import math
import shutil
from itertools import combinations
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .segmentation.provenance import git_commit
from .storage import digest, now, read, write

METHOD = 'auto05-hull-maxmin-v1'
DEFAULTS = {'test_fraction': 0.15, 'validation_fraction': 0.10, 'neighbors_k': 4,
            'min_train_neighbors': 2, 'min_train_panoramas': 3, 'max_combinations': 200_000,
            'expected_partition_sha256': None}
SETS = ('train', 'validation', 'test')
STAGES = ('import_sfm', 'auto_split', 'split_gates', 'split_report')


def parameters(config):
    unknown = set(config.get('split', {})) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown split parameters: {unknown}')
    return {**DEFAULTS, **config.get('split', {})}


def partition_sha256(sets):
    canonical = {name: sorted(sets[name]) for name in SETS}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


def check(name, passed, evidence, unknown=False):
    return {'name': name, 'result': 'UNKNOWN' if unknown else 'PASS' if passed else 'FAIL',
            'evidence': evidence}


def plane(centers):
    """2-D coordinates on the two principal axes; deterministic signs; unit RMS radius."""
    ids = sorted(centers)
    x = np.asarray([centers[i] for i in ids], float)
    x = x - x.mean(axis=0)
    _, _, axes = np.linalg.svd(x, full_matrices=False)
    xy = x @ axes[:2].T
    for k in range(2):
        column = np.round(xy[:, k], 12)
        if column[np.argmax(np.abs(column))] < 0:
            xy[:, k] = -xy[:, k]
    scale = np.sqrt((xy ** 2).sum(axis=1).mean()) or 1.
    return {i: xy[n] / scale for n, i in enumerate(ids)}


def hull(points):
    """Strict convex-hull vertex ids (monotone chain); collinear points are not vertices."""
    ordered = sorted(points, key=lambda i: (round(points[i][0], 12), round(points[i][1], 12), i))
    if len(ordered) < 3:
        return list(ordered)

    def cross(o, a, b):
        return ((points[a][0] - points[o][0]) * (points[b][1] - points[o][1])
                - (points[a][1] - points[o][1]) * (points[b][0] - points[o][0]))
    lower, upper = [], []
    for chain, sequence in ((lower, ordered), (upper, ordered[::-1])):
        for i in sequence:
            while len(chain) >= 2 and cross(chain[-2], chain[-1], i) <= 1e-12:
                chain.pop()
            chain.append(i)
    return lower[:-1] + upper[:-1]


def inside(point, polygon):
    """Strictly inside a counter-clockwise convex polygon."""
    if len(polygon) < 3:
        return False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]) <= 1e-12:
            return False
    return True


def connected(nodes, edges):
    nodes = set(nodes)
    if not nodes:
        return False
    seen, stack = set(), [min(nodes)]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack += [b for a, b in edges if a == node and b in nodes]
        stack += [a for a, b in edges if b == node and a in nodes]
    return seen == nodes


def admissible(held, ids, xy, edges, p):
    train = [i for i in ids if i not in held]
    train_hull = [xy[i] for i in hull({i: xy[i] for i in train})]
    reasons = []
    for h in held:
        nearest = sorted((i for i in ids if i != h),
                         key=lambda i: (float(np.linalg.norm(xy[i] - xy[h])), i))[:p['neighbors_k']]
        if sum(i in train for i in nearest) < p['min_train_neighbors']:
            reasons.append(f'{h}: fewer than {p["min_train_neighbors"]} train stations among '
                           f'{p["neighbors_k"]} nearest')
        if not inside(xy[h], train_hull):
            reasons.append(f'{h}: outside the train hull (extrapolation)')
        if not any((a == h and b in train) or (b == h and a in train) for a, b in edges):
            reasons.append(f'{h}: no covisibility edge with train')
    if not connected(train, edges):
        reasons.append('train covisibility graph disconnected')
    return reasons


def propose(centers, edges, p):
    """Deterministic partition, or status UNKNOWN with reasons. Pure function."""
    ids = sorted(centers)
    edges = {tuple(sorted(e)) for e in edges}
    n = len(ids)
    n_test = max(1, round(p['test_fraction'] * n))
    n_val = max(1, round(p['validation_fraction'] * n))
    base = {'method': METHOD, 'parameters': p, 'panoramas': ids,
            'counts': {'train': n - n_test - n_val, 'validation': n_val, 'test': n_test}}
    if n - n_test - n_val < p['min_train_panoramas']:
        return {**base, 'status': 'unknown', 'reasons': [f'{n} panoramas: too few for three sets']}
    xy = plane(centers)
    vertices = hull(xy)
    candidates = [i for i in ids if i not in vertices]
    size = n_test + n_val
    if len(candidates) < size:
        return {**base, 'status': 'unknown', 'hull_vertices': vertices,
                'reasons': [f'{len(candidates)} interior panoramas for {size} held-out']}
    total = math.comb(len(candidates), size)
    if total > p['max_combinations']:
        return {**base, 'status': 'unknown', 'reasons': [f'{total} combinations exceed max_combinations']}
    best, rejected = None, 0
    for held in combinations(candidates, size):
        if admissible(held, ids, xy, edges, p):
            rejected += 1
            continue
        spread = min(float(np.linalg.norm(xy[a] - xy[b])) for a, b in combinations(held, 2)) if size > 1 else 0.
        key = (-round(spread, 9), held)
        if best is None or key < best[0]:
            best = (key, held, spread)
    layout = {i: [round(float(v), 6) for v in xy[i]] for i in ids}
    if best is None:
        return {**base, 'status': 'unknown', 'hull_vertices': vertices, 'layout': layout,
                'reasons': [f'none of {total} held-out sets satisfies the coverage constraints']}
    _, held, spread = best
    order = sorted(held, key=lambda i: (round(float(xy[i][0]), 9), i))
    sets = {'train': [i for i in ids if i not in held], 'validation': [], 'test': []}
    for rank, h in enumerate(order):
        preferred = 'test' if rank % 2 == 0 else 'validation'
        other = 'validation' if preferred == 'test' else 'test'
        target = preferred if len(sets[preferred]) < base['counts'][preferred] else other
        sets[target].append(h)
    return {**base, 'status': 'proposed', **{k: sorted(v) for k, v in sets.items()},
            'hull_vertices': sorted(vertices), 'layout': layout,
            'min_heldout_distance_rms_units': round(spread, 6),
            'admissible_sets': total - rejected, 'evaluated_sets': total}


def face_assignment(sets, views):
    """Every face (SfM image) inherits its panorama's set."""
    owners = {}
    for name in SETS:
        for pano in sets.get(name) or []:
            owners.setdefault(pano, []).append(name)
    # A panorama listed twice is reported as such, never silently given to one set.
    return {v['sfm_name']: '+'.join(owners[v['panorama_id']]) if v['panorama_id'] in owners else None
            for v in views}


def validate(split, panoramas, poses_sha256, views):
    """Structural checks, re-run at freeze time and again before any training."""
    sets = {name: split.get(name) or [] for name in SETS}
    union = [p for name in SETS for p in sets[name]]
    faces = face_assignment(sets, views)
    by_pano = {}
    for v in views:
        by_pano.setdefault(v['panorama_id'], set()).add(faces[v['sfm_name']])
    stored = split.get('partition_sha256')
    return [
        check('non_empty_sets', all(sets[name] for name in SETS),
              {name: len(sets[name]) for name in SETS}),
        check('disjoint_sets', len(union) == len(set(union)), 'no panorama in two sets'),
        check('complete_cover', sorted(union) == sorted(panoramas),
              {'missing': sorted(set(panoramas) - set(union)), 'extra': sorted(set(union) - set(panoramas))}),
        check('faces_grouped_by_panorama',
              all(len(s) == 1 and s <= set(SETS) for s in by_pano.values()) and set(by_pano) == set(panoramas),
              f'{len(faces)} faces, one set per panorama'),
        check('poses_unchanged', split.get('poses_sha256') == poses_sha256,
              {'recorded': split.get('poses_sha256'), 'current': poses_sha256}),
        check('partition_hash', stored == partition_sha256(sets),
              {'recorded': stored, 'recomputed': partition_sha256(sets)}),
    ]


def heldout_only_points(tracks, train):
    """Point ids observed by no train panorama: never used to initialize training."""
    return sorted(pid for pid, panos in tracks.items() if not set(panos) & set(train))


def expected_train_files(sets, views, sfm_run, semantic_run):
    """The exact training files: every face image and the appearance mask of each train panorama."""
    train = set(sets['train'])
    images = sorted((v['panorama_id'], f"{sfm_run}/{v['image']}") for v in views if v['panorama_id'] in train)
    masks = {p: f'{semantic_run}/segmentation/fused/{p}/appearance_mask.png' for p in sorted(train)}
    return images, masks


def recorded_digests(output_root, run_name, stage):
    """SHA-256 recorded by a source run for its completed stage artifacts, keyed by output path."""
    entry = read(output_root / run_name / 'run.json')['stages'].get(stage, {})
    if entry.get('status') != 'completed':
        return {}
    return {f'{run_name}/{rel}': sha for rel, sha in entry['artifacts'].items()}


def file_record(output_root, rel, recorded):
    return {'path': rel, 'resolved': str((output_root / rel).resolve()), 'sha256': recorded.get(rel)}


def train_inputs(sets, views, tracks, sfm_run, semantic_run, output_root, recorded):
    images, masks = expected_train_files(sets, views, sfm_run, semantic_run)
    excluded = heldout_only_points(tracks, sets['train'])
    return {'schema_version': 2, 'panoramas': sorted(sets['train']),
            'sfm_run': sfm_run, 'semantic_run': semantic_run,
            'paths_relative_to': 'the runs output directory',
            'images': [{'panorama_id': p, **file_record(output_root, rel, recorded)} for p, rel in images],
            'masks': {p: file_record(output_root, rel, recorded) for p, rel in masks.items()},
            'init_points': {
                'eligible_point_ids': sorted(set(tracks) - set(excluded)),
                'excluded_heldout_only_point_ids': excluded,
                'empty_policy': 'an empty eligible set is FAIL: no random or untraceable initialization',
                'positions': 'triangulated by the masked SfM with all 13 panoramas: held-out '
                             'stations contributed to camera and point estimation (transductive)',
                'colors': 'NOT YET RECOMPUTED. The gsplat adapter must recompute colors from train '
                          'observations only, and test it, before claiming color separation.'}}


def verify_train_files(inputs, output_root):
    """Existence and SHA-256 of every training file. Must run again when training starts."""
    entries = [*(inputs.get('images') or []), *(inputs.get('masks') or {}).values()]
    missing = [e['path'] for e in entries if not (Path(output_root) / e['path']).is_file()]
    unrecorded = [e['path'] for e in entries if not e.get('sha256')]
    changed = [e['path'] for e in entries if e['path'] not in missing and e.get('sha256')
               and digest(Path(output_root) / e['path']) != e['sha256']]
    return check('train_files_intact', bool(entries) and not (missing or unrecorded or changed),
                 {'files': len(entries), 'missing': missing, 'unrecorded': unrecorded, 'changed': changed})


def separation_checks(sets, inputs, views, tracks, sfm_run, semantic_run):
    """Validate the manifest content against what the frozen sets imply, not its own claims."""
    expected_images, expected_masks = expected_train_files(sets, views, sfm_run, semantic_run)
    images = sorted((e['panorama_id'], e['path']) for e in inputs.get('images') or [])
    masks = {p: e['path'] for p, e in (inputs.get('masks') or {}).items()}
    init = inputs.get('init_points') or {}
    eligible = init.get('eligible_point_ids') or []
    excluded = init.get('excluded_heldout_only_point_ids') or []
    train = set(sets['train'])
    support = bool(eligible) and all(pid in tracks and set(tracks[pid]) & train for pid in eligible)
    partition = (not set(eligible) & set(excluded) and set(eligible) | set(excluded) == set(tracks)
                 and sorted(excluded) == heldout_only_points(tracks, train))
    wrong_masks = sorted(p for p in set(masks) | set(expected_masks) if masks.get(p) != expected_masks.get(p))
    return [
        check('train_panoramas_exact', inputs.get('panoramas') == sorted(train), inputs.get('panoramas')),
        check('train_images_exact', images == expected_images and len(images) == len(set(images)),
              {'expected': len(expected_images), 'listed': len(images),
               'unexpected': sorted(set(images) - set(expected_images))[:20],
               'missing': sorted(set(expected_images) - set(images))[:20]}),
        check('train_masks_match_panoramas', not wrong_masks,
              {'expected': len(expected_masks), 'mismatched': wrong_masks}),
        check('init_points_have_train_support', support and partition,
              {'eligible': len(eligible), 'excluded_heldout_only': len(excluded),
               'empty': not eligible, 'consistent_with_tracks': partition}),
    ]


def permissions(sources_ok, poses_ok, split_ok, separation_ok):
    """Exploratory training needs sources and poses; evaluated training needs everything."""
    exploratory = 'PASS' if sources_ok and poses_ok else 'UNKNOWN'
    evaluated = 'PASS' if sources_ok and poses_ok and split_ok and separation_ok else 'UNKNOWN'
    return {'panorama_delivery': 'PASS', 'exploratory_training': exploratory,
            'evaluated_training': evaluated, 'guided_3d_navigation': 'UNKNOWN',
            'free_3d_navigation': 'UNKNOWN', 'product_delivery': 'UNKNOWN'}


def verify_split_source(record, gates, config):
    """Problems preventing evaluated training from this split, or an empty list."""
    expected = config.get('split', {}).get('expected_partition_sha256')
    sets = {name: record.get(name) or [] for name in SETS}
    problems = []
    if record.get('status') != 'frozen':
        problems.append(f"split status is {record.get('status')}, not frozen")
    if not expected:
        problems.append('config split.expected_partition_sha256 is not pinned')
    elif record.get('partition_sha256') != expected:
        problems.append(f"partition {record.get('partition_sha256')} != pinned {expected}")
    if partition_sha256(sets) != record.get('partition_sha256'):
        problems.append('split lists do not match their recorded partition hash')
    if not (gates.get('gsplat_allowed') or {}).get('evaluated') or \
            gates.get('permissions', {}).get('evaluated_training') != 'PASS':
        problems.append('split gates do not allow evaluated training')
    for key in ('sfm_run', 'semantic_run'):
        if record.get(key) != config.get(key):
            problems.append(f'split {key} {record.get(key)} != config {config.get(key)}')
    return problems


# ---- Run stages ---------------------------------------------------------------

def verified_copy(source, entry, rel, target):
    if entry.get('status') != 'completed' or digest(source / rel) != entry['artifacts'].get(rel):
        raise RuntimeError(f'{source.name}/{rel} changed, missing or stage incomplete')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / rel, target)
    return target


def import_sfm(run):
    """Copy, after hash verification, the masked SfM artifacts this experiment relies on."""
    run.require('audit')
    source = (run.config_dir / run.config['output'] / run.config['sfm_run']).resolve()
    state = read(source / 'run.json')
    stages = state['stages']
    expected_semantic = run.provenance['semantic_run']
    if state['provenance'].get('semantic_run') != expected_semantic:
        raise RuntimeError(f"{source.name} was not fed by semantic run {expected_semantic}")
    if state['provenance']['inputs'] != run.provenance['inputs']:
        raise RuntimeError(f'{source.name} used different input panoramas')
    target = run.path / 'sfm_import'
    outputs = [verified_copy(source, stages['diagnose'], rel, target / rel)
               for rel in ('poses.json', 'quality.json')]
    outputs.append(verified_copy(source, stages['prepare'], 'views.json', target / 'views.json'))
    outputs.append(verified_copy(source, stages['matching'], 'sfm/matches.json', target / 'sfm/matches.json'))
    component = read(target / 'quality.json')['largest_component']
    sparse = sorted(rel for rel in stages['mapping']['artifacts'] if rel.startswith(f'sfm/sparse/{component}/'))
    outputs += [verified_copy(source, stages['mapping'], rel, target / rel) for rel in sparse]
    manifest = target / 'sfm_import.json'
    write(manifest, {'schema_version': 1, 'sfm_run': run.config['sfm_run'],
                     'sfm_run_fingerprint': state['fingerprint'], 'semantic_run': expected_semantic,
                     'component': component,
                     'files': {str(p.relative_to(target)): digest(p) for p in outputs},
                     'imported_at': now()})
    return outputs + [manifest]


def tracks_from_model(path):
    import pycolmap as pc
    rec = pc.Reconstruction(path)
    return {int(pid): sorted({Path(rec.images[e.image_id].name).stem for e in point.track.elements})
            for pid, point in rec.points3D.items()}


def source_status(run):
    semantic = read(run.path / 'semantic_masks.json')
    quality = read(run.path / 'sfm_import/quality.json')
    poses = read(run.path / 'sfm_import/poses.json')['poses']
    component = quality['largest_component']
    registered = [p for p in poses if p['status'] == 'registered' and p['component'] == component]
    expected = [p.stem for p in run.sources]
    return [
        check('semantic_masks_accepted', semantic['status'] == 'ACCEPTED' and semantic['accepted_masks'],
              f"{run.config['semantic_run']}: {semantic['status']}, manifests hash-verified at import"),
        check('sfm_imported_intact', True, f"{run.config['sfm_run']}: artifacts hash-verified at import"),
        check('poses_available', sorted(p['panorama_id'] for p in registered) == sorted(expected),
              f'{len(registered)}/{len(expected)} panoramas in component {component}'),
    ], {p['panorama_id']: p['center_world'] for p in registered}


def auto_split(run):
    run.require('auto_mask')
    run.require('import_sfm')
    p = parameters(run.config)
    source_checks, centers = source_status(run)
    target = run.path / 'sfm_import'
    poses_sha256 = digest(target / 'poses.json')
    views = read(target / 'views.json')['views']
    edges = [(e['source'], e['target']) for e in read(target / 'sfm/matches.json')['edges']
             if e['inlier_observations'] > 0]
    expected = [s.stem for s in run.sources]
    commit = git_commit()
    proposal = (propose(centers, edges, p) if source_checks[2]['result'] == 'PASS'
                else {'status': 'unknown', 'method': METHOD, 'parameters': p,
                      'reasons': ['reconstructed positions missing for some panoramas']})
    split = {'schema_version': 2, 'unit': 'panorama', 'seed': run.config['seed'], 'randomness': 'none',
             'sfm_run': run.config['sfm_run'], 'semantic_run': run.config['semantic_run'],
             'poses_sha256': poses_sha256, 'git_commit': commit, **proposal}
    checks = list(source_checks)
    inputs_path = run.path / 'train_inputs.json'
    if proposal['status'] == 'proposed':
        sets = {name: proposal[name] for name in SETS}
        split['partition_sha256'] = partition_sha256(sets)
        split['face_assignment'] = face_assignment(sets, views)
        component = read(target / 'sfm_import.json')['component']
        tracks = tracks_from_model(target / f'sfm/sparse/{component}')
        sfm_run, semantic_run = run.config['sfm_run'], run.config['semantic_run']
        output_root = run.path.parent
        recorded = {**recorded_digests(output_root, sfm_run, 'prepare'),
                    **recorded_digests(output_root, semantic_run, 'auto_mask')}
        inputs = train_inputs(sets, views, tracks, sfm_run, semantic_run, output_root, recorded)
        write(inputs_path, inputs)
        checks += validate(split, expected, poses_sha256, views)
        checks += separation_checks(sets, inputs, views, tracks, sfm_run, semantic_run)
        checks.append(verify_train_files(inputs, output_root))
        pinned = p['expected_partition_sha256']
        checks.append(check('matches_pinned_partition', pinned is None or pinned == split['partition_sha256'],
                            {'pinned': pinned, 'computed': split['partition_sha256']}))
        checks.append(check('clean_commit', bool(commit) and not commit.endswith('-dirty'), commit))
    else:
        checks.append(check('partition_feasible', False, proposal['reasons'], unknown=True))
        write(inputs_path, {'schema_version': 1, 'panoramas': None, 'reason': 'no valid partition'})
    results = {c['result'] for c in checks}
    split['status'] = ('frozen' if results == {'PASS'} else
                       'rejected' if 'FAIL' in results else 'unknown')
    split['frozen_at'] = now() if split['status'] == 'frozen' else None
    split['checks'] = checks
    split['protocol'] = PROTOCOL
    path = run.path / 'split.json'
    write(path, split)
    layout = run.path / 'split_layout.png'
    draw_layout(split, layout)
    return [path, inputs_path, layout]


PROTOCOL = {
    'poses': 'Camera poses come from the masked SfM run on all 13 panoramas: validation and '
             'test stations took part in camera and point estimation (transductive poses).',
    'training_losses': 'train panoramas only; no validation or test image in any loss',
    'initialization': 'SfM points with at least one train observation; points seen only from '
                      'validation/test are excluded; colors to be recomputed from train views by the '
                      'gsplat adapter (not yet executed)',
    'validation': 'parameter and checkpoint selection only',
    'test': 'reserved for the final evaluation; never used for selection',
    'metrics': 'no quality threshold is defined; metrics are reported, not judged',
}


def draw_layout(split, path):
    canvas = Image.new('RGB', (900, 700), '#f4f1e8')
    draw = ImageDraw.Draw(canvas)
    layout = split.get('layout') or {}
    colors = {'train': '#3b6ea5', 'validation': '#d08a00', 'test': '#c83d2f'}
    if layout:
        xy = np.asarray(list(layout.values()))
        low, high = xy.min(axis=0), xy.max(axis=0)
        scale = min(760 / max(high[0] - low[0], 1e-9), 520 / max(high[1] - low[1], 1e-9))
        owner = {p: name for name in SETS for p in split.get(name) or []}
        for pano, (x, y) in layout.items():
            px, py = 70 + (x - low[0]) * scale, 630 - (y - low[1]) * scale
            color = colors.get(owner.get(pano), '#777777')
            draw.ellipse((px - 9, py - 9, px + 9, py + 9), fill=color)
            draw.text((px + 12, py - 7), pano, fill='#1d252c')
    draw.text((20, 15), f"AUTO-05 {split['status']}: bleu train, orange validation, rouge test", fill='#1d252c')
    draw.text((20, 35), 'Axes principaux des centres reconstruits; echelle arbitraire', fill='#4d5961')
    canvas.save(path)


def split_gates(run):
    run.require('auto_split')
    split = read(run.path / 'split.json')
    source_checks, _ = source_status(run)
    sources_ok = all(c['result'] == 'PASS' for c in source_checks[:2])
    poses_ok = source_checks[2]['result'] == 'PASS'
    views = read(run.path / 'sfm_import/views.json')['views']
    poses_sha256 = digest(run.path / 'sfm_import/poses.json')
    expected = [s.stem for s in run.sources]
    if split['status'] == 'frozen':
        # Re-validated now: a frozen partition cannot change silently.
        current = validate(split, expected, poses_sha256, views)
        # Manifest content and training files are re-verified now, not trusted from auto_split.
        inputs = read(run.path / 'train_inputs.json')
        component = read(run.path / 'sfm_import/sfm_import.json')['component']
        tracks = tracks_from_model(run.path / f'sfm_import/sfm/sparse/{component}')
        sets = {name: split[name] for name in SETS}
        separation = separation_checks(sets, inputs, views, tracks,
                                       run.config['sfm_run'], run.config['semantic_run'])
        separation.append(verify_train_files(inputs, run.path.parent))
    else:
        current = [check('partition_frozen', False, f"split status {split['status']}",
                         unknown=split['status'] == 'unknown')]
        separation = [check('data_separation', False, 'no valid partition', unknown=True)]
    split_ok = split['status'] == 'frozen' and all(c['result'] == 'PASS' for c in current)
    separation_ok = all(c['result'] == 'PASS' for c in separation)
    quality = read(run.path / 'sfm_import/quality.json')
    gates = source_checks + current + separation + [
        check('navigation_segments', False, 'AUTO-04 navigation builder not implemented', unknown=True)]
    granted = permissions(sources_ok, poses_ok, split_ok, separation_ok)
    path = run.path / 'gate_results.json'
    write(path, {'schema_version': 2, 'gates': gates, 'permissions': granted,
                 'gsplat_allowed': {'exploratory': granted['exploratory_training'] == 'PASS',
                                    'evaluated': granted['evaluated_training'] == 'PASS'},
                 'j1_passed': quality['j1_passed'],
                 'unknown_policy': 'No UNKNOWN permission is accepted as PASS',
                 'exploratory_definition': 'may use all 13 panoramas; no held-out metric may be reported',
                 'evaluated_definition': 'frozen AUTO-05 partition; train-only losses and initialization',
                 'pending_before_training': [
                     'training launcher must call split.verify_train_files again before reading inputs',
                     'gsplat adapter must recompute point colors from train observations and test it']})
    return [path]


def split_report(run):
    run.require('split_gates')
    split = read(run.path / 'split.json')
    gates = read(run.path / 'gate_results.json')
    lines = [f"# Partition AUTO-05 — {run.run_id}", '',
             f"Statut : **{split['status']}**. Méthode `{split['method']}`, sans tirage aléatoire.",
             f"Sources : segmentation `{split['semantic_run']}`, SfM masqué `{split['sfm_run']}` "
             '(artefacts importés après vérification des empreintes, non réexécutés).',
             f"Commit : `{split['git_commit']}`. Empreinte des poses : `{split['poses_sha256']}`.",
             f"Empreinte de la partition : `{split.get('partition_sha256')}`.", '',
             '| Ensemble | Panoramas |', '|---|---|']
    for name in SETS:
        lines.append(f"| {name} | {', '.join(split.get(name) or []) or '—'} |")
    lines += ['', '![Disposition spatiale](split_layout.png)', '',
              'Disposition : centres reconstruits projetés sur leurs deux axes principaux. Les sommets '
              f"de l’enveloppe convexe restent en entraînement : {', '.join(split.get('hull_vertices') or [])}.",
              '', '## Contrôles automatiques', '', '| Contrôle | Résultat | Détail |', '|---|---|---|']
    for c in split['checks'] + [g for g in gates['gates'] if g not in split['checks']]:
        evidence = json.dumps(c['evidence'], ensure_ascii=False) if not isinstance(c['evidence'], str) else c['evidence']
        lines.append(f"| {c['name']} | {c['result']} | {evidence} |")
    lines += ['', '## Autorisations', '']
    lines += [f'- {k} : {v}' for k, v in gates['permissions'].items()]
    lines += [f"- j1_passed : {gates['j1_passed']} (non modifié)", '', '## Protocole d’évaluation', '']
    lines += [f'- {k} : {v}' for k, v in split['protocol'].items()]
    lines += ['', '## Limites restantes', '',
              '- Les poses des vues réservées proviennent du SfM conjoint sur les 13 panoramas.',
              '- Positions des points d’initialisation triangulées avec toutes les vues ; couleurs à '
              'recalculer depuis l’entraînement seul.',
              '- Échelle arbitraire, repère non aligné sur la verticale ; distances relatives seulement.',
              '- Aucun seuil de qualité de rendu n’est défini ; navigation et livraison produit : UNKNOWN.',
              '- Paramètres structurels de partition non calibrés (fractions, voisinage).', '']
    path = run.path / 'split_report.md'
    path.write_text('\n'.join(lines))
    return [path]
