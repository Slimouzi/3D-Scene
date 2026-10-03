"""CPU diagnostic stages. No Gaussian or benchmark artifacts are fabricated."""
from collections import Counter, defaultdict
from pathlib import Path
import shutil
import numpy as np
import cv2
from PIL import Image, ImageDraw
import pycolmap as pc
from pycolmap import panorama
from .geometry import homogeneous, project, ownership
from .segmentation import fusion
from .segmentation.faces import prepare_faces
from .storage import digest, now, read, write


def rotations():
    return panorama.get_virtual_rotations(4, (-35., 0., 35.))


def camera(run):
    return panorama.create_virtual_camera(pano_width=run.config['erp_width'],
        pano_height=run.config['erp_width'] // 2, hfov_deg=90., vfov_deg=90.)


def audit(run):
    records = []
    for path in run.sources:
        with Image.open(path) as image:
            image.load()
            if image.width != image.height * 2:
                raise ValueError(f'{path.name}: expected 2:1 equirectangular image')
            exif = image.getexif()
            if exif.get(274, 1) != 1:
                raise ValueError(f'{path.name}: unsupported EXIF rotation')
            for mask in run.mask_paths.get(path.stem, {}).values():
                with Image.open(mask) as m:
                    if m.size != image.size:
                        raise ValueError(f'{mask.name}: mask must match original dimensions')
            records.append({'panorama_id': path.stem, 'path': path.name,
                            'sha256': run.provenance['inputs'][path.name],
                            'width': image.width, 'height': image.height,
                            'bytes': path.stat().st_size, 'model': exif.get(272),
                            'mask_review': 'unreviewed',
                            'provided_masks': list(run.mask_paths.get(path.stem, {}))})
    if len({r['sha256'] for r in records}) != len(records):
        raise ValueError('Duplicate source content: remove duplicated captures')
    write(run.path / 'capture.json', {'schema_version': 1, 'images': records,
        'warning': 'Geometry and RGB exclusions require manual review. No automatic semantic masks.'})
    return [run.path / 'capture.json']


def seg_faces(run):
    return prepare_faces(run, rotations())


def seg_trial(run):
    if run.config.get('auto_mask_backend') != 'sam3':
        raise RuntimeError('seg-trial requires "auto_mask_backend": "sam3" in the config')
    return fusion.trial(run)


def auto_mask(run):
    run.require('audit')
    backend = run.config.get('auto_mask_backend', 'unavailable')
    if run.config.get('semantic_run'):
        return fusion.import_semantics(run)
    if backend == 'sam3':
        return fusion.all_panoramas(run)
    records = []
    for source in run.sources:
        records.append({'panorama_id': source.stem, 'status': 'UNKNOWN',
                        'geometry_valid': None, 'rgb_valid': None,
                        'semantic_unknown': 'all_pixels',
                        'mask_artifacts': {'geometry_valid': None, 'rgb_valid': None,
                                           'semantic_unknown': None},
                        'backend': backend,
                        'reason': 'No qualified segmentation backend is configured'
                                  if backend == 'unavailable'
                                  else 'Backend adapter not implemented'})
    manifest = {'schema_version': 1, 'status': 'UNKNOWN', 'backend': backend,
                'prompts': ['mirror', 'window', 'glass door', 'glass partition',
                            'person', 'animal', 'tripod', 'floor', 'wall', 'sofa',
                            'chair', 'table'],
                'error_policy': 'fail_closed_unknown', 'panoramas': records,
                'accepted_masks': False,
                'next_action': 'Install and qualify SAM 3 or an approved alternative; do not use white fallback masks.'}
    path = run.path / 'semantic_masks.json'
    write(path, manifest)
    return [path]


def save_image(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def prepare(run):
    run.require('audit')
    width = run.config['erp_width']
    size = width // 4
    rr = rotations()
    own = [ownership(r, rr, size, i) for i, r in enumerate(rr)]
    views, outputs = [], []
    sheet = Image.new('RGB', (6 * 192, len(run.sources) * 216), '#18202b')
    draw = ImageDraw.Draw(sheet)
    for pano_i, path in enumerate(run.sources):
        print(f'prepare {path.name}', flush=True)
        with Image.open(path) as raw:
            original = raw.convert('RGB')
            erp = original.resize((width, width // 2), Image.Resampling.LANCZOS)
            erp_path = run.path / 'prepare/erp' / (path.stem + '.png')
            save_image(erp_path, np.asarray(erp))
            outputs.append(erp_path)
            source_masks = {}
            for kind in ('geometry', 'rgb'):
                source_path = run.mask_paths.get(path.stem, {}).get(kind)
                if source_path:
                    with Image.open(source_path) as m:
                        # Conservatively exclude the Lanczos resizing footprint.
                        src = np.asarray(m.convert('L'))
                        radius = int(np.ceil(3 * original.width / width)) + 2
                        padded = np.pad(src, ((radius, radius), (radius, radius)), mode='wrap')
                        padded[:radius] = padded[radius]
                        padded[-radius:] = padded[-radius - 1]
                        eroded = cv2.erode(padded, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
                        source_masks[kind] = eroded[radius:-radius, radius:-radius]
            for i, rotation in enumerate(rr):
                name = f'pano_camera{i}/{path.stem}.png'
                image_path = run.path / 'prepare/images' / name
                pixels = project(erp, rotation, size)
                save_image(image_path, pixels)
                outputs.append(image_path)
                masks = {}
                for kind in ('rgb', 'geometry'):
                    if kind in source_masks:
                        valid = project(source_masks[kind], rotation, size, mask=True)
                    else:
                        valid = np.full((size, size), 255, dtype=np.uint8)
                    masks[kind] = valid
                # Geometry never admits pixels excluded for RGB.
                feature_mask = own[i] & masks['geometry'] & masks['rgb']
                paths = {}
                for kind, value in [('rgb', masks['rgb']), ('geometry', masks['geometry']),
                                    ('features', feature_mask)]:
                    p = run.path / f'prepare/masks/{kind}' / (name + '.png')
                    save_image(p, value)
                    outputs.append(p)
                    paths[kind] = str(p.relative_to(run.path))
                views.append({'view_id': f'{path.stem}_sfm_{i:02d}', 'panorama_id': path.stem,
                    'role': 'sfm', 'image': str(image_path.relative_to(run.path)), 'sfm_name': name,
                    'masks': paths, 'width': size, 'height': size,
                    'K': [[size / 2, 0, size / 2], [0, size / 2, size / 2], [0, 0, 1]],
                    'T_face_from_panorama': homogeneous(rotation).tolist(),
                    'feature_valid_fraction': float(np.mean(feature_mask > 0))})
                if i < 6:
                    thumb = pixels.copy()
                    thumb[feature_mask == 0] //= 4
                    sheet.paste(Image.fromarray(thumb).resize((192, 192)), (i * 192, pano_i * 216))
            draw.text((5, pano_i * 216 + 196), f'{path.stem} — 6/12 vues, masque features', fill='white')
    preview = run.path / 'prepare/masks_preview.jpg'
    sheet.save(preview, quality=85)
    write(run.path / 'views.json', {'schema_version': 1, 'convention': 'x-right y-down z-forward; pixel centers +0.5', 'views': views})
    write(run.path / 'rig.json', {'schema_version': 1, 'reference_face': 0,
        'T_panorama_from_rig': homogeneous(rr[0].T).tolist(),
        'fixed_intrinsics': True, 'fixed_sensor_from_rig': True,
        'cameras': [{'prefix': f'pano_camera{i}/',
                     'T_face_from_rig': homogeneous(r @ rr[0].T).tolist()} for i, r in enumerate(rr)]})
    return outputs + [preview, run.path / 'views.json', run.path / 'rig.json']


def features(run):
    run.require('prepare')
    database = run.path / 'sfm/database_extraction.db'
    if database.exists():
        raise RuntimeError('Incomplete feature database exists; use a new run-id (never delete it automatically)')
    database.parent.mkdir(parents=True, exist_ok=True)
    cam = camera(run)
    opts = pc.FeatureExtractionOptions(use_gpu=False, num_threads=run.config['num_threads'])
    opts.sift.max_num_features = run.config['max_features']
    pc.extract_features(database, run.path / 'prepare/images',
        camera_mode=pc.CameraMode.PER_FOLDER,
        reader_options=pc.ImageReaderOptions(mask_path=run.path / 'prepare/masks/features',
            camera_model=cam.model_name, camera_params=cam.params_to_string()), extraction_options=opts)
    rig = panorama.create_pano_rig_config(rotations())
    for c in rig.cameras:
        c.camera = cam
    with pc.Database.open(database) as db:
        pc.apply_rig_config([rig], db)
        summary = {'images': db.num_images(), 'keypoints': db.num_keypoints()}
    write(run.path / 'sfm/features.json', summary)
    return [run.path / 'sfm/features.json', database]


def matching(run):
    run.require('features')
    extraction_database = run.path / 'sfm/database_extraction.db'
    database = run.path / 'sfm/database_matching.db'
    matched_database = run.path / 'sfm/database_matched.db'
    shutil.copy2(extraction_database, database)
    opts = pc.FeatureMatchingOptions(use_gpu=False, num_threads=run.config['num_threads'],
        rig_verification=True, skip_image_pairs_in_same_frame=True)
    pc.match_exhaustive(database, matching_options=opts)
    with pc.Database.open(database) as db:
        images = {im.image_id: Path(im.name).stem for im in db.read_all_images()}
        pair_ids, counts = db.read_two_view_geometry_num_inliers()
        edges = Counter()
        same_center = 0
        for pair, count in zip(pair_ids, counts):
            a, b = pc.pair_id_to_image_pair(pair)
            if images[a] == images[b]:
                same_center += 1
            elif count:
                edges[tuple(sorted((images[a], images[b])))] += int(count)
    if same_center:
        raise RuntimeError('Same-center pairs survived matching')
    shutil.copy2(database, matched_database)
    write(run.path / 'sfm/matches.json', {'same_center_verified_pairs': same_center,
        'edges': [{'source': a, 'target': b, 'inlier_observations': n} for (a, b), n in sorted(edges.items())]})
    return [run.path / 'sfm/matches.json', matched_database]


def mapping(run):
    run.require('matching')
    folder = run.path / 'sfm/sparse'
    if folder.exists():
        raise RuntimeError('Mapping output already exists; use a new run-id to retain previous attempt')
    folder.mkdir(parents=True)
    archived_database = run.path / 'sfm/database_matched.db'
    mapping_database = run.path / 'sfm/database_mapping_working.db'
    archived_digest = digest(archived_database)
    shutil.copy2(archived_database, mapping_database)
    opts = pc.IncrementalPipelineOptions(num_threads=run.config['num_threads'],
        random_seed=run.config['seed'], ba_refine_sensor_from_rig=False,
        ba_refine_focal_length=False, ba_refine_principal_point=False, ba_refine_extra_params=False,
        max_runtime_seconds=run.config['mapping_max_seconds'])
    opts.mapper.abs_pose_refine_focal_length = False
    opts.mapper.abs_pose_refine_extra_params = False
    models = pc.incremental_mapping(mapping_database, run.path / 'prepare/images', folder, opts)
    if digest(archived_database) != archived_digest:
        raise RuntimeError('Mapping mutated the archived matched database')
    write(run.path / 'sfm/models.json', {'components': sorted(models), 'count': len(models)})
    return [run.path / 'sfm/models.json'] + sorted(p for p in folder.rglob('*') if p.is_file())


def component_center_path(records, expected, component_id):
    selected = {record['panorama_id']: record for record in records
                if record.get('component') == component_id and record['status'] == 'registered'}
    order = [panorama_id for panorama_id in expected if panorama_id in selected]
    centers = np.asarray([selected[panorama_id]['center_world'] for panorama_id in order],
                         dtype=float).reshape((-1, 3))
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1) if len(centers) >= 2 else np.array([])
    return {'component': component_id, 'panorama_order': order,
            'coordinate_y_range': ([float(centers[:, 1].min()), float(centers[:, 1].max())]
                                   if len(centers) else None),
            'coordinate_y_span': float(np.ptp(centers[:, 1])) if len(centers) else None,
            'consecutive_distance_range': ([float(steps.min()), float(steps.max())]
                                           if len(steps) else None),
            'consecutive_distance_median': float(np.median(steps)) if len(steps) else None,
            'units': 'arbitrary; reconstruction coordinates only',
            'projection_axes': 'X/Z in the reconstruction frame; equal aspect required',
            'navigation_validated': False,
            'review_reason': 'No floor, obstacle or free-space annotation is available'}


def diagnose(run):
    run.require('mapping')
    models = read(run.path / 'sfm/models.json')['components']
    expected = [p.stem for p in run.sources]
    components, records, errors_all = [], [], []
    for model_id in models:
        rec = pc.Reconstruction(run.path / 'sfm/sparse' / str(model_id))
        pano_images = defaultdict(list)
        for im in rec.images.values():
            if im.has_pose:
                pano_images[Path(im.name).stem].append(im)
        errors = []
        for pano_id, images in sorted(pano_images.items()):
            # The reference sensor is face 0, not the ERP coordinate frame.
            frame_pose = images[0].frame.rig_from_world.matrix()
            T_pano_from_world = homogeneous(rotations()[0].T) @ np.vstack((frame_pose, [0, 0, 0, 1]))
            center = np.linalg.inv(T_pano_from_world)[:3, 3]
            residuals = []
            for im in images:
                R = im.cam_from_world().rotation.matrix()
                t = im.cam_from_world().translation
                for pt in im.points2D:
                    if pt.has_point3D():
                        xyz = R @ rec.points3D[pt.point3D_id].xyz + t
                        if xyz[2] > 0:
                            xy = im.camera.img_from_cam(xyz)
                            residuals.append(float(np.linalg.norm(xy - pt.xy)))
            errors.extend(residuals)
            records.append({'panorama_id': pano_id, 'component': model_id, 'status': 'registered',
                'T_panorama_from_world': T_pano_from_world.tolist(), 'center_world': center.tolist(),
                'observations': len(residuals),
                'median_reprojection_px': float(np.median(residuals)) if residuals else None})
        angles, track_lengths = [], []
        for pt in rec.points3D.values():
            centers = {Path(rec.images[e.image_id].name).stem:
                       rec.images[e.image_id].projection_center() for e in pt.track.elements}
            track_lengths.append(len(centers))
            if len(centers) >= 2:
                dirs = pt.xyz - np.array(list(centers.values()))
                dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-15)
                angles.append(float(np.rad2deg(np.arccos(np.clip(np.min(dirs @ dirs.T), -1, 1)))))
        components.append({'id': model_id, 'panoramas': sorted(pano_images), 'points3D': rec.num_points3D(),
            'median_reprojection_px': float(np.median(errors)) if errors else None,
            'median_track_distinct_panoramas': float(np.median(track_lengths)) if track_lengths else None,
            'median_max_triangulation_angle_deg': float(np.median(angles)) if angles else None})
        errors_all.extend(errors)
    largest = max(components, key=lambda c: len(c['panoramas']), default=None)
    registered = {r['panorama_id'] for r in records}
    records += [{'panorama_id': p, 'component': None, 'status': 'unregistered',
                 'T_panorama_from_world': None} for p in expected if p not in registered]
    component_paths = [component_center_path(records, expected, component['id'])
                       for component in components]
    center_path = (next((path for path in component_paths
                         if path['component'] == largest['id']),
                        component_center_path([], expected, None))
                   if largest else component_center_path([], expected, None))
    write(run.path / 'poses.json', {'schema_version': 1, 'units': 'arbitrary',
        'convention': 'T_panorama_from_world; column vectors; camera x-right y-down z-forward',
        'components_have_independent_world_frames': True, 'poses': records})
    enough = largest is not None and len(largest['panoramas']) >= min(12, len(expected))
    # Numeric registration alone cannot qualify a navigation route or room coverage.
    quality = {'schema_version': 1, 'kind': 'diagnostic',
        'decision': 'requires_spatial_review' if enough else 'insufficient_registration',
        'j1_passed': False, 'registered_unique_panoramas': len(registered), 'input_panoramas': len(expected),
        'components': components, 'largest_component': largest['id'] if largest else None,
        'center_path': center_path, 'center_paths_by_component': component_paths,
        'coverage_by_zone': {'value': None, 'reason': 'Navigation route and spatial coverage not reviewed'},
        'mask_review': {'value': None, 'reason': 'Semantic exclusions not reviewed'},
        'heldout_metrics': {'value': None, 'reason': 'All-input diagnostic; no benchmark split or training'},
        'next_action': 'Review centers, matches, mirrors and coverage before freezing the benchmark split.'}
    write(run.path / 'quality.json', quality)
    return [run.path / 'poses.json', run.path / 'quality.json']


def report(run):
    run.require('diagnose')
    q = read(run.path / 'quality.json')
    visualization = run.path / 'diagnostic_centers.png'
    canvas = Image.new('RGB', (1200, 800), '#f4f1e8')
    draw = ImageDraw.Draw(canvas)
    component_id = q['largest_component']
    rec = (pc.Reconstruction(run.path / 'sfm/sparse' / str(component_id))
           if component_id is not None else None)
    centers = {pose['panorama_id']: np.asarray(pose['center_world'])
               for pose in read(run.path / 'poses.json')['poses']
               if pose['component'] == component_id and pose['status'] == 'registered'}
    points = (np.asarray([point.xyz for point in rec.points3D.values()], dtype=float).reshape((-1, 3))
              if rec else np.empty((0, 3), dtype=float))
    center_values = np.asarray([value[[0, 2]] for value in centers.values()], dtype=float).reshape((-1, 2))
    samples = np.vstack((points[:, [0, 2]], center_values)) if len(points) or len(center_values) else np.zeros((1, 2))
    low = samples.min(axis=0)
    high = samples.max(axis=0)
    span = np.maximum(high - low, 1e-9)

    scale = min(1040 / span[0], 640 / span[1])
    origin = np.array([600., 400.]) - ((low + high) / 2) * np.array([scale, -scale])

    def pixel(value):
        return (int(origin[0] + value[0] * scale), int(origin[1] - value[2] * scale))

    for point in points[::max(1, len(points) // 3000)]:
        x, y = pixel(point)
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill='#8c8c87')
    for panorama_id, center in sorted(centers.items()):
        x, y = pixel(center)
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill='#c83d2f', outline='#641d18', width=2)
        draw.text((x + 10, y - 8), panorama_id, fill='#1d252c')
    draw.text((80, 30), 'Diagnostic poses: centres (rouge) et nuage sparse (gris)', fill='#1d252c')
    draw.text((80, 55), 'Projection X/Z du repere de reconstruction; echelle arbitraire', fill='#4d5961')
    canvas.save(visualization)
    text = ['# Diagnostic initial du salon', '', f"Décision : **{q['decision']}**. J1 non validé.",
            f"Panoramas recalés : {q['registered_unique_panoramas']}/{q['input_panoramas']}.", '',
            '| Composante | Panoramas | Points 3D | Résidu médian (px) | Angle médian maximal |',
            '|---|---:|---:|---:|---:|']
    for c in q['components']:
        text.append(f"| {c['id']} | {len(c['panoramas'])} | {c['points3D']} | {c['median_reprojection_px']} | {c['median_max_triangulation_angle_deg']} |")
    text += ['', 'Ces poses proviennent de toutes les images : elles ne constituent pas un benchmark indépendant.',
        'Les masques sémantiques, la couverture et le parcours restent à contrôler. Les composantes ont des repères indépendants.',
        f"Centres : coordonnée Y sur {q['center_path']['coordinate_y_span'] if q['center_path']['coordinate_y_span'] is not None else 'non mesurable'} unité, repère non aligné sur une verticale physique; navigation non validée.",
        '![Centres et nuage sparse](diagnostic_centers.png)',
        'Aucun Gaussian Splatting entraîné à ce stade. Prochaine étape : revue spatiale puis partition par panorama.', '']
    path = run.path / 'report.md'
    path.write_text('\n'.join(text))
    return [path, visualization]


def partition(run):
    run.require('diagnose')
    run.require('auto_gates')
    gates = read(run.path / 'gate_results.json')
    if run.config.get('semantic_run'):
        # A masked SfM experiment must not inherit the provisional pre-mask partition.
        path = run.path / 'split.json'
        write(path, {'schema_version': 1, 'status': 'not_proposed', 'unit': 'panorama',
                     'train': None, 'validation': None, 'test': None,
                     'reason': 'Deterministic spatial partition (AUTO-05) not implemented; '
                               'the provisional pre-mask split is not reused',
                     'research_training': gates['permissions']['research_training'],
                     'proposed_at': None, 'frozen_at': None})
        return [path]
    expected = [p.stem for p in run.sources]
    proposed = {'train': ['R0010004', 'R0010005', 'R0010006', 'R0010008', 'R0010009',
                          'R0010010', 'R0010012', 'R0010013', 'R0010015', 'R0010016'],
                'validation': ['R0010011'], 'test': ['R0010007', 'R0010014']}
    assigned = [p for values in proposed.values() for p in values]
    if sorted(assigned) != sorted(expected):
        raise RuntimeError('Proposed panorama partition does not cover the input exactly')
    quality = read(run.path / 'quality.json')
    can_freeze = gates['permissions']['research_training'] == 'PASS'
    split = {'schema_version': 1, 'status': 'frozen' if can_freeze else 'provisional', 'unit': 'panorama',
             'protocol': 'strict_holdout', 'poses': 'exploratory_all_input',
             'train': proposed['train'], 'validation': proposed['validation'],
             'test': proposed['test'],
             'justification': 'Initial R&D proposal; spatial review is still required',
             'spatial_review_required': not can_freeze,
             'navigation_validated': quality['center_path']['navigation_validated'],
             'proposed_at': now(), 'frozen_at': now() if can_freeze else None,
             'freeze_reason': 'All research-training gates passed' if can_freeze else gates['decision_reason']}
    path = run.path / 'split.json'
    write(path, split)
    return [path]


def auto_gates(run):
    run.require('diagnose')
    masks = read(run.path / 'semantic_masks.json')
    quality = read(run.path / 'quality.json')
    registered_pass = quality['registered_unique_panoramas'] == quality['input_panoramas']
    gates = [
        {'name': 'source_audit', 'result': 'PASS', 'evidence': 'audit completed'},
        {'name': 'semantic_masks', 'result': 'PASS' if masks['accepted_masks'] else
         'FAIL' if masks.get('status') == 'REJECTED' else 'UNKNOWN',
         'evidence': masks['next_action']},
        {'name': 'pose_registration', 'result': 'PASS' if registered_pass else 'FAIL',
         'evidence': f"{quality['registered_unique_panoramas']}/{quality['input_panoramas']} panoramas registered"},
        {'name': 'navigation_segments', 'result': 'UNKNOWN',
         'evidence': quality['center_path']['review_reason']},
    ]
    extra = semantic_gates(run, masks, quality)
    # Partition freeze and gsplat stay blocked until every segmentation gate passes.
    training = all(g['result'] == 'PASS' for g in gates[:3] + extra)
    gates += extra
    permissions = {
        'panorama_delivery': 'PASS',
        'research_training': 'PASS' if training else 'UNKNOWN',
        'guided_3d_navigation': 'UNKNOWN', 'free_3d_navigation': 'UNKNOWN',
        'product_delivery': 'UNKNOWN'}
    path = run.path / 'gate_results.json'
    write(path, {'schema_version': 1, 'decision': 'accept_restricted',
                 'decision_reason': 'Unknown controls restrict permissions without requesting recapture',
                 'gates': gates, 'permissions': permissions,
                 'gsplat_allowed': permissions['research_training'] == 'PASS',
                 'unknown_policy': 'No UNKNOWN permission is accepted as PASS',
                 'recapture_required': False})
    return [path]


def semantic_gates(run, masks, quality):
    """Preconditions for freezing the partition and launching gsplat."""
    names = ('semantic_all_panoramas', 'glass_mirror_multiview',
             'unknown_glass_constraints_recorded', 'validation_hashes')
    if masks.get('backend') != 'sam3':
        return [{'name': n, 'result': 'UNKNOWN', 'evidence': 'SAM 3 backend not configured'}
                for n in names]
    processed = [p for p in masks['panoramas'] if p['status'] == 'OK']
    multiview = masks['multiview_panoramas']
    navigation = read(run.path / 'navigation_constraints.json')
    # Recorded, not validated: guided/free navigation permissions stay UNKNOWN.
    propagated = ({p['panorama_id'] for p in navigation['panoramas']
                   if p['labels'] or p['all_pixels_unknown']}
                  if navigation.get('status') == 'advisory' and not navigation['navigation_validated']
                  else set())
    hashed = read(run.path / 'mask_provenance.json')['model_and_code_hashed']
    return [
        {'name': names[0], 'result': 'PASS' if len(processed) == quality['input_panoramas'] else 'FAIL',
         'evidence': f"{len(processed)}/{quality['input_panoramas']} panoramas segmented without failure"},
        {'name': names[1], 'result': 'PASS' if all(len(v) >= 2 for v in multiview.values()) else 'FAIL',
         'evidence': {g: f'{len(v)} panoramas' for g, v in multiview.items()}},
        {'name': names[2], 'result': 'PASS' if propagated == {p.stem for p in run.sources} else 'FAIL',
         'evidence': f"advisory constraints, consumed_by={navigation['consumed_by']}; "
                     'navigation not validated'},
        {'name': names[3], 'result': 'PASS' if hashed else 'FAIL',
         'evidence': 'clean git commit, checkpoint SHA-256, qualified GPU environment and '
                     'code digests in mask_provenance.json'},
    ]
