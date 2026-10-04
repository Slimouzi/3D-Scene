import random
import tempfile
import unittest
from pathlib import Path
import numpy as np
from theta_pipeline import split
from theta_pipeline.storage import read, write

# 13 stations: 7 on an outer ring (hull) and 6 inside, on a slightly tilted plane.
ANGLES = np.deg2rad(np.arange(7) * 360 / 7)
CENTERS = {f'P{i:02d}': [3 * np.cos(a), .05 * np.sin(2 * a), 2 * np.sin(a)] for i, a in enumerate(ANGLES)}
CENTERS.update({f'P{7 + i:02d}': [1.2 * np.cos(a), .02, .8 * np.sin(a)]
                for i, a in enumerate(np.deg2rad(np.arange(6) * 60 + 15))})


def neighbour_edges(centers, k=4):
    ids = sorted(centers)
    edges = set()
    for i in ids:
        near = sorted(ids, key=lambda j: np.linalg.norm(np.subtract(centers[i], centers[j])))[1:k + 1]
        edges |= {tuple(sorted((i, j))) for j in near}
    return sorted(edges)


def views(panoramas, faces=12):
    return [{'panorama_id': p, 'sfm_name': f'pano_camera{f}/{p}.png',
             'image': f'prepare/images/pano_camera{f}/{p}.png', 'width': 8, 'height': 8,
             'T_face_from_panorama': np.eye(4).tolist()} for p in panoramas for f in range(faces)]


def pano_color(pano):
    k = sorted(CENTERS).index(pano)
    return (10 * k, 255 - 10 * k, 7 * k)


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.edges = neighbour_edges(CENTERS)
        self.proposal = split.propose(CENTERS, self.edges, split.DEFAULTS)
        self.sets = {name: self.proposal[name] for name in split.SETS}

    def frozen(self):
        record = {**self.sets, 'poses_sha256': 'p' * 64, 'partition_sha256': split.partition_sha256(self.sets)}
        return record

    def test_partition_is_valid_and_hull_stays_in_train(self):
        self.assertEqual(self.proposal['status'], 'proposed')
        self.assertEqual([len(self.sets[n]) for n in split.SETS], [10, 1, 2])
        self.assertTrue(set(self.proposal['hull_vertices']) <= set(self.sets['train']))
        self.assertEqual(set(self.proposal['hull_vertices']), {f'P{i:02d}' for i in range(7)})
        checks = split.validate(self.frozen(), sorted(CENTERS), 'p' * 64, views(CENTERS))
        self.assertEqual({c['result'] for c in checks}, {'PASS'}, checks)

    def test_reproducible_under_input_order_and_scale(self):
        for seed in range(5):
            items = list(CENTERS.items())
            random.Random(seed).shuffle(items)
            edges = [e[::-1] if k % 2 else e for k, e in enumerate(self.edges)]
            random.Random(seed).shuffle(edges)
            result = split.propose(dict(items), edges, split.DEFAULTS)
            self.assertEqual({n: result[n] for n in split.SETS}, self.sets)
        scaled = {k: list(np.multiply(v, 37.5) + [4, -2, 9]) for k, v in CENTERS.items()}
        result = split.propose(scaled, self.edges, split.DEFAULTS)
        self.assertEqual({n: result[n] for n in split.SETS}, self.sets)

    def test_sets_disjoint_and_faces_grouped(self):
        union = [p for n in split.SETS for p in self.sets[n]]
        self.assertEqual(len(union), len(set(union)))
        faces = split.face_assignment(self.sets, views(CENTERS))
        for pano in CENTERS:
            owners = {faces[f'pano_camera{f}/{pano}.png'] for f in range(12)}
            self.assertEqual(len(owners), 1)
        overlapping = {**self.sets, 'test': self.sets['test'] + [self.sets['train'][0]]}
        record = {**overlapping, 'poses_sha256': 'p' * 64, 'partition_sha256': split.partition_sha256(overlapping)}
        result = {c['name']: c['result'] for c in split.validate(record, sorted(CENTERS), 'p' * 64, views(CENTERS))}
        self.assertEqual(result['disjoint_sets'], 'FAIL')
        self.assertEqual(result['faces_grouped_by_panorama'], 'FAIL')

    def test_empty_or_incomplete_partitions_rejected(self):
        for broken in ({**self.sets, 'validation': []},
                       {**self.sets, 'train': self.sets['train'][1:]}):
            record = {**broken, 'poses_sha256': 'p' * 64, 'partition_sha256': split.partition_sha256(broken)}
            results = {c['result'] for c in split.validate(record, sorted(CENTERS), 'p' * 64, views(CENTERS))}
            self.assertIn('FAIL', results)
        few = {k: CENTERS[k] for k in sorted(CENTERS)[:4]}
        self.assertEqual(split.propose(few, neighbour_edges(few, 2), split.DEFAULTS)['status'], 'unknown')

    def test_incompatible_data_is_unknown(self):
        # Interior stations with no covisibility edge to train cannot be held out.
        hull_only = [e for e in self.edges if e[0] < 'P07' and e[1] < 'P07']
        result = split.propose(CENTERS, hull_only, split.DEFAULTS)
        self.assertEqual(result['status'], 'unknown')
        self.assertNotIn('train', result)

    def test_modified_poses_or_partition_after_freeze_detected(self):
        record = self.frozen()
        names = lambda checks: {c['name']: c['result'] for c in checks}
        self.assertEqual(names(split.validate(record, sorted(CENTERS), 'q' * 64, views(CENTERS)))['poses_unchanged'],
                         'FAIL')
        tampered = {**record, 'train': record['train'][1:], 'test': record['test'] + record['train'][:1]}
        self.assertEqual(names(split.validate(tampered, sorted(CENTERS), 'p' * 64, views(CENTERS)))['partition_hash'],
                         'FAIL')

    def test_evaluated_training_refused_without_valid_partition(self):
        granted = split.permissions(sources_ok=True, poses_ok=True, split_ok=False, separation_ok=False)
        self.assertEqual(granted['exploratory_training'], 'PASS')
        self.assertEqual(granted['evaluated_training'], 'UNKNOWN')
        self.assertEqual(split.permissions(True, True, True, False)['evaluated_training'], 'UNKNOWN')
        self.assertEqual(split.permissions(False, True, True, True)['evaluated_training'], 'UNKNOWN')
        granted = split.permissions(True, True, True, True)
        self.assertEqual(granted['evaluated_training'], 'PASS')
        for key in ('guided_3d_navigation', 'free_3d_navigation', 'product_delivery'):
            self.assertEqual(granted[key], 'UNKNOWN')

    def manifest(self, tracks, root=Path('/runs')):
        images, masks = split.expected_train_files(self.sets, views(CENTERS), 'sfm-run', 'sem-run')
        recorded = {rel: 'd' * 64 for _, rel in images} | {rel: 'd' * 64 for rel in masks.values()}
        return split.train_inputs(self.sets, views(CENTERS), tracks, 'sfm-run', 'sem-run', root, recorded)

    def separation(self, inputs, tracks):
        return {c['name']: c['result'] for c in
                split.separation_checks(self.sets, inputs, views(CENTERS), tracks, 'sfm-run', 'sem-run')}

    def test_training_inputs_contain_no_heldout_data(self):
        held = self.sets['validation'] + self.sets['test']
        train = self.sets['train']
        tracks = {1: [train[0], train[1]], 2: [train[0], held[0]], 3: [held[0], held[1]], 4: [held[2]]}
        inputs = self.manifest(tracks)
        self.assertEqual(inputs['init_points']['excluded_heldout_only_point_ids'], [3, 4])
        self.assertEqual(inputs['init_points']['eligible_point_ids'], [1, 2])
        self.assertFalse(any(h in e['path'] for e in inputs['images'] for h in held))
        self.assertTrue(all(e['sha256'] and e['resolved'].startswith('/') for e in inputs['images']))
        self.assertEqual(set(self.separation(inputs, tracks).values()), {'PASS'})

    def test_separation_rejects_forged_manifests(self):
        train, held = self.sets['train'], self.sets['validation'] + self.sets['test']
        tracks = {1: [train[0]], 2: [held[0]]}
        inputs = self.manifest(tracks)
        unknown_image = {**inputs, 'images': inputs['images'] + [
            {'panorama_id': train[0], 'path': 'elsewhere/unknown.png', 'resolved': '/x', 'sha256': 'd' * 64}]}
        self.assertEqual(self.separation(unknown_image, tracks)['train_images_exact'], 'FAIL')
        heldout_image = {**inputs, 'images': inputs['images'][1:] + [
            {**inputs['images'][0], 'path': f'sfm-run/prepare/images/pano_camera0/{held[0]}.png'}]}
        self.assertEqual(self.separation(heldout_image, tracks)['train_images_exact'], 'FAIL')
        swapped_mask = {**inputs, 'masks': {**inputs['masks'], train[0]: {
            **inputs['masks'][train[0]], 'path': f'sem-run/segmentation/fused/{held[0]}/appearance_mask.png'}}}
        self.assertEqual(self.separation(swapped_mask, tracks)['train_masks_match_panoramas'], 'FAIL')
        extra_mask = {**inputs, 'masks': {**inputs['masks'], held[0]: inputs['masks'][train[0]]}}
        self.assertEqual(self.separation(extra_mask, tracks)['train_masks_match_panoramas'], 'FAIL')
        no_init = {**inputs, 'init_points': {**inputs['init_points'], 'eligible_point_ids': []}}
        self.assertEqual(self.separation(no_init, tracks)['init_points_have_train_support'], 'FAIL')
        empty = self.manifest({2: [held[0]]})        # every point is held-out only
        self.assertEqual(self.separation(empty, {2: [held[0]]})['init_points_have_train_support'], 'FAIL')
        unknown_point = {**inputs, 'init_points': {**inputs['init_points'], 'eligible_point_ids': [1, 99]}}
        self.assertEqual(self.separation(unknown_point, tracks)['init_points_have_train_support'], 'FAIL')
        self.assertEqual(self.separation({}, tracks)['train_images_exact'], 'FAIL')

    def test_training_files_verified_by_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            from theta_pipeline.storage import digest
            images, masks = split.expected_train_files(self.sets, views(CENTERS), 'sfm-run', 'sem-run')
            recorded = {}
            for rel in [r for _, r in images] + list(masks.values()):
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_bytes(rel.encode())
                recorded[rel] = digest(root / rel)
            inputs = split.train_inputs(self.sets, views(CENTERS), {1: self.sets['train'][:1]},
                                        'sfm-run', 'sem-run', root, recorded)
            self.assertEqual(split.verify_train_files(inputs, root)['result'], 'PASS')
            (root / images[0][1]).write_bytes(b'modified')
            result = split.verify_train_files(inputs, root)
            self.assertEqual(result['result'], 'FAIL')
            self.assertEqual(result['evidence']['changed'], [images[0][1]])
            (root / images[0][1]).unlink()
            self.assertEqual(split.verify_train_files(inputs, root)['evidence']['missing'], [images[0][1]])
            unrecorded = split.train_inputs(self.sets, views(CENTERS), {1: self.sets['train'][:1]},
                                            'sfm-run', 'sem-run', root, {})
            self.assertEqual(split.verify_train_files(unrecorded, root)['result'], 'FAIL')
            self.assertEqual(split.verify_train_files({}, root)['result'], 'FAIL')

    def test_layout_and_hash_are_canonical(self):
        reordered = {n: list(reversed(self.sets[n])) for n in split.SETS}
        self.assertEqual(split.partition_sha256(reordered), split.partition_sha256(self.sets))
        with tempfile.TemporaryDirectory() as temp:
            split.draw_layout({**self.proposal, 'status': 'frozen'}, Path(temp) / 'layout.png')
            self.assertTrue((Path(temp) / 'layout.png').stat().st_size > 0)
            write(Path(temp) / 'x.json', self.proposal)
            self.assertEqual(read(Path(temp) / 'x.json')['test'], self.sets['test'])


if __name__ == '__main__':
    unittest.main()


class SplitFixture(unittest.TestCase):
    """Synthetic ACCEPTED semantic run and masked SfM run, with real files and run.json hashes."""

    def setUp(self):
        from PIL import Image
        from theta_pipeline.storage import digest
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = root = Path(temp.name)
        (root / 'input').mkdir()
        for k, pano in enumerate(sorted(CENTERS)):
            image = np.zeros((64, 128, 3), np.uint8)
            image[0, 0] = k
            Image.fromarray(image).save(root / f'input/{pano}.png')
        base = {'schema_version': 1, 'kind': 'diagnostic', 'input': 'input', 'output': 'runs',
                'erp_width': 128, 'num_threads': 1, 'seed': 0, 'max_features': 128,
                'mapping_max_seconds': 10, 'masks': {}}
        sem = root / 'runs/sem'
        manifests = {}
        panoramas = []
        for pano in sorted(CENTERS):
            mask = sem / f'segmentation/fused/{pano}/geometry_mask.png'
            mask.parent.mkdir(parents=True)
            Image.fromarray(np.full((64, 128), 255, np.uint8)).save(mask)
            manifests[str(mask.relative_to(sem))] = digest(mask)
            appearance = mask.with_name('appearance_mask.png')
            Image.fromarray(np.full((64, 128), 255, np.uint8)).save(appearance)
            manifests[str(appearance.relative_to(sem))] = digest(appearance)
            panoramas.append({'panorama_id': pano, 'artifacts': {'geometry_mask': str(mask.relative_to(sem))}})
        for name in ('mask_consistency.json', 'mask_provenance.json', 'navigation_constraints.json'):
            write(sem / name, {'schema_version': 1})
        write(sem / 'semantic_masks.json', {'status': 'ACCEPTED', 'accepted_masks': True,
                                            'panoramas': panoramas})
        for name in ('semantic_masks.json', 'mask_consistency.json', 'mask_provenance.json',
                     'navigation_constraints.json'):
            manifests[name] = digest(sem / name)
        write(sem / 'run.json', {'stages': {'auto_mask': {'status': 'completed', 'artifacts': manifests}}})

        sfm = root / 'runs/sfm'
        write(sfm / 'poses.json', {'poses': [{'panorama_id': p, 'status': 'registered', 'component': 0,
                                              'center_world': c} for p, c in CENTERS.items()]})
        write(sfm / 'quality.json', {'largest_component': 0, 'j1_passed': False})
        write(sfm / 'views.json', {'views': views(CENTERS)})
        for view in views(CENTERS):
            (sfm / view['image']).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((8, 8, 3), pano_color(view['panorama_id']), np.uint8)).save(sfm / view['image'])
        write(sfm / 'sfm/matches.json', {'edges': [{'source': a, 'target': b, 'inlier_observations': 50}
                                                   for a, b in neighbour_edges(CENTERS)]})
        (sfm / 'sfm/sparse/0').mkdir(parents=True)
        (sfm / 'sfm/sparse/0/points3D.bin').write_bytes(b'synthetic')
        art = lambda *rels: {'status': 'completed', 'artifacts': {r: digest(sfm / r) for r in rels}}
        self.base, self.sem_digest = base, digest(sem / 'semantic_masks.json')
        inputs = {p.name: digest(p) for p in sorted((root / 'input').iterdir())}
        write(sfm / 'run.json', {'fingerprint': 'f' * 64, 'provenance': {
            'semantic_run': {'run': 'sem', 'semantic_masks_sha256': self.sem_digest}, 'inputs': inputs},
            'stages': {'diagnose': art('poses.json', 'quality.json'), 'prepare': art('views.json', *[v['image'] for v in views(CENTERS)]),
                       'matching': art('sfm/matches.json'), 'mapping': art('sfm/sparse/0/points3D.bin')}})
        self.config = root / 'split.json'
        write(self.config, {**base, 'semantic_run': 'sem', 'sfm_run': 'sfm', 'split': {}})
        train_pano = sorted(CENTERS)[0]
        self.tracks = {1: [train_pano], 2: sorted(CENTERS)[7:9]}
        self.patches = [(split, 'tracks_from_model', lambda path: self.tracks),
                        (split, 'git_commit', lambda: 'a' * 40)]
        self.originals = [(m, n, getattr(m, n)) for m, n, _ in self.patches]
        for module, name, value in self.patches:
            setattr(module, name, value)
        self.addCleanup(lambda: [setattr(m, n, v) for m, n, v in self.originals])

    def run_split(self, run):
        from theta_pipeline import stages
        run.stage('audit', stages.audit)
        run.stage('auto_mask', stages.auto_mask, requires=('audit',))
        run.stage('import_sfm', split.import_sfm, requires=('audit',))
        run.stage('auto_split', split.auto_split, requires=('auto_mask', 'import_sfm'))
        run.stage('split_gates', split.split_gates, requires=('auto_split',))
        run.stage('split_report', split.split_report, requires=('split_gates',))



class SplitExperimentTests(SplitFixture):
    """End to end on synthetic source runs: import, freeze, gates, report, tamper detection."""

    def test_frozen_split_authorizes_evaluated_training(self):
        from theta_pipeline.storage import Run
        with Run(self.config, 'exp').locked() as run:
            self.run_split(run)
            record = read(run.path / 'split.json')
            self.assertEqual(record['status'], 'frozen', record['checks'])
            self.assertEqual(record['git_commit'], 'a' * 40)
            for key in ('method', 'parameters', 'seed', 'poses_sha256', 'partition_sha256', 'face_assignment'):
                self.assertIn(key, record)
            gates = read(run.path / 'gate_results.json')
            self.assertEqual(gates['gsplat_allowed'], {'exploratory': True, 'evaluated': True})
            self.assertFalse(gates['j1_passed'])
            self.assertEqual(gates['permissions']['guided_3d_navigation'], 'UNKNOWN')
            inputs = read(run.path / 'train_inputs.json')
            held = record['validation'] + record['test']
            self.assertFalse(any(h in e['path'] for e in inputs['images'] for h in held))
            self.assertEqual(len(inputs['images']), 12 * len(record['train']))
            self.assertTrue(all(len(e['sha256']) == 64 for e in [*inputs['images'], *inputs['masks'].values()]))
            heldout_only = [pid for pid, panos in self.tracks.items() if not set(panos) & set(record['train'])]
            self.assertEqual(inputs['init_points']['excluded_heldout_only_point_ids'], heldout_only)
            report = (run.path / 'split_report.md').read_text()
            for name in ('train', 'validation', 'test', 'PASS', 'SfM conjoint', 'split_layout.png'):
                self.assertIn(name, report)
            # A partition modified after freezing is detected, never trusted silently.
            tampered = {**record, 'test': record['test'][:1] + record['train'][:1]}
            write(run.path / 'split.json', tampered)
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                run.stage('split_report', split.split_report, requires=('split_gates',))

    def test_pinned_partition_mismatch_is_rejected(self):
        from theta_pipeline.storage import Run
        write(self.config, {**self.base, 'semantic_run': 'sem', 'sfm_run': 'sfm',
                            'split': {'expected_partition_sha256': '0' * 64}})
        with Run(self.config, 'pinned').locked() as run:
            self.run_split(run)
            self.assertEqual(read(run.path / 'split.json')['status'], 'rejected')
            gates = read(run.path / 'gate_results.json')
            self.assertFalse(gates['gsplat_allowed']['evaluated'])
            self.assertTrue(gates['gsplat_allowed']['exploratory'])

    def test_training_file_modified_after_split_blocks_evaluated_training(self):
        from theta_pipeline import stages
        from theta_pipeline.storage import Run
        with Run(self.config, 'late').locked() as run:
            run.stage('audit', stages.audit)
            run.stage('auto_mask', stages.auto_mask, requires=('audit',))
            run.stage('import_sfm', split.import_sfm, requires=('audit',))
            run.stage('auto_split', split.auto_split, requires=('auto_mask', 'import_sfm'))
            self.assertEqual(read(run.path / 'split.json')['status'], 'frozen')
            image = read(run.path / 'train_inputs.json')['images'][0]['path']
            (self.root / 'runs' / image).write_bytes(b'modified after freeze')
            run.stage('split_gates', split.split_gates, requires=('auto_split',))
            gates = read(run.path / 'gate_results.json')
            self.assertFalse(gates['gsplat_allowed']['evaluated'])
            intact = {g['name']: g for g in gates['gates']}['train_files_intact']
            self.assertEqual(intact['evidence']['changed'], [image])

    def test_modified_source_run_is_refused(self):
        from theta_pipeline.storage import Run
        write(self.root / 'runs/sfm/poses.json', {'poses': []})
        with Run(self.config, 'tampered').locked() as run:
            from theta_pipeline import stages
            run.stage('audit', stages.audit)
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                run.stage('import_sfm', split.import_sfm, requires=('audit',))


class ExtendValidationTests(unittest.TestCase):
    """AUTO-05 extension: fixed test, exactly one train panorama moved to validation."""

    def setUp(self):
        edges = neighbour_edges(CENTERS)
        base = split.propose(CENTERS, edges, split.DEFAULTS)
        self.base = {name: base[name] for name in split.SETS}
        self.base['partition_sha256'] = split.partition_sha256(self.base)
        self.edges = edges
        self.p = {**split.DEFAULTS, 'base_split_run': 'base', 'base_partition_sha256': self.base['partition_sha256'],
                  'fixed_test': self.base['test'], 'add_validation': 1}

    def test_counts_disjoint_and_fixed_test(self):
        result = split.extend(CENTERS, self.edges, self.p, self.base)
        self.assertEqual(result['status'], 'proposed')
        self.assertEqual(result['test'], sorted(self.base['test']))
        self.assertEqual([len(result[n]) for n in split.SETS], [9, 2, 2])
        self.assertFalse(set(result['test']) & set(result['validation']))
        moved = result['moved_to_validation']
        self.assertEqual(len(moved), 1)
        self.assertIn(moved[0], self.base['train'])
        self.assertNotIn(moved[0], result['hull_vertices'])
        self.assertEqual(sorted(sum((result[n] for n in split.SETS), [])), sorted(CENTERS))
        checks = split.extension_checks(self.base, {n: result[n] for n in split.SETS}, self.p)
        self.assertEqual({c['result'] for c in checks}, {'PASS'})

    def test_reproducible_under_order_and_scale(self):
        expected = split.extend(CENTERS, self.edges, self.p, self.base)
        for seed in range(4):
            items = list(CENTERS.items())
            random.Random(seed).shuffle(items)
            edges = [e[::-1] for e in self.edges]
            random.Random(seed).shuffle(edges)
            scaled = {k: list(np.multiply(v, 12.5) + [1, 2, 3]) for k, v in items}
            result = split.extend(scaled, edges, self.p, self.base)
            self.assertEqual({n: result[n] for n in split.SETS}, {n: expected[n] for n in split.SETS})

    def test_wrong_test_or_count_is_rejected(self):
        self.assertEqual(split.extend(CENTERS, self.edges, {**self.p, 'add_validation': 0}, self.base)['status'],
                         'unknown')
        result = split.extend(CENTERS, self.edges, self.p, self.base)
        sets = {n: result[n] for n in split.SETS}
        moved_twice = {**sets, 'validation': sets['validation'] + sets['train'][:1], 'train': sets['train'][1:]}
        checks = {c['name']: c['result'] for c in split.extension_checks(self.base, moved_twice, self.p)}
        self.assertEqual(checks['validation_extends_base'], 'FAIL')
        changed_test = {**sets, 'test': sets['test'][:1] + sets['train'][:1], 'train': sets['train'][1:]}
        checks = {c['name']: c['result'] for c in split.extension_checks(self.base, changed_test, self.p)}
        self.assertEqual(checks['test_fixed'], 'FAIL')

    def test_versioned_configs(self):
        root = Path(__file__).resolve().parents[1] / 'configs'
        import json
        original = json.loads((root / 'salon-split.json').read_text())
        self.assertEqual(original['split'], {'test_fraction': 0.15, 'validation_fraction': 0.1, 'neighbors_k': 4,
                                             'min_train_neighbors': 2, 'expected_partition_sha256': None})
        v2 = json.loads((root / 'salon-split-v2.json').read_text())
        self.assertEqual(v2['split']['base_split_run'], 'salon-split-005')
        self.assertEqual(v2['split']['base_partition_sha256'],
                         '3949e717cb4d730c77ba302c3974224462e33425fd18e89a67afaea88291aaae')
        self.assertEqual(v2['split']['fixed_test'], ['R0010008', 'R0010014'])
        self.assertEqual(v2['split']['add_validation'], 1)
        self.assertEqual({k: v for k, v in v2.items() if k != 'split'},
                         {k: v for k, v in original.items() if k != 'split'})


class ExtendExperimentTests(SplitFixture):
    """Extension end to end; the base split run is never modified."""

    def snapshot(self, folder):
        from theta_pipeline.storage import digest
        return {str(p.relative_to(folder)): digest(p) for p in sorted(folder.rglob('*')) if p.is_file()}

    def base_run(self):
        from theta_pipeline.storage import Run
        with Run(self.config, 'exp').locked() as run:
            self.run_split(run)
        return read(self.root / 'runs/exp/split.json')

    def extension_config(self, base, **changes):
        write(self.root / 'ext.json', {**self.base, 'semantic_run': 'sem', 'sfm_run': 'sfm', 'split': {
            'base_split_run': 'exp', 'base_partition_sha256': base['partition_sha256'],
            'fixed_test': base['test'], 'add_validation': 1, **changes}})
        return self.root / 'ext.json'

    def test_extension_freezes_and_leaves_base_untouched(self):
        from theta_pipeline.storage import Run
        base = self.base_run()
        self.assertEqual(base['status'], 'frozen')
        before = self.snapshot(self.root / 'runs/exp')
        with Run(self.extension_config(base), 'ext').locked() as run:
            self.run_split(run)
            record = read(run.path / 'split.json')
        self.assertEqual(record['status'], 'frozen', record['checks'])
        self.assertEqual(record['method'], split.METHOD_EXTEND)
        self.assertEqual(record['test'], sorted(base['test']))
        self.assertEqual(len(record['validation']), len(base['validation']) + 1)
        self.assertNotEqual(record['partition_sha256'], base['partition_sha256'])
        self.assertEqual(record['base']['partition_sha256'], base['partition_sha256'])
        self.assertEqual(self.snapshot(self.root / 'runs/exp'), before)

    def test_mismatched_base_or_test_refused(self):
        from theta_pipeline.storage import Run
        base = self.base_run()
        before = self.snapshot(self.root / 'runs/exp')
        for name, changes, message in (('h', {'base_partition_sha256': '0' * 64}, 'pinned'),
                                       ('t', {'fixed_test': base['train'][:2]}, 'fixed_test')):
            with Run(self.extension_config(base, **changes), name).locked() as run:
                from theta_pipeline import stages
                run.stage('audit', stages.audit)
                run.stage('auto_mask', stages.auto_mask, requires=('audit',))
                run.stage('import_sfm', split.import_sfm, requires=('audit',))
                with self.assertRaisesRegex(RuntimeError, message):
                    run.stage('auto_split', split.auto_split, requires=('auto_mask', 'import_sfm'))
        self.assertEqual(self.snapshot(self.root / 'runs/exp'), before)
