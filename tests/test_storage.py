import json
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from theta_pipeline import stages
from theta_pipeline.storage import Run, write


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'input').mkdir()
        Image.new('RGB', (128, 64)).save(self.root / 'input/a.png')
        self.config = self.root / 'config.json'
        write(self.config, {'schema_version': 1, 'kind': 'diagnostic', 'input': 'input',
            'output': 'runs', 'erp_width': 128, 'num_threads': 1, 'seed': 0,
            'max_features': 128, 'mapping_max_seconds': 10, 'masks': {}})

    def test_changed_source_rejected_and_original_unchanged(self):
        with Run(self.config, 'test').locked():
            pass
        Image.new('RGB', (128, 64), 'red').save(self.root / 'input/a.png')
        with self.assertRaisesRegex(ValueError, 'changed'):
            with Run(self.config, 'test').locked():
                pass

    def test_cache_validates_content_and_does_not_rerun(self):
        calls = []
        def action(run):
            calls.append(1)
            target = run.path / 'result.json'
            write(target, {'okay': True})
            return [target]
        with Run(self.config, 'test').locked() as run:
            run.stage('a', action)
            run.stage('a', action)
            self.assertEqual(len(calls), 1)
            (run.path / 'result.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError, 'changed/missing'):
                run.stage('a', action)

    def test_cached_stage_revalidates_dependencies(self):
        def source(run):
            target = run.path / 'source.json'
            write(target, {'okay': True})
            return [target]

        def derived(run):
            target = run.path / 'derived.json'
            write(target, {'okay': True})
            return [target]

        with Run(self.config, 'dependencies').locked() as run:
            run.stage('source', source)
            run.stage('middle', derived, requires=('source',))
            run.stage('derived', derived, requires=('middle',))
            (run.path / 'source.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError, 'changed/missing'):
                run.stage('derived', derived, requires=('middle',))

    def test_empty_reconstruction_publishes_diagnostic(self):
        def empty_mapping(run):
            target = run.path / 'sfm/models.json'
            write(target, {'components': [], 'count': 0})
            return [target]

        with Run(self.config, 'empty').locked() as run:
            run.stage('mapping', empty_mapping)
            run.stage('diagnose', stages.diagnose, requires=('mapping',))
            run.stage('report', stages.report, requires=('diagnose',))
            quality = json.loads((run.path / 'quality.json').read_text())
            self.assertEqual(quality['decision'], 'insufficient_registration')
            self.assertIsNone(quality['center_path']['coordinate_y_span'])
            self.assertTrue((run.path / 'report.md').is_file())

    def test_auto_mask_fails_closed_when_backend_is_unavailable(self):
        with Run(self.config, 'auto-mask').locked() as run:
            run.stage('audit', stages.audit)
            run.stage('auto_mask', stages.auto_mask, requires=('audit',))
            masks = json.loads((run.path / 'semantic_masks.json').read_text())
            self.assertEqual(masks['status'], 'UNKNOWN')
            self.assertFalse(masks['accepted_masks'])
            self.assertIsNone(masks['panoramas'][0]['geometry_valid'])
            self.assertIsNone(masks['panoramas'][0]['rgb_valid'])

    def test_failure_persisted(self):
        def fail(run):
            raise RuntimeError('expected failure')
        with Run(self.config, 'test').locked() as run:
            with self.assertRaises(RuntimeError):
                run.stage('bad', fail)
            state = json.loads((run.path / 'run.json').read_text())
            self.assertEqual(state['stages']['bad']['status'], 'failed')
            self.assertIn('expected failure', state['stages']['bad']['error'])

    def test_interrupted_stage_is_recovered_by_next_process(self):
        script = """
from theta_pipeline.storage import Run
with Run(r'{config}', 'crash').locked() as run:
    run.stage('native', lambda current: os.kill(os.getpid(), signal.SIGTERM))
""".format(config=self.config)
        script = 'import os, signal\n' + script
        result = subprocess.run([sys.executable, '-c', script], check=False)
        self.assertEqual(result.returncode, -signal.SIGTERM)
        with Run(self.config, 'crash').locked() as run:
            self.assertEqual(run.state['stages']['native']['status'], 'failed')
            self.assertIn('Previous process terminated', run.state['stages']['native']['error'])

    def test_path_escape_and_input_overlap_rejected(self):
        with self.assertRaises(ValueError):
            Run(self.config, '../escape')
        config = json.loads(self.config.read_text())
        config['output'] = 'input'
        write(self.config, config)
        with self.assertRaises(ValueError):
            Run(self.config, 'test')

    def test_exclusive_lock(self):
        with Run(self.config, 'test').locked():
            with self.assertRaisesRegex(RuntimeError, 'Another process'):
                with Run(self.config, 'test').locked():
                    pass
