"""Run isolation, atomic manifests, provenance and verified stage caching."""
import hashlib
import json
import os
import platform
import resource
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import fcntl


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


class Run:
    def __init__(self, config_path, run_id):
        if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in run_id):
            raise ValueError('run-id must contain only letters, digits, - or _')
        config_path = Path(config_path).resolve()
        self.config = read(config_path)
        allowed = {'schema_version', 'kind', 'input', 'output', 'erp_width', 'num_threads',
                   'seed', 'max_features', 'mapping_max_seconds', 'masks', 'auto_mask_backend'}
        if set(self.config) - allowed:
            raise ValueError(f'Unknown config keys: {set(self.config) - allowed}')
        if self.config.get('kind') != 'diagnostic':
            raise ValueError('Only diagnostic runs implemented; no benchmark claims allowed')
        if self.config.get('schema_version') != 1:
            raise ValueError('Unsupported schema_version')
        width = self.config['erp_width']
        if not isinstance(width, int) or width < 128 or width % 4:
            raise ValueError('erp_width must be an integer multiple of 4, >=128')
        for key in ('num_threads', 'max_features', 'mapping_max_seconds'):
            if self.config[key] <= 0:
                raise ValueError(f'{key} must be positive')
        self.input = (config_path.parent / self.config['input']).resolve()
        self.path = (config_path.parent / self.config['output'] / run_id).resolve()
        if self.path == self.input or self.input in self.path.parents or self.path in self.input.parents:
            raise ValueError('Input and run directories must not contain each other')
        self.sources = sorted(p for p in self.input.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
        if not self.sources:
            raise ValueError('No input images')
        if len({p.stem for p in self.sources}) != len(self.sources):
            raise ValueError('Duplicate panorama stems')
        self.mask_paths = {}
        mask_hashes = {}
        for pano, kinds in self.config.get('masks', {}).items():
            if pano not in {p.stem for p in self.sources} or set(kinds) - {'geometry', 'rgb'}:
                raise ValueError(f'Invalid mask entry: {pano}')
            self.mask_paths[pano] = {}
            mask_hashes[pano] = {}
            for kind, name in kinds.items():
                path = (config_path.parent / name).resolve()
                self.mask_paths[pano][kind] = path
                mask_hashes[pano][kind] = digest(path)
        self.provenance = {
            'config': self.config, 'inputs': {p.name: digest(p) for p in self.sources},
            'masks': mask_hashes,
            'code': {p.name: digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
            'versions': {p: version(p) for p in ('numpy', 'Pillow', 'pycolmap', 'opencv-python-headless')},
            'python': platform.python_version(),
        }
        self.identity = fingerprint(self.provenance)
        self.path.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id

    @contextmanager
    def locked(self):
        with (self.path / '.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('Another process is using this run') from None
            manifest = self.path / 'run.json'
            if manifest.exists():
                self.state = read(manifest)
                if self.state['fingerprint'] != self.identity:
                    raise ValueError('Inputs/config/code/environment changed: use a new run-id')
                recovered = False
                for entry in self.state['stages'].values():
                    if entry.get('status') == 'running':
                        entry.update(status='failed',
                                     error='Previous process terminated before stage completion',
                                     recovered_at=now())
                        recovered = True
                if recovered:
                    write(manifest, self.state)
            else:
                self.state = {'schema_version': 1, 'run_id': self.run_id, 'kind': 'diagnostic',
                              'created_at': now(), 'fingerprint': self.identity,
                              'provenance': self.provenance, 'stages': {},
                              'input_root_from_run': os.path.relpath(self.input, self.path),
                              'platform': platform.platform()}
                write(manifest, self.state)
            yield self

    def require(self, name, seen=None):
        seen = set() if seen is None else seen
        if name in seen:
            raise RuntimeError(f'Cyclic stage dependency: {name}')
        seen.add(name)
        entry = self.state['stages'].get(name, {})
        if entry.get('status') != 'completed':
            raise RuntimeError(f'Run stage {name} first')
        for dependency in entry.get('requires', []):
            self.require(dependency, seen.copy())
        for rel, expected in entry['artifacts'].items():
            p = self.path / rel
            if not p.is_file() or digest(p) != expected:
                raise RuntimeError(f'Artifact changed/missing: {rel}; use a new run-id')

    def stage(self, name, action, requires=()):
        for dependency in requires:
            self.require(dependency)
        if self.state['stages'].get(name, {}).get('status') == 'completed':
            self.require(name)
            print(f'{name}: verified cache', flush=True)
            return
        entry = {'status': 'running', 'started_at': now(), 'requires': list(requires)}
        self.state['stages'][name] = entry
        write(self.path / 'run.json', self.state)
        start = time.monotonic()
        try:
            outputs = action(self)
            entry['artifacts'] = {str(p.relative_to(self.path)): digest(p) for p in outputs}
            entry['status'] = 'completed'
        except BaseException as error:
            entry.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            entry.update(finished_at=now(), duration_s=time.monotonic() - start,
                         process_peak_rss_bytes=peak if platform.system() == 'Darwin' else peak * 1024)
            write(self.path / 'run.json', self.state)
        print(f'{name}: completed ({entry["duration_s"]:.1f}s)', flush=True)
