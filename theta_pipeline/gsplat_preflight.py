"""Start-of-training checks for the gsplat trainer. No torch import at module level.

Re-verifies, from the files on disk: the GPU environment against gsplat.lock.txt, a clean
git commit equal to the code that prepared the inputs, the frozen pinned partition, the
evaluated-training permission, every prepared artifact hash, and the train/held-out
separation of cameras. Files actually read later are hash-checked again on load.
"""
import platform
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from . import split as auto05
from .segmentation.provenance import PACKAGE, code_digests, git_commit
from .storage import digest, read

LOCK = PACKAGE.parent / 'requirements/gsplat.lock.txt'


def pins(path=LOCK):
    text = Path(path).read_text()
    header = dict(re.findall(r'^# (python|cuda|torch|gsplat): (\S+)', text, re.M))
    return {**header, 'numpy': re.search(r'^numpy==(\S+)$', text, re.M).group(1)}


def render_probe():
    """Rasterize one red Gaussian in front of an 8x8 camera on CUDA; checks the center pixel."""
    import torch
    from gsplat import rasterization
    d = 'cuda'
    renders, alphas, _ = rasterization(
        means=torch.tensor([[0., 0., 2.]], device=d), quats=torch.tensor([[1., 0., 0., 0.]], device=d),
        scales=torch.full((1, 3), .2, device=d), opacities=torch.tensor([.9], device=d),
        colors=torch.tensor([[1., 0., 0.]], device=d), viewmats=torch.eye(4, device=d)[None],
        Ks=torch.tensor([[[8., 0., 4.], [0., 8., 4.], [0., 0., 1.]]], device=d), width=8, height=8)
    torch.cuda.synchronize()
    center = renders[0, 4, 4].tolist()
    if not (alphas[0, 4, 4, 0] > .5 and center[0] > .5 and center[1] < .1 and center[2] < .1):
        raise RuntimeError(f'unexpected probe render: rgb {center}, alpha {float(alphas[0, 4, 4, 0])}')
    return {'center_rgb': [round(v, 4) for v in center], 'alpha': round(float(alphas[0, 4, 4, 0]), 4)}


def qualify(path=LOCK):
    expected = pins(path)
    found = {'python': platform.python_version(), 'platform': f'{platform.system()} {platform.machine()}'}
    for name in ('torch', 'gsplat', 'numpy'):
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = None
    problems = []
    try:
        import torch
        found.update(cuda=torch.version.cuda, cuda_available=torch.cuda.is_available(),
                     device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except ImportError as error:
        found.update(cuda=None, cuda_available=False, device=None)
        problems.append(f'torch import failed: {error}')
    try:
        from gsplat import rasterization  # noqa: F401
    except Exception as error:
        problems.append(f'gsplat import failed: {error}')
    else:
        # Importing gsplat does not load its CUDA backend (packaging, setuptools, csrc):
        # only a real render proves the environment can train.
        if found['cuda_available']:
            try:
                found['render_probe'] = render_probe()
            except Exception as error:
                problems.append(f'gsplat CUDA render failed: {type(error).__name__}: {error}')
    if found['platform'] != 'Linux x86_64':
        problems.append(f"platform {found['platform']} is not Linux x86_64")
    if not found['cuda_available']:
        problems.append('CUDA unavailable')
    if not str(found['cuda'] or '').startswith(expected['cuda']):
        problems.append(f"CUDA {found['cuda']} != {expected['cuda']}")
    for name in ('python', 'torch', 'gsplat', 'numpy'):
        if found[name] != expected[name]:
            problems.append(f'{name} {found[name]} != {expected[name]}')
    return {'qualified': not problems, 'problems': problems, 'expected': expected, 'found': found}


def verify_prep(prep):
    """Problems with a prepared gsplat experiment run, or an empty list."""
    prep = Path(prep)
    output = prep.parent
    state = read(prep / 'run.json')
    problems = []
    for stage in ('import_split', 'gsplat_prepare'):
        entry = state['stages'].get(stage, {})
        if entry.get('status') != 'completed':
            problems.append(f'stage {stage} not completed')
            continue
        for rel, sha in entry['artifacts'].items():
            if not (prep / rel).is_file() or digest(prep / rel) != sha:
                problems.append(f'{rel} changed or missing')
    if problems:
        return problems
    config = state['provenance']['config']
    record = read(prep / 'split_import/split.json')
    gates = read(prep / 'split_import/gate_results.json')
    manifest = read(prep / 'gsplat_inputs/gsplat_inputs.json')
    problems += auto05.verify_split_source(record, gates, config)
    if manifest['partition_sha256'] != record['partition_sha256']:
        problems.append('gsplat_inputs partition differs from the imported split')
    for rel, sha in manifest['files'].items():
        if digest(output / rel) != sha:
            problems.append(f'{rel} changed since preparation')
    inputs = read(prep / 'split_import/train_inputs.json')
    faces = len(inputs['images']) // max(len(record['train']), 1)
    for group in auto05.SETS:
        cameras = read(prep / f'gsplat_inputs/cameras_{group}.json')['cameras']
        wrong = [c['name'] for c in cameras if c['set'] != group or c['panorama_id'] not in record[group]]
        if wrong:
            problems.append(f'cameras_{group}.json lists cameras outside {group}: {wrong[:5]}')
        if len(cameras) != faces * len(record[group]):
            problems.append(f'{len(cameras)} {group} cameras for {len(record[group])} panoramas '
                            f'of {faces} faces')
    train_images = sorted(e['path'] for e in inputs['images'])
    camera_images = sorted(c['image']['path'] for c in
                           read(prep / 'gsplat_inputs/cameras_train.json')['cameras'])
    if camera_images != train_images:
        problems.append('cameras_train.json images differ from the split train_inputs.json')
    return problems


# Training and read-only analysis modules may change between preparation and training (the
# training commit is clean and recorded); every other module produced or checked the inputs.
TRAINING_MODULES = {'gsplat_train.py', 'gsplat_preflight.py', 'gsplat_inspect.py', 'gsplat_compare.py',
                    'gsplat_divergence.py', 'gsplat_sheet.py', 'gsplat_camera_check.py'}


def code_differences(prep_code, current):
    """Changed files, split into preparation code (must be identical) and training code."""
    changed = sorted(k for k in set(prep_code) | set(current) if prep_code.get(k) != current.get(k))
    return {'preparation': [k for k in changed if k not in TRAINING_MODULES],
            'training': [k for k in changed if k in TRAINING_MODULES]}


def code_matches_prep(prep):
    """True when the code that prepared the inputs is unchanged (training modules may differ)."""
    state = read(Path(prep) / 'run.json')
    return not code_differences(state['provenance']['code'], code_digests())['preparation']


def preflight(prep):
    """Everything required before the first training step; returns a report dict."""
    commit = git_commit()
    environment = qualify()
    problems = list(environment['problems'])
    if commit is None or commit.endswith('-dirty'):
        problems.insert(0, f'git commit {commit}: training needs a clean fixed commit')
    differences = code_differences(read(Path(prep) / 'run.json')['provenance']['code'], code_digests())
    if differences['preparation']:
        problems.append(f"preparation code differs from the code that prepared the inputs: "
                        f"{differences['preparation']}")
    problems += verify_prep(prep)
    return {'ok': not problems, 'problems': problems, 'git_commit': commit, 'environment': environment,
            'training_code_changed_since_preparation': differences['training']}


def load_verified(output, record):
    """Path of a file to read, after checking its SHA-256 against its record."""
    path = Path(output) / record['path']
    if not record.get('sha256') or digest(path) != record['sha256']:
        raise RuntimeError(f"{record['path']} changed, missing or unrecorded")
    return path
