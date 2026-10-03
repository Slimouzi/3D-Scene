"""GPU environment qualification against requirements/gpu.lock.txt.

Any mismatch makes the run UNKNOWN: it is not a failure of the scene, and it never
authorizes a mask. Imports torch/triton lazily so the CPU environment can call it.
"""
import json
import platform
import re
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from .provenance import PACKAGE

LOCK = PACKAGE.parent / 'requirements/gpu.lock.txt'


def pins(path=LOCK):
    text = Path(path).read_text()
    header = dict(re.findall(r'^# (python|cuda|triton|sam3-commit): (\S+)', text, re.M))
    packages = dict(re.findall(r'^([A-Za-z0-9_.-]+)==(\S+)$', text, re.M))
    return {'python': header['python'], 'cuda': header['cuda'], 'triton': header['triton'],
            'sam3_commit': header['sam3-commit'], 'torch': packages['torch'],
            'torchvision': packages['torchvision'], 'numpy': packages['numpy']}


def installed_sam3_commit():
    try:
        direct = distribution('sam3').read_text('direct_url.json')
    except PackageNotFoundError:
        return None
    return json.loads(direct).get('vcs_info', {}).get('commit_id') if direct else None


def qualify(path=LOCK):
    expected = pins(path)
    found = {'python': platform.python_version(), 'platform': platform.platform(),
             'sam3_commit': installed_sam3_commit()}
    for name in ('torch', 'torchvision', 'numpy', 'triton'):
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = None
    problems = []
    try:
        import torch
        found['cuda'] = torch.version.cuda
        found['cuda_available'] = torch.cuda.is_available()
        found['device'] = torch.cuda.get_device_name(0) if found['cuda_available'] else None
    except ImportError as error:
        found.update(cuda=None, cuda_available=False, device=None)
        problems.append(f'torch import failed: {error}')
    try:
        import triton  # noqa: F401
    except ImportError as error:
        problems.append(f'triton import failed: {error}')
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        problems.append(f'platform {platform.system()} {platform.machine()} is not linux x86_64')
    if not found['cuda_available']:
        problems.append('CUDA unavailable')
    if not str(found['cuda'] or '').startswith(expected['cuda']):
        problems.append(f"CUDA {found['cuda']} != {expected['cuda']}")
    for name in ('python', 'torch', 'torchvision', 'numpy', 'triton', 'sam3_commit'):
        value = (found[name] or '').split('+')[0]
        if value != expected[name]:
            problems.append(f'{name} {found[name]} != {expected[name]}')
    return {'qualified': not problems, 'problems': problems, 'expected': expected, 'found': found}
