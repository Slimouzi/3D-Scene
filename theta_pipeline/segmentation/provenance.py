"""Code and repository identity shared by the GPU runner and CPU fusion."""
import subprocess
from pathlib import Path
from ..storage import digest

PACKAGE = Path(__file__).resolve().parents[1]


def git_commit():
    """HEAD sha, suffixed -dirty when tracked files differ; None outside a repository."""
    try:
        def git(*args):
            return subprocess.run(['git', *args], cwd=PACKAGE.parent, capture_output=True,
                                  text=True, check=True).stdout.strip()
        dirty = git('status', '--porcelain', '--untracked-files=no')
        return git('rev-parse', 'HEAD') + ('-dirty' if dirty else '')
    except (OSError, subprocess.CalledProcessError):
        return None


def code_digests():
    return {str(p.relative_to(PACKAGE)): digest(p) for p in sorted(PACKAGE.rglob('*.py'))}
