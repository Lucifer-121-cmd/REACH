"""Overwrite guard for everything that writes into ``results/``.

One systemic guard instead of six one-off patches. The rule is simple:

    **A file that is tracked in git is never overwritten unless the caller
    passes ``force=True``.**

Rationale: every irreplaceable artefact in this repo is tracked --
the 21 ablation aggregates, the three NSGA-II caches, every ``checkpoint_gen*``
and ``pareto_gen*``, the ``*_LLM.csv`` audit trails and all of
``results/statistics``. Untracked files are regenerable by definition, so they
are free to overwrite. That single test separates "irreplaceable" from
"derived" without maintaining a hand-written path list that will drift.

Writes are atomic: content goes to a temporary file in the same directory and is
then ``os.replace``-d into position, so an interrupted write cannot truncate a
tracked file.

Usage::

    from utils.safe_write import safe_write_json, OverwriteRefused

    safe_write_json(path, payload)                # refuses if tracked
    safe_write_json(path, payload, force=True)    # explicit override
"""

from __future__ import annotations

import functools
import json
import logging
import os
import subprocess
import tempfile
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

__all__ = [
    'OverwriteRefused',
    'repo_root',
    'is_git_tracked',
    'assert_writable',
    'safe_write_json',
    'safe_write_text',
    'safe_write_bytes',
    'safe_savefig',
    'run_id_dir',
]


class OverwriteRefused(RuntimeError):
    """Raised when a write would clobber a git-tracked file without ``force``."""


@functools.lru_cache(maxsize=1)
def repo_root() -> str:
    """Absolute path of the repository root (parent of ``utils/``)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@functools.lru_cache(maxsize=1)
def _tracked_paths() -> frozenset:
    """Every path tracked at HEAD, as repo-relative POSIX strings.

    Queried once per process. If git is unavailable the set is empty, which
    fails *open* -- deliberately, so that a tarball export without .git still
    runs. The guard's job is to catch accidents, not to be a security boundary.
    """
    try:
        out = subprocess.run(
            ['git', 'ls-files', '-z'],
            cwd=repo_root(), capture_output=True, text=True, timeout=60, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("safe_write: git unavailable (%s); overwrite guard is INACTIVE", exc)
        return frozenset()
    return frozenset(p for p in out.split('\0') if p)


def is_git_tracked(path: str) -> bool:
    """True if ``path`` is tracked at HEAD."""
    try:
        rel = os.path.relpath(os.path.abspath(path), repo_root())
    except ValueError:                      # different drive / unrelated path
        return False
    if rel.startswith(os.pardir):           # outside the repo
        return False
    return rel.replace(os.sep, '/') in _tracked_paths()


def assert_writable(path: str, *, force: bool = False, what: str = 'file') -> None:
    """Raise :class:`OverwriteRefused` if writing ``path`` would clobber history.

    Passing ``force=True`` logs the override at WARNING so it appears in the run
    log rather than silently.
    """
    if not os.path.exists(path):
        return
    if not is_git_tracked(path):
        logger.debug("safe_write: overwriting untracked %s %s", what, path)
        return
    if force:
        logger.warning("safe_write: FORCED overwrite of git-tracked %s %s", what, path)
        return
    raise OverwriteRefused(
        f"refusing to overwrite git-tracked {what}: {path}\n"
        f"  This file is part of the tracked record -- the ablation aggregates,\n"
        f"  NSGA-II caches, checkpoints and pareto archives are unrecoverable.\n"
        f"  Write to a run-id directory instead, or pass force=True / --force if\n"
        f"  you genuinely intend to replace it. Recover with:\n"
        f"    git checkout -- {os.path.relpath(os.path.abspath(path), repo_root())}"
    )


def _atomic_write(path: str, data, mode: str, encoding: Optional[str]) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or '.'
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix='.tmp-', suffix='.part')
    try:
        with os.fdopen(fd, mode, encoding=encoding) as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def safe_write_text(path: str, text: str, *, force: bool = False,
                    encoding: str = 'utf-8') -> str:
    """Atomically write ``text``, refusing to clobber tracked files."""
    assert_writable(path, force=force)
    _atomic_write(path, text, 'w', encoding)
    logger.info("wrote %s (%d bytes)", path, len(text.encode(encoding)))
    return path


def safe_write_bytes(path: str, blob: bytes, *, force: bool = False) -> str:
    """Atomically write ``blob``, refusing to clobber tracked files."""
    assert_writable(path, force=force)
    _atomic_write(path, blob, 'wb', None)
    logger.info("wrote %s (%d bytes)", path, len(blob))
    return path


def safe_write_json(path: str, obj: Any, *, force: bool = False, indent: int = 2,
                    default=float, sort_keys: bool = False) -> str:
    """Atomically write ``obj`` as JSON, refusing to clobber tracked files."""
    return safe_write_text(
        path,
        json.dumps(obj, indent=indent, default=default, sort_keys=sort_keys),
        force=force,
    )


def safe_savefig(fig, path: str, *, force: bool = False, **kwargs) -> str:
    """``fig.savefig(path)`` behind the overwrite guard.

    ``fig`` may be a Figure or the ``matplotlib.pyplot`` module.
    """
    assert_writable(path, force=force, what='figure')
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    fig.savefig(path, **kwargs)
    logger.info("wrote %s", path)
    return path


def run_id_dir(base: str, run_id: Optional[str] = None, *,
               create: bool = True) -> str:
    """Return ``base/<run_id>``, defaulting ``run_id`` to a UTC timestamp.

    Run-id directories are the positive half of the guard: instead of arguing
    about whether an overwrite is safe, write somewhere new. The timestamp is
    generated here rather than by the caller so every stage of one pipeline can
    share an explicit id.
    """
    if run_id is None:
        from datetime import datetime, timezone
        run_id = datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%SZ')
    path = os.path.join(base, run_id)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def newest_run_dir(base: str) -> Optional[str]:
    """Most recent ``run_*`` directory under ``base``, or None."""
    if not os.path.isdir(base):
        return None
    runs = sorted(d for d in os.listdir(base)
                  if d.startswith('run_') and os.path.isdir(os.path.join(base, d)))
    return os.path.join(base, runs[-1]) if runs else None
