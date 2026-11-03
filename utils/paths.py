"""Canonical output-root resolution: campaign output lives on /data, not on /.

Why this module exists
----------------------
``/`` is a small root device; ``/data`` is a separate block device with the
capacity for campaign output. Campaign output therefore belongs on ``/data``,
reachable from the repo through a symlink so that day-to-day access is
unchanged.

    runs/  -> /data     everything a campaign writes          on /data
    results/            the quotable subset, git-tracked     in the repo

Layout under the runs root::

    <runs>/<run_id>/<stage>/...        per-stage campaign output
    <runs>/<run_id>/RUN_METADATA.json  what produced this run

Override the target with ``REACH_NMPC_RUNS_DIR`` (absolute path; the legacy
``EP2MPC_RUNS_DIR`` spelling still works). Everything else resolves from
here, so there is exactly one place that decides where output goes.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Dict, Optional

#: Repo root (parent of utils/).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Name of the symlink inside the repo. Gitignored.
RUNS_LINK_NAME = 'runs'

#: Where the symlink points unless REACH_NMPC_RUNS_DIR overrides it.
#: (The legacy EP2MPC_RUNS_DIR spelling is still honoured for compatibility.)
DEFAULT_RUNS_TARGET = '/data/ai/temp_20260712/REACH-nmpc/runs'

#: Refuse to start a campaign with less than this much space on the target.
MIN_FREE_GB = 5.0


class RunsRootError(RuntimeError):
    """Raised when the output root is missing, mis-linked, or unwritable."""


def runs_link() -> str:
    """Absolute path of the in-repo symlink (may not exist yet)."""
    return os.path.join(REPO_ROOT, RUNS_LINK_NAME)


def runs_target() -> str:
    """Absolute path of the real storage directory on /data.

    ``REACH_NMPC_RUNS_DIR`` overrides the default; the legacy
    ``EP2MPC_RUNS_DIR`` spelling still works so existing shells/scripts
    that export it keep resolving to the same target.
    """
    env = (os.environ.get('REACH_NMPC_RUNS_DIR')
           or os.environ.get('EP2MPC_RUNS_DIR')
           or DEFAULT_RUNS_TARGET)
    return os.path.abspath(env)


def free_gb(path: str) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1024 ** 3


def ensure_runs_link(*, create: bool = True) -> str:
    """Return the in-repo runs path, creating and validating the symlink.

    Validates rather than assumes: a symlink that exists but points somewhere
    unexpected is a worse failure than one that is missing, because output would
    silently land on the wrong device.
    """
    link, target = runs_link(), runs_target()

    if create:
        os.makedirs(target, exist_ok=True)
    if not os.path.isdir(target):
        raise RunsRootError(
            f"runs target {target} does not exist and could not be created. "
            f"Set EP2MPC_RUNS_DIR to a writable directory on a device with room.")
    if not os.access(target, os.W_OK):
        raise RunsRootError(f"runs target {target} is not writable")

    if os.path.islink(link):
        actual = os.path.realpath(link)
        if actual != os.path.realpath(target):
            raise RunsRootError(
                f"{link} is a symlink to {actual}, not to {target}. Refusing to "
                f"silently retarget it -- remove the link deliberately if that is "
                f"what you intend.")
    elif os.path.exists(link):
        raise RunsRootError(
            f"{link} exists and is NOT a symlink ({'dir' if os.path.isdir(link) else 'file'}). "
            f"Refusing to replace it. Move it aside first.")
    elif create:
        os.symlink(target, link)

    return link


def runs_root(*, create: bool = True) -> str:
    """The path campaign writers should use as their output root."""
    return ensure_runs_link(create=create)


def run_dir(run_id: str, stage: Optional[str] = None, *, create: bool = True) -> str:
    """``<runs>/<run_id>[/<stage>]``, created on demand.

    ``run_id`` is validated: it becomes a directory name and is threaded into
    every sidecar, so a path separator or ``..`` in it would scatter output.
    """
    if not run_id or os.sep in run_id or run_id in ('.', '..') or run_id.startswith('-'):
        raise ValueError(f"invalid run_id {run_id!r}: must be a single path segment")
    if stage and (os.sep in stage or stage in ('.', '..')):
        raise ValueError(f"invalid stage {stage!r}")
    root = runs_root(create=create)
    path = os.path.join(root, run_id) if not stage else os.path.join(root, run_id, stage)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def check_space(min_free_gb: float = MIN_FREE_GB) -> Dict[str, object]:
    """Free space on the runs target, and whether it clears ``min_free_gb``."""
    target = runs_target()
    probe = target if os.path.isdir(target) else os.path.dirname(target)
    while probe and not os.path.isdir(probe):
        probe = os.path.dirname(probe)
    free = free_gb(probe or '/')
    total = shutil.disk_usage(probe or '/').total / 1024 ** 3
    return {'target': target, 'probe': probe, 'free_gb': free,
            'total_gb': total, 'min_free_gb': min_free_gb,
            'ok': free >= min_free_gb,
            'same_device_as_repo': os.stat(probe or '/').st_dev == os.stat(REPO_ROOT).st_dev}


def write_run_metadata(run_id: str, extra: Optional[dict] = None,
                       *, force: bool = False) -> str:
    """Stamp ``<runs>/<run_id>/RUN_METADATA.json`` so a run directory is self-describing."""
    from utils.provenance import git_commit_info, utc_now_iso, dependency_versions
    from utils.safe_write import safe_write_json

    d = run_dir(run_id)
    payload = {
        'run_id': run_id,
        'created_utc': utc_now_iso(),
        'runs_target': runs_target(),
        'repo_root': REPO_ROOT,
        'git': git_commit_info(),
        'space': check_space(),
        'note': 'Campaign output only. The git-tracked record and the A1 freeze '
                'stay under results/ on the repo device; nothing here is tracked.',
    }
    if extra:
        payload.update(extra)
    return safe_write_json(os.path.join(d, 'RUN_METADATA.json'), payload, force=force)
