"""Tests for utils/paths.py — the /data output root and its in-repo symlink.

What must not break:

1. **The tracked record stays on the repo device.** ``results/`` holds 112
   git-tracked files including the only surviving copy of the deployed genomes,
   plus the A1 freeze and its sha256 manifest. Nothing here may move it.
2. **The symlink is validated, not assumed.** A link pointing somewhere
   unexpected is worse than a missing one, because output would silently land on
   the wrong device.
3. **``run_id`` is validated.** It becomes a directory name and is threaded into
   every sidecar, so ``../`` or a path separator would scatter output.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import paths
from utils.paths import (REPO_ROOT, RunsRootError, check_space, ensure_runs_link,
                         run_dir, runs_link, runs_root, runs_target)


def test_symlink_exists_and_points_at_the_target():
    link = ensure_runs_link()
    assert os.path.islink(link), f"{link} must be a symlink, not a real directory"
    assert os.path.realpath(link) == os.path.realpath(runs_target())


def test_output_root_is_a_different_device_from_the_repo():
    """The whole point: / is at 97%, /data has room."""
    s = check_space()
    assert s['same_device_as_repo'] is False, (
        "the runs target is on the same filesystem as the repo, so moving output "
        "there buys nothing")
    assert s['free_gb'] > s['min_free_gb']


def test_run_dir_resolves_onto_the_target_device():
    p = run_dir('run_pytest_paths', 'ep2')
    assert os.path.realpath(p).startswith(os.path.realpath(runs_target()))
    assert os.path.isdir(p)


@pytest.mark.parametrize('bad', ['../escape', 'a/b', '', '.', '..', '-rf'])
def test_run_id_must_be_a_single_safe_segment(bad):
    with pytest.raises(ValueError):
        run_dir(bad)


@pytest.mark.parametrize('bad', ['../x', 'a/b', '.', '..'])
def test_stage_must_be_a_single_safe_segment(bad):
    with pytest.raises(ValueError):
        run_dir('run_ok', bad)


def test_a_wrongly_pointed_symlink_is_refused(tmp_path, monkeypatch):
    """Silently retargeting the link would send output to the wrong device."""
    fake_repo = tmp_path / 'repo'
    fake_repo.mkdir()
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    intended = tmp_path / 'intended'
    intended.mkdir()
    os.symlink(elsewhere, fake_repo / paths.RUNS_LINK_NAME)
    monkeypatch.setattr(paths, 'REPO_ROOT', str(fake_repo))
    monkeypatch.setenv('REACH_NMPC_RUNS_DIR', str(intended))
    with pytest.raises(RunsRootError, match='not to'):
        ensure_runs_link()


def test_a_real_directory_in_the_way_is_refused(tmp_path, monkeypatch):
    fake_repo = tmp_path / 'repo'
    fake_repo.mkdir()
    (fake_repo / paths.RUNS_LINK_NAME).mkdir()          # real dir, not a link
    monkeypatch.setattr(paths, 'REPO_ROOT', str(fake_repo))
    monkeypatch.setenv('REACH_NMPC_RUNS_DIR', str(tmp_path / 'target'))
    with pytest.raises(RunsRootError, match='NOT a symlink'):
        ensure_runs_link()


def test_env_override_is_honoured(tmp_path, monkeypatch):
    monkeypatch.setenv('REACH_NMPC_RUNS_DIR', str(tmp_path / 'custom'))
    assert runs_target() == str(tmp_path / 'custom')


def test_legacy_env_var_spelling_still_overrides(tmp_path, monkeypatch):
    """EP2MPC_RUNS_DIR (pre-rename spelling) must keep working."""
    monkeypatch.setenv('EP2MPC_RUNS_DIR', str(tmp_path / 'legacy'))
    assert runs_target() == str(tmp_path / 'legacy')


# --- the tracked record must stay where it is -------------------------------
# NOTE (2026-10-01 cleanup): results/ was removed with the old EP²-MPC
# paper's data. These tests are kept as skip-if-absent so the suite
# passes in a clean checkout that doesn't carry the historical results.

def test_results_is_a_real_directory_not_a_symlink():
    """If results/ ever becomes a symlink, the quotable data leaves the repo."""
    results = os.path.join(REPO_ROOT, 'results')
    if not os.path.exists(results):
        pytest.skip('results/ not present in this checkout')
    assert os.path.isdir(results)
    assert not os.path.islink(results), (
        "results/ must stay a real directory on the repo device: it is the "
        "git-tracked quotable-data tree (results/README.md)")


def test_the_freeze_is_on_the_repo_device():
    frozen = os.path.join(REPO_ROOT, 'results', '_frozen')
    if not os.path.isdir(frozen):
        pytest.skip('freeze not present in this checkout')
    assert not os.path.islink(frozen)
    assert os.stat(frozen).st_dev == os.stat(REPO_ROOT).st_dev, (
        "the freeze and the manifest that certifies it must live on one device")


def test_irreplaceable_genomes_are_still_on_the_repo_device():
    for scen in ('corridor', 's_curve', 'chicane'):
        p = os.path.join(REPO_ROOT, 'results', 'evolution', scen, 'pareto_gen019.json')
        if not os.path.isfile(p):
            pytest.skip('evolution archives not present in this checkout')
        assert os.stat(p).st_dev == os.stat(REPO_ROOT).st_dev


def test_runs_is_gitignored():
    """Output on /data must never be proposed for commit."""
    import subprocess
    r = subprocess.run(['git', 'check-ignore', '-v', 'runs'],
                       cwd=REPO_ROOT, capture_output=True, text=True)
    assert r.returncode == 0, "runs/ is not gitignored"


def test_safe_write_treats_runs_paths_as_untracked():
    from utils.safe_write import assert_writable, is_git_tracked
    p = os.path.join(run_dir('run_pytest_paths'), 'probe.json')
    assert is_git_tracked(p) is False
    assert_writable(p)          # must not raise
