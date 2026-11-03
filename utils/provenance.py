"""Provenance + manifest helpers for reproducible EP²-MPC experiment runs."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_root_from_file(file_path: str) -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(file_path)))


def load_json(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)


def dependency_versions() -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            for line in (proc.stdout or "").splitlines():
                if "==" in line:
                    name, ver = line.split("==", 1)
                    if name and ver:
                        out[name.strip()] = ver.strip()
    except Exception:
        return {}
    return out


def base_metadata(
    *,
    repo_root: str,
    config_path_a: str,
    config_path_b: str,
    seeds: Optional[List[int]] = None,
    scenario_set: Optional[List[str]] = None,
) -> dict:
    cfg_hashes = {}
    for key, path in (
        ("global_config_sha256", config_path_a),
        ("fallback_config_sha256", config_path_b),
    ):
        try:
            cfg_hashes[key] = sha256_file(path)
        except Exception:
            cfg_hashes[key] = None
    return {
        "commit_sha": os.environ.get("GITHUB_SHA"),
        "config_hashes": cfg_hashes,
        "seeds": list(seeds or []),
        "scenario_set": list(scenario_set or []),
        "runtime_environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
        },
        "dependency_versions": dependency_versions(),
        "repo_root": repo_root,
    }


MANDATORY_METADATA_FIELDS = (
    "commit_sha",
    "config_hashes",
    "seeds",
    "scenario_set",
    "runtime_environment",
    "dependency_versions",
)


def validate_metadata_fields(metadata: dict) -> Dict[str, Any]:
    missing = []
    inconsistent = []
    for field in MANDATORY_METADATA_FIELDS:
        if field not in metadata:
            missing.append(field)
            continue
        val = metadata.get(field)
        if val in (None, "", [], {}):
            inconsistent.append(field)
    cfg = metadata.get("config_hashes", {})
    if not isinstance(cfg, dict):
        inconsistent.append("config_hashes")
    else:
        if not cfg.get("global_config_sha256") or not cfg.get("fallback_config_sha256"):
            inconsistent.append("config_hashes")
    return {
        "pass": len(missing) == 0 and len(inconsistent) == 0,
        "missing": missing,
        "inconsistent": inconsistent,
    }


def init_or_load_manifest(path: str, *, run_id: str, metadata: dict) -> dict:
    existing = load_json(path)
    if existing is None:
        return {
            "schema_version": "1.1",
            "run_id": run_id,
            "created_at_utc": utc_now_iso(),
            "metadata": metadata,
            "scripts_invoked": [],
            "expected_artifacts": {},
            "artifact_sha256": {},
            "metadata_checklist": validate_metadata_fields(metadata),
            "status": "in_progress",
        }
    return existing


def append_script_invocation(
    manifest: dict,
    *,
    script_name: str,
    command: List[str],
    parameters: Dict[str, Any],
    exit_code: int,
    outputs: Optional[List[str]] = None,
) -> None:
    manifest.setdefault("scripts_invoked", []).append(
        {
            "script": script_name,
            "command": command,
            "parameters": parameters,
            "exit_code": int(exit_code),
            "outputs": list(outputs or []),
            "timestamp_utc": utc_now_iso(),
        }
    )


def append_expected_artifacts(manifest: dict, group: str, artifacts: List[str]) -> None:
    manifest.setdefault("expected_artifacts", {})
    manifest["expected_artifacts"].setdefault(group, [])
    for a in artifacts:
        if a not in manifest["expected_artifacts"][group]:
            manifest["expected_artifacts"][group].append(a)


def finalize_manifest(manifest: dict) -> dict:
    expected = manifest.get("expected_artifacts", {}) or {}
    missing = {}
    checksums = {}
    for group, paths in expected.items():
        miss = []
        group_sums = {}
        for p in paths:
            if not os.path.isfile(p):
                miss.append(p)
                continue
            try:
                group_sums[p] = sha256_file(p)
            except Exception:
                miss.append(p)
        if miss:
            missing[group] = miss
        checksums[group] = group_sums
    manifest["artifact_sha256"] = checksums
    manifest["missing_artifacts"] = missing
    manifest["metadata_checklist"] = validate_metadata_fields(manifest.get("metadata", {}))
    manifest["status"] = "passed" if (not missing and manifest["metadata_checklist"]["pass"]) else "failed"
    manifest["finalized_at_utc"] = utc_now_iso()
    return manifest


def check_metadata_consistency_with_previous(previous: Optional[dict], current: dict) -> Dict[str, Any]:
    if not previous:
        return {"pass": True, "mismatches": {}}
    keys = ("config_hashes", "scenario_set")
    mismatches: Dict[str, Dict[str, Any]] = {}
    prev_meta = previous.get("metadata", {}) if isinstance(previous, dict) else {}
    cur_meta = current.get("metadata", {})
    for k in keys:
        if prev_meta.get(k) != cur_meta.get(k):
            mismatches[k] = {"previous": prev_meta.get(k), "current": cur_meta.get(k)}
    return {"pass": len(mismatches) == 0, "mismatches": mismatches}


# ---------------------------------------------------------------------------
# Per-artefact provenance sidecars
#
# The rejected manuscript printed 126.7 J in a figure and 135.0 J in the table
# because a reader silently picked up data from a different run and produced
# output that looked fresh. Two mechanisms prevent the recurrence: readers take a
# REQUIRED --results-dir with no default and no fallback (so stale input is
# impossible rather than merely unlikely), and every artefact they emit gets a
# sidecar recording exactly what went in. The sidecar matters because it survives
# someone copying the PDF out of its run directory.
# ---------------------------------------------------------------------------

class MissingResultsDir(SystemExit):
    """Raised as a hard exit when a required --results-dir is absent or wrong."""


def require_results_dir(path: Optional[str], *, argname: str = '--results-dir') -> str:
    """Validate an explicitly-supplied results directory. No default, no fallback.

    Deliberately strict: a reader must never guess where its input lives.
    """
    if not path:
        raise MissingResultsDir(
            f"{argname} is required and has no default. Pass the run directory "
            f"whose data you intend to read, e.g.\n"
            f"    {argname} results/statistics/run_20260901T120000Z\n"
            f"Refusing to fall back to the flat tracked path: that is how a stale "
            f"read produced 126.7 J in a figure and 135.0 J in the table of the "
            f"rejected submission."
        )
    if not os.path.isdir(path):
        raise MissingResultsDir(f"{argname}={path!r} is not a directory")
    return os.path.abspath(path)


def git_commit_info(repo_root: Optional[str] = None) -> Dict[str, Any]:
    """Current commit, branch and whether the working tree is dirty."""
    root = repo_root or repo_root_from_file(__file__)
    def _git(*args):
        try:
            p = subprocess.run(['git', *args], cwd=root, capture_output=True,
                               text=True, timeout=30)
            return p.stdout.strip() if p.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None
    status = _git('status', '--porcelain')
    return {
        'commit': _git('rev-parse', 'HEAD'),
        'commit_short': _git('rev-parse', '--short', 'HEAD'),
        'branch': _git('rev-parse', '--abbrev-ref', 'HEAD'),
        'dirty': bool(status) if status is not None else None,
        'dirty_files': sorted(l[3:] for l in status.splitlines())[:50] if status else [],
    }


def sidecar_path(output_path: str) -> str:
    """``fig3_adaptive_horizon.pdf`` -> ``fig3_adaptive_horizon.provenance.json``."""
    stem, _ = os.path.splitext(output_path)
    return stem + '.provenance.json'


def write_sidecar(output_path: str, *, run_id: str, inputs: List[str],
                  config: Optional[dict] = None, extra: Optional[dict] = None,
                  script: Optional[str] = None, force: bool = False) -> str:
    """Write ``<output>.provenance.json`` next to an artefact.

    ``inputs`` are hashed so a later reader can prove which bytes produced the
    figure, not merely which path was named.
    """
    from utils.safe_write import safe_write_json

    recorded_inputs = []
    for path in inputs:
        entry: Dict[str, Any] = {'path': path, 'exists': os.path.isfile(path)}
        if entry['exists']:
            entry['sha256'] = sha256_file(path)
            entry['bytes'] = os.path.getsize(path)
        recorded_inputs.append(entry)

    payload = {
        'artifact': os.path.basename(output_path),
        'run_id': run_id,
        'generated_utc': utc_now_iso(),
        'script': script or os.path.basename(sys.argv[0] or '?'),
        'argv': list(sys.argv),
        'git': git_commit_info(),
        'host': platform.node(),
        'python': sys.version.split()[0],
        'inputs': recorded_inputs,
        'effective_config': config if config is not None else {},
    }
    if extra:
        payload.update(extra)
    return safe_write_json(sidecar_path(output_path), payload, force=force)
