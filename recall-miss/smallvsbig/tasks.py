"""Build and load the R7 task union.

Two families, one workspace layout (``state/small-vs-big/work/<id>/workspace``):

- ``agent-distill``: the 44 held-out records from agent-distill's
  ``state/tasks.jsonl``.  30 are injected bugs (export ``base_commit`` and apply
  the recorded AST mutation); 14 are the shootout's mined tasks (export
  ``parent`` and overlay the fix commit's test files).
- ``shootout``: the 14 tasks from employee-shootout's ``tasks/tasks.json.txt``,
  built the same mined way.  These 14 are the same repos/commits as the 14
  shootout-origin agent-distill records, but they carry their own ids (the R7
  spec asks for both sets; the overlap is reported in docs/small-vs-big.md).

Both builders only ever *read* the source repos; every byte is written under the
repo's gitignored ``state/small-vs-big``.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from . import config
from .mutations import apply_mutation

TEST_RE = re.compile(r"(^|/)(tests?/|conftest\.py$|test_[^/]*\.py$|[^/]*_test\.py$|pytest\.ini$)")
SPEC_FILES = {
    "agent-distill": config.TASKS / "agent-distill-heldout.jsonl.txt",
    "shootout": config.TASKS / "shootout.json.txt",
}


def repo_path(name: str) -> Path:
    if name == "orchestrator":
        return Path.home() / "theprojects" / "orchestrator"
    return config.QUEUE / name


def is_test(path: str) -> bool:
    return bool(TEST_RE.search(path))


def git(repo: Path, *args: str, timeout: int = 180) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} in {repo}: {r.stderr.strip()[:200]}")
    return r.stdout


def _git_bytes(repo: Path, spec: str) -> bytes | None:
    r = subprocess.run(["git", "-C", str(repo), "show", spec], capture_output=True, timeout=180)
    return r.stdout if r.returncode == 0 else None


def _export_ignored(repo: Path, rel: str) -> bool:
    r = subprocess.run(["git", "-C", str(repo), "check-attr", "export-ignore", "--", rel],
                       capture_output=True, text=True, timeout=60)
    return r.returncode == 0 and r.stdout.strip().endswith(": set")


def _worktree_files(repo: Path) -> list[str]:
    tracked = git(repo, "ls-files", "-z").split("\0")
    others = git(repo, "ls-files", "-z", "--others", "--exclude-standard").split("\0")
    return sorted({f for f in tracked + others if f})


def _overlay_export_ignored(repo: Path, dest: Path) -> None:
    """Restore the tracked files `git archive` drops because of export-ignore."""
    for rel in _worktree_files(repo):
        out = dest / rel
        if out.exists() or not _export_ignored(repo, rel):
            continue
        src = repo / rel
        if src.is_file():
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)


def _clear_pycache(source_file: Path) -> None:
    cache = source_file.parent / "__pycache__"
    if cache.is_dir():
        shutil.rmtree(cache, ignore_errors=True)


def purge_pycache(root: Path) -> int:
    """Drop every __pycache__ under a built workspace (stale-pyc guard)."""
    n = 0
    for cache in root.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
        n += 1
    return n


# --------------------------------------------------------------------- specs
def load_specs() -> list[dict]:
    """The 44 agent-distill held-out records + the 14 shootout records (58 entries)."""
    specs: list[dict] = []
    for line in SPEC_FILES["agent-distill"].read_text().splitlines():
        if line.strip():
            t = json.loads(line)
            t["source"] = "agent-distill"
            t["test_cmd"] = t.get("check_cmd") or "python3 -m pytest -q"
            specs.append(t)
    for t in json.loads(SPEC_FILES["shootout"].read_text()):
        t = dict(t)
        t["source"] = "shootout"
        t["origin"] = "shootout"
        t["split"] = "heldout"
        specs.append(t)
    seen: set[str] = set()
    out = []
    for t in specs:
        if t["id"] in seen:
            continue
        seen.add(t["id"])
        out.append(t)
    return out


def workspace_of(task: dict) -> Path:
    return config.WORK / task["id"] / "workspace"


# --------------------------------------------------------------------- build
def build_one(task: dict, dest: Path, force: bool = False) -> dict:
    """Materialize one task's pre-fix workspace under ``dest/workspace``."""
    meta_path = dest / "task.json"
    if meta_path.is_file() and not force:
        return json.loads(meta_path.read_text())
    repo = repo_path(task["repo"])
    workspace = dest / "workspace"
    if dest.exists():
        shutil.rmtree(dest)
    workspace.mkdir(parents=True)

    test_files: list[str] = []
    if task.get("origin") == "injected":
        archive = subprocess.run(["git", "-C", str(repo), "archive", task["base_commit"]],
                                 capture_output=True, timeout=300)
        if archive.returncode != 0:
            raise RuntimeError(f"git archive {task['base_commit']} failed in {repo}")
        subprocess.run(["tar", "-x", "-C", str(workspace)], input=archive.stdout, timeout=300)
        target = workspace / task["target_file"]
        target.write_text(apply_mutation(target.read_text(errors="replace"), task["mutation"]))
        _clear_pycache(target)
    else:
        archive = subprocess.run(["git", "-C", str(repo), "archive", task["parent"]],
                                 capture_output=True, timeout=300)
        if archive.returncode != 0:
            raise RuntimeError(f"git archive {task['parent']} failed in {repo}")
        subprocess.run(["tar", "-x", "-C", str(workspace)], input=archive.stdout, timeout=300)
        changed = [x for x in git(repo, "diff", "--name-only", task["parent"], task["commit"]).splitlines() if x]
        test_files = [f for f in changed if is_test(f)]
        for rel in test_files:
            blob = _git_bytes(repo, f"{task['commit']}:{rel}")
            p = workspace / rel
            if blob is None:
                if p.exists():
                    p.unlink()
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(blob)

    _overlay_export_ignored(repo, workspace)
    failing = _failing(workspace, task["test_cmd"], task.get("failing"))
    purge_pycache(workspace)
    meta = config.sanitize({**task, "workspace": str(workspace.relative_to(config.REPO_ROOT)),
                            "test_files": test_files, "failing": failing})
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def _failing(workspace: Path, test_cmd: str, recorded: list | None) -> list[str]:
    try:
        r = subprocess.run(["bash", "-lc", test_cmd], cwd=workspace, capture_output=True,
                           text=True, timeout=900)
    except subprocess.SubprocessError:
        return list(recorded or [])
    out = r.stdout + r.stderr
    found = [m.group(2) for m in re.finditer(r"(?m)^(FAILED|ERROR)\s+(\S+)", out)]
    return found or list(recorded or [])


def build_all(force: bool = False, ids: list[str] | None = None) -> list[dict]:
    specs = load_specs()
    if ids:
        want = set(ids)
        specs = [t for t in specs if t["id"] in want]
    manifest = []
    for t in specs:
        dest = config.WORK / t["id"]
        try:
            manifest.append(build_one(t, dest, force=force))
        except Exception as e:  # noqa: BLE001 - one broken task must not sink the set
            print(f"  !! {t['id']}: {e}", file=sys.stderr)
            manifest.append({"id": t["id"], "error": str(e)})
    config.WORK.mkdir(parents=True, exist_ok=True)
    (config.WORK / "manifest.json").write_text(json.dumps(manifest, indent=2))
    ok = sum(1 for m in manifest if not m.get("error"))
    print(f"built {ok}/{len(manifest)} tasks -> {config.WORK.relative_to(config.REPO_ROOT)}")
    return manifest


def load_manifest() -> list[dict]:
    path = config.WORK / "manifest.json"
    if not path.is_file():
        return []
    return json.loads(path.read_text())


def _failing_ok(meta: dict) -> bool:
    return bool(meta.get("failing"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the R7 task union (derived, gitignored).")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tasks", nargs="*", default=None)
    a = ap.parse_args(argv)
    manifest = build_all(force=a.force, ids=a.tasks)
    bad = [m["id"] for m in manifest if m.get("error")]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
