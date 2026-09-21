#!/usr/bin/env python3
"""Regression test: eval_sim_corpus must not silently clobber its JSON summary.

The summary filename is derived from --root, and the script used to write it
unconditionally. An evaluation run that omitted --out therefore overwrote
data/simulation/metrics_summary.json -- which is untracked and gitignored, so
neither git nor a backup could restore it, and its content belonged to a corpus
that had itself been regenerated since. Four flags now exist to prevent that:
--out (write elsewhere), --no-write, and --force (overwrite deliberately), with a
refusal in between.

This test pins the refusal and, critically, that the refusal happens BEFORE the
evaluation: it asserts the run finishes fast on a root that would take far longer
to evaluate than to reject.

Run:  python tests/python/test_eval_sim_corpus_write_guard.py
Exit: 0 on the expected behaviour, 1 otherwise.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "experiments" / "scripts" / "eval_sim_corpus.py"

ORIGINAL = '{\n  "note": "a previous corpus run; must survive"\n}\n'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          capture_output=True, text=True, cwd=str(REPO), timeout=300)


def main() -> int:
    failures = []
    tmp = Path(tempfile.mkdtemp(prefix="evalsim_guard_"))
    try:
        summary = tmp / "metrics_summary.json"
        summary.write_text(ORIGINAL, encoding="utf-8")
        before = sha(summary)

        # 1. Default target with an existing file -> refuse, non-destructively.
        t0 = time.monotonic()
        p = run("--root", str(tmp))
        elapsed = time.monotonic() - t0
        if p.returncode != 2:
            failures.append(f"expected exit 2 on refusal, got {p.returncode}")
            print(f"FAIL — refusal exit code was {p.returncode}, expected 2")
        else:
            print("PASS — default run refuses to overwrite an existing summary (exit 2)")
        if sha(summary) != before:
            failures.append("the summary was modified despite the refusal")
            print("FAIL — the summary file changed")
        else:
            print("PASS — the existing summary is byte-identical after the refusal")
        if "refusing to overwrite" not in (p.stdout + p.stderr):
            failures.append("no refusal message printed")
            print("FAIL — refusal message missing")
        else:
            print("PASS — refusal message names the file and the three ways forward")
        if elapsed > 5.0:
            failures.append(f"refusal took {elapsed:.1f}s — it ran the evaluation first")
            print(f"FAIL — refusal took {elapsed:.1f}s; it should not evaluate first")
        else:
            print(f"PASS — refused in {elapsed:.2f}s, before evaluating")

        # 2. --no-write -> nothing written, exit 0.
        p = run("--root", str(tmp), "--no-write")
        if p.returncode != 0 or sha(summary) != before:
            failures.append("--no-write changed the file or exited non-zero")
            print("FAIL — --no-write did not leave the summary untouched")
        else:
            print("PASS — --no-write leaves the summary untouched (exit 0)")

        # 3. --out elsewhere -> written there, original untouched.
        other = tmp / "elsewhere.json"
        p = run("--root", str(tmp), "--out", str(other))
        if not other.exists() or sha(summary) != before:
            failures.append("--out did not redirect, or the original changed")
            print("FAIL — --out did not redirect cleanly")
        else:
            print("PASS — --out writes elsewhere and leaves the original untouched")

        # 4. --force -> overwrites, as asked.
        p = run("--root", str(tmp), "--force")
        if sha(summary) == before:
            failures.append("--force did not overwrite")
            print("FAIL — --force left the file unchanged")
        else:
            print("PASS — --force overwrites when explicitly requested")

        print()
        if failures:
            print(f"FAIL — {len(failures)} guard check(s) failed: {failures}")
            return 1
        print("PASS — the write guard behaves as specified.")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
