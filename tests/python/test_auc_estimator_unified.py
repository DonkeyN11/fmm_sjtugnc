#!/usr/bin/env python3
"""Regression test: every AUC in the pipeline uses ONE estimator.

The pipeline carried two. `verify_paper_numbers.auc` is the tie-corrected
Mann-Whitney rank statistic; seven other modules integrated the ROC step curve
with np.trapz / np.trapezoid. They disagree whenever scores tie, and tied scores
are the norm here: an epoch with no ground-truth edge scores a sentinel, and a
saturated posterior ties at 1.0. On the sigma-sweep corpus at sigma=1 m the two
give 0.4813 (trapz) and 0.6955 (rank) on identical labels -- chance versus real
discrimination. Table II reported the trapz column while every other table
reported the rank column, so the paper compared two matchers under two different
estimators depending on which table you read.

Commit f728584 adopted the rank statistic and recorded that the other scripts
"were never migrated". This test is the migration's guard.

⚠️ IT DELIBERATELY NEVER IMPORTS THE MODULES UNDER TEST. Most of them execute a
full pipeline at module scope -- `exp6_redraw_all.py` regenerates nine PNGs under
experiments/output/exp6_real/ merely by being imported, which is how this test
first announced itself by overwriting those figures twice. So the module under
test is parsed, the single function of interest is lifted out of the AST, and only
that function is executed. Nothing else in the file runs.

Run:  python tests/python/test_auc_estimator_unified.py
Exit: 0 on parity, 1 on divergence.
"""
from __future__ import annotations

import ast
import sys
import textwrap
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "experiments" / "scripts"

# (module, helper name). Every one must return the canonical rank AUC.
DELEGATES = [
    ("exp3_full_matching", "compute_roc_auc"),
    ("exp4_sigma_mismatch", "compute_roc_auc"),
    ("exp5_degraded_conditions", "compute_roc_auc"),
    ("exp6_real_accuracy", "auc"),
    ("exp6_redraw_all", "auc_score"),
    ("compare_real_metrics_enu", "auc"),
    ("utils", "compute_roc_auc"),
    ("exp7_statistical_tests", "auc_score"),
]

TRAPZ_NAMES = {"trapz", "trapezoid"}
CANONICAL_FN = "roc_curve_and_auc"


def parse_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def find_live_trapz(tree: ast.Module) -> list[int]:
    """Line numbers of live np.trapz/np.trapezoid CALLS anywhere in the module."""
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else (
            fn.id if isinstance(fn, ast.Name) else None)
        if name in TRAPZ_NAMES:
            lines.append(node.lineno)
    return sorted(lines)


def extract_function(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def call_lifted_function(tree: ast.Module, name: str, *args):
    """Exec ONLY `name` from the module and call it. Module-level code never runs."""
    fn = extract_function(tree, name)
    if fn is None:
        raise LookupError(f"no top-level function named {name!r}")
    wrapper = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(wrapper)
    code = compile(wrapper, filename=f"<lifted {name}>", mode="exec")
    # np is what these bodies reference; sys.path is set so their in-function
    # `from verify_paper_numbers import ...` resolves the way it does at runtime.
    namespace: dict = {"np": np, "__name__": f"_lifted_{name}"}
    exec(code, namespace)  # noqa: S102 - deliberate, and the body is ours
    return namespace[name](*args)


def describe_module_sideeffects(tree: ast.Module) -> list[str]:
    """Statement-level module-scope work: the reason this test lifts functions."""
    found = []
    for node in tree.body:
        if isinstance(node, (ast.For, ast.While, ast.With)):
            found.append(type(node).__name__)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            f = node.value.func
            nm = f.attr if isinstance(f, ast.Attribute) else (
                f.id if isinstance(f, ast.Name) else "?")
            if nm in ("savefig", "to_csv", "write_text", "write", "print"):
                found.append(f"{nm}()")
        if isinstance(node, ast.Assign):
            for sub in ast.walk(node.value):
                if isinstance(sub, ast.Call):
                    f = sub.func
                    nm = f.attr if isinstance(f, ast.Attribute) else (
                        f.id if isinstance(f, ast.Name) else "")
                    if nm in ("savefig", "to_csv"):
                        found.append(f"{nm}()")
    return sorted(set(found))


def main() -> int:
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))

    failures: list[str] = []
    trees = {}
    for module_name, _ in DELEGATES:
        trees[module_name] = parse_module(SCRIPTS / f"{module_name}.py")

    # ── 1. STATIC: no live trapz ROC integration, and delegation is present ──
    for module_name, tree in trees.items():
        live = find_live_trapz(tree)
        if live:
            failures.append(f"{module_name}:static")
            print(f"FAIL [static] — {module_name} still calls trapz at line(s) {live}")
        else:
            print(f"PASS [static] — {module_name}: no live trapz integration.")

    for module_name, attr in DELEGATES:
        tree = trees[module_name]
        fn = extract_function(tree, attr)
        if fn is None:
            failures.append(f"{module_name}:missing")
            print(f"FAIL [static] — {module_name} has no top-level {attr}()")
            continue
        body_src = ast.get_source_segment(
            (SCRIPTS / f"{module_name}.py").read_text(), fn) or ""
        if "verify_paper_numbers" not in body_src:
            failures.append(f"{module_name}:notdelegating")
            print(f"FAIL [static] — {module_name}.{attr} does not import from "
                  f"verify_paper_numbers")

    # ── 2. DYNAMIC: the lifted helpers agree with the canonical estimator ────
    rng = np.random.default_rng(20260922)
    n = 400
    scores = np.round(rng.random(n), 1)      # heavy quantization -> many ties
    labels = (rng.random(n) < 0.35).astype(float)

    canon_tree = trees["exp3_full_matching"]   # any tree can reach the canonical fn
    _ = canon_tree
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_canon", SCRIPTS / "verify_paper_numbers.py")
    assert spec and spec.loader
    canon = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(canon)             # verify_paper_numbers is import-safe
    expected = float(canon.auc(labels, scores))

    trapz_fn = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    order = np.argsort(scores)[::-1]
    ls = labels[order]
    n_pos, n_neg = ls.sum(), n - ls.sum()
    if trapz_fn is not None and n_pos > 0 and n_neg > 0:
        tpr = np.concatenate([[0], np.cumsum(ls) / n_pos])
        fpr = np.concatenate([[0], np.cumsum(1 - ls) / n_neg])
        tv = float(trapz_fn(tpr, fpr))
        print(f"     tie-heavy set: rank {expected:.6f} vs trapz {tv:.6f} "
              f"(delta {abs(expected - tv):.6f}) — the divergence this guards")

    _, _, pair = canon.roc_curve_and_auc(labels, scores)
    if abs(pair - expected) > 1e-12:
        failures.append("canonical:pair")
        print(f"FAIL [dynamic] — roc_curve_and_auc returned {pair!r}, auc() {expected!r}")
    else:
        print("PASS [dynamic] — canonical curve/scalar pair agree.")

    for module_name, attr in DELEGATES:
        try:
            result = call_lifted_function(trees[module_name], attr,
                                          labels.copy(), scores.copy())
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{module_name}:error")
            print(f"FAIL [dynamic] — {module_name}.{attr} raised "
                  f"{type(exc).__name__}: {exc}")
            continue
        if isinstance(result, tuple) and len(result) == 3:
            scalars = [x for x in result
                       if np.isscalar(x) or np.ndim(x) == 0]
            got = float(scalars[0]) if scalars else float("nan")
        else:
            got = float(result)
        if abs(got - expected) > 1e-12:
            failures.append(module_name)
            print(f"FAIL [dynamic] — {module_name}.{attr} gave {got!r}, "
                  f"canonical {expected!r} (delta {abs(got - expected):.6f})")
        else:
            print(f"PASS [dynamic] — {module_name}.{attr} matches canonical.")

    # ── 3. Report which modules would have side-effected had we imported them ─
    print()
    for module_name, tree in trees.items():
        se = describe_module_sideeffects(tree)
        if se:
            print(f"NOTE — {module_name} does module-scope work on import "
                  f"({', '.join(se)}); lifted, not imported.")

    print()
    if failures:
        print(f"FAIL — {len(failures)} AUC site(s) diverged from the canonical estimator.")
        return 1
    print("PASS — every AUC in the pipeline uses the tie-corrected rank statistic.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
