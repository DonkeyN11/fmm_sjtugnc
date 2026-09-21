#!/usr/bin/env python3
"""Regression test: the three simulation drivers must build the SAME FMM baseline config.

The whole point of comparing CaMM against "the classical HMM baseline" is that the baseline
is one object. exp4_sigma_mismatch.py drifted: it searched with <r>0.03</r>, which in a
degree-based network is ~3,340 m rather than the stated 300 m, set
<reverse_tolerance>0.0 instead of 0.1, and omitted <tw_use_viterbi>, so its reported ECE
was a forward-posterior ECE while every other table reported the native Viterbi-score ECE.

The drift was invisible because each script carried its own copy of the XML template and
nothing compared them. This test is that comparison.

Run:  python tests/python/test_fmm_config_parity.py
Exit: 0 on parity, 1 on divergence.
"""
from __future__ import annotations

import importlib.util
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "experiments" / "scripts"

# The parameters that must agree. <step> and output fields are presentation; these are the
# ones that change what is matched or how its confidence is scored.
BASELINE_KEYS = ("k", "r", "gps_error", "reverse_tolerance", "tw_use_viterbi")

DRIVERS = {
    "exp3_full_matching": "the sigma sweep (Table II)",
    "exp4_sigma_mismatch": "the sigma-mismatch study (Table IX)",
    "exp5_degraded_conditions": "the degraded-condition study (Table VII)",
}


def load(module_name: str):
    path = SCRIPTS / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"_parity_{module_name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def baseline_params(module_name: str) -> dict[str, str]:
    mod = load(module_name)
    xml_path = mod.build_fmm_xml("/dev/null", "/dev/null", "/dev/null", "/dev/null")
    try:
        params = ET.parse(xml_path).getroot().find("parameters")
        return {k: (params.findtext(k) if params.find(k) is not None else "<absent>")
                for k in BASELINE_KEYS}
    finally:
        Path(xml_path).unlink(missing_ok=True)


def main() -> int:
    observed = {}
    for name in DRIVERS:
        try:
            observed[name] = baseline_params(name)
        except Exception as exc:  # noqa: BLE001 - report, do not mask
            print(f"FAIL  {name}: could not build config ({type(exc).__name__}: {exc})")
            return 1

    reference = observed["exp3_full_matching"]
    failures = []
    for name, params in observed.items():
        for key in BASELINE_KEYS:
            if params[key] != reference[key]:
                failures.append((name, key, params[key], reference[key]))

    print(f"{'driver':<28} " + "  ".join(f"{k:>16}" for k in BASELINE_KEYS))
    for name, params in observed.items():
        print(f"{name:<28} " + "  ".join(f"{params[k]:>16}" for k in BASELINE_KEYS))
    print()

    if failures:
        print("FAIL — baseline config diverged between simulation drivers:")
        for name, key, got, want in failures:
            print(f"  {name}: <{key}> is {got!r}, exp3 uses {want!r}  ({DRIVERS[name]})")
        return 1

    print("PASS — all three drivers build an identical FMM baseline configuration.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
