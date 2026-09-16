# Obsolete scripts — do not run

These scripts measured mechanisms that have been removed from the CMM
implementation so that it becomes a direct translation of the paper's §III
formulas. They are kept for provenance only. They are **not** runnable in any
meaningful sense, and — this is the important part — **they will not tell you
that**.

## Why re-running them fails silently

Each script writes a temporary XML config and invokes `build/cmm` on it. The
configuration loaders read every key with an explicit `get(key, default)` and
have no `else` branch, so **a key the loader does not know is ignored without a
warning, without an error, and without a non-zero exit code**:

- `CovarianceMapMatchConfig::load_from_xml` — [src/mm/cmm/cmm_algorithm.cpp:584](../../src/mm/cmm/cmm_algorithm.cpp#L584)
- the `<fields>` loop in `ResultConfig::load_from_xml` — [src/config/result_config.cpp:120](../../src/config/result_config.cpp#L120)

So a sweep over a removed knob runs to completion and returns identical results
for every value of that knob. The output is a flat curve that looks like a real
negative result. `subprocess.run(..., check=True)` does not catch it, because
the binary exits 0.

If you need one of these measurements again, the mechanism has to come back
first — which per the project's decisions requires a corresponding change to the
paper.

## The scripts

| Script | Measured | Removed by | Status of the claim |
|---|---|---|---|
| `run_temperature_controlled.py` | The adaptive softmax temperature, isolated by running the same binary twice with `temperature_adapt` on and off | Decision 1 | Measured a no-op before removal (ECE 0.0410 → 0.0400, AUC 0.7204 → 0.7203). The "Mechanisms deliberately absent" table in [CLAUDE.md](../../CLAUDE.md) carries the result. |
| `exp1_lag_sweep.py` | `lag_steps` sweep on one trajectory | Decision 4 | The claim survives as the fixed-lag entry in the same table (lag degrades TW calibration on real data with tight HPL: ECE 0.040 → 0.26 at L = 20). |
| `exp1_multitraj_lag_sweep.py` | `lag_steps` sweep across all trajectories | Decision 4 | As above. |
| `exp2_synthetic_validation.py` | `lag_steps` sweep on synthetic data, compared against a matching FMM baseline | Decision 4 | The sweep *is* the experiment, so there is nothing to keep. All of its input paths still resolve, which is exactly why it is dangerous to leave in place: it would run and print a flat table rather than fail. |
| `exp3_multiplier_sweep.py` | `protection_level_multiplier` and `phmi_pl_multiplier` sweeps | Decision 3 | Superseded by `experiments/scripts/exp3_full_matching.py`. The search radius is now exactly $\mathrm{HPL}_i$, so there is no multiplier to sweep. |
| `exp3_phmi_analysis.py` | PHMI-enabled vs baseline ECE, tabulated per lag, plus a PL-coverage analysis | Decision 4 | The PHMI half is still meaningful — PHMI grouping was kept — but the script is organised around the lag axis and its input path `data/real_vehicle/cmm_input_points.csv` no longer exists (the table now lives at `data/real_vehicle/hainan_06/`). Archive rather than rewrite: the surviving PHMI behaviour is documented in the Level 3 section of [CLAUDE.md](../../CLAUDE.md). |
| `exp4_ablation_ece.py` | A1–A4 ablation, where A3 = + fixed-lag smoothing and A4 = + lag + PHMI | Decisions 3 and 4 | Two of the four configurations no longer exist, so the A1–A4 decomposition is void. The surviving ablation studies are `experiments/scripts/{exp4_sigma_mismatch,exp5_degraded_conditions}.py`. |
| `exp8_ablation_components.py` | Cumulative component ablation whose configuration D was "+ entropy-aware temperature scaling τ + background state" | Decisions 1 and 7 | Both components of D are gone. Note that `direction_penalty` (component C) is still a live config key, so that half of the study is reproducible. |
| `regenerate_camm_iso_fixed_sigma.py` | Regeneration of the "CaMM-iso with FMM-matching fixed σ" variant | Decision 6 | Void on two counts: it sets `temperature_adapt`, and its premise — matching a fixed σ floor — no longer exists now that the `MIN_SIGMA` floor is replaced by a covariance-validity test plus an isotropic 5 m fallback. |
| `run_cmm_matching.py` | Standalone CMM batch runner | Never worked | Different failure mode from the rest: its config schema is fabricated and matches no loader, so it does not merely ignore keys — it cannot run at all. It writes `<id_column>`, `<source_column>`, `<target_column>` and `<gps_format>`, none of which any loader reads, so the network falls back to `NetworkConfig`'s defaults `id`/`source`/`target` while the Hainan shapefile uses `key`/`u`/`v`. Measured: `Field not found: id index -1, source index -1, target index -1` → `CMM application failed`. It also carries five removed keys. |
| `compare_temperature_ece.py` | The temperature ablation table: `cmm_result.csv.bak0806` (adaptive temperature off) vs `cmm_result.csv` (on), over σ ∈ {5, 15, 25} m × interval ∈ {1, 2, 5, 10} s | Decision 1 | The one archived script whose *inputs still exist* — all 24 `cmm_result.csv.bak0806` files are present, so it runs and prints a table. It is archived because neither side can be regenerated from the current code, so the table is frozen evidence rather than a repeatable measurement. Both result sets are retained under `data/simulation/`. |
| `exp3_deep_dive.py` | PHMI bin-level calibration, `phmi_pl_multiplier = 5` vs baseline | Decision 3 | Void on three counts, and this one does not even import: it opens `data/real_vehicle/cmm_input_points.csv` at module level (line 47), a path that no longer exists, so it raises `FileNotFoundError` immediately. Its other two inputs are `mr/multiplier_sweep/cmm_mult05.csv` and `mr/multitraj_sweep/cmm_all_lag000.csv` — the decision-3 and decision-4 sweeps. The surviving PHMI behaviour is documented in the Level 3 section of [CLAUDE.md](../../CLAUDE.md). |

## Related

`experiments/scripts/` is the live pipeline. Scripts there that set removed keys
incidentally — rather than being defined by them — were repaired in place rather
than archived; see the "Command-line and XML keys" note in
[CLAUDE.md](../../CLAUDE.md).
