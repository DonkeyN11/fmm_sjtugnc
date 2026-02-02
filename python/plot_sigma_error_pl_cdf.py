#!/usr/bin/env python3
"""
Plot per-trajectory error distributions (with covariance ellipse and PL),
Stanford-style error vs PL plot, and CDF of matching errors for a given sigma_pr.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required to run this script.") from exc


def _read_csv_auto(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    if len(df.columns) == 1:
        df = pd.read_csv(path, sep=",")
    return df


def _pick_id_seq_columns(df: pd.DataFrame) -> Tuple[str, str]:
    if {"id", "seq"}.issubset(df.columns):
        return "id", "seq"
    if {"traj_id", "point_idx"}.issubset(df.columns):
        return "traj_id", "point_idx"
    raise ValueError("Missing id/seq columns in input data.")


def _pick_xy_columns(df: pd.DataFrame) -> Tuple[str, str]:
    if {"x_m", "y_m"}.issubset(df.columns):
        return "x_m", "y_m"
    if {"x", "y"}.issubset(df.columns):
        return "x", "y"
    raise ValueError("Missing x/y columns in input data.")


def _load_sigma_map(metadata_path: Path) -> Dict[int, float]:
    import json

    with metadata_path.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    sigma_map: Dict[int, float] = {}
    for traj in meta.get("trajectories", []):
        traj_id = traj.get("id")
        if traj_id is None:
            continue
        sigma = traj.get("sigma_pr", traj.get("sigma_m", traj.get("sigma")))
        if sigma is None:
            continue
        sigma_map[int(traj_id)] = float(sigma)
    return sigma_map


def _cov_from_points(points: pd.DataFrame) -> np.ndarray:
    sde = points["sde"].astype(float)
    sdn = points["sdn"].astype(float)
    sdne = points.get("sdne", pd.Series(0.0, index=points.index)).astype(float)
    cov_xx = float(np.nanmean(sde ** 2))
    cov_yy = float(np.nanmean(sdn ** 2))
    cov_xy = float(np.nanmean(sdne))
    return np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=float)


def _covariance_ellipse(cov: np.ndarray, n_std: float = 2.0) -> Tuple[float, float, float]:
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    width, height = 2 * n_std * np.sqrt(np.maximum(eigvals, 0.0))
    angle = math.degrees(math.atan2(eigvecs[1, 0], eigvecs[0, 0]))
    return width, height, angle


def _ellipse_points(center: Tuple[float, float], width: float, height: float, angle_deg: float) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * math.pi, 240)
    angle = math.radians(angle_deg)
    x = 0.5 * width * np.cos(theta)
    y = 0.5 * height * np.sin(theta)
    rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    pts = np.stack([x, y], axis=1) @ rot.T
    pts[:, 0] += center[0]
    pts[:, 1] += center[1]
    return pts


def _chi2_inv_approx(prob: float, dof: int = 2) -> float:
    if not (0.0 < prob < 1.0):
        raise ValueError("chi-square probability must be in (0, 1)")
    if dof <= 0:
        raise ValueError("chi-square dof must be positive")
    z = np.sqrt(2.0) * math.erfcinv(2.0 * (1.0 - prob)) if hasattr(math, "erfcinv") else None
    if z is None:
        from statistics import NormalDist

        z = NormalDist().inv_cdf(prob)
    k = float(dof)
    return k * (1.0 - 2.0 / (9.0 * k) + z * math.sqrt(2.0 / (9.0 * k))) ** 3


def _limits_from_errors(errors: np.ndarray, percentile: float = 99.0) -> float:
    lim = np.percentile(np.abs(errors), percentile)
    return max(float(lim), 1e-6)


def _parse_linestring(text: str) -> List[Tuple[float, float]]:
    if not isinstance(text, str):
        return []
    txt = text.strip()
    if not txt.upper().startswith("LINESTRING"):
        return []
    open_idx = txt.find("(")
    close_idx = txt.rfind(")")
    if open_idx == -1 or close_idx == -1 or close_idx <= open_idx:
        return []
    body = txt[open_idx + 1 : close_idx]
    coords: List[Tuple[float, float]] = []
    for token in body.split(","):
        parts = token.strip().split()
        if len(parts) != 2:
            continue
        try:
            coords.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    return coords


def _collect_traj_errors(
    obs: pd.DataFrame, truth: pd.DataFrame, traj_id: int
) -> Tuple[np.ndarray, np.ndarray]:
    id_col, seq_col = _pick_id_seq_columns(obs)
    x_col, y_col = _pick_xy_columns(obs)
    t_x, t_y = _pick_xy_columns(truth)
    p = obs[obs[id_col] == traj_id].set_index(seq_col)
    t = truth[truth[id_col] == traj_id].set_index(seq_col)
    aligned = p.join(t[[t_x, t_y]], rsuffix="_truth")
    x_truth = f"{t_x}_truth" if t_x in p.columns else t_x
    y_truth = f"{t_y}_truth" if t_y in p.columns else t_y
    err_x = aligned[x_col] - aligned[x_truth]
    err_y = aligned[y_col] - aligned[y_truth]
    errors = np.stack([err_x.values, err_y.values], axis=1)
    errors = errors[np.all(np.isfinite(errors), axis=1)]
    return errors, p


def plot_traj_error_panels(
    obs: pd.DataFrame,
    truth: pd.DataFrame,
    traj_ids: Sequence[int],
    output: Path,
    percentile: float = 99.0,
) -> None:
    rows = math.ceil(len(traj_ids) / 4)
    cols = min(4, max(1, len(traj_ids)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 3.4), squeeze=False)

    all_errors = []
    for traj_id in traj_ids:
        errors, _ = _collect_traj_errors(obs, truth, traj_id)
        if errors.size:
            all_errors.append(errors)
    if all_errors:
        all_errors = np.vstack(all_errors)
        lim = _limits_from_errors(all_errors, percentile)
    else:
        lim = 1.0

    for ax, traj_id in zip(axes.flat, traj_ids):
        errors, p = _collect_traj_errors(obs, truth, traj_id)
        if errors.size == 0:
            ax.axis("off")
            ax.set_title(f"traj {traj_id}", fontsize=9)
            continue
        mu = errors.mean(axis=0)
        errors_c = errors - mu
        if errors_c.shape[0] >= 1000:
            ax.hexbin(errors_c[:, 0], errors_c[:, 1], gridsize=35, cmap="Blues", mincnt=1)
        else:
            ax.scatter(errors_c[:, 0], errors_c[:, 1], s=4, alpha=0.25, color="tab:blue", edgecolors="none")
        ax.axhline(0, color="0.7", lw=0.6)
        ax.axvline(0, color="0.7", lw=0.6)

        cov = _cov_from_points(p)
        w, h, ang = _covariance_ellipse(cov, n_std=2.0)
        ell = _ellipse_points((0.0, 0.0), w, h, ang)
        ax.plot(ell[:, 0], ell[:, 1], color="black", lw=1.6, linestyle="--")

        if "protection_level" in p.columns:
            pl = float(np.nanmean(p["protection_level"].astype(float)))
        else:
            chi2 = _chi2_inv_approx(0.999, dof=2)
            eig = np.linalg.eigvalsh(cov)
            pl = math.sqrt(chi2 * float(np.max(eig)))
        if math.isfinite(pl):
            theta = np.linspace(0, 2 * math.pi, 200)
            circ = np.column_stack([pl * np.cos(theta), pl * np.sin(theta)])
            ax.plot(circ[:, 0], circ[:, 1], color="tab:orange", lw=1.4)

        ax.set_xlim(-lim * 1.2, lim * 1.2)
        ax.set_ylim(-lim * 1.2, lim * 1.2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"traj {traj_id}", fontsize=9)
        ax.set_xlabel("Error X (m)", fontsize=8)
        ax.set_ylabel("Error Y (m)", fontsize=8)
        ax.tick_params(labelsize=7)

    for ax in axes.flat[len(traj_ids) :]:
        ax.axis("off")

    fig.suptitle("Per-trajectory error distribution with covariance ellipse and PL", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_stanford(
    obs: pd.DataFrame,
    truth: pd.DataFrame,
    traj_ids: Sequence[int],
    output: Path,
) -> None:
    id_col, seq_col = _pick_id_seq_columns(obs)
    x_col, y_col = _pick_xy_columns(obs)
    t_x, t_y = _pick_xy_columns(truth)
    merged = obs[obs[id_col].isin(traj_ids)].merge(truth, on=[id_col, seq_col], suffixes=("", "_truth"))
    err = np.hypot(
        (merged[x_col] - merged[f"{t_x}_truth"]).astype(float),
        (merged[y_col] - merged[f"{t_y}_truth"]).astype(float),
    )
    if "protection_level" in merged.columns:
        pl = merged["protection_level"].astype(float)
    else:
        covs = np.stack([_cov_from_points(merged.loc[idx:idx]) for idx in merged.index], axis=0)
        chi2 = _chi2_inv_approx(0.999, dof=2)
        pl = np.sqrt(chi2 * np.max(np.linalg.eigvalsh(covs), axis=1))
    err = err.to_numpy()
    pl = np.asarray(pl, dtype=float)
    mask = np.isfinite(err) & np.isfinite(pl)
    err = err[mask]
    pl = pl[mask]
    if err.size == 0:
        raise ValueError("No valid samples for Stanford plot.")

    max_val = max(np.percentile(err, 99.5), np.percentile(pl, 99.5))
    max_val = max(max_val, 1.0)
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.scatter(err, pl, s=6, alpha=0.25, color="tab:blue", edgecolors="none")
    ax.plot([0, max_val], [0, max_val], color="black", lw=1.5, linestyle="--", label="PL=Error")
    miss = np.sum(err > pl)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_xlabel("Position error (m)")
    ax.set_ylabel("Protection level (m)")
    ax.set_title(f"Stanford plot (violations: {miss}/{err.size})")
    ax.legend(loc="upper left")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _collect_match_errors(
    match_df: pd.DataFrame,
    truth_points: Dict[int, np.ndarray],
) -> np.ndarray:
    errors = []
    for _, row in match_df.iterrows():
        traj_id = int(row["id"])
        if traj_id not in truth_points:
            continue
        coords = _parse_linestring(row.get("pgeom", ""))
        if not coords:
            continue
        truth = truth_points[traj_id]
        match = np.asarray(coords, dtype=float)
        n = min(len(truth), len(match))
        if n <= 0:
            continue
        diff = match[:n, :2] - truth[:n, :2]
        err = np.hypot(diff[:, 0], diff[:, 1])
        errors.append(err)
    if not errors:
        return np.array([], dtype=float)
    return np.concatenate(errors, axis=0)


def plot_match_error_cdf(
    truth: pd.DataFrame,
    cmm_result: Path,
    fmm_result: Path,
    traj_ids: Sequence[int],
    output: Path,
) -> None:
    id_col, seq_col = _pick_id_seq_columns(truth)
    x_col, y_col = _pick_xy_columns(truth)
    truth_points: Dict[int, np.ndarray] = {}
    for traj_id, group in truth[truth[id_col].isin(traj_ids)].groupby(id_col):
        coords = group.sort_values(seq_col)[[x_col, y_col]].to_numpy(dtype=float)
        truth_points[int(traj_id)] = coords

    cmm_df = _read_csv_auto(cmm_result)
    fmm_df = _read_csv_auto(fmm_result)
    cmm_df = cmm_df[cmm_df["id"].isin(traj_ids)]
    fmm_df = fmm_df[fmm_df["id"].isin(traj_ids)]

    cmm_err = _collect_match_errors(cmm_df, truth_points)
    fmm_err = _collect_match_errors(fmm_df, truth_points)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    if cmm_err.size:
        xs = np.sort(cmm_err)
        ys = np.arange(1, xs.size + 1) / xs.size
        ax.plot(xs, ys, color="tab:blue", lw=2, label=f"CMM (N={xs.size})")
    if fmm_err.size:
        xs = np.sort(fmm_err)
        ys = np.arange(1, xs.size + 1) / xs.size
        ax.plot(xs, ys, color="tab:orange", lw=2, label=f"FMM (N={xs.size})")

    ax.set_xlabel("Matching error to truth (m)")
    ax.set_ylabel("CDF")
    ax.set_title("CDF of matching error")
    ax.set_yscale("log")
    ax.set_xlim(left=1e-3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot error/PL stats for a given sigma_pr.")
    parser.add_argument("--dataset_root", type=Path, required=True, help="Dataset root directory.")
    parser.add_argument("--sigma_pr", type=float, required=True, help="sigma_pr to filter trajectories.")
    parser.add_argument("--output_dir", type=Path, default=Path("plots/sigma_stats"), help="Output directory.")
    parser.add_argument("--observations", type=Path, default=None, help="Override observations.csv path.")
    parser.add_argument("--truth_points", type=Path, default=None, help="Override ground_truth_points.csv path.")
    parser.add_argument("--metadata", type=Path, default=None, help="Override metadata.json path.")
    parser.add_argument("--cmm_result", type=Path, default=None, help="Override cmm_result.csv path.")
    parser.add_argument("--fmm_result", type=Path, default=None, help="Override fmm_result.csv path.")
    parser.add_argument("--traj_ids", type=int, nargs="*", default=None, help="Optional explicit traj ids.")
    parser.add_argument("--percentile", type=float, default=99.0, help="Percentile for plot limits.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_root = args.dataset_root
    obs_path = args.observations or dataset_root / "raw_data" / "observations.csv"
    truth_path = args.truth_points or dataset_root / "raw_data" / "ground_truth_points.csv"
    metadata_path = args.metadata or dataset_root / "raw_data" / "metadata.json"
    cmm_path = args.cmm_result or dataset_root / "mm_result" / "cmm_result.csv"
    fmm_path = args.fmm_result or dataset_root / "mm_result" / "fmm_result.csv"

    obs = _read_csv_auto(obs_path)
    truth = _read_csv_auto(truth_path)
    sigma_map = _load_sigma_map(metadata_path)

    if args.traj_ids:
        traj_ids = [int(t) for t in args.traj_ids]
    else:
        traj_ids = [
            tid for tid, sigma in sigma_map.items() if math.isclose(float(sigma), args.sigma_pr, rel_tol=1e-6, abs_tol=1e-9)
        ]
    if not traj_ids:
        raise ValueError("No trajectories found for specified sigma_pr.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_traj_error_panels(
        obs,
        truth,
        traj_ids,
        output_dir / f"sigma_{args.sigma_pr:g}_traj_error_panels.png",
        percentile=args.percentile,
    )

    plot_stanford(
        obs,
        truth,
        traj_ids,
        output_dir / f"sigma_{args.sigma_pr:g}_stanford.png",
    )

    plot_match_error_cdf(
        truth,
        cmm_path,
        fmm_path,
        traj_ids,
        output_dir / f"sigma_{args.sigma_pr:g}_match_error_cdf.png",
    )

    print(f"Saved plots to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
