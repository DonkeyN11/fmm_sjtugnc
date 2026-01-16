#!/usr/bin/env python3
"""
Generate a chain of evidence plots for covariance-based emission vs isotropic model.

Plots:
  1) Emission ellipse vs isotropic circle at a single epoch
  2) Consistency suite (Mahalanobis hist, P-P, whitening, radial CDF)
  3) Viterbi log-likelihood margin CDF (requires n_best_trustworthiness)
  4) Position error vs PL envelope
  5) Summary metrics (consistency, margin rejection, PL tightness)
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from statistics import NormalDist

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - optional dependency guard
    raise SystemExit("matplotlib is required to run this script.") from exc

try:
    import geopandas as gpd
    from shapely.geometry import box, LineString, MultiLineString
except ImportError:
    gpd = None  # type: ignore
    box = None  # type: ignore
    LineString = None  # type: ignore
    MultiLineString = None  # type: ignore


CHI2_2_CDF = lambda x: 1.0 - math.exp(-0.5 * max(x, 0.0))


def _rayleigh_cdf(r: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-9)
    return 1.0 - np.exp(-(r ** 2) / (2.0 * sigma ** 2))


@dataclass
class PlotInputs:
    observations: Path
    truth_points: Path
    metadata: Path
    cmm_result: Path | None
    fmm_result: Path | None
    shapefile: Path | None


def _read_csv_auto(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    if len(df.columns) == 1:
        df = pd.read_csv(path, sep=",")
    return df


def _pick_xy_columns(df: pd.DataFrame) -> Tuple[str, str]:
    if {"x_m", "y_m"}.issubset(df.columns):
        return "x_m", "y_m"
    if {"x", "y"}.issubset(df.columns):
        return "x", "y"
    raise ValueError("Missing x/y columns in input data.")


def _pick_id_seq_columns(df: pd.DataFrame) -> Tuple[str, str]:
    if {"id", "seq"}.issubset(df.columns):
        return "id", "seq"
    if {"traj_id", "point_idx"}.issubset(df.columns):
        return "traj_id", "point_idx"
    raise ValueError("Missing id/seq columns in input data.")


def _chi2_inv_approx(prob: float, dof: int = 2) -> float:
    if not (0.0 < prob < 1.0):
        raise ValueError("chi-square probability must be in (0, 1)")
    if dof <= 0:
        raise ValueError("chi-square dof must be positive")
    # Wilson-Hilferty approximation
    z = NormalDist().inv_cdf(prob)
    k = float(dof)
    return k * (1.0 - 2.0 / (9.0 * k) + z * math.sqrt(2.0 / (9.0 * k))) ** 3


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


def _cov_from_row(row: pd.Series) -> np.ndarray:
    sde = float(row["sde"])
    sdn = float(row["sdn"])
    sdne = float(row.get("sdne", 0.0))
    return np.array([[sde * sde, sdne], [sdne, sdn * sdn]], dtype=float)


def _mahalanobis_d2(err: np.ndarray, cov: np.ndarray) -> float:
    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return float("nan")
    return float(err.T @ inv @ err)


def _whiten_error(err: np.ndarray, cov: np.ndarray) -> np.ndarray | None:
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        return None
    return np.linalg.solve(L, err)


def _parse_n_best_series(series: pd.Series) -> List[Tuple[float, float]]:
    values: List[Tuple[float, float]] = []
    for cell in series:
        if not isinstance(cell, str):
            continue
        parts = [p for p in cell.split("|") if p]
        for token in parts:
            token = token.strip().lstrip("(").rstrip(")")
            nums = [n.strip() for n in token.split(",") if n.strip()]
            if len(nums) < 2:
                continue
            try:
                best = float(nums[0])
                second = float(nums[1])
            except ValueError:
                continue
            if math.isfinite(best) and math.isfinite(second):
                values.append((best, second))
    return values


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _determine_utm_epsg(lon_deg: float, lat_deg: float) -> int | None:
    if not (np.isfinite(lon_deg) and np.isfinite(lat_deg)):
        return None
    if lat_deg <= -80.0 or lat_deg >= 84.0:
        return None
    zone = int(math.floor((lon_deg + 180.0) / 6.0)) + 1
    zone = max(1, min(zone, 60))
    base = 32600 if lat_deg >= 0.0 else 32700
    return base + zone


def _load_road_background(shapefile: Path, center: Tuple[float, float], buffer_m: float) -> List[np.ndarray]:
    if gpd is None or box is None:
        raise SystemExit("geopandas/shapely are required for plot 1 background.")
    gdf = gpd.read_file(shapefile)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326, allow_override=True)
    lonlat = gdf.to_crs(4326) if gdf.crs.to_epsg() != 4326 else gdf
    min_lon, min_lat, max_lon, max_lat = lonlat.total_bounds
    epsg = _determine_utm_epsg((min_lon + max_lon) * 0.5, (min_lat + max_lat) * 0.5)
    if epsg is None:
        raise SystemExit("Failed to determine UTM EPSG for shapefile.")
    projected = lonlat.to_crs(epsg)
    cx, cy = center
    bbox = box(cx - buffer_m, cy - buffer_m, cx + buffer_m, cy + buffer_m)
    clipped = projected[projected.geometry.intersects(bbox)]
    segments: List[np.ndarray] = []
    for geom in clipped.geometry:
        if isinstance(geom, LineString):
            segments.append(np.asarray(geom.coords, dtype=float))
        elif isinstance(geom, MultiLineString):
            for part in geom.geoms:
                segments.append(np.asarray(part.coords, dtype=float))
    return segments


def plot_emission_ellipse(
    df_obs: pd.DataFrame,
    df_truth: pd.DataFrame,
    sigma_map: Dict[int, float],
    traj_id: int,
    seq: int,
    shapefile: Path | None,
    buffer_m: float,
    output: Path,
) -> None:
    id_col, seq_col = _pick_id_seq_columns(df_obs)
    x_col, y_col = _pick_xy_columns(df_obs)
    t_x, t_y = _pick_xy_columns(df_truth)

    obs = df_obs[(df_obs[id_col] == traj_id) & (df_obs[seq_col] == seq)]
    truth = df_truth[(df_truth[id_col] == traj_id) & (df_truth[seq_col] == seq)]
    if obs.empty or truth.empty:
        raise ValueError("Requested traj_id/seq not found in observations/truth.")
    obs_row = obs.iloc[0]
    truth_row = truth.iloc[0]

    center = (float(obs_row[x_col]), float(obs_row[y_col]))
    cov = _cov_from_row(obs_row)
    sigma_iso = sigma_map.get(int(traj_id), float("nan"))
    if not math.isfinite(sigma_iso):
        sigma_iso = math.sqrt(max(cov[0, 0] + cov[1, 1], 1e-9) * 0.5)

    chi2_95 = _chi2_inv_approx(0.95, dof=2)

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    if shapefile is not None:
        try:
            segments = _load_road_background(shapefile, center, buffer_m)
            for seg in segments:
                ax.plot(seg[:, 0], seg[:, 1], color="0.85", lw=0.8, zorder=1)
        except SystemExit as exc:
            print(f"[plot 1] {exc}")

    ax.scatter([center[0]], [center[1]], s=30, color="black", zorder=5, label="GNSS obs")
    ax.scatter(
        [float(truth_row[t_x])],
        [float(truth_row[t_y])],
        s=30,
        color="tab:green",
        zorder=5,
        label="Ground truth",
    )

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    angle = math.degrees(math.atan2(eigvecs[1, 0], eigvecs[0, 0]))

    theta = np.linspace(0, 2 * math.pi, 200)
    for m in (1.0, 2.0, 3.0):
        r = m
        w = 2 * r * math.sqrt(eigvals[0])
        h = 2 * r * math.sqrt(eigvals[1])
        ell = _ellipse_points(center, w, h, angle, theta)
        ax.plot(ell[:, 0], ell[:, 1], color="0.5", lw=1.0, linestyle="--")

    r95 = math.sqrt(chi2_95)
    w95 = 2 * r95 * math.sqrt(eigvals[0])
    h95 = 2 * r95 * math.sqrt(eigvals[1])
    ell95 = _ellipse_points(center, w95, h95, angle, theta)
    ax.plot(ell95[:, 0], ell95[:, 1], color="tab:blue", lw=2, label="Covariance 95%")

    r_iso = math.sqrt(chi2_95) * sigma_iso
    circle = np.column_stack([center[0] + r_iso * np.cos(theta), center[1] + r_iso * np.sin(theta)])
    ax.plot(circle[:, 0], circle[:, 1], color="tab:orange", lw=1.8, label="Isotropic 95%")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Emission contours: covariance-based vs isotropic")
    ax.legend(loc="upper right")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _ellipse_points(center: Tuple[float, float], width: float, height: float, angle_deg: float, theta: np.ndarray) -> np.ndarray:
    cx, cy = center
    angle = math.radians(angle_deg)
    x = 0.5 * width * np.cos(theta)
    y = 0.5 * height * np.sin(theta)
    rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    pts = np.stack([x, y], axis=1) @ rot.T
    pts[:, 0] += cx
    pts[:, 1] += cy
    return pts


def plot_mahalanobis_hist(
    errors: np.ndarray,
    covs: np.ndarray,
    sigma_iso: np.ndarray,
    output: Path,
    x_max: float | None,
    hist_alpha: float,
    fit_alpha: float,
) -> Tuple[float, float]:
    d2_cov = []
    d2_iso = []
    for err, cov, sig in zip(errors, covs, sigma_iso):
        if not np.all(np.isfinite(err)) or not np.all(np.isfinite(cov)):
            continue
        d2_cov.append(_mahalanobis_d2(err, cov))
        if sig > 0 and math.isfinite(sig):
            d2_iso.append(float(err[0] ** 2 + err[1] ** 2) / (sig * sig))
    d2_cov = np.asarray(d2_cov, dtype=float)
    d2_iso = np.asarray(d2_iso, dtype=float)
    d2_cov = d2_cov[np.isfinite(d2_cov)]
    d2_iso = d2_iso[np.isfinite(d2_iso)]
    if d2_cov.size == 0 or d2_iso.size == 0:
        raise ValueError("Insufficient samples for Mahalanobis histogram.")

    if x_max is not None:
        d2_cov = d2_cov[d2_cov <= x_max]
        d2_iso = d2_iso[d2_iso <= x_max]
        hist_max = x_max
    else:
        hist_max = max(10.0, np.percentile(np.concatenate([d2_cov, d2_iso]), 99.5))
    if d2_cov.size == 0 or d2_iso.size == 0:
        raise ValueError("Insufficient samples after applying x-axis limit.")

    xs = np.linspace(0.0, hist_max, 400)
    chi_pdf = 0.5 * np.exp(-0.5 * xs)
    bins = 60
    if x_max is not None:
        bins = max(30, min(160, int(round(hist_max * 5))))

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.hist(d2_cov, bins=bins, density=True, alpha=hist_alpha, color="tab:blue", label="Covariance-based")
    ax.hist(d2_iso, bins=bins, density=True, alpha=hist_alpha, color="tab:orange", label="Isotropic")
    ax.plot(xs, chi_pdf, color="black", lw=2, alpha=fit_alpha, label=r"$\chi^2(2)$ PDF")
    ax.set_xlabel(r"Mahalanobis distance$^2$")
    ax.set_ylabel("PDF")
    ax.set_title("Normalized innovation distribution")
    if x_max is not None:
        ax.set_xlim(0.0, x_max)
    ax.legend(loc="upper right")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)

    ks_cov = _ks_stat(d2_cov)
    ks_iso = _ks_stat(d2_iso)
    return ks_cov, ks_iso


def plot_consistency_overview(
    errors: np.ndarray,
    covs: np.ndarray,
    sigma_iso: np.ndarray,
    output: Path,
    x_max: float | None,
    hist_alpha: float,
    fit_alpha: float,
    max_points: int = 3000,
) -> None:
    d2_cov = []
    d2_iso = []
    whitened = []
    radial = []
    sigma_eq = []
    for err, cov, sig in zip(errors, covs, sigma_iso):
        if not np.all(np.isfinite(err)) or not np.all(np.isfinite(cov)):
            continue
        d2_cov.append(_mahalanobis_d2(err, cov))
        if sig > 0 and math.isfinite(sig):
            d2_iso.append(float(err[0] ** 2 + err[1] ** 2) / (sig * sig))
        radial.append(float(np.hypot(err[0], err[1])))
        sigma_eq.append(math.sqrt(max(0.5 * float(np.trace(cov)), 1e-9)))
        if len(whitened) < max_points:
            w = _whiten_error(err, cov)
            if w is not None and np.all(np.isfinite(w)):
                whitened.append(w)

    d2_cov = np.asarray(d2_cov, dtype=float)
    d2_iso = np.asarray(d2_iso, dtype=float)
    d2_cov = d2_cov[np.isfinite(d2_cov)]
    d2_iso = d2_iso[np.isfinite(d2_iso)]
    radial = np.asarray(radial, dtype=float)
    sigma_eq = np.asarray(sigma_eq, dtype=float)
    if d2_cov.size == 0 or d2_iso.size == 0:
        raise ValueError("Insufficient samples for consistency overview.")

    if x_max is not None:
        d2_cov = d2_cov[d2_cov <= x_max]
        d2_iso = d2_iso[d2_iso <= x_max]
        hist_max = x_max
    else:
        hist_max = max(10.0, np.percentile(np.concatenate([d2_cov, d2_iso]), 99.5))
    bins = 60
    if x_max is not None:
        bins = max(30, min(160, int(round(hist_max * 5))))

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    ax_hist, ax_pp, ax_white, ax_rad = axes.flat

    xs = np.linspace(0.0, hist_max, 400)
    chi_pdf = 0.5 * np.exp(-0.5 * xs)
    ax_hist.hist(d2_cov, bins=bins, density=True, alpha=hist_alpha, color="tab:blue", label="Covariance-based")
    ax_hist.hist(d2_iso, bins=bins, density=True, alpha=hist_alpha, color="tab:orange", label="Isotropic")
    ax_hist.plot(xs, chi_pdf, color="black", lw=2, alpha=fit_alpha, label=r"$\chi^2(2)$ PDF")
    ax_hist.set_xlabel(r"Mahalanobis distance$^2$")
    ax_hist.set_ylabel("PDF")
    ax_hist.set_title("Normalized innovation")
    if x_max is not None:
        ax_hist.set_xlim(0.0, x_max)
    ax_hist.legend(loc="upper right")

    def _pp_plot(ax, samples: np.ndarray, label: str, color: str) -> None:
        sorted_vals = np.sort(samples)
        n = sorted_vals.size
        emp = np.arange(1, n + 1) / n
        theo = np.array([CHI2_2_CDF(x) for x in sorted_vals])
        ax.plot(theo, emp, color=color, lw=2, label=label)

    ax_pp.plot([0, 1], [0, 1], color="0.5", lw=1.2, linestyle="--")
    _pp_plot(ax_pp, d2_cov, "Covariance-based", "tab:blue")
    _pp_plot(ax_pp, d2_iso, "Isotropic", "tab:orange")
    ax_pp.set_xlabel("Theoretical CDF")
    ax_pp.set_ylabel("Empirical CDF")
    ax_pp.set_title("P-P plot (chi-square)")
    ax_pp.legend(loc="lower right")

    if whitened:
        w = np.asarray(whitened, dtype=float)
        ax_white.scatter(w[:, 0], w[:, 1], s=6, alpha=0.25, color="tab:blue", edgecolors="none")
        theta = np.linspace(0, 2 * math.pi, 200)
        ax_white.plot(np.cos(theta), np.sin(theta), color="black", lw=1.5, label="Unit circle")
        ax_white.plot(2 * np.cos(theta), 2 * np.sin(theta), color="black", lw=1.0, linestyle="--", label="2σ")
        ax_white.set_aspect("equal", adjustable="box")
        ax_white.set_xlabel("Whitened X")
        ax_white.set_ylabel("Whitened Y")
        ax_white.set_title("Whitened errors")
        ax_white.legend(loc="upper right", fontsize=8)
    else:
        ax_white.axis("off")

    r_sorted = np.sort(radial)
    emp_cdf = np.arange(1, r_sorted.size + 1) / r_sorted.size
    r_grid = np.linspace(0.0, np.percentile(radial, 99.5), 300)
    cov_cdf = _rayleigh_cdf(r_grid[:, None], sigma_eq[None, :]).mean(axis=1)
    iso_sigma = float(np.nanmean(sigma_iso))
    iso_cdf = _rayleigh_cdf(r_grid, np.full_like(r_grid, iso_sigma))
    ax_rad.plot(r_sorted, emp_cdf, color="tab:blue", lw=2, label="MC radial CDF")
    ax_rad.plot(r_grid, cov_cdf, color="tab:blue", lw=1.5, linestyle="--", label="Rayleigh (cov)")
    ax_rad.plot(r_grid, iso_cdf, color="tab:orange", lw=1.5, linestyle="--", label="Rayleigh (iso)")
    ax_rad.set_xlabel(r"$|e|$ (m)")
    ax_rad.set_ylabel("CDF")
    ax_rad.set_title("Radial error distribution")
    ax_rad.legend(loc="lower right", fontsize=8)

    fig.suptitle("Consistency overview: Monte Carlo vs LS covariance", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_direction_consistency(
    df_obs: pd.DataFrame,
    df_truth: pd.DataFrame,
    sigma_map: Dict[int, float],
    output: Path,
) -> None:
    id_col, seq_col = _pick_id_seq_columns(df_obs)
    x_col, y_col = _pick_xy_columns(df_obs)
    t_x, t_y = _pick_xy_columns(df_truth)
    merged = df_obs.merge(df_truth, on=[id_col, seq_col], suffixes=("", "_truth"))
    merged = merged.sort_values([id_col, seq_col])
    if merged.empty:
        raise ValueError("No samples for direction consistency plot.")

    along = []
    cross = []
    along_sig = []
    cross_sig = []
    for _, group in merged.groupby(id_col):
        coords = group[[f"{t_x}_truth", f"{t_y}_truth"]].to_numpy(dtype=float)
        if coords.shape[0] < 2:
            continue
        diffs = np.gradient(coords, axis=0)
        for (dx, dy), (_, row) in zip(diffs, group.iterrows()):
            norm = math.hypot(dx, dy)
            if norm <= 1e-9:
                continue
            t = np.array([dx / norm, dy / norm])
            n = np.array([-t[1], t[0]])
            err = np.array([row[x_col] - row[f"{t_x}_truth"], row[y_col] - row[f"{t_y}_truth"]], dtype=float)
            cov = _cov_from_row(row)
            along.append(float(err @ t))
            cross.append(float(err @ n))
            along_sig.append(math.sqrt(max(float(t.T @ cov @ t), 1e-9)))
            cross_sig.append(math.sqrt(max(float(n.T @ cov @ n), 1e-9)))

    along = np.asarray(along)
    cross = np.asarray(cross)
    along_sig = np.asarray(along_sig)
    cross_sig = np.asarray(cross_sig)
    if along.size == 0 or cross.size == 0:
        raise ValueError("Insufficient samples for direction consistency plot.")

    sigma_iso = merged[id_col].map(sigma_map).astype(float).values
    sigma_iso = sigma_iso[np.isfinite(sigma_iso)]
    sigma_iso = float(np.nanmean(sigma_iso)) if sigma_iso.size else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, data, sig, title in [
        (axes[0], along, along_sig, "Along-road error"),
        (axes[1], cross, cross_sig, "Cross-road error"),
    ]:
        ax.hist(data, bins=60, density=True, alpha=0.5, color="tab:blue", label="MC (cov)")
        xs = np.linspace(np.percentile(data, 1), np.percentile(data, 99), 300)
        cov_sig = float(np.nanmean(sig))
        cov_pdf = (1.0 / (math.sqrt(2 * math.pi) * cov_sig)) * np.exp(-0.5 * (xs / cov_sig) ** 2)
        ax.plot(xs, cov_pdf, color="tab:blue", lw=1.8, linestyle="--", label="N(0, σ_cov)")
        if math.isfinite(sigma_iso):
            iso_pdf = (1.0 / (math.sqrt(2 * math.pi) * sigma_iso)) * np.exp(
                -0.5 * (xs / sigma_iso) ** 2
            )
            ax.plot(xs, iso_pdf, color="tab:orange", lw=1.8, linestyle="--", label="N(0, σ_iso)")
        ax.set_title(title)
        ax.set_xlabel("Meters")
        ax.set_ylabel("PDF")
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Direction-aligned error consistency")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _ks_stat(samples: np.ndarray) -> float:
    if samples.size == 0:
        return float("nan")
    sorted_vals = np.sort(samples)
    n = sorted_vals.size
    ecdf = np.arange(1, n + 1) / n
    cdf = np.array([CHI2_2_CDF(x) for x in sorted_vals])
    return float(np.max(np.abs(ecdf - cdf)))


def plot_margin_cdf(
    cmm_df: pd.DataFrame | None,
    fmm_df: pd.DataFrame | None,
    output: Path,
) -> Tuple[float | None, float | None]:
    cov_margin = None
    iso_margin = None

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    if cmm_df is not None and "n_best_trustworthiness" in cmm_df.columns:
        pairs = _parse_n_best_series(cmm_df["n_best_trustworthiness"])
        if pairs:
            cov_margin = np.array([best - second for best, second in pairs], dtype=float)
            cov_margin = cov_margin[np.isfinite(cov_margin)]
            xs = np.sort(cov_margin)
            ys = np.arange(1, xs.size + 1) / xs.size
            ax.plot(xs, ys, color="tab:blue", lw=2, label="Covariance-based")
    if fmm_df is not None and "n_best_trustworthiness" in fmm_df.columns:
        pairs = _parse_n_best_series(fmm_df["n_best_trustworthiness"])
        if pairs:
            iso_margin = np.array([best - second for best, second in pairs], dtype=float)
            iso_margin = iso_margin[np.isfinite(iso_margin)]
            xs = np.sort(iso_margin)
            ys = np.arange(1, xs.size + 1) / xs.size
            ax.plot(xs, ys, color="tab:orange", lw=2, label="Isotropic")

    ax.set_xlabel(r"$\Delta L = L_{best} - L_{2nd}$")
    ax.set_ylabel("CDF")
    ax.set_title("Viterbi log-likelihood margin")
    ax.legend(loc="lower right")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)

    return cov_margin, iso_margin


def plot_error_vs_pl(
    df_obs: pd.DataFrame,
    df_truth: pd.DataFrame,
    sigma_map: Dict[int, float],
    traj_id: int,
    output: Path,
    false_alarm: float = 1e-3,
) -> Tuple[float, float]:
    id_col, seq_col = _pick_id_seq_columns(df_obs)
    x_col, y_col = _pick_xy_columns(df_obs)
    t_x, t_y = _pick_xy_columns(df_truth)
    obs = df_obs[df_obs[id_col] == traj_id].sort_values(seq_col)
    truth = df_truth[df_truth[id_col] == traj_id].sort_values(seq_col)
    if obs.empty or truth.empty:
        raise ValueError("Trajectory not found for error vs PL plot.")
    merged = obs.merge(truth, on=[id_col, seq_col], suffixes=("", "_truth"))
    err_x = merged[x_col] - merged[f"{t_x}_truth"]
    err_y = merged[y_col] - merged[f"{t_y}_truth"]
    err = np.hypot(err_x.values, err_y.values)

    if "protection_level" in merged.columns:
        pl_cov = merged["protection_level"].astype(float).values
    else:
        covs = np.stack([_cov_from_row(row) for _, row in merged.iterrows()], axis=0)
        chi2 = _chi2_inv_approx(1.0 - false_alarm, dof=2)
        pl_cov = np.array([math.sqrt(chi2 * max(np.linalg.eigvalsh(cov))) for cov in covs], dtype=float)
    sigma = sigma_map.get(int(traj_id), float("nan"))
    if not math.isfinite(sigma):
        sigma = float(np.nanmean(np.sqrt(merged["sde"] ** 2 + merged["sdn"] ** 2) / math.sqrt(2)))
    chi2 = _chi2_inv_approx(1.0 - false_alarm, dof=2)
    pl_iso = np.full_like(err, math.sqrt(chi2) * sigma, dtype=float)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(err, color="black", lw=1.5, label="Position error")
    ax.plot(pl_cov, color="tab:blue", lw=2, label="PL (covariance-based)")
    ax.plot(pl_iso, color="tab:orange", lw=2, label="PL (isotropic)")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Meters")
    ax.set_title(f"Integrity envelope (traj {traj_id})")
    ax.legend(loc="upper right")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)

    pl95_cov = float(np.nanpercentile(pl_cov, 95))
    pl95_iso = float(np.nanpercentile(pl_iso, 95))
    return pl95_cov, pl95_iso


def plot_summary_metrics(
    ks_cov: float,
    ks_iso: float,
    margin_cov: np.ndarray | None,
    margin_iso: np.ndarray | None,
    pl95_cov: float,
    pl95_iso: float,
    margin_threshold: float,
    output: Path,
) -> None:
    labels = ["Consistency (1-KS)", "Margin > thr", "PL 95% (inv)"]
    cov_scores = []
    iso_scores = []

    cov_scores.append(1.0 - ks_cov if math.isfinite(ks_cov) else float("nan"))
    iso_scores.append(1.0 - ks_iso if math.isfinite(ks_iso) else float("nan"))

    if margin_cov is not None and margin_cov.size:
        cov_scores.append(float(np.mean(margin_cov > margin_threshold)))
    else:
        cov_scores.append(float("nan"))
    if margin_iso is not None and margin_iso.size:
        iso_scores.append(float(np.mean(margin_iso > margin_threshold)))
    else:
        iso_scores.append(float("nan"))

    cov_scores.append(1.0 / pl95_cov if pl95_cov > 0 else float("nan"))
    iso_scores.append(1.0 / pl95_iso if pl95_iso > 0 else float("nan"))

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, cov_scores, width, label="Covariance-based", color="tab:blue")
    ax.bar(x + width / 2, iso_scores, width, label="Isotropic", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Score (higher is better)")
    ax.set_title("Methodology summary metrics")
    ax.legend(loc="upper right")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def build_inputs(dataset_root: Path, args: argparse.Namespace) -> PlotInputs:
    obs = args.observations or dataset_root / "raw_data" / "observations.csv"
    truth = args.truth_points or dataset_root / "raw_data" / "ground_truth_points.csv"
    metadata = args.metadata or dataset_root / "raw_data" / "metadata.json"
    cmm = args.cmm_result or dataset_root / "mm_result" / "cmm_result.csv"
    fmm = args.fmm_result or dataset_root / "mm_result" / "fmm_result.csv"
    shapefile = args.shapefile
    return PlotInputs(Path(obs), Path(truth), Path(metadata), Path(cmm) if cmm else None, Path(fmm) if fmm else None, Path(shapefile) if shapefile else None)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot methodology evidence figures.")
    parser.add_argument("--dataset_root", type=Path, required=True, help="Dataset root directory.")
    parser.add_argument("--observations", type=Path, default=None, help="Override observations.csv path.")
    parser.add_argument("--truth_points", type=Path, default=None, help="Override ground_truth_points.csv path.")
    parser.add_argument("--metadata", type=Path, default=None, help="Override metadata.json path.")
    parser.add_argument("--cmm_result", type=Path, default=None, help="Override cmm_result.csv path.")
    parser.add_argument("--fmm_result", type=Path, default=None, help="Override fmm_result.csv path.")
    parser.add_argument("--shapefile", type=Path, default=None, help="Shapefile for plot 1 background.")
    parser.add_argument("--traj_id", type=int, default=1, help="Trajectory id for plots 1/4.")
    parser.add_argument("--seq", type=int, default=0, help="Sequence index for plot 1.")
    parser.add_argument("--buffer_m", type=float, default=300.0, help="Road buffer (m) for plot 1.")
    parser.add_argument("--output_dir", type=Path, default=Path("plots/methodology"), help="Output directory.")
    parser.add_argument("--plots", nargs="*", default=["1", "2", "3", "4", "5"], help="Which plots to generate.")
    parser.add_argument("--margin_threshold", type=float, default=1.0, help="Threshold for margin rejection rate.")
    parser.add_argument("--mahalanobis_xmax", type=float, default=None, help="Max x-axis value for fig2.")
    parser.add_argument("--hist_alpha", type=float, default=0.6, help="Alpha for histogram bins in fig2.")
    parser.add_argument("--fit_alpha", type=float, default=0.9, help="Alpha for chi-square fit curve in fig2.")
    parser.add_argument("--overview_max_points", type=int, default=3000, help="Max points for whitened scatter.")
    parser.add_argument("--no_show", action="store_true", help="Do not display plots.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plots = set(args.plots)
    inputs = build_inputs(args.dataset_root, args)

    df_obs = _read_csv_auto(inputs.observations)
    df_truth = _read_csv_auto(inputs.truth_points)
    sigma_map = _load_sigma_map(inputs.metadata)

    _ensure_output_dir(args.output_dir)

    ks_cov = float("nan")
    ks_iso = float("nan")
    pl95_cov = float("nan")
    pl95_iso = float("nan")
    margin_cov = None
    margin_iso = None

    if "1" in plots:
        plot_emission_ellipse(
            df_obs,
            df_truth,
            sigma_map,
            args.traj_id,
            args.seq,
            inputs.shapefile,
            args.buffer_m,
            args.output_dir / "fig1_emission_ellipse.png",
        )

    if "2" in plots:
        id_col, seq_col = _pick_id_seq_columns(df_obs)
        x_col, y_col = _pick_xy_columns(df_obs)
        t_x, t_y = _pick_xy_columns(df_truth)
        merged = df_obs.merge(df_truth, on=[id_col, seq_col], suffixes=("", "_truth"))
        err_x = merged[x_col] - merged[f"{t_x}_truth"]
        err_y = merged[y_col] - merged[f"{t_y}_truth"]
        errors = np.stack([err_x.values, err_y.values], axis=1)
        covs = np.stack([_cov_from_row(row) for _, row in merged.iterrows()], axis=0)
        sigma_iso = merged[id_col].map(sigma_map).fillna(merged["sde"]).astype(float).values
        ks_cov, ks_iso = plot_mahalanobis_hist(
            errors,
            covs,
            sigma_iso,
            args.output_dir / "fig2_mahalanobis_hist.png",
            args.mahalanobis_xmax,
            args.hist_alpha,
            args.fit_alpha,
        )
        plot_consistency_overview(
            errors,
            covs,
            sigma_iso,
            args.output_dir / "fig2_consistency_overview.png",
            args.mahalanobis_xmax,
            args.hist_alpha,
            args.fit_alpha,
            max_points=args.overview_max_points,
        )
        plot_direction_consistency(
            df_obs,
            df_truth,
            sigma_map,
            args.output_dir / "fig2_direction_consistency.png",
        )

    if "3" in plots:
        cmm_df = _read_csv_auto(inputs.cmm_result) if inputs.cmm_result and inputs.cmm_result.exists() else None
        fmm_df = _read_csv_auto(inputs.fmm_result) if inputs.fmm_result and inputs.fmm_result.exists() else None
        margin_cov, margin_iso = plot_margin_cdf(
            cmm_df,
            fmm_df,
            args.output_dir / "fig3_viterbi_margin_cdf.png",
        )

    if "4" in plots:
        pl95_cov, pl95_iso = plot_error_vs_pl(
            df_obs,
            df_truth,
            sigma_map,
            args.traj_id,
            args.output_dir / "fig4_error_vs_pl.png",
        )

    if "5" in plots:
        plot_summary_metrics(
            ks_cov,
            ks_iso,
            margin_cov,
            margin_iso,
            pl95_cov,
            pl95_iso,
            args.margin_threshold,
            args.output_dir / "fig5_summary_metrics.png",
        )

    if not args.no_show:
        print(f"Figures saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
