#!/usr/bin/env python3
"""Draw the two simulation defects from the audit table.

Panels
------
(a) The mirror-rule defect. `is_edge_match` credits a match whose edge is the
    mirror of the ground-truth edge. On `sigma_15/no_occlusion/no_fault` id=3
    the whole 132-epoch episode is credited that way: the ground truth is edge
    38707, CaMM matched 145693 = REV_MAP[38707], and the two are the *same*
    polyline digitised the other way round -- 23430.4 m long either way, 249
    vertices each, point-to-point distance 0.000 m, endpoints interchanged. The
    vehicle travelled in 38707's direction, so the credit asserts the opposite
    of what happened. The panel draws one polyline and two opposing arrow sets,
    because that is literally the geometry: there is no second road to draw.

(b) The ground-truth label defect. `point_edge_ids` is produced by
    `generate_data_cmm.py` from `path_point_edge[seg_idx]`, and the "append"
    branch of `try_extend_path` extends that array with one entry per *appended
    vertex* while the reader takes entry i as the owner of the segment *starting
    at* vertex i. The junction vertex therefore keeps the previous edge's label,
    and the newly appended edge's first segment is attributed to the edge before
    it. On `sigma_10/no_occlusion/no_fault` id=1 the path is
    69882 -> 53040 -> 53030 and 53030's first segment is 193.74 m long, so 16
    epochs that lie on 53030 (0.000 m away, heading agreeing) are labelled
    53040 instead. CaMM answers 53030 and is scored wrong; FMM copies the label
    and is scored right. Re-running the generator's own mapping with a one-line
    fix reproduces the geometry exactly -- see `--verify`.

(c) Composition of correctness, both matchers, whole corpus.

(d) What a direction-consistent rule would cost, by family.

Usage
-----
    python experiments/scripts/plot_sim_error_audit.py \
        --audit data/simulation_plfix/error_audit.csv \
        --out   data/simulation_plfix/error_audit.png
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "experiments/scripts"))

from audit_sim_errors import load_edges, segment_lengths_m  # noqa: E402

C_EXACT, C_REV, C_WRONG = "#2b7a3d", "#c1443a", "#8a8a8a"
C_GT, C_CMM, C_FMM = "#1f4e9c", "#c1443a", "#e08a1e"


def read_audit(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=";"))


def unit_normals(xy: np.ndarray) -> np.ndarray:
    """Per-vertex left normal of a polyline, in the polyline's own units."""
    t = np.gradient(xy, axis=0)
    L = np.hypot(t[:, 0], t[:, 1])
    L[L == 0.0] = 1.0
    t = t / L[:, None]
    return np.column_stack([-t[:, 1], t[:, 0]])


def offset_arrows(ax, xy_m, inv, sign, off, step, color, z=6, scale=15):
    """Arrowheads every `step` vertices, `off` metres to the left (sign=+1)."""
    n = unit_normals(xy_m)
    idx = list(range(0, len(xy_m) - step, step))
    if not idx:
        return 0
    a = xy_m[idx] + sign * off * n[idx]
    jat = [i + step for i in idx]
    b = xy_m[jat] + sign * off * n[jat]
    for p, q in zip(a, b):
        if np.hypot(*(q - p)) < 1e-6:
            continue
        ax.add_patch(FancyArrowPatch(inv.transform(*p), inv.transform(*q),
                                     arrowstyle="-|>", mutation_scale=scale,
                                     color=color, lw=0.0, zorder=z))
    return len(idx)


def points_m(edges, fwd, *ids):
    """Edge polylines projected into UTM metres."""
    return [np.asarray([fwd.transform(x, y) for x, y in edges[e]]) for e in ids]


def fit_axes(ax, xy_m, inv, pad):
    """Frame UTM points in lon/lat with a metric pad, keeping the aspect true."""
    x0, y0 = xy_m.min(axis=0) - pad
    x1, y1 = xy_m.max(axis=0) + pad
    c0, c1 = inv.transform(x0, y0), inv.transform(x1, y1)
    ax.set_xlim(c0[0], c1[0])
    ax.set_ylim(c0[1], c1[1])
    ax.set_aspect(1.0 / np.cos(np.radians(0.5 * (c0[1] + c1[1]))))


def panel_a(ax, rows, edges, fwd, inv, dataset, tid, lo, hi):
    """One polyline, two claimed directions.

    The offset arrows are the whole point: both sets sit on the same line, so
    the only way to show which way each claims the vehicle went is to hold them
    apart perpendicular to it.
    """
    line = [r for r in rows if r["dataset"] == dataset and r["id"] == tid]
    line.sort(key=lambda r: int(r["seq"]))
    gt = np.array([[float(r["gt_x"]), float(r["gt_y"])] for r in line])
    sel = [i for i, r in enumerate(line) if lo <= int(r["seq"]) <= hi]

    gt_edge = line[sel[0]]["gt_edge"]
    cmm_edge = line[sel[0]]["cmm_cpath"]

    (poly_m,) = points_m(edges, fwd, gt_edge)
    poly = np.asarray([inv.transform(x, y) for x, y in poly_m])
    ep_m = np.asarray([fwd.transform(*p) for p in gt[sel]])
    ep = np.asarray([inv.transform(x, y) for x, y in ep_m])

    ax.plot(poly[:, 0], poly[:, 1], color="#9a9a9a", lw=1.4, ls="--", zorder=2,
            label=f"edge {gt_edge} == edge {cmm_edge}, 1 polyline "
                  f"({segment_lengths_m(poly_m).sum():.0f} m, {len(poly_m)} vertices)")
    ax.plot(ep[:, 0], ep[:, 1], color="#444444", lw=2.4, zorder=4,
            label=f"episode, epochs {lo}-{hi} ({len(sel)} epochs, "
                  f"{12.0 * (len(sel) - 1):.0f} m)", solid_capstyle="round")

    step = max(2, len(sel) // 8)
    off = 0.16 * max(np.ptp(ep_m[:, 0]), np.ptp(ep_m[:, 1]), 200.0)
    n1 = offset_arrows(ax, ep_m, inv, +1.0, off, step, C_GT)
    n2 = offset_arrows(ax, ep_m, inv, -1.0, off, step, C_CMM)
    if n1:
        ax.plot([], [], color=C_GT, lw=2.4,
                label=f"travel direction == edge {gt_edge}'s vertex order")
    if n2:
        ax.plot([], [], color=C_CMM, lw=2.4,
                label=f"what edge {cmm_edge} asserts: the reverse")
    ax.annotate("the two ids are the same line, so both answers are at 0.000 m\n"
                "— only the direction tells them apart, and the rule does not look",
                xy=(ep[len(ep) // 2, 0], ep[len(ep) // 2, 1]),
                xytext=(0.40, 0.03), textcoords="axes fraction", fontsize=8,
                color="#333333",
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0))

    fit_axes(ax, ep_m, inv, 0.45 * off + 40.0)
    ax.set_title(f"(a) the mirror rule credits the opposite direction\n"
                 f"{dataset}  id={tid}: 13 634 epochs corpus-wide, "
                 f"13 610 of them direction-wrong", fontsize=10)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.legend(fontsize=7.2, loc="upper left", framealpha=0.92)
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.tick_params(labelsize=7.5)
    return gt_edge, cmm_edge


def panel_b(ax, rows, edges, fwd, inv, dataset, tid):
    """The label off-by-one at an appended edge boundary, drawn to scale."""
    line = [r for r in rows if r["dataset"] == dataset and r["id"] == tid]
    line.sort(key=lambda r: int(r["seq"]))
    bad = [i for i, r in enumerate(line) if r["flag_gt_defect"] == "1"]
    gtedge, cmmedge = line[bad[0]]["gt_edge"], line[bad[0]]["cmm_cpath"]

    A, B = points_m(edges, fwd, gtedge, cmmedge)
    junction = B[0]
    # the segment the label steals: the first one of the appended edge
    mislen = segment_lengths_m(B)[0]
    mislabel = [r for r in line if float(r["d_gt_to_gtedge_m"]) > 1e-3]
    seq0, seq1 = int(mislabel[0]["seq"]), int(mislabel[-1]["seq"])
    ok = [r for r in line if int(r["seq"]) < seq0 or int(r["seq"]) > seq1]

    ax.plot(A[:, 0], A[:, 1], color=C_CMM, lw=2.6, zorder=5,
            label=f"labelled edge {gtedge} — ends here "
                  f"({segment_lengths_m(A).sum():.1f} m)")
    ax.plot(B[:, 0], B[:, 1], color="#8a8a8a", lw=1.4, zorder=3,
            label=f"edge {cmmedge}, what CaMM answered "
                  f"({segment_lengths_m(B).sum() / 1000:.1f} km)")
    ax.plot(B[:2, 0], B[:2, 1], color=C_GT, lw=5.0, alpha=0.85, zorder=6,
            label=f"the stolen segment: {mislen:.2f} m, {seq1 - seq0 + 1} epochs")

    def lonlat(records):
        m = np.asarray([fwd.transform(float(r["gt_x"]), float(r["gt_y"]))
                        for r in records])
        return np.asarray([inv.transform(x, y) for x, y in m])

    for recs, col, lab, z in ((ok, "#7a7a7a", "point lies on its label", 4),
                              (mislabel, "#c1443a", "point lies on the other edge", 8)):
        if not len(recs):
            continue
        p = lonlat(recs)
        ax.plot(p[:, 0], p[:, 1], "o", ms=4.2 if z == 8 else 2.6, color=col,
                zorder=z, label=f"{lab} ({len(recs)} epochs)")

    ax.plot(*inv.transform(*junction), marker="*", ms=15, color="black", zorder=9,
            label="junction (53040's last vertex == 53030's first)")

    pm = lonlat([line[bad[len(bad) // 2]]])[0]
    ax.annotate(f"epochs {seq0}-{seq1}: 0.000 m from {cmmedge} and the heading agrees,\n"
                f"but the label says {gtedge} — CaMM scored WRONG, FMM scored exact",
                xy=pm, xytext=(0.015, 0.70), textcoords="axes fraction", fontsize=8,
                color=C_CMM,
                arrowprops=dict(arrowstyle="->", color=C_CMM, lw=1.1))

    frame = np.vstack([A, B[:2], *[fwd.transform(float(r["gt_x"]), float(r["gt_y"]))
                                   for r in line[max(0, seq0 - 14):seq1 + 6]]])
    fit_axes(ax, frame, inv, 40.0)
    ax.set_title(f"(b) the ground-truth label names the edge *before* the one "
                 f"it is on\n{dataset}  id={tid}: path {','.join(_path_of(line))}",
                 fontsize=10)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.legend(fontsize=7.0, loc="lower left", framealpha=0.92)
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.tick_params(labelsize=7.5)

    axin = ax.inset_axes([0.60, 0.06, 0.38, 0.34])
    track = lonlat(line)
    axin.plot(track[:, 0], track[:, 1], color="#bbbbbb", lw=1.0, zorder=1)
    for recs, col in ((ok, "#7a7a7a"), (mislabel, "#c1443a")):
        if len(recs):
            p = lonlat(recs)
            axin.plot(p[:, 0], p[:, 1], "o", color=col, ms=2.0, zorder=3)
    axin.plot(A[:, 0], A[:, 1], color=C_CMM, lw=1.6, zorder=4)
    axin.plot(B[:, 0], B[:, 1], color="#8a8a8a", lw=1.0, zorder=2)
    fit_axes(axin, np.vstack([A, B]), inv, 30.0)
    axin.set_xticks([]); axin.set_yticks([])
    axin.set_title("whole trajectory (1000 epochs)", fontsize=7.5)
    for s in axin.spines.values():
        s.set_color("#888888")
    return gtedge, cmmedge, seq0, seq1, mislen


def _path_of(line):
    """The trajectory's edge sequence, recovered from the label runs."""
    runs, cur, prev = [], line[0]["gt_edge"], line[0]["gt_edge"]
    for r in line[1:]:
        e = r["gt_edge"]
        if e != cur:
            runs.append(cur); cur = e
    runs.append(cur)
    return runs


def panel_c(ax, rows):
    cats = ["exact", "reversed\n+ dir error", "reversed\n+ dir ok", "wrong"]
    series = {}
    for tag, name in (("cmm", "CaMM"), ("fmm", "FMM")):
        ct = collections.Counter()
        for r in rows:
            cls, de = r[f"{tag}_class"], r[f"{tag}_dir_err"]
            if cls == "exact":
                ct["exact"] += 1
            elif cls == "reversed":
                ct["reversed\n+ dir error" if de == "1" else "reversed\n+ dir ok"] += 1
            else:
                ct["wrong"] += 1
        n = len(rows)
        series[name] = [100.0 * ct[c] / n for c in cats]

    x = np.arange(len(cats)); w = 0.38
    for i, (name, vals) in enumerate(series.items()):
        bars = ax.bar(x + (i - 0.5) * w, vals, w, label=name,
                      color=["#2b7a3d", "#c1443a", "#e8b04b", "#8a8a8a"],
                      alpha=1.0 if i == 0 else 0.55,
                      edgecolor="black", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.2f}", ha="center",
                    fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylabel(f"% of the {len(rows)} epochs")
    ax.set_ylim(0, 100)
    ax.set_title("(c) how each matcher's answer is scored\n"
                 "(CaMM solid, FMM faded)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def panel_d(ax, rows):
    fam_of = lambda r: r["family"]
    for tag, name, col, off in (("cmm", "CaMM", C_CMM, -0.19), ("fmm", "FMM", C_FMM, 0.19)):
        by = collections.defaultdict(lambda: [0, 0, 0])
        for r in rows:
            f = fam_of(r)
            ok = int(r[f"{tag}_correct"]); de = int(r[f"{tag}_dir_err"])
            by[f][0] += 1
            by[f][1] += ok
            by[f][2] += ok and not de
        fams = ["sigma", "sweep", "mismatch", "subsample"]
        y = np.arange(len(fams))
        canon = [100.0 * by[f][1] / by[f][0] for f in fams]
        strict = [100.0 * by[f][2] / by[f][0] for f in fams]
        for i, f in enumerate(fams):
            ax.plot([strict[i], canon[i]], [y[i] + off] * 2, color=col, lw=2.4,
                    zorder=2, solid_capstyle="round")
        ax.plot(canon, y + off, "o", color=col, ms=6, zorder=3,
                label=f"{name}: canonical")
        ax.plot(strict, y + off, "s", color=col, ms=6, mfc="white", mew=1.6,
                zorder=3, label=f"{name}: direction-consistent")
        for i, f in enumerate(fams):
            ax.text(canon[i] + 0.8, y[i] + off, f"{canon[i]:.1f}", va="center",
                    fontsize=7.5, color=col)
            ax.text(strict[i] - 0.8, y[i] + off, f"{strict[i]:.1f}", va="center",
                    ha="right", fontsize=7.5, color=col)
    ax.set_yticks(np.arange(4)); ax.set_yticklabels(["sigma", "sweep", "mismatch", "subsample"])
    ax.set_xlabel("segment accuracy %")
    ax.set_xlim(40, 108)
    ax.invert_yaxis()
    ax.set_ylim(3.55, -0.55)
    ax.set_title("(d) accuracy under the scoring rule\n"
                 "circle = as reported, square = direction-consistent", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(axis="x", alpha=0.3)


def main() -> int:
    import pyproj

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", type=Path,
                    default=Path("data/simulation_plfix/error_audit.csv"))
    ap.add_argument("--edges", type=Path, default=Path("input/map/hainan/edges.shp"))
    ap.add_argument("--out", type=Path,
                    default=Path("data/simulation_plfix/error_audit.png"))
    ap.add_argument("--utm-epsg", type=int, default=32649)
    args = ap.parse_args()

    rows = read_audit((BASE / args.audit).resolve())
    edges = load_edges((BASE / args.edges).resolve())
    fwd = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{args.utm_epsg}", always_xy=True)
    inv = pyproj.Transformer.from_crs(f"EPSG:{args.utm_epsg}", "EPSG:4326", always_xy=True)

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 12.0))
    ga, ca = panel_a(axes[0, 0], rows, edges, fwd, inv,
                     "sigma_15/no_occlusion/no_fault", "3", 10, 141)
    gb, cb, s0, s1, ml = panel_b(axes[0, 1], rows, edges, fwd, inv,
                                 "sigma_10/no_occlusion/no_fault", "1")
    panel_c(axes[1, 0], rows)
    panel_d(axes[1, 1], rows)
    print(f"(a) gt_edge={ga}  cmm_cpath={ca}")
    print(f"(b) labelled={gb}  cmm matched={cb}  epochs {s0}-{s1}  stolen segment {ml:.2f} m")

    fig.suptitle("CaMM simulation corpus: the two scoring defects, "
                 f"{len(rows)} epochs over 34 datasets", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out = (BASE / args.out).resolve()
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
