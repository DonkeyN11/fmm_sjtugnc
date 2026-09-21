#!/usr/bin/env python3
"""One row per simulation epoch: the fix, the ground truth, and both matchers.

Written to make two defects inspectable rather than merely countable.

Defect 2 -- the reverse-edge scoring rule
-----------------------------------------
`verify_paper_numbers.is_edge_match` accepts a match when the matched edge is
the *mirror* of the ground-truth edge (`REV_MAP`). That rule exists because a
road network often carries the two directions of a street as two edges: a
trajectory that travelled west on the eastbound edge is not an error, it is an
artefact of which carriageway the label happened to name. But the same rule
also accepts a match that travelled the *wrong way* -- the mirrored edge is
geometrically the same polyline, so nothing in the ids says which happened.

This script separates them by measuring direction, which the ids cannot. For
every epoch it records

    cmm_class      exact | reversed | gt_rev | wrong | unresolved
    cmm_head_deg   heading of the matched edge's polyline, as digitised
    truth_head_deg heading of the ground-truth track over +-1 epoch
    cmm_dir_err    1 when the two differ by more than 90 degrees

`reversed` alone is not an error, and `exact` alone is not a proof of one: a
one-way edge matched `exact` in a trajectory travelling backwards is equally
wrong. `cmm_dir_err` is the test that does not depend on the labelling, and is
therefore the column to count.

Defect 4 -- ground-truth labels that do not sit on their own edge
----------------------------------------------------------------
`point_edge_ids` is derived per point, and on a minority of epochs the label
does not name the edge the point is actually on. The signature is a true
position tens of metres from the polyline it is labelled with while sitting
essentially on a different one. `d_gt_to_gtedge_m` measures exactly that -- the
distance from the ground-truth point to its own labelled edge -- and refers to
neither matcher, so it is an independent check on the label.
`flag_gt_defect` marks > `--gt-defect-m` (default 50).

Units and cost
--------------
Distances are metres in UTM 49N (EPSG:32649), the zone the generator projects
into; `--utm-epsg` overrides. Edge polylines are projected once and cached --
there are 152547 of them and the corpus has 247030 epochs, so per-epoch
projection is not affordable.

Usage
-----
    python experiments/scripts/audit_sim_errors.py \
        --root data/simulation_plfix --cmm-suffix _plfix --fmm-suffix _audit \
        --out data/simulation_plfix/error_audit.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "experiments/scripts"))

from verify_paper_numbers import REV_MAP  # noqa: E402  the canonical mirror rule

TW_SENTINEL = -900.0
CORRECT_CLASSES = ("exact", "reversed", "gt_rev")

COLUMNS = [
    "dataset", "family", "id", "seq", "timestamp",
    "gnss_x", "gnss_y", "gt_x", "gt_y", "gt_edge",
    "cmm_cpath", "cmm_tw", "cmm_class", "cmm_status", "cmm_head_deg",
    "fmm_cpath", "fmm_tw", "fmm_class", "fmm_status", "fmm_head_deg",
    "truth_head_deg", "gt_edge_head_deg",
    "err_gnss_m", "d_gnss_to_gtedge_m", "d_gt_to_gtedge_m",
    "d_gt_to_cmm_m", "d_gt_to_fmm_m",
    "len_gt_m", "len_cmm_m", "len_fmm_m", "gt_edge_n", "gt_edge_len_m",
    "cmm_correct", "fmm_correct", "flag_reverse_cmm", "flag_reverse_fmm",
    "cmm_dir_err", "fmm_dir_err", "flag_gt_defect",
]


# ── geometry ──────────────────────────────────────────────────────────────

def load_edges(shapefile: Path, field: str = "key") -> dict:
    """edge id -> (N, 2) lon/lat array.

    The datasource is parked in a module global on purpose: letting the
    reference fall out of scope frees the layer mid-iteration and segfaults.
    """
    from osgeo import ogr
    ds = ogr.Open(str(shapefile))
    if ds is None:
        raise SystemExit(f"cannot open {shapefile}")
    out = {}
    for feat in ds.GetLayer(0):
        gid = feat.GetField(field)
        geom = feat.GetGeometryRef()
        if gid is None or geom is None:
            continue
        pts = geom.GetPoints()
        if len(pts) >= 2:
            out[str(gid)] = np.asarray([(p[0], p[1]) for p in pts], dtype=float)
    globals()["_EDGE_DS"] = ds
    return out


def segment_lengths_m(xy: np.ndarray) -> np.ndarray:
    return np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))


def point_to_polyline_m(p: np.ndarray, xy: np.ndarray) -> tuple:
    """(distance in metres, arc length, local tangent bearing in degrees).

    Every segment gets a projection, and the closest one also supplies the
    direction: the bearing of *that segment*, not of the polyline's net
    displacement. The distinction is not pedantic -- edge 79844 on this network
    is 17.5 km long with a single 6.7 km segment and a 3.51 deg net bearing,
    while the segment a vehicle is actually on can point anywhere. Using the net
    displacement would call a vehicle driving along that edge perpendicular to
    it.
    """
    a, b = xy[:-1], xy[1:]
    ab = b - a
    seg2 = np.einsum("ij,ij->i", ab, ab)
    seg2 = np.where(seg2 > 0.0, seg2, 1.0)
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / seg2, 0.0, 1.0)
    foot = a + t[:, None] * ab
    d = np.hypot(foot[:, 0] - p[0], foot[:, 1] - p[1])
    k = int(np.argmin(d))
    seg = segment_lengths_m(xy)
    bear = float(np.degrees(np.arctan2(ab[k, 0], ab[k, 1])))
    return float(d[k]), float(seg[:k].sum() + t[k] * seg[k]), bear


def angle_diff_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


# ── loading ───────────────────────────────────────────────────────────────

def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=";"))


def parse_linestring(wkt: str) -> np.ndarray:
    """LINESTRING (x y, x y, ...) -> (N, 2). The generator emits 'x y' pairs."""
    body = wkt[wkt.index("(") + 1: wkt.rindex(")")]
    return np.asarray([[float(a), float(b)]
                       for a, b in (c.split() for c in body.split(","))], dtype=float)


def load_dataset(data_dir: Path, names: dict) -> dict:
    """The ground-truth track is read from `ground_truth.csv`'s LINESTRING.

    Its seq-th vertex *is* the seq-th ground-truth point: on the datasets that
    also ship `ground_truth_points.csv` the two agree to under a millimetre
    (the file rounds to 8 decimals), so the linestring is the same information
    without the family split -- `sigma_01_to_30` has no
    `ground_truth_points.csv`, and reading the truth from a different source
    per family is how the sweep silently scored as empty elsewhere.
    """
    gt_edge, gt_m, obs_xy, obs_ts, cmm, fmm = {}, {}, {}, {}, {}, {}

    for row in read_csv(data_dir / "ground_truth.csv"):
        tid = row["id"].strip()
        try:
            edges = json.loads(row["point_edge_ids"])
        except (json.JSONDecodeError, TypeError, KeyError):
            continue
        for seq, e in enumerate(edges):
            gt_edge[(tid, seq)] = str(e)
        if "geom" in row:
            line = parse_linestring(row["geom"])
            for seq, xy in enumerate(line):
                gt_m[(tid, seq)] = xy

    ts2seq = {}
    for row in read_csv(data_dir / "observations.csv"):
        tid, seq = row["id"].strip(), int(row["seq"])
        obs_xy[(tid, seq)] = (float(row["x"]), float(row["y"]))
        obs_ts[(tid, seq)] = float(row["timestamp"])
        ts2seq[(tid, int(round(float(row["timestamp"]))))] = seq

    for tag, dest in (("cmm", cmm), ("fmm", fmm)):
        path = data_dir / names[tag]
        if not path.exists():
            continue
        for row in read_csv(path):
            tid = row["id"].strip()
            seq = ts2seq.get((tid, int(round(float(row["timestamp"])))))
            if seq is None:
                continue  # an epoch this dataset never emitted
            try:
                tw = float(row.get("trustworthiness", TW_SENTINEL) or TW_SENTINEL)
            except (TypeError, ValueError):
                tw = TW_SENTINEL
            dest[(tid, seq)] = (row.get("cpath", "").strip(), tw, row.get("status", ""))

    return {"gt_edge": gt_edge, "gt_m": gt_m, "obs_xy": obs_xy, "obs_ts": obs_ts,
            "cmm": cmm, "fmm": fmm}


def classify(cpath: str, gtedge: str) -> str:
    if not cpath:
        return "unresolved"
    if cpath == gtedge:
        return "exact"
    if REV_MAP.get(cpath) == gtedge:
        return "reversed"
    if REV_MAP.get(gtedge) == cpath:
        return "gt_rev"
    return "wrong"


def family_of(rel: str) -> str:
    if rel.startswith("sigma_mismatch/"):
        return "mismatch"
    if "subsample_" in rel:
        return "subsample"
    if rel.startswith("sigma_01_to_30"):
        return "sweep"
    return "sigma"


# ── main ──────────────────────────────────────────────────────────────────

class Geo:
    """Lazy projection cache: one UTM polyline, length and heading per edge."""

    def __init__(self, edges: dict, fwd, inv):
        self._edges = edges
        self._fwd = fwd
        self._inv = inv
        self._cache: dict[str, tuple] = {}

    def get(self, eid: str):
        if eid not in self._cache:
            lonlat = self._edges.get(eid)
            if lonlat is None:
                self._cache[eid] = None
            else:
                xy = np.asarray([self._fwd.transform(a, b) for a, b in lonlat])
                self._cache[eid] = (xy, float(segment_lengths_m(xy).sum()), len(lonlat))
        return self._cache[eid]

    def to_m(self, lon: float, lat: float) -> np.ndarray:
        x, y = self._fwd.transform(lon, lat)
        return np.array([x, y])

    def to_deg(self, m: np.ndarray) -> tuple[float, float]:
        return self._inv.transform(float(m[0]), float(m[1]))


def build(root: Path, geo: Geo, names: dict, gt_defect_m: float) -> list[dict]:
    rows = []
    for data_dir in sorted(p.parent for p in root.rglob("observations.csv")):
        rel = str(data_dir.relative_to(root))
        fam = family_of(rel)
        d = load_dataset(data_dir, names)
        gt_e, gt_m_all = d["gt_edge"], d["gt_m"]
        obs_xy, obs_ts = d["obs_xy"], d["obs_ts"]
        n_before = len(rows)

        for (tid, seq) in sorted(gt_e, key=lambda k: (int(k[0]), k[1])):
            gtedge = gt_e[(tid, seq)]
            if (tid, seq) not in gt_m_all or (tid, seq) not in obs_xy:
                continue
            gt_m = gt_m_all[(tid, seq)]
            ox, oy = obs_xy[(tid, seq)]
            o_m = geo.to_m(ox, oy)
            gx, gy = geo.to_deg(gt_m)

            rec = dict.fromkeys(COLUMNS, "")
            rec.update({
                "dataset": rel, "family": fam, "id": tid, "seq": seq,
                "timestamp": f"{obs_ts.get((tid, seq), float('nan')):.6f}",
                "gnss_x": f"{ox:.8f}", "gnss_y": f"{oy:.8f}",
                "gt_x": f"{gx:.8f}", "gt_y": f"{gy:.8f}", "gt_edge": gtedge,
                "err_gnss_m": f"{np.hypot(*(gt_m - o_m)):.3f}",
            })

            # the direction the truth was travelling: a central difference over
            # the neighbouring ground-truth points, so one noisy epoch cannot
            # decide it. Falls back to a forward difference at the ends.
            prev, nxt = gt_m_all.get((tid, seq - 1)), gt_m_all.get((tid, seq + 1))
            if prev is not None and nxt is not None:
                h = nxt - prev
            elif nxt is not None:
                h = nxt - gt_m
            elif prev is not None:
                h = gt_m - prev
            else:
                h = None
            truth_head = None
            if h is not None and np.any(h):
                truth_head = float(np.degrees(np.arctan2(h[0], h[1])))
                rec["truth_head_deg"] = f"{truth_head:.2f}"

            g = geo.get(gtedge)
            if g is not None:
                gt_poly, gt_len, gt_n = g
                dgt, _, gtbear = point_to_polyline_m(gt_m, gt_poly)
                rec["d_gt_to_gtedge_m"] = f"{dgt:.3f}"
                rec["d_gnss_to_gtedge_m"] = f"{point_to_polyline_m(o_m, gt_poly)[0]:.3f}"
                rec["len_gt_m"] = f"{gt_len:.3f}"
                rec["gt_edge_n"] = gt_n
                rec["gt_edge_len_m"] = f"{gt_len:.3f}"
                rec["gt_edge_head_deg"] = f"{gtbear:.2f}"
                rec["flag_gt_defect"] = int(dgt > gt_defect_m)

            for tag in ("cmm", "fmm"):
                cpath, tw, status = d[tag].get((tid, seq), ("", TW_SENTINEL, ""))
                cls = classify(cpath, gtedge)
                rec[f"{tag}_cpath"] = cpath
                rec[f"{tag}_tw"] = "" if tw <= TW_SENTINEL else f"{tw:.6f}"
                rec[f"{tag}_class"] = cls
                rec[f"{tag}_status"] = status
                rec[f"{tag}_correct"] = int(cls in CORRECT_CLASSES)
                rec[f"flag_reverse_{tag}"] = int(cls == "reversed")

                cg = geo.get(cpath)
                if cg is not None:
                    cpoly, clen, _ = cg
                    dgm, _, bear = point_to_polyline_m(gt_m, cpoly)
                    rec[f"d_gt_to_{tag}_m"] = f"{dgm:.3f}"
                    rec[f"len_{tag}_m"] = f"{clen:.3f}"
                    rec[f"{tag}_head_deg"] = f"{bear:.2f}"
                    if truth_head is not None:
                        rec[f"{tag}_dir_err"] = int(angle_diff_deg(bear, truth_head) > 90.0)
            rows.append(rec)

        print(f"  {rel:46s} {len(rows) - n_before:7d} rows", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/simulation_plfix"))
    ap.add_argument("--cmm-suffix", default="_plfix")
    ap.add_argument("--fmm-suffix", default="_audit")
    ap.add_argument("--shapefile", type=Path,
                    default=Path("input/map/hainan/edges.shp"))
    ap.add_argument("--utm-epsg", type=int, default=32649)
    ap.add_argument("--gt-defect-m", type=float, default=50.0)
    ap.add_argument("--out", type=Path,
                    default=Path("data/simulation_plfix/error_audit.csv"))
    args = ap.parse_args()

    import pyproj

    shapefile = (BASE / args.shapefile).resolve()
    print(f"edges : {shapefile}")
    edges = load_edges(shapefile)
    print(f"        {len(edges)} polylines")

    fwd = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{args.utm_epsg}",
                                      always_xy=True)
    inv = pyproj.Transformer.from_crs(f"EPSG:{args.utm_epsg}", "EPSG:4326",
                                      always_xy=True)
    geo = Geo(edges, fwd, inv)
    names = {"cmm": f"cmm_result{args.cmm_suffix}.csv",
             "fmm": f"fmm_result{args.fmm_suffix}.csv"}

    root = (BASE / args.root).resolve()
    print(f"corpus: {root}")
    rows = build(root, geo, names, args.gt_defect_m)

    out = (BASE / args.out).resolve()
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter=";")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}  ({len(rows)} rows, {out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
