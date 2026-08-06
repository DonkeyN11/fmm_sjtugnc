#!/usr/bin/env python3
"""Detailed TW distribution plots for simulation and real-vehicle experiments."""
import csv, json, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / "data/simulation"
ALIGNED = next((ROOT / "data/real_vehicle").rglob("aligned.csv"), None)
REV = json.load(open(ROOT / "experiments/config/reverse_edge_map.json"))
REV = {str(k): str(v) for k, v in REV.items()}
def em(m,t): return str(m)==str(t) or REV.get(str(m))==str(t)

OUT_SIM = ROOT / "docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/figs/tw_dist_simulation.png"
OUT_REAL = ROOT / "docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/figs/tw_dist_real.png"
DPI = 300
plt.rcParams.update({"font.size":8,"axes.labelsize":9,"axes.titlesize":10,
    "legend.fontsize":7,"figure.dpi":DPI,"savefig.dpi":DPI,"savefig.bbox":"tight"})

# ═══════════════════════════════════════════════════════════════
# Figure 1: Simulation — per-sigma TW histogram
# ═══════════════════════════════════════════════════════════════
sigmas = sorted([d.name for d in SIM.glob("sigma_*") if d.is_dir() and "mismatch" not in d.name],
                key=lambda s: int(s.replace("sigma_","")))

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()
bins = np.linspace(0, 1, 51)  # 50 bins → 0.02 resolution

for idx, s in enumerate(sigmas):
    f = SIM / s / "no_occlusion/no_fault/cmm_result.csv"
    if not f.exists(): continue
    tws, corrects = [], []
    with open(f) as fh:
        for row in csv.DictReader(fh, delimiter=";"):
            try: tws.append(float(row["trustworthiness"]))
            except: pass
    tws = np.array(tws)
    ax = axes[idx]
    ax.hist(tws, bins=bins, color="#2166ac", alpha=0.75, edgecolor="white", lw=0.3)
    ax.axvline(tws.mean(), color="#C0392B", ls="--", lw=1.2, label=f"μ={tws.mean():.3f}")
    ax.axvline(np.median(tws), color="#E67E22", ls=":", lw=1.2, label=f"med={np.median(tws):.3f}")
    # Mark the 0.5 line
    ax.axvline(0.5, color="#888888", ls=":", lw=0.5, alpha=0.5)
    ax.set_title(f"σ={s.replace('sigma_','')} m (n={len(tws)})")
    ax.set_xlabel("TW"); ax.set_ylabel("Count")
    ax.legend(fontsize=6)
    ax.set_xlim(0, 1)

# Hide extra subplot
if len(sigmas) < 8:
    for idx in range(len(sigmas), 8):
        axes[idx].set_visible(False)

fig.suptitle("CaMM Trustworthiness Distribution — Simulation (50 bins)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_SIM, dpi=DPI)
plt.close(fig)
print(f"Saved {OUT_SIM}")

# ═══════════════════════════════════════════════════════════════
# Figure 2: Real-vehicle — per-trajectory TW histogram
# ═══════════════════════════════════════════════════════════════
if ALIGNED:
    traj_data = {}
    with open(ALIGNED) as fh:
        for row in csv.DictReader(fh, delimiter=";"):
            tid = row["id"].strip()
            try: tw = float(row["cmm_tw"])
            except: continue
            gt = row.get("gt_edge","").strip()
            cp = row.get("cmm_cpath","").strip()
            correct = em(cp, gt)
            x = row.get("cmm_x","").strip()
            if tid not in traj_data: traj_data[tid] = []
            traj_data[tid].append({"tw":tw, "correct":correct, "x":float(x) if x else 0})

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    bins = np.linspace(0, 1, 51)

    for idx, tid in enumerate([11,12,13,14,21,22,23]):
        seg = traj_data.get(str(tid), [])
        if not seg: continue
        tws = np.array([r["tw"] for r in seg])
        tws_c = np.array([r["tw"] for r in seg if r["correct"]])
        tws_w = np.array([r["tw"] for r in seg if not r["correct"]])

        ax = axes[idx]
        ax.hist(tws_c, bins=bins, color="#27AE60", alpha=0.6, label=f"Correct ({len(tws_c)})")
        ax.hist(tws_w, bins=bins, color="#C0392B", alpha=0.6, label=f"Wrong ({len(tws_w)})")
        ax.axvline(tws.mean(), color="#2166ac", ls="--", lw=1.2, label=f"μ={tws.mean():.3f}")
        ax.axvline(tws_c.mean() if len(tws_c) else 0, color="#27AE60", ls=":", lw=1)
        ax.axvline(tws_w.mean() if len(tws_w) else 0, color="#C0392B", ls=":", lw=1)
        ax.axvline(0.5, color="#888888", ls=":", lw=0.5, alpha=0.5)

        # Annotate key epochs
        min_idx = np.argmin(tws)
        ax.annotate(f"min={tws[min_idx]:.4f}", xy=(tws[min_idx], 0),
                    xytext=(tws[min_idx], ax.get_ylim()[1]*0.5 if idx==0 else 0),
                    fontsize=6, color="#C0392B", ha="center",
                    arrowprops=dict(arrowstyle="->", color="#C0392B", lw=0.8))

        ax.set_title(f"Traj {tid} (n={len(tws)}, TW={tws.mean():.3f})")
        ax.set_xlabel("TW"); ax.set_ylabel("Count")
        ax.legend(fontsize=5.5, loc="upper left")
        ax.set_xlim(0, 1)

    # Hide extra subplot
    axes[7].set_visible(False)
    fig.suptitle("CaMM Trustworthiness Distribution — Real Vehicle (50 bins, green=correct, red=wrong)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_REAL, dpi=DPI)
    plt.close(fig)
    print(f"Saved {OUT_REAL}")

    # Also print detailed bin-level stats for Traj 22 (most problematic)
    seg22 = traj_data.get("22", [])
    tws22 = np.array([r["tw"] for r in seg22])
    print(f"\n=== Traj 22 detailed bins ===")
    for i in range(50):
        lo, hi = i/50, (i+1)/50
        mask = (tws22 >= lo) & (tws22 < hi)
        cnt = mask.sum()
        if cnt > 0:
            c_in_bin = sum(1 for r in seg22 if lo <= r["tw"] < hi and r["correct"])
            w_in_bin = sum(1 for r in seg22 if lo <= r["tw"] < hi and not r["correct"])
            bar = "█" * int(cnt/len(tws22)*100)
            print(f"  [{lo:.2f},{hi:.2f}): n={cnt:4d}  correct={c_in_bin:4d}  wrong={w_in_bin:4d}  {bar}")
else:
    print("aligned.csv not found — skipping real-vehicle figure")
