#!/usr/bin/env python3
"""CaMM Framework figure — IEEE journal style with LaTeX-rendered formulas."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

OUT = "/home/ncz/fmm_sjtugnc/docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/figs/tmm_framework"
DPI = 300

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "text.usetex": False,  # use mathtext — same visual quality for simple formulas
    "mathtext.fontset": "stix",  # STIX fonts look like LaTeX
    "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
})

# ── Color palette ──
C = {
    'blue':   {'bg': '#DCEAF5', 'txt': '#1565C0'},
    'teal':   {'bg': '#CFE9E6', 'txt': '#00695C'},
    'green':  {'bg': '#E1EFE0', 'txt': '#2E7D32'},
    'orange': {'bg': '#FAE9DA', 'txt': '#E65100'},
    'purple': {'bg': '#EFE5F2', 'txt': '#6A1B9A'},
    'dark':   {'bg': '#2C333A', 'txt': '#FFFFFF'},
}

def stage_box(ax, x, y, w, h, color_key, title, subtitle=""):
    bg, tc = C[color_key]['bg'], C[color_key]['txt']
    rect = patches.FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.06", facecolor=bg, edgecolor='#AAAAAA',
        linewidth=1.2, linestyle='--')
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 0.13, title, ha='center', va='center',
            fontsize=13, fontweight='bold', color=tc)
    if subtitle:
        ax.text(x + w/2, y + 0.06, subtitle, ha='center', va='center',
                fontsize=8.5, color='gray', alpha=0.8)

def formula_box(ax, cx, cy, w, h, formulas):
    """White box with LaTeX formulas (list of (text, fs, bold) tuples)."""
    rect = patches.FancyBboxPatch((cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.04", facecolor='white',
        edgecolor='#BBBBBB', linewidth=1.0)
    ax.add_patch(rect)
    n = len(formulas)
    for i, (txt, fs, bold) in enumerate(formulas):
        y_off = h/2 - h/(n+1) * (i + 0.5)
        fw = 'bold' if bold else 'normal'
        ax.text(cx, cy + y_off, txt, ha='center', va='center',
                fontsize=fs, fontweight=fw, color='#222222')

def arrow(ax, x1, y1, x2, y2, label="", lw=1.8, color='#555555'):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw))
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.02, label, ha='center', va='bottom', fontsize=8, color=color)

# ── Create figure ──
fig, ax = plt.subplots(figsize=(16, 8))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

# Layout
cols = 5; pad = 0.012; gap = 0.01
col_w = (1.0 - 2*pad - (cols-1)*gap) / cols
col_x = [pad + i*(col_w + gap) for i in range(cols)]
stage_top_y = 0.38; stage_h = 0.52
bot_y = 0.08; bot_h = 0.20

# ── Stage 1: Input Data ──
cx0 = col_x[0]
stage_box(ax, cx0, stage_top_y, col_w, stage_h, 'blue', 'Input Data')
formula_box(ax, cx0+col_w/2, stage_top_y+0.38, col_w*0.82, 0.13,
    [("GNSS Raw Observations (RINEX)", 10.5, True)])
formula_box(ax, cx0+col_w/2, stage_top_y+0.20, col_w*0.82, 0.09,
    [("NMEA GST: Covariance Matrix", 10.5, False)])

# ── Stage 2: GNSS Processing ──
cx1 = col_x[1]
stage_box(ax, cx1, stage_top_y, col_w, stage_h, 'teal', 'GNSS Processing')
formula_box(ax, cx1+col_w/2, stage_top_y+0.38, col_w*0.82, 0.14,
    [(r"WLS: $\widehat{\Delta\mathbf{x}} = (\mathbf{H}^{\top}\mathbf{WH})^{-1}\mathbf{H}^{\top}\mathbf{Wy}$", 9, False),
     (r"$\mathbf{\Sigma}_x = (\mathbf{H}^{\top}\mathbf{WH})^{-1}$", 9, False)])
formula_box(ax, cx1+col_w/2, stage_top_y+0.20, col_w*0.82, 0.16,
    [(r"RAIM Integrity Monitor", 10.5, True),
     (r"$\chi^2$ Test ($P_{\mathrm{FA}}\!=\!10^{-5}$), FDE", 9.5, False),
     (r"$\mathrm{HPL}_i = K \cdot \sigma_{\mathrm{major}}$", 9.5, False)])

# ── Stage 3: HMM Map Matching ──
cx2 = col_x[2]
stage_box(ax, cx2, stage_top_y, col_w, stage_h, 'green', 'HMM Map Matching')
formula_box(ax, cx2+col_w/2, stage_top_y+0.42, col_w*0.82, 0.09,
    [(r"(1) Candidate Generation", 10, True),
     (r"$r_i = \mathrm{HPL}_i$, Mahalanobis Proj.", 9, False)])
formula_box(ax, cx2+col_w/2, stage_top_y+0.28, col_w*0.82, 0.09,
    [(r"(2) Emission Probability", 10, True),
     (r"$p(z_i|x) \propto \exp(-\frac{1}{2}\mathbf{d}^{\top}\mathbf{\Sigma}_i^{-1}\mathbf{d})$", 9, False)])
formula_box(ax, cx2+col_w/2, stage_top_y+0.14, col_w*0.82, 0.09,
    [(r"(3) HMM Inference", 10, True),
     (r"Forward $\alpha_t(i)$ + Viterbi $\delta_t(i)$", 9.5, False)])

# ── Stage 4: Trustworthiness ──
cx3 = col_x[3]
stage_box(ax, cx3, stage_top_y, col_w, stage_h, 'orange', 'Trustworthiness')
formula_box(ax, cx3+col_w/2, stage_top_y+0.38, col_w*0.82, 0.16,
    [(r"Filtering Posterior (TW)", 10, True),
     (r"$\mathrm{tw}_t = P(x_t=i^*\!\mid\!z_{1:t})$", 9.5, False),
     (r"$= \alpha_t(i^*) / \sum_j\alpha_t(j)$", 9.5, False)])
formula_box(ax, cx3+col_w/2, stage_top_y+0.14, col_w*0.82, 0.12,
    [(r"Info-Theoretic Metrics", 10, True),
     (r"$H_t = -\sum_i p_i\log_2 p_i$   $\Delta H_t = H_{\mathrm{prior}}\!-\!H_t$", 8.5, False)])

# ── Stage 5: Validation ──
cx4 = col_x[4]
stage_box(ax, cx4, stage_top_y, col_w, stage_h, 'purple', 'Validation')
formula_box(ax, cx4+col_w/2, stage_top_y+0.42, col_w*0.82, 0.10,
    [(r"Accuracy: Seg. 96.9\%, Err. 5.6 m", 10, True)])
formula_box(ax, cx4+col_w/2, stage_top_y+0.26, col_w*0.82, 0.10,
    [(r"Calibration: ECE = 0.069 (36\%$\downarrow$)", 10, True)])
formula_box(ax, cx4+col_w/2, stage_top_y+0.10, col_w*0.82, 0.10,
    [(r"Integrity: Stanford, $\sigma$ Sweep, Real", 10, True)])

# ── Horizontal arrows ──
arrow_y = stage_top_y + 0.30
for i in range(cols - 1):
    arrow(ax, col_x[i] + col_w, arrow_y, col_x[i+1], arrow_y, lw=2.0)

# ── Downstream ──
bot_x = pad - 0.005; bot_w = 1.0 - 2*(pad - 0.005)
stage_box(ax, bot_x, bot_y, bot_w, bot_h, 'dark', 'Downstream Integration \& Safety-Critical Applications')
ax.text(bot_x+bot_w/2, bot_y + bot_h/2,
    "Integrity Monitoring  ·  Multi-Sensor Fusion (IMU/Odometer)  ·  Autonomous Driving  ·  HD Mapping  ·  Road Pricing",
    ha='center', va='center', fontsize=12, color='white', alpha=0.85)

# ── Downward arrows ──
for i in range(cols):
    arrow(ax, col_x[i] + col_w/2, stage_top_y - 0.01,
          col_x[i] + col_w/2, bot_y + bot_h + 0.01, color='#777777', lw=1.5)

# ── Title ──
ax.text(0.5, 0.97, "Covariance-aware Map Matching (CaMM) — Framework Overview",
        ha='center', va='top', fontsize=17, fontweight='bold', color='#222222')

# ── CaMM vs HMM badge ──
ax.text(0.82, 0.96, "CaMM vs. Classical HMM",
        ha='center', va='top', fontsize=9, fontweight='bold', color='#555555',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='#BBBBBB', alpha=0.9))
ax.text(0.82, 0.93, "Isotropic → Anisotropic | Fixed $r$ → HPL-Adaptive | Uncalibrated → Calibrated",
        ha='center', va='top', fontsize=7.5, color='#777777')

# ── Color legend ──
leg_y = 0.01
legend_items = [
    (C['blue']['bg'],   'Input Data'),
    (C['teal']['bg'],   'GNSS Processing'),
    (C['green']['bg'],  'HMM Map Matching'),
    (C['orange']['bg'], 'Trustworthiness'),
    (C['purple']['bg'], 'Validation'),
    (C['dark']['bg'],   'Downstream Apps'),
]
leg_x = 0.06
for color_bg, text in legend_items:
    rect = patches.Rectangle((leg_x, leg_y + 0.005), 0.013, 0.011,
        facecolor=color_bg, edgecolor='#AAAAAA', linewidth=0.8, linestyle='--')
    ax.add_patch(rect)
    ax.text(leg_x + 0.018, leg_y + 0.01, text, ha='left', va='center', fontsize=9, color='#333333')
    leg_x += 0.155

# ── Save ──
for fmt in ["pdf", "png"]:
    fig.savefig(f"{OUT}.{fmt}", format=fmt)
    print(f"Saved {OUT}.{fmt}")
plt.close(fig)
