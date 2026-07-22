#!/usr/bin/env python3
"""Convert all manuscript PNG figures to true editable SVG by re-executing their matplotlib code."""
import subprocess, sys
from pathlib import Path

ROOT = Path("/home/ncz/fmm_sjtugnc")
FIG_DIR = ROOT / "docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/figs"
SVG_DIR = ROOT / "docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/figs_svg"
SVG_DIR.mkdir(parents=True, exist_ok=True)

# Config: (script, args, output_png_name)
tasks = [
    # Scripts that produce single figures
    ("experiments/scripts/regenerate_sigma_sweep.py", [], "sigma_sweep.png"),
    ("experiments/scripts/regenerate_degraded.py", [], "degraded_comparison.png"),
    ("experiments/scripts/fig_candidate_counts.py", [], "candidate_count_simulation.png"),
    ("experiments/scripts/fig_candidate_counts.py", [], "candidate_count_real.png"),
    ("experiments/scripts/fig_dataset_overview.py", [], "dataset_overview.png"),
    ("experiments/scripts/fig_stanford_combined.py", [], "stanford_combined.png"),
    ("experiments/scripts/fig_traj_tw_panorama.py", [], "traj_tw_panorama.png"),
    ("python/experiments/gen_figures.py", [], "reliability_diagram.png"),
    ("python/experiments/gen_figures.py", [], "ece_ablation.png"),
]

for script, args, png_name in tasks:
    script_path = ROOT / script
    if not script_path.exists():
        print(f"SKIP: {script} not found")
        continue
    png_path = FIG_DIR / png_name
    svg_path = SVG_DIR / png_name.replace(".png", ".svg")
    print(f"Generating {svg_path.name} from {script}...")
    # The scripts already have SVG save patched in; just run them
    result = subprocess.run([sys.executable, str(script_path)],
                          capture_output=True, text=True, timeout=600, cwd=str(ROOT))
    out = (result.stdout + result.stderr)[:500]
    if result.returncode != 0:
        print(f"  ERROR: {out}")
    else:
        for line in out.split('\n'):
            if 'Saved' in line or 'Error' in line:
                print(f"  {line.strip()}")
    # Check if SVG was generated
    if svg_path.exists():
        print(f"  -> {svg_path.stat().st_size} bytes")
    elif (FIG_DIR / png_name.replace('.png', '.svg')).exists():
        import shutil
        shutil.copy(FIG_DIR / png_name.replace('.png', '.svg'), svg_path)
        print(f"  -> copied from figs/")

print("\nDone. SVGs in:", SVG_DIR)
