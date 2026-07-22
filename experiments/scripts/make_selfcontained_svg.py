#!/usr/bin/env python3
"""Replace only math <text> elements in the pipeline SVG with 2x LaTeX formula images."""
import re, base64, io, os
import xml.etree.ElementTree as ET
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

SRC = "/home/ncz/fmm_sjtugnc/docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/figs_svg/tmm_pipeline.svg"
OUT_SVG = "/home/ncz/fmm_sjtugnc/docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/figs_svg/tmm_framework.svg"
OUT_PNG = "/home/ncz/fmm_sjtugnc/docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/figs/tmm_framework.png"
CACHE = "/tmp/formula_cache"; os.makedirs(CACHE, exist_ok=True)

# ── Formula renderer ──
def render(latex, fs=9):
    key = f"{latex}_{fs}"
    cp = os.path.join(CACHE, f"{abs(hash(key)) & 0x7fffffff}.png")
    if os.path.exists(cp):
        with open(cp, 'rb') as f: return base64.b64encode(f.read()).decode()
    safe = latex.replace(r'\!\big', '').replace(r'\big', '').replace(r'\,', '')
    fig, ax = plt.subplots(figsize=(0.01, 0.01)); ax.axis('off')
    try:
        ax.text(0, 0, safe, fontsize=fs, ha='left', va='bottom', math_fontfamily='stix')
    except:
        try: ax.text(0, 0, safe, fontsize=fs, ha='left', va='bottom')
        except: plt.close(fig); return None
    fig.canvas.draw()
    bb = ax.get_window_extent(); plt.close(fig)
    w_in, h_in = bb.width/fig.dpi + 0.12, bb.height/fig.dpi + 0.10
    fig, ax = plt.subplots(figsize=(w_in, h_in)); ax.axis('off')
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    try:
        ax.text(0.04, 0.15, safe, fontsize=fs, ha='left', va='bottom',
                transform=ax.transAxes, math_fontfamily='stix')
    except:
        ax.text(0.04, 0.15, safe, fontsize=fs, ha='left', va='bottom', transform=ax.transAxes)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', transparent=True, pad_inches=0.02)
    plt.close(fig); buf.seek(0)
    with open(cp, 'wb') as f: f.write(buf.getvalue())
    return base64.b64encode(buf.getvalue()).decode()

# ── Read original SVG ──
with open(SRC, 'r') as f:
    svg = f.read()

# Add xlink namespace for Adobe Illustrator compatibility
svg = svg.replace(
    '<svg xmlns="http://www.w3.org/2000/svg"',
    '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"'
)

# ── Map each math <text> element to a LaTeX formula ──
# Extract x, y, and full text from math <text> elements
# Build a (LaTeX, x, y, fs) mapping

# Use a careful regex: match <text x="..." y="..."> ... <tspan baseline-shift...> ... </text>
text_re = re.compile(
    r'<text\s+x="([^"]*)"\s+y="([^"]*)"[^>]*>(.*?)</text>',
    re.DOTALL
)

formula_map = []
for m in text_re.finditer(svg):
    x, y, body = m.group(1), m.group(2), m.group(3)
    if 'baseline-shift' not in body:
        continue  # not a math formula

    # Get all character data (strip tags)
    raw = re.sub(r'<[^>]+>', '', body).strip()
    if not raw: continue

    # Build LaTeX from the raw text
    latex = raw
    # Systematic replacements
    subs = [
        ('zi', 'z_i'), ('Σi', r'\mathbf{\Sigma}_i'), ('HPLi', r'\mathrm{HPL}_i'),
        ('twt', r'\mathrm{tw}_t'), ('αt', r'\alpha_t'), ('δt', r'\delta_t'),
        ('Ht', 'H_t'), ('ΔHt', r'\Delta H_t'), ('Hprior', r'H_{\mathrm{prior}}'),
        ('σmajor', r'\sigma_{\mathrm{major}}'), ('log₂', r'\log_2'),
        ('p(zi', 'p(z_i'), ('10⁻⁵', '10^{-5}'), ('PFA', r'P_{\mathrm{FA}}'),
        ('·', r'\cdot'), ('½', r'\frac{1}{2}'), ('dᵀ', r'\mathbf{d}^{\top}'),
        ('Σ⁻¹', r'\mathbf{\Sigma}^{-1}'), ('∝', r'\propto'), ('Σⱼ', r'\sum_j'),
        ('pi', 'p_i'), ('χ²', r'\chi^2'), ('X*', 'X^*'),
        ('ΔH', r'\Delta H'), ('{zi', r'\{z_i'), ('{twt', r'\{\mathrm{tw}_t'),
        ('=1..n', '=1..n'), ('t=1..n', 't=1..n'),
    ]
    for old, new in subs:
        latex = latex.replace(old, new)
    latex = f'${latex}$'

    # Clean up double brackets
    latex = re.sub(r'\{_\{', '_{', latex)

    formula_map.append((latex, float(x), float(y), raw))
    print(f"  [{x},{y}] {raw[:50]} -> {latex[:70]}")

# ── Remove math text elements ──
# Use precise non-greedy regex for each individual <text> with baseline-shift
def remove_math_text(svg_content):
    # Match ONE <text> element that contains baseline-shift
    pattern = re.compile(
        r'<text\s+x="[^"]*"\s+y="[^"]*"[^>]*>((?:(?!</text>).)*?<tspan[^>]*baseline-shift[^>]*>.*?)</text>',
        re.DOTALL
    )
    # Apply repeatedly until no more matches
    prev = None
    while prev != svg_content:
        prev = svg_content
        svg_content = pattern.sub('', svg_content, count=1)
    return svg_content

svg_clean = remove_math_text(svg)

# ── Inject formula images ──
images_xml = '\n  <!-- LaTeX-rendered formula images (2x scale) -->\n'
for latex, x, y, raw in formula_map:
    b64 = render(latex, fs=10)
    if not b64:
        print(f"  SKIP: {latex[:30]}")
        continue
    try:
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
        w_px, h_px = img.size
    except:
        w_px, h_px = 300, 50
    # 2x scale compared to original
    scale = 1.15
    w, h = w_px * scale, h_px * scale
    tag = (f'<image x="{x - w/2}" y="{y - h*0.65}" width="{w}" height="{h}" '
           f'xlink:href="data:image/png;base64,{b64}" preserveAspectRatio="xMidYMid meet"/>')
    images_xml += '  ' + tag + '\n'
    print(f"  -> {w_px}x{h_px}px @ 2x")

insert_pos = svg_clean.rfind('</svg>')
if insert_pos < 0:
    insert_pos = len(svg_clean)
    svg_clean += '\n</svg>'
svg_final = svg_clean[:insert_pos] + images_xml + svg_clean[insert_pos:]

# ── Save ──
with open(OUT_SVG, 'w') as f:
    f.write(svg_final)
print(f"\nSaved {OUT_SVG} ({len(svg_final)} bytes)")

# Convert to PNG
try:
    import cairosvg
    cairosvg.svg2png(url=OUT_SVG, write_to=OUT_PNG, output_width=2200, output_height=1400)
    print(f"Saved {OUT_PNG}")
except Exception as e:
    print(f"PNG: {e}")
