"""Sitecustomize: monkey-patch matplotlib to auto-save SVG alongside every PNG save."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
_orig = plt.Figure.savefig
def _patched(self, fname, *a, **kw):
    _orig(self, fname, *a, **kw)
    s = str(fname)
    if s.endswith('.png'):
        try: _orig(self, s.replace('.png','.svg'), format='svg')
        except: pass
plt.Figure.savefig = _patched
