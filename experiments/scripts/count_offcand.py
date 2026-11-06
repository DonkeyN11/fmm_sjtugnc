#!/usr/bin/env python3
"""Inventory off-candidate epochs across the whole real corpus (all trajectories)."""
import csv, math
from collections import defaultdict
import numpy as np
from osgeo import ogr

per = defaultdict(list)
with open('data/real_vehicle/hainan_06/processed/aligned.csv', newline='') as fh:
    for r in csv.DictReader(fh, delimiter=';'):
        if r['uni_seq'].strip():
            per[r['id'].strip()].append(r)

lat0 = float(np.mean([float(r['obs_y']) for rs in per.values() for r in rs if r['obs_y'].strip()]))
lon0 = float(np.mean([float(r['obs_x']) for rs in per.values() for r in rs if r['obs_x'].strip()]))
kx = 111320.0 * math.cos(math.radians(lat0)); ky = 110540.0
print('frame lon0=%.5f lat0=%.5f ids=%s' % (lon0, lat0, sorted(per)))

AX=[]; AY=[]; BX=[]; BY=[]; R0=[]
ds = ogr.Open('input/map/hainan/edges.shp'); ly = ds.GetLayer(0)
for fi, f in enumerate(ly):
    g = f.GetGeometryRef()
    if g is None: continue
    for gi in range(g.GetGeometryCount() or 1):
        gg = g.GetGeometryRef(gi) if g.GetGeometryCount() else g
        if gg is None: continue
        for i in range(gg.GetPointCount()-1):
            x1=(gg.GetX(i)-lon0)*kx;   y1=(gg.GetY(i)-lat0)*ky
            x2=(gg.GetX(i+1)-lon0)*kx; y2=(gg.GetY(i+1)-lat0)*ky
            AX.append(x1); AY.append(y1); BX.append(x2); BY.append(y2)
            if fi == 0: R0.append((x1,y1,x2,y2))
AX=np.array(AX); AY=np.array(AY); BX=np.array(BX); BY=np.array(BY)
R0 = np.array(R0)
DX=BX-AX; DY=BY-AY; L2=DX*DX+DY*DY; L2S=np.where(L2>1e-12,L2,1.0)
MX=(AX+BX)/2; MY=(AY+BY)/2
print('segments: %d' % len(AX))

def seg_min(px, py, ax, ay, bx, by):
    dx=bx-ax; dy=by-ay; l2=dx*dx+dy*dy
    l2s=np.where(l2>1e-12,l2,1.0)
    t=np.clip(((px-ax)*dx+(py-ay)*dy)/l2s, 0.0, 1.0)
    return float(np.min(np.hypot(px-(ax+t*dx), py-(ay+t*dy))))

hdr = '%-4s %6s %7s %7s %7s %7s %9s %9s' % ('id','n','gt=0','gt!=0','d>5m','d>10m','gt0&d>10','gt0&d<=3')
print(hdr); print('-'*len(hdr))
T=defaultdict(float); row0_min=1e18
for tid in sorted(per):
    n=n0=n1=d5=d10=ca=cb=0
    for r in per[tid]:
        if not r['obs_x'].strip() or not r['gt_x'].strip(): continue
        px=(float(r['gt_x'])-lon0)*kx; py=(float(r['gt_y'])-lat0)*ky
        m=(np.abs(MX-px)<600.0)&(np.abs(MY-py)<600.0)
        if not m.any(): continue
        d=seg_min(px,py,AX[m],AY[m],BX[m],BY[m])
        g=(r.get('gt_edge') or '').strip()
        is0 = g in ('0','0.0','')
        n+=1; n0+=is0; n1+=(not is0)
        d5+=int(d>5.0); d10+=int(d>10.0)
        ca+=int(is0 and d>10.0); cb+=int(is0 and d<=3.0)
        if is0 and len(R0): row0_min=min(row0_min, seg_min(px,py,R0[:,0],R0[:,1],R0[:,2],R0[:,3]))
    print('%-4s %6d %7d %7d %7d %7d %9d %9d' % (tid,n,n0,n1,d5,d10,ca,cb))
    T['n']+=n; T['n0']+=n0; T['d10']+=d10; T['ca']+=ca; T['cb']+=cb
print('-'*len(hdr))
print('TOTAL n=%d  gt_edge==0: %d (%.1f%%)  d>10m: %d (%.1f%%)  both: %d (%.1f%%)  gt0&d<=3m: %d'
      % (T['n'],T['n0'],100*T['n0']/T['n'],T['d10'],100*T['d10']/T['n'],T['ca'],100*T['ca']/T['n'],T['cb']))
print('sentinel check: min distance from a gt_edge==0 epoch to feature row 0 = %.0f m' % row0_min)
