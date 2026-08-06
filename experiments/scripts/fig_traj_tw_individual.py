#!/usr/bin/env python3
"""Generate 8 individual editable SVG subplots from the panorama figure."""
import csv, json, math
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import shapefile, requests
from PIL import Image
from io import BytesIO

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data/real_vehicle/hainan_06/processed"
MAP_SHP = PROJECT / "input/map/hainan/edges.shp"
OUT_DIR = PROJECT / "docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/figs_svg"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DPI = 300
plt.rcParams.update({"font.size":9,"axes.labelsize":10,"axes.titlesize":11,
    "legend.fontsize":8,"xtick.labelsize":8,"ytick.labelsize":8,
    "figure.dpi":DPI,"savefig.dpi":DPI,"savefig.bbox":"tight"})

# ── Satellite tiles ──
def deg2num(lon,lat,z):
    n=2**z; return int((lon+180)/360*n), int((1-math.log(math.tan(math.radians(lat))+1/math.cos(math.radians(lat)))/math.pi)/2*n)
def num2deg(xt,yt,z):
    n=2.**z; return xt/n*360-180, math.degrees(math.atan(math.sinh(math.pi*(1-2*yt/n))))
TC={}
def get_sat(lo,hi,bo,to,z):
    x0,y0=deg2num(lo,bo,z); x1,y1=deg2num(hi,to,z)
    x0,x1=min(x0,x1),max(x0,x1); y0,y1=min(y0,y1),max(y0,y1)
    c,r=x1-x0+1,y1-y0+1
    if c*r>64: return get_sat(lo,hi,bo,to,z-1)
    st=Image.new('RGB',(c*256,r*256))
    for dy in range(r):
        for dx in range(c):
            tx,ty=x0+dx,y0+dy; k=(z,tx,ty)
            if k not in TC:
                try:
                    resp=requests.get(f'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{ty}/{tx}',timeout=5)
                    TC[k]=Image.open(BytesIO(resp.content)) if resp.status_code==200 else None
                except: TC[k]=None
            if TC[k]: st.paste(TC[k],(dx*256,dy*256))
    l,t=num2deg(x0,y0,z); r,b=num2deg(x1+1,y1+1,z)
    return st,(l,r,b,t)

# ── Data loading ──
sf=shapefile.Reader(str(MAP_SHP))
all_roads=[np.array(s.points) for s in sf.shapes() if len(s.points)>=2]
REV=json.load(open(PROJECT/"experiments/config/reverse_edge_map.json"))
def em(m,t): return str(m)==str(t) or REV.get(str(m))==str(t)
gt_edges={}
with open(DATA/"ground_truth.csv") as f:
    for row in csv.DictReader(f,delimiter=";"): gt_edges[(row["id"].strip(),int(row["seq"]))]=row["edge_id"].strip()
traj_data={}
with open(DATA/"aligned.csv") as f:
    for row in csv.DictReader(f,delimiter=";"):
        tid=row["id"].strip(); useq=int(row["uni_seq"])
        if (tid,useq) not in gt_edges: continue
        if gt_edges[(tid,useq)] in ("0","-1"): continue
        x=row.get("cmm_x","").strip()
        if not x: continue
        tw=float(row.get("cmm_tw","0") or 0)
        if tid not in traj_data: traj_data[tid]=[]
        traj_data[tid].append({"seq":useq,"x":float(x),"y":float(row["cmm_y"]),
            "ox":float(row["obs_x"]),"oy":float(row["obs_y"]),
            "tw":tw,"correct":em(row.get("cmm_cpath","").strip(),gt_edges[(tid,useq)])})
for tid in traj_data: traj_data[tid].sort(key=lambda r:r["seq"])

# ── Plotting function ──
R50M=50.0/111320.0
def find_lowest(seg,w=50):
    if len(seg)<w: return 0,len(seg)
    tws=np.array([s["tw"] for s in seg]); bs,bm=0,float('inf')
    for i in range(len(seg)-w+1):
        m=tws[i:i+w].mean()
        if m<bm: bm,bs=m,i
    return bs,bs+w

def plot_one_map(seg, tid, out_path):
    """Single satellite TW map for one trajectory."""
    if not seg: return
    xs=[s["x"] for s in seg]; ys=[s["y"] for s in seg]
    tws=np.array([s["tw"] for s in seg])
    corrects=np.array([s["correct"] for s in seg])
    min_idx=int(np.argmin(tws)); mtx, mty = xs[min_idx], ys[min_idx]
    ws,we=find_lowest(seg,min(60,len(seg))); low_seg=seg[ws:we]

    # Viewport: tight for short trajs, extremes for long
    if len(seg)<500:
        xs_a=np.array(xs); ys_a=np.array(ys)
        cx=(xs_a.min()+xs_a.max())/2; cy=(ys_a.min()+ys_a.max())/2
        pad=max(xs_a.max()-xs_a.min(),ys_a.max()-ys_a.min())*0.65+0.001
    else:
        corr=np.where(corrects==1)[0]; wrong=np.where(corrects==0)[0]
        if len(corr) and len(wrong):
            bp=corr[np.argmax(tws[corr])]; wp=wrong[np.argmin(tws[wrong])]
            cx=(xs[bp]+xs[wp])/2; cy=(ys[bp]+ys[wp])/2
            pad=max(abs(xs[bp]-xs[wp]),abs(ys[bp]-ys[wp]))*0.75+0.004
        else:
            cx,cy=float(np.mean(xs)),float(np.mean(ys)); pad=0.004
    lo,hi=cx-pad,cx+pad; bo,to=cy-pad,cy+pad

    fig,ax=plt.subplots(figsize=(8,7))
    ax.set_xlim(lo,hi); ax.set_ylim(bo,to); ax.set_aspect("equal")
    try:
        span=max(hi-lo,to-bo); z=18 if span<0.002 else (17 if span<0.005 else 16)
        img,ext=get_sat(lo,hi,bo,to,z); ax.imshow(img,extent=ext,zorder=0,alpha=0.9,interpolation='bilinear')
    except: pass
    for r in all_roads:
        rmx,rmx2=float(r[:,0].min()),float(r[:,0].max())
        rmy,rmy2=float(r[:,1].min()),float(r[:,1].max())
        if rmx<=hi and rmx2>=lo and rmy<=to and rmy2>=bo:
            ax.plot(r[:,0],r[:,1],color="#ffffff",lw=1.0,alpha=0.4,zorder=1)
    norm=plt.Normalize(0,1)
    pts=np.array([xs,ys]).T.reshape(-1,1,2)
    lc=LineCollection(np.concatenate([pts[:-1],pts[1:]],axis=1),cmap="RdYlGn",norm=norm,linewidth=8.0,zorder=3)
    lc.set_array(tws[1:]); ax.add_collection(lc)
    sm=plt.cm.ScalarMappable(cmap="RdYlGn",norm=norm); sm.set_array([])
    cbar=plt.colorbar(sm,ax=ax,shrink=0.5,aspect=18,pad=0.015); cbar.set_label("tw",fontsize=8); cbar.ax.tick_params(labelsize=6)
    ell=Ellipse((mtx,mty),R50M*2,R50M*2,facecolor='none',edgecolor='#FFD700',linewidth=5,linestyle='--',zorder=6)
    ax.add_patch(ell)
    ax.scatter(xs[0],ys[0],s=80,c="#FFD700",marker="o",zorder=5,ec="black",lw=1.5)
    ax.set_xticks([]); ax.set_yticks([])
    mt=np.mean(tws); acc=np.mean(corrects)*100
    ax.set_title(f"Traj {tid} | TW={mt:.3f} | Acc={acc:.0f}% | {len(seg)} ep",fontsize=11,fontweight="bold")
    # Inset
    ir=R50M*2.5
    axi=inset_axes(ax,width="28%",height="28%",loc="lower left",bbox_to_anchor=(0.02,0.02,1,1),bbox_transform=ax.transAxes)
    axi.set_xlim(mtx-ir,mtx+ir); axi.set_ylim(mty-ir,mty+ir); axi.set_aspect("equal")
    for r in all_roads:
        rmx,rmx2=float(r[:,0].min()),float(r[:,0].max())
        rmy,rmy2=float(r[:,1].min()),float(r[:,1].max())
        if rmx<=mtx+ir and rmx2>=mtx-ir and rmy<=mty+ir and rmy2>=mty-ir:
            axi.plot(r[:,0],r[:,1],color="#555555",lw=1.2,alpha=0.6)
    axi.scatter(xs,ys,c=tws,cmap="RdYlGn",norm=norm,s=12,zorder=5,edgecolors="none")
    axi.scatter(mtx,mty,s=30,c="none",marker="o",edgecolors="#FFD700",linewidth=2,zorder=6)
    axi.set_xticks([]); axi.set_yticks([])
    ml=np.mean([s["tw"] for s in low_seg])
    axi.set_title(f"Min TW={ml:.3f}",fontsize=7,color="#C0392B",pad=1)
    fig.tight_layout()
    fig.savefig(out_path,format="svg"); plt.close(fig)
    print(f"Saved {out_path}")

def plot_tw_curves(out_path):
    """All 7 TW curves end-to-end."""
    TRAJ_IDS=[11,12,13,14,21,22,23]
    COLS=plt.cm.tab10(np.linspace(0,1,7))
    fig,ax=plt.subplots(figsize=(16,4))
    offset=0; boundaries=[0]
    for i,tid in enumerate(TRAJ_IDS):
        seg=traj_data.get(str(tid),[])
        if not seg: continue
        n=len(seg); cs=np.arange(n)+offset
        tws=np.array([r["tw"] for r in seg])
        ax.plot(cs,tws,color=COLS[i],lw=0.8,alpha=0.85,label=f"T{tid}")
        offset+=n; boundaries.append(offset)
    for i,b in enumerate(boundaries[1:-1]):
        ax.axvline(x=b,color="#CCCCCC",ls=":",lw=0.5,alpha=0.6)
        mid=(boundaries[i]+boundaries[i+1])/2
        ax.text(mid,1.02,f"T{TRAJ_IDS[i]}",ha="center",va="bottom",fontsize=6,color=COLS[i],transform=ax.get_xaxis_transform())
    ax.axhline(y=0.5,color="gray",ls="--",lw=0.8)
    ax.set_ylabel("Trustworthiness"); ax.set_xlabel("Cumulative epoch index")
    ax.set_ylim(-0.02,1.08); ax.set_xlim(0,boundaries[-1])
    ax.set_title("All 7 Trajectories — Trustworthiness Timeline (end-to-end)",fontsize=11,fontweight="bold")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path,format="svg"); plt.close(fig)
    print(f"Saved {out_path}")

# ── Generate all 8 SVGs ──
for tid in [11,12,13,14,21,22,23]:
    seg=traj_data.get(str(tid),[])
    plot_one_map(seg, str(tid), OUT_DIR / f"traj{tid}_tw_map.svg")

plot_tw_curves(OUT_DIR / "all_tw_curves.svg")
print("\nDone. All SVGs in:", OUT_DIR)
