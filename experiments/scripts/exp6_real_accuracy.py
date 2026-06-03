#!/usr/bin/env python3
"""Exp6: Real-vehicle matching accuracy — CMM vs FMM on all 7 trajectories."""
import csv, json, math, re
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DPI=200
COLOR_CMM="#2166ac"; COLOR_FMM="#b2182b"
plt.rcParams.update({"font.size":8,"axes.labelsize":9,"axes.titlesize":10,
    "legend.fontsize":7,"xtick.labelsize":7,"ytick.labelsize":7,
    "figure.dpi":DPI,"savefig.dpi":DPI,"savefig.bbox":"tight"})

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data/real_data"
OUT=ROOT/"output/exp6_real"; OUT.mkdir(parents=True,exist_ok=True)

REV=json.load(open(ROOT/"config/reverse_edge_map.json"))
def em(m,t): return str(m)==str(t) or REV.get(str(m))==str(t)
def hdist(lon1,lat1,lon2,lat2):
    mlat=math.radians((lat1+lat2)/2)
    dx=(lon1-lon2)*111320*math.cos(mlat); dy=(lat1-lat2)*111320
    return math.sqrt(dx*dx+dy*dy)
def tf(v):
    try: return float(v)
    except: return None

def ece(confs,labels,n=10):
    N=len(confs); e=0.0; m=0.0; bins=[]
    for i in range(n):
        lo,hi=i/n,(i+1)/n; mask=(confs>=lo)&(confs<hi)
        if i==n-1: mask=(confs>=lo)&(confs<=hi)
        idx=np.where(mask)[0]; nb=len(idx)
        if nb==0: continue
        mc=float(np.mean(confs[idx])); acc=float(np.mean(labels[idx]))
        gap=abs(mc-acc); e+=nb/N*gap; m=max(m,gap)
        bins.append({"n":nb,"mean_conf":mc,"accuracy":acc,"lo":lo,"hi":hi})
    return e,m,bins

def auc(labels,scores):
    order=np.argsort(scores)[::-1]; ls=labels[order]
    n_pos=np.sum(ls); n_neg=len(ls)-n_pos
    if n_pos==0 or n_neg==0: return 0.5,None,None
    tpr=np.cumsum(ls)/n_pos; fpr=np.cumsum(1-ls)/n_neg
    tpr=np.concatenate([[0],tpr]); fpr=np.concatenate([[0],fpr])
    return float(np.trapz(tpr,fpr)),fpr,tpr

# Load GT
gt_edges={}
with open(DATA/"ground_truth.csv",newline="") as f:
    for row in csv.DictReader(f,delimiter=";"):
        gt_edges[(row["id"].strip(),int(row["seq"]))]=row["edge_id"].strip()

# Load RTK positions
rtk_pos={}
with open(DATA/"ground_truth_points.csv",newline="") as f:
    for row in csv.DictReader(f,delimiter=";"):
        ts=str(round(float(row["timestamp"])))
        rtk_pos[(row["id"].strip(),ts)]=(float(row["x"]),float(row["y"]))

# Process aligned data (all trajectories)
cmm_errs,cmm_corr,cmm_tws,cmm_lbls=[],[],[],[]
fmm_errs,fmm_corr,fmm_tws,fmm_lbls=[],[],[],[]
per_traj=defaultdict(lambda:{"cmm_ok":0,"cmm_n":0,"cmm_errs":[],"fmm_ok":0,"fmm_n":0,"fmm_errs":[]})

with open(DATA/"aligned.csv",newline="") as f:
    for row in csv.DictReader(f,delimiter=";"):
        tid=row["id"].strip(); useq=int(row["uni_seq"]); ts_norm=str(round(float(row["timestamp"])))
        if (tid,useq) not in gt_edges: continue
        gt_eid=gt_edges[(tid,useq)]
        if gt_eid in ("0","-1"): continue  # no-road or not-moved, exclude

        # CMM
        cmm_x=row.get("cmm_x",""); cmm_tw=tf(row.get("cmm_tw","0"))
        cmm_cpath=row.get("cmm_cpath","").strip()
        if cmm_x:
            correct=em(cmm_cpath,gt_eid)
            cmm_corr.append(1.0 if correct else 0.0); cmm_tws.append(cmm_tw or 0.5)
            cmm_lbls.append(1.0 if correct else 0.0)
            per_traj[tid]["cmm_ok"]+=int(correct); per_traj[tid]["cmm_n"]+=1
            if (tid,ts_norm) in rtk_pos:
                rtk_lon,rtk_lat=rtk_pos[(tid,ts_norm)]
                cmm_errs.append(hdist(float(cmm_x),float(row["cmm_y"]),rtk_lon,rtk_lat))
                per_traj[tid]["cmm_errs"].append(cmm_errs[-1])

        # FMM
        fmm_x=row.get("fmm_x",""); fmm_tw=tf(row.get("fmm_tw","0"))
        fmm_cpath=row.get("fmm_cpath","").strip()
        if fmm_x:
            correct=em(fmm_cpath,gt_eid)
            fmm_corr.append(1.0 if correct else 0.0); fmm_tws.append(fmm_tw or 0.5)
            fmm_lbls.append(1.0 if correct else 0.0)
            per_traj[tid]["fmm_ok"]+=int(correct); per_traj[tid]["fmm_n"]+=1
            if (tid,ts_norm) in rtk_pos:
                rtk_lon,rtk_lat=rtk_pos[(tid,ts_norm)]
                fmm_errs.append(hdist(float(fmm_x),float(row["fmm_y"]),rtk_lon,rtk_lat))
                per_traj[tid]["fmm_errs"].append(fmm_errs[-1])

# Compute metrics
cL=np.array(cmm_lbls); fL=np.array(fmm_lbls)
cT=np.array(cmm_tws); fT=np.array(fmm_tws)
cE=np.array(cmm_errs); fE=np.array(fmm_errs)

cmm_ece,cmm_mce,cmm_bins=ece(cT,cL)
fmm_ece,fmm_mce,fmm_bins=ece(fT,fL)
cmm_auc_val,cmm_fpr,cmm_tpr=auc(cL,cT)
fmm_auc_val,fmm_fpr,fmm_tpr=auc(fL,fT)

print(f"{'='*60}")
print(f"Real-Vehicle: All 7 Trajectories")
print(f"{'='*60}")
print(f"  Edge accuracy:  CMM={np.mean(cL)*100:.1f}%  FMM={np.mean(fL)*100:.1f}%")
print(f"  ECE:            CMM={cmm_ece:.4f}       FMM={fmm_ece:.4f}")
print(f"  AUC:            CMM={cmm_auc_val:.3f}       FMM={fmm_auc_val:.3f}")
print(f"  Pos err (m):    CMM mean={np.mean(cE):.1f} P95={np.percentile(cE,95):.1f}")
print(f"                  FMM mean={np.mean(fE):.1f} P95={np.percentile(fE,95):.1f}")
print(f"  Eval epochs:    {len(cL)}")

# Per-trajectory table
print(f"\n{'Traj':>5} {'Eval':>6} {'CMM%':>7} {'FMM%':>7}")
for tid in sorted(per_traj.keys(),key=int):
    s=per_traj[tid]
    print(f"{tid:>5} {s['cmm_n']:>6} {s['cmm_ok']/s['cmm_n']*100:>6.1f}% {s['fmm_ok']/s['fmm_n']*100:>6.1f}%")

# Plot
fig,axes=plt.subplots(2,3,figsize=(14,8))
ax=axes[0,0]
tids=sorted(per_traj.keys(),key=int)
x=np.arange(len(tids)); w=0.35
cmm_accs=[per_traj[t]["cmm_ok"]/per_traj[t]["cmm_n"]*100 for t in tids]
fmm_accs=[per_traj[t]["fmm_ok"]/per_traj[t]["fmm_n"]*100 for t in tids]
ax.bar(x-w/2,cmm_accs,w,color=COLOR_CMM,label="CMM",edgecolor="white",lw=0.5)
ax.bar(x+w/2,fmm_accs,w,color=COLOR_FMM,label="FMM",edgecolor="white",lw=0.5)
for i,(cv,fv) in enumerate(zip(cmm_accs,fmm_accs)):
    ax.text(i-w/2,cv+1,f"{cv:.0f}",ha="center",fontsize=6,fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(tids); ax.set_ylabel("Accuracy (%)")
ax.set_title(f"(a) Segment Accuracy (n={len(cL)})"); ax.legend(); ax.grid(alpha=0.3,axis="y")
ax.set_ylim(0,105)

ax=axes[0,1]; bins=np.linspace(0,min(np.percentile(np.concatenate([cE,fE]),98),50),40)
ax.hist(cE,bins=bins,alpha=0.5,color=COLOR_CMM,label=f"CMM (μ={np.mean(cE):.1f}m)")
ax.hist(fE,bins=bins,alpha=0.5,color=COLOR_FMM,label=f"FMM (μ={np.mean(fE):.1f}m)")
ax.set_xlabel("Position error (m)"); ax.set_ylabel("Count")
ax.set_title("(b) Position Error"); ax.legend(); ax.grid(alpha=0.3)

ax=axes[0,2]
ax.bar(["CMM","FMM"],[cmm_ece,fmm_ece],color=[COLOR_CMM,COLOR_FMM],edgecolor="white",lw=0.5)
for i,v in enumerate([cmm_ece,fmm_ece]): ax.text(i,v+0.01,f"{v:.3f}",ha="center",fontweight="bold")
ax.set_ylabel("ECE"); ax.set_title("(c) ECE"); ax.grid(alpha=0.3,axis="y")

ax=axes[1,0]; ax.plot([0,1],[0,1],"k--",lw=0.8)
for label,bins,color in [("CMM",cmm_bins,COLOR_CMM),("FMM",fmm_bins,COLOR_FMM)]:
    confs=[b["mean_conf"] for b in bins if b["n"]>0]; accs=[b["accuracy"] for b in bins if b["n"]>0]
    if confs: ax.plot(confs,accs,"o-",color=color,lw=1.2,ms=5,label=f"{label} (ECE={cmm_ece if label=='CMM' else fmm_ece:.3f})")
ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy"); ax.set_title("(d) Reliability"); ax.legend(); ax.grid(alpha=0.3)

ax=axes[1,1]; ax.plot([0,1],[0,1],"k--",lw=0.8)
if cmm_fpr is not None: ax.plot(cmm_fpr,cmm_tpr,color=COLOR_CMM,lw=1.5,label=f"CMM AUC={cmm_auc_val:.3f}")
if fmm_fpr is not None: ax.plot(fmm_fpr,fmm_tpr,color=COLOR_FMM,lw=1.5,label=f"FMM AUC={fmm_auc_val:.3f}")
ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("(e) ROC"); ax.legend(); ax.grid(alpha=0.3)

ax=axes[1,2]
for errs,color,label in [(cE,COLOR_CMM,"CMM"),(fE,COLOR_FMM,"FMM")]:
    se=np.sort(errs); cdf=np.arange(1,len(se)+1)/len(se)
    ax.plot(se,cdf,color=color,lw=1.5,label=label)
ax.axhline(0.5,color="gray",lw=0.8,ls="--"); ax.axhline(0.95,color="gray",lw=0.8,ls="--")
ax.set_xlabel("Position error (m)"); ax.set_ylabel("CDF"); ax.set_title("(f) Error CDF"); ax.legend(); ax.grid(alpha=0.3)
ax.set_xlim(0,np.percentile(np.concatenate([cE,fE]),99))

fig.suptitle("Real-Vehicle: CMM vs FMM (7 Trajectories, k=16)",fontsize=11,fontweight="bold")
fig.tight_layout(); fig.savefig(OUT/"all_traj_accuracy.png",dpi=DPI); plt.close()
print(f"\nSaved: {OUT}/all_traj_accuracy.png")
