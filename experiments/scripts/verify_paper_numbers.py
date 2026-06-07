#!/usr/bin/env python3
"""
Verification script: compare CMM/FMM output against paper claims.
Matches by TIMESTAMP, uses reverse edge map, excludes gt_edge ∈ {0, -1, ''}.

Usage:
    python experiments/scripts/verify_paper_numbers.py

Outputs a comprehensive comparison of all metrics against paper claims.
"""

import csv, json, math, numpy as np
from pathlib import Path
from collections import defaultdict
import re

BASE = Path(__file__).resolve().parents[2]
REV_MAP = json.load(open(BASE / 'experiments/config/reverse_edge_map.json'))

# Paper claims (from LaTeX as of 2026-06-07)
PAPER = {
    'accuracy': {
        '11': (2377, 93.8, 87.1), '12': (133, 100, 0), '13': (1908, 98.1, 94.2),
        '14': (352, 100, 79.5), '21': (3516, 98.7, 92.1), '22': (2062, 93.4, 83.3),
        '23': (2704, 98.6, 88.3), 'ALL': (13052, 96.9, 88.0),
    },
    'tw_sep': {'cmm_corr': 0.925, 'cmm_wrong': 0.633, 'cmm_sep': 0.292,
               'fmm_corr': 0.996, 'fmm_wrong': 0.909, 'fmm_sep': 0.087, 'ratio': 3.3},
    'ece': {'cmm': 0.069, 'fmm': 0.107},
    'auc': {'cmm': 0.600, 'fmm': 0.965},
    'pos_err': {'cmm_mean': 5.6, 'cmm_p95': 13.5, 'fmm_mean': 9.4, 'fmm_p95': 40.3},
    'fmr': {'cmm': 3.1, 'fmm': 12.0},
}

def is_edge_match(m, g):
    m=m.strip(); g=g.strip()
    if m==g: return True
    if REV_MAP.get(m,'')==g: return True
    if REV_MAP.get(g,'')==m: return True
    return False

def parse_point(wkt):
    m=re.match(r'POINT\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)',str(wkt))
    return (float(m.group(1)),float(m.group(2))) if m else (None,None)

def haversine_m(lon1,lat1,lon2,lat2):
    mlat=math.radians((lat1+lat2)/2)
    return math.hypot((lon1-lon2)*111320*math.cos(mlat),(lat1-lat2)*111320)

def ece(confs,corrects,n=10):
    e=0.0;N=len(confs)
    for i in range(n):
        lo,hi=i/n,(i+1)/n
        mask=(confs>=lo)&(confs<hi) if i<n-1 else (confs>=lo)&(confs<=hi)
        nb=mask.sum()
        if nb==0:continue
        e+=nb/N*abs(confs[mask].mean()-corrects[mask].mean())
    return e

def auc(labels,scores):
    order=np.argsort(scores)[::-1];ls=labels[order]
    n_pos=ls.sum();n_neg=len(ls)-n_pos
    if n_pos==0 or n_neg==0:return 0.5
    return float(np.trapz(np.cumsum(ls)/n_pos,np.cumsum(1-ls)/n_neg))

def main():
    # Load GT by timestamp
    gt_by_ts = {}
    with open(BASE/'experiments/data/real_data/aligned.csv',newline='') as f:
        for r in csv.DictReader(f,delimiter=';'):
            gt_e = r['gt_edge'].strip()
            if gt_e in ('0','-1',''): continue
            try: gx=float(r['gt_x']); gy=float(r['gt_y'])
            except: continue
            gt_by_ts[r['timestamp'].strip()] = (r['id'].strip(), gt_e, gx, gy)

    # Load CMM/FMM by timestamp
    def load_mr(path):
        d = {}
        with open(path,newline='') as f:
            for r in csv.DictReader(f,delimiter=';'):
                d[r['timestamp'].strip()] = (
                    r['cpath'].strip(), float(r['trustworthiness']),
                    r.get('ogeom',''), r.get('pgeom','')
                )
        return d

    cmm = load_mr(BASE/'experiments/data/real_data/cmm_result.csv')
    fmm = load_mr(BASE/'experiments/data/real_data/fmm_result.csv')

    # Evaluate
    cmm_pt = defaultdict(lambda:{'c':0,'t':0,'perr':[]})
    fmm_pt = defaultdict(lambda:{'c':0,'t':0,'perr':[]})
    cmm_tw_c=[]; cmm_tw_w=[]
    fmm_tw_c=[]; fmm_tw_w=[]

    for ts,(tid,gt_e,gx,gy) in gt_by_ts.items():
        if ts in cmm:
            cpath,tw,ogeom,pgeom = cmm[ts]
            cmm_pt[tid]['t'] += 1
            if is_edge_match(cpath,gt_e): cmm_pt[tid]['c']+=1; cmm_tw_c.append(tw)
            else: cmm_tw_w.append(tw)
            px,py = parse_point(pgeom) if pgeom else (None,None)
            if px is not None: cmm_pt[tid]['perr'].append(haversine_m(px,py,gx,gy))

        if ts in fmm:
            fpath,ftw,_,fpgeom = fmm[ts]
            fmm_pt[tid]['t'] += 1
            if is_edge_match(fpath,gt_e): fmm_pt[tid]['c']+=1; fmm_tw_c.append(ftw)
            else: fmm_tw_w.append(ftw)
            px,py = parse_point(fpgeom) if fpgeom else (None,None)
            if px is not None: fmm_pt[tid]['perr'].append(haversine_m(px,py,gx,gy))

    cmm_tw_c=np.array(cmm_tw_c); cmm_tw_w=np.array(cmm_tw_w)
    fmm_tw_c=np.array(fmm_tw_c); fmm_tw_w=np.array(fmm_tw_w)

    # Print comparison
    print("="*80)
    print("  PAPER NUMBER VERIFICATION")
    print("="*80)
    print(f"\n{'Metric':<40} {'Paper':>12} {'Computed':>12} {'Δ':>10} {'Status':>10}")
    print("-"*88)

    tc={'cc':0,'ct':0,'fc':0,'ft':0}
    for tid in sorted(cmm_pt.keys()):
        cd=cmm_pt[tid]; fd=fmm_pt.get(tid,{'c':0,'t':0})
        tc['cc']+=cd['c']; tc['ct']+=cd['t']; tc['fc']+=fd['c']; tc['ft']+=fd['t']

    for tid in sorted(cmm_pt.keys()):
        cd=cmm_pt[tid]; pn,pc,pf = PAPER['accuracy'].get(tid,(0,0,0))
        ca=cd['c']/cd['t']*100 if cd['t']>0 else 0
        check(f"Traj{tid} CMM Acc (%)", pc, ca)
        fa=fmm_pt[tid]['c']/fmm_pt[tid]['t']*100 if fmm_pt[tid]['t']>0 else 0
        check(f"Traj{tid} FMM Acc (%)", pf, fa)

    ca_all=tc['cc']/tc['ct']*100; fa_all=tc['fc']/tc['ft']*100
    check("CMM Accuracy (%)", PAPER['accuracy']['ALL'][1], ca_all)
    check("FMM Accuracy (%)", PAPER['accuracy']['ALL'][2], fa_all)
    check("Eval epochs", PAPER['accuracy']['ALL'][0], tc['ct'])

    sep_c=cmm_tw_c.mean()-cmm_tw_w.mean(); sep_f=fmm_tw_c.mean()-fmm_tw_w.mean()
    check("CMM TW corr mean", PAPER['tw_sep']['cmm_corr'], cmm_tw_c.mean())
    check("CMM TW wrong mean", PAPER['tw_sep']['cmm_wrong'], cmm_tw_w.mean())
    check("CMM TW separation", PAPER['tw_sep']['cmm_sep'], sep_c)
    check("FMM TW separation", PAPER['tw_sep']['fmm_sep'], sep_f)
    check("TW Ratio", PAPER['tw_sep']['ratio'], sep_c/sep_f if sep_f>0 else 0)

    c_corr=np.concatenate([np.ones(len(cmm_tw_c)),np.zeros(len(cmm_tw_w))])
    c_all=np.concatenate([cmm_tw_c,cmm_tw_w])
    f_corr=np.concatenate([np.ones(len(fmm_tw_c)),np.zeros(len(fmm_tw_w))])
    f_all=np.concatenate([fmm_tw_c,fmm_tw_w])
    check("CMM ECE", PAPER['ece']['cmm'], ece(c_all,c_corr))
    check("FMM ECE", PAPER['ece']['fmm'], ece(f_all,f_corr))
    check("CMM AUC", PAPER['auc']['cmm'], auc(c_corr,c_all))
    check("FMM AUC", PAPER['auc']['fmm'], auc(f_corr,f_all))

    all_cperr=np.concatenate([np.array(d['perr']) for d in cmm_pt.values() if d['perr']])
    all_fperr=np.concatenate([np.array(d['perr']) for d in fmm_pt.values() if d['perr']])
    check("CMM pos err mean", PAPER['pos_err']['cmm_mean'], all_cperr.mean())
    check("CMM pos err P95", PAPER['pos_err']['cmm_p95'], np.percentile(all_cperr,95))
    check("FMM pos err mean", PAPER['pos_err']['fmm_mean'], all_fperr.mean())
    check("FMM pos err P95", PAPER['pos_err']['fmm_p95'], np.percentile(all_fperr,95))

    fmr_c=len(cmm_tw_w)/(len(cmm_tw_c)+len(cmm_tw_w))*100
    check("CMM FMR (%)", PAPER['fmr']['cmm'], fmr_c)

def check(name, paper_v, comp_v):
    delta = comp_v - paper_v
    if abs(delta) < 0.005: status = "✅ EXACT"
    elif abs(delta) < 0.02: status = "✅ OK"
    elif abs(delta) < 0.5: status = "⚠️ REVIEW"
    else: status = "❌ UPDATE"
    print(f"{name:<40} {paper_v:>12.4f} {comp_v:>12.4f} {delta:>+10.4f} {status:>10}")

if __name__ == '__main__':
    main()
