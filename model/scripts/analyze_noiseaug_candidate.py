"""Evaluate AB train-noise augmentation against fixed DI on clean and low-SNR.

The candidate is a new registered arm (AB only gets 10 dB training noise). The
outer test has already been observed elsewhere, so all outcomes are exploratory
and no retry/search is performed after this report.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "results" / "studies"
CAND = "physics3_designB_branch_noiseaug10_v1"
BASE = "noise_designB_branch_v1"
PRIMARY = "ab_emstgat"
OPP = "di_emstgat"

def load(name):
    return json.loads((S/name/"per_seed_records.json").read_text(encoding="utf-8"))

def pair(a,b):
    seeds=sorted(set(a)&set(b)); x=np.array([a[s] for s in seeds]); y=np.array([b[s] for s in seeds]); d=(x-y)*100
    return {"seeds":seeds,"delta_pp":[round(float(v),4) for v in d],"mean_pp":round(float(d.mean()),4),"sd_pp":round(float(d.std(ddof=1)),4),"paired_t_p":float(stats.ttest_rel(x,y).pvalue),"wilcoxon_p":float(stats.wilcoxon(x,y).pvalue),"wins":f"{int((d>0).sum())}/{len(d)}"}

def main():
    c=load(CAND); n=load(BASE)
    cv={int(r['seed']):float(r['accuracy']) for r in c if r['model']==PRIMARY}
    dv={int(r['seed']):float(r['accuracy']) for r in c if r['model']==OPP}
    print('clean candidate records',len(c),'seeds',sorted({r['seed'] for r in c}))
    print('AB clean',round(np.mean(list(cv.values()))*100,4),'DI clean',round(np.mean(list(dv.values()))*100,4),'AB-DI',pair(cv,dv))
    # Candidate main run is clean. To assess low-SNR, the noise evaluator needs
    # frozen candidate models; this runner does not save weights. Therefore do
    # not fabricate a noise result here. Report the honest blocking fact.
    print('low-SNR frozen candidate evaluation: NOT YET RUN (main runner does not preserve fitted weights)')
    print('fixed baseline low-SNR values from noise_designB_branch_v1 are retained; candidate needs a frozen-checkpoint noise runner')
    out={'candidate_run':CAND,'control_noise_run':BASE,'clean_ab_vs_di':pair(cv,dv),'low_snr_status':'blocked_pending_frozen_candidate_checkpoints','note':'Do not infer low-SNR behavior from clean test result.'}
    dest=S/CAND/'noiseaug_candidate_statistics.json'; dest.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8'); print('Saved:',dest)
if __name__=='__main__': main()
